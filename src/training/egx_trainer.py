"""
src/training/egx_trainer.py
────────────────────────────
EGX-Powered Trainer — rewritten for U-Net VAE autoencoder.

Key upgrades over previous version:
  ✓ Mixed precision (FP16) via torch.cuda.amp — cuts VRAM ~40%, +30% speed
  ✓ Gradient accumulation (GRAD_ACCUM_STEPS=4) — effective batch=32 on 4GB GPU
  ✓ Calls loss.forward_verbose() — logs MSE, SSIM, KL, β every batch
  ✓ Calls loss.set_epoch() — KL beta annealing
  ✓ Handles new model signature: x_hat, z, mu, logvar = model(x)
  ✓ Reconstruction sharpness metric (Laplacian variance) logged per epoch
  ✓ GPU memory printed every N batches
  ✓ AdamW optimizer (better weight decay than Adam)
  ✓ Cosine annealing LR with warm restarts
"""

import os
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.training.loss import CombinedLoss, get_loss_fn
from src.training.callbacks import ModelCheckpoint, LRSchedulerCallback

# ── EGX imports (graceful fallback if not installed) ─────────────────────────
try:
    from egx.api.callbacks import (
        TrainingCallback, CallbackHandler,
        NaNDetectionCallback, EarlyStoppingCallback as EGXES,
    )
    _EGX_AVAILABLE = True
except ImportError:
    _EGX_AVAILABLE = False
    # Minimal stubs so the rest of the file works without EGX
    class TrainingCallback:
        def on_train_begin(self, **kw): pass
        def on_epoch_begin(self, **kw): pass
        def on_epoch_end(self, **kw):   pass
        def on_evaluate_end(self, **kw):pass
        def on_train_end(self, **kw):   pass

    class CallbackHandler:
        def __init__(self, cbs): self.cbs = list(cbs)
        def add(self, cb):       self.cbs.append(cb)
        def fire(self, event, **kw):
            for cb in self.cbs:
                getattr(cb, event, lambda **k: None)(**kw)

    class NaNDetectionCallback(TrainingCallback): pass

    class EGXES(TrainingCallback):
        should_stop = False
        best_value  = float("inf")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_mem() -> str:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved()  / 1e9
        return f"{a:.2f}/{r:.2f}GB"
    return "CPU"


def _sharpness(imgs: torch.Tensor) -> float:
    """
    Laplacian variance of reconstructions.
    Higher = sharper. Use to track reconstruction quality numerically.
    No need to open PNG files — watch this number climb each epoch.
    """
    lap = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=imgs.dtype, device=imgs.device
    ).view(1, 1, 3, 3)
    out = nn.functional.conv2d(
        imgs[:4].mean(1, keepdim=True).clamp(-1, 1), lap, padding=1
    )
    return out.var().item()


SEP  = "─" * 85
SEP2 = "═" * 85


# ── Logging callback ──────────────────────────────────────────────────────────

class AutoencoderLoggingCallback(TrainingCallback):
    def __init__(self):
        self._epoch_start = 0.0
        self._train_start = 0.0

    def on_train_begin(self, trainer, **kw):
        self._train_start = time.time()

    def on_epoch_begin(self, trainer, epoch, **kw):
        self._epoch_start = time.time()

    def on_epoch_end(self, trainer, epoch, metrics, **kw):
        pass   # detailed print is done inside fit()

    def on_train_end(self, trainer, result, **kw):
        total = time.time() - self._train_start
        print(f"\n  EGX Training Complete — {total/60:.1f} min total")


# ── Checkpoint callback ───────────────────────────────────────────────────────

class ReconstructionCheckpointCallback(TrainingCallback):
    def __init__(self, save_path: str = config.BEST_MODEL_PATH):
        self.save_path  = save_path
        self.best_score: Optional[float] = None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_evaluate_end(self, trainer, metrics, **kw):
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        if self.best_score is None or val_loss < self.best_score:
            prev            = self.best_score
            self.best_score = val_loss
            model = getattr(trainer, "_model", None)
            if model is not None:
                torch.save({
                    "state_dict": model.state_dict(),
                    "val_loss":   val_loss,
                    "latent_dim": getattr(model, "latent_dim", config.LATENT_DIM),
                    "metrics":    metrics,
                }, self.save_path)
                config.save_snapshot()
                prev_str = "inf" if prev is None else f"{prev:.6f}"
                print(f"  [Checkpoint] val_loss {prev_str} → {val_loss:.6f}  "
                      f"✔ saved")


# ── LR scheduler callback ─────────────────────────────────────────────────────

class LRSchedulerEGXCallback(TrainingCallback):
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def on_evaluate_end(self, trainer, metrics, **kw):
        val_loss = metrics.get("eval_loss")
        if val_loss is not None:
            self._scheduler.step(val_loss)


# ── Main EGX Trainer ──────────────────────────────────────────────────────────

class EGXAutoencoderTrainer:
    """
    Production-grade autoencoder trainer.
    Handles the new model signature: x_hat, z, mu, logvar = model(x)
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, loss_name: str = None):

        self.device       = get_device()
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # Loss — always CombinedLoss with KL support
        self.loss_fn: CombinedLoss = get_loss_fn(loss_name)

        # AdamW — better generalisation than Adam via decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with warm restarts
        # T_0 = restart every 15 epochs; T_mult=2 doubles period each restart
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15, T_mult=2,
            eta_min=config.LEARNING_RATE * 0.01,
        )

        # Mixed precision scaler (FP16 on CUDA, disabled on CPU)
        self._use_amp = torch.cuda.is_available()
        self.scaler   = GradScaler(enabled=self._use_amp)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": [],
            "train_acc":  [], "val_acc":  [],
        }

        # EGX callbacks
        self._checkpoint_cb = ReconstructionCheckpointCallback()
        self._log_cb        = AutoencoderLoggingCallback()
        self._lr_cb         = LRSchedulerEGXCallback(self.scheduler)

        print("[EGX] Trainer initialised")
        print(f"[EGX] Mixed precision (FP16): {self._use_amp}")
        print(f"[EGX] Grad accumulation steps: {config.GRAD_ACCUM_STEPS}")
        print(f"[EGX] Effective batch size: "
              f"{config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader,
                   training: bool, epoch: int = 0) -> Tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        total_acc  = 0.0
        num_batches = len(loader)
        total_samples = len(loader.dataset)
        samples_done  = 0
        batch_times   = []
        phase = "TRAIN" if training else "VALID"

        print(f"\n  [{phase}] {num_batches} batches | "
              f"{total_samples} samples | GPU: {_gpu_mem()}")
        print(f"  {SEP}")

        ctx = torch.enable_grad() if training else torch.no_grad()
        t0  = time.time()

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with ctx:
            for i, (images, _) in enumerate(loader, 1):
                bt = time.time()
                images = images.to(self.device, non_blocking=True)
                samples_done += images.size(0)

                with autocast(enabled=self._use_amp):
                    x_hat, z, mu, logvar = self.model(images)
                    components = self.loss_fn.forward_verbose(
                        x_hat, images, mu, logvar
                    )
                    # Scale loss by accum steps so effective LR stays constant
                    loss = components["loss"] / config.GRAD_ACCUM_STEPS

                if training:
                    self.scaler.scale(loss).backward()

                    # Step only every GRAD_ACCUM_STEPS batches
                    if i % config.GRAD_ACCUM_STEPS == 0 or i == num_batches:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        grad_norm = 0.0
                else:
                    grad_norm = 0.0

                # Unscaled loss for logging
                raw_loss = components["loss"].item()
                total_loss += raw_loss

                x_hat_01  = (x_hat.detach() + 1) / 2
                images_01  = (images + 1) / 2
                batch_acc  = 100.0 * (
                    1.0 - torch.mean(torch.abs(x_hat_01 - images_01)).item()
                )
                total_acc += batch_acc

                batch_times.append(time.time() - bt)
                avg_bt   = sum(batch_times) / len(batch_times)
                eta_min  = (num_batches - i) * avg_bt / 60

                # ── Per-batch print ──────────────────────────────────────────
                print_every = 5 if training else 10
                if i % print_every == 0 or i == num_batches:
                    pct = i / num_batches * 100
                    if training:
                        print(
                            f"  Batch {i:>4}/{num_batches} [{pct:5.1f}%] | "
                            f"Loss={raw_loss:.5f} | "
                            f"MSE={components['mse']:.5f} | "
                            f"SSIM={components['ssim']:.5f} | "
                            f"KL={components['kl']:.5f} | "
                            f"β={components['beta']:.5f} | "
                            f"‖∇‖={grad_norm:.3f} | "
                            f"Acc={batch_acc:.1f}% | "
                            f"Imgs={samples_done}/{total_samples} | "
                            f"ETA={eta_min:.1f}m | "
                            f"GPU={_gpu_mem()}"
                        )
                    else:
                        print(
                            f"  Val {i:>4}/{num_batches} [{pct:5.1f}%] | "
                            f"Loss={raw_loss:.5f} | "
                            f"MSE={components['mse']:.5f} | "
                            f"SSIM={components['ssim']:.5f} | "
                            f"KL={components['kl']:.5f} | "
                            f"Acc={batch_acc:.1f}%"
                        )

        avg_loss = total_loss / num_batches
        avg_acc  = total_acc  / num_batches
        elapsed  = time.time() - t0
        print(f"  {SEP}")
        print(f"  [{phase}] avg_loss={avg_loss:.5f} | "
              f"avg_acc={avg_acc:.2f}% | time={elapsed:.1f}s | "
              f"avg_batch={sum(batch_times)/len(batch_times)*1000:.0f}ms")

        return avg_loss, avg_acc

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(self) -> Dict[str, List[float]]:
        # Build callback stack
        callbacks = [self._log_cb, self._checkpoint_cb, self._lr_cb]
        if _EGX_AVAILABLE:
            callbacks.append(NaNDetectionCallback(halt_on_nan=False,
                                                   max_nan_count=10))
        cb = CallbackHandler(callbacks)

        es_cb = EGXES(
            patience=config.EARLY_STOP_PATIENCE,
            min_delta=1e-4,
            metric_name="eval_loss",
            greater_is_better=False,
        ) if _EGX_AVAILABLE else EGXES()
        cb.add(es_cb)

        self._model = self.model   # so checkpoint cb can access it
        cb.fire("on_train_begin", trainer=self)

        n_train   = len(self.train_loader.dataset)
        n_val     = len(self.val_loader.dataset)
        n_params  = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)

        print("\n" + SEP2)
        print("  EGX POWERED — Pulmonary Anomaly Detection (U-Net VAE)")
        print(SEP2)
        print(f"  Device          : {self.device}")
        print(f"  GPU memory      : {_gpu_mem()}")
        print(f"  Loss            : {config.LOSS_FUNCTION}  "
              f"(α={config.LOSS_ALPHA}, β_max={config.KL_BETA_MAX})")
        print(f"  Epochs          : {config.EPOCHS}")
        print(f"  Batch size      : {config.BATCH_SIZE}  "
              f"(effective: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS})")
        print(f"  LR              : {config.LEARNING_RATE}  "
              f"(cosine warm-restarts, T0=15)")
        print(f"  Mixed precision : {self._use_amp}")
        print(f"  Train samples   : {n_train}  ({len(self.train_loader)} batches)")
        print(f"  Val samples     : {n_val}  ({len(self.val_loader)} batches)")
        print(f"  Parameters      : {n_params:,}  "
              f"(trainable: {trainable:,}  ~{n_params*4/1e6:.1f} MB)")
        print(SEP2 + "\n")

        t_total = time.time()

        for epoch in range(config.EPOCHS):
            ep_start = time.time()

            # Update KL beta annealing
            self.loss_fn.set_epoch(epoch)

            cb.fire("on_epoch_begin", trainer=self, epoch=epoch)

            print(f"\n{SEP2}")
            print(f"  Epoch [{epoch+1:>3}/{config.EPOCHS}]  "
                  f"β={self.loss_fn.beta:.5f}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                  f"GPU={_gpu_mem()}")
            print(SEP2)

            train_loss, train_acc = self._run_epoch(
                self.train_loader, training=True, epoch=epoch
            )
            val_loss, val_acc = self._run_epoch(
                self.val_loader, training=False, epoch=epoch
            )

            # Sharpness on last val batch (reconstructions)
            self.model.eval()
            with torch.no_grad():
                imgs, _ = next(iter(self.val_loader))
                imgs    = imgs.to(self.device)
                with autocast(enabled=self._use_amp):
                    x_hat, _, _, _ = self.model(imgs)
                sharp = _sharpness(x_hat.float())

            lr = self.optimizer.param_groups[0]["lr"]
            ep_time = time.time() - ep_start

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            best_val = min(self.history["val_loss"])
            is_best  = val_loss <= best_val
            marker   = "  ★ BEST" if is_best else ""

            # ── Epoch summary table ──────────────────────────────────────────
            print(f"\n{SEP2}")
            print(f"  EPOCH [{epoch+1:>3}/{config.EPOCHS}] — "
                  f"{ep_time:.1f}s{marker}")
            print(SEP)
            print(f"  {'Metric':<20} {'Train':>12} {'Val':>12}")
            print(f"  {'-'*44}")
            print(f"  {'Total Loss':<20} {train_loss:>12.5f} {val_loss:>12.5f}")
            print(f"  {'Best Val Loss':<20} {'—':>12} {best_val:>12.5f}")
            print(f"  {'Accuracy':<20} {train_acc:>11.2f}% {val_acc:>11.2f}%")
            print(f"  {'Sharpness (↑)':<20} {'—':>12} {sharp:>12.6f}")
            print(f"  {'KL β':<20} {self.loss_fn.beta:>12.5f} {'—':>12}")
            print(f"  {'LR':<20} {lr:>12.2e} {'—':>12}")
            print(f"  {'GPU Memory':<20} {_gpu_mem():>24}")
            print(f"  {'Progress':<20} "
                  f"  {epoch+1}/{config.EPOCHS} "
                  f"({(epoch+1)/config.EPOCHS*100:.1f}%)")
            print(SEP2)

            # Fire callbacks
            epoch_metrics = {
                "train_loss_epoch": train_loss, "val_loss": val_loss,
                "train_acc_epoch":  train_acc,  "val_acc":  val_acc,
                "lr": lr, "epoch": epoch + 1, "sharpness": sharp,
            }
            eval_metrics = {"eval_loss": val_loss, "val_loss": val_loss}

            cb.fire("on_epoch_end",    trainer=self,
                    epoch=epoch, metrics=epoch_metrics)
            cb.fire("on_evaluate_end", trainer=self,
                    metrics=eval_metrics)
            cb.fire("on_step_end",     trainer=self,
                    step=epoch, loss=train_loss, lr=lr)

            # Early stopping check
            if getattr(es_cb, "should_stop", False):
                print(f"\n  [EGX] Early stopping at epoch {epoch+1}  "
                      f"best_val={getattr(es_cb,'best_value',best_val):.6f}")
                break

        # ── Final summary ────────────────────────────────────────────────────
        total_time = time.time() - t_total
        epochs_done = len(self.history["train_loss"])
        best_val    = min(self.history["val_loss"])

        cb.fire("on_train_end", trainer=self,
                result={"final_loss": self.history["train_loss"][-1],
                        "best_val_loss": best_val})

        print(f"\n{SEP2}")
        print("  EGX TRAINING COMPLETED")
        print(SEP)
        print(f"  Total time      : {total_time/60:.1f} min")
        print(f"  Epochs done     : {epochs_done}/{config.EPOCHS}")
        print(f"  Best val loss   : {best_val:.6f}")
        print(f"  Final train loss: {self.history['train_loss'][-1]:.6f}")
        print(f"  Final val loss  : {self.history['val_loss'][-1]:.6f}")
        print(f"  Avg epoch time  : {total_time/epochs_done:.1f}s")
        print(f"  Checkpoint      : {config.BEST_MODEL_PATH}")
        print(SEP2 + "\n")

        return self.history