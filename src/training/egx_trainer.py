"""
src/training/egx_trainer.py
────────────────────────────
EGX-Powered Trainer for Pulmonary Anomaly Detection.

Integrates the EGX (Elastic Guardian X) framework to bring production-grade
training features to the autoencoder pipeline.
"""

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.training.loss import get_loss_fn
from src.training.callbacks import ModelCheckpoint, LRSchedulerCallback

# ── EGX Imports ──────────────────────────────────────────────────────
from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig
from egx.api.callbacks import (
    TrainingCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    NaNDetectionCallback,
    GradientClipCallback,
)


# ══════════════════════════════════════════════════════════════════════
#  Custom EGX Callbacks for Autoencoder Training
# ══════════════════════════════════════════════════════════════════════


class AutoencoderLoggingCallback(TrainingCallback):
    """
    Rich logging callback tailored for autoencoder reconstruction training.
    """

    def __init__(self):
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0

    def on_train_begin(self, trainer, **kwargs):
        self._train_start = time.time()
        print("=" * 55)
        print("  EGX Training Session — Pulmonary Anomaly Detection")
        print("=" * 55)

    def on_epoch_begin(self, trainer, epoch, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        elapsed = time.time() - self._epoch_start
        train_loss = metrics.get("train_loss_epoch", 0.0)
        val_loss = metrics.get("val_loss", None)
        lr = metrics.get("lr", None)

        parts = [f"Epoch [{epoch + 1:03d}/{config.EPOCHS}]"]
        parts.append(f"train={train_loss:.5f}")
        if val_loss is not None:
            parts.append(f"val={val_loss:.5f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"({elapsed:.1f}s)")
        print("  ".join(parts))

    def on_train_end(self, trainer, result, **kwargs):
        total = time.time() - self._train_start
        print("=" * 55)
        print(f"  EGX Training Complete — {total / 60:.1f} min total")
        print(f"  Final loss: {result.get('final_loss', 0):.6f}")
        print("=" * 55)


class ReconstructionCheckpointCallback(TrainingCallback):
    """
    Save best model checkpoint based on validation loss.
    """

    def __init__(self, save_path: str = config.BEST_MODEL_PATH):
        self.save_path = save_path
        self.best_score: Optional[float] = None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_evaluate_end(self, trainer, metrics, **kwargs):
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if self.best_score is None or val_loss < self.best_score:
            prev = self.best_score
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

                # Fixed for Windows console (replaced ∞ with "inf")
                print(
                    f"[EGX Checkpoint] val_loss "
                    f"{'inf' if prev is None else f'{prev:.6f}'} → {val_loss:.6f}"
                )


class LRSchedulerEGXCallback(TrainingCallback):
    """
    Wraps ReduceLROnPlateau into the EGX callback lifecycle.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=1e-6,
        )
        self._optimizer = optimizer
        self._prev_lr = optimizer.param_groups[0]["lr"]

    def on_evaluate_end(self, trainer, metrics, **kwargs):
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        self._scheduler.step(val_loss)
        lr = self._optimizer.param_groups[0]["lr"]
        if lr < self._prev_lr:
            print(f"[EGX LRScheduler] LR {self._prev_lr:.2e} → {lr:.2e}")
            self._prev_lr = lr


# ══════════════════════════════════════════════════════════════════════
#  EGX-Powered Trainer
# ══════════════════════════════════════════════════════════════════════


class EGXAutoencoderTrainer:
    """
    Production-grade autoencoder trainer powered by EGX.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_name: str = None,
    ):
        self.device       = get_device()
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = get_loss_fn(loss_name)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )

        # ── History tracking (compatible with original Trainer) ──
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": [],
        }

        # ── EGX Callbacks ──
        self._checkpoint_cb = ReconstructionCheckpointCallback()
        self._lr_scheduler_cb = LRSchedulerEGXCallback(self.optimizer)
        self._autoencoder_log_cb = AutoencoderLoggingCallback()

        # ── EGX Configuration ──
        self._egx_config = EGXConfig(
            num_epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            max_grad_norm=1.0,
            eval_strategy="epoch",
            early_stopping_patience=config.EARLY_STOP_PATIENCE,
            early_stopping_threshold=1e-4,
            logging_steps=10,
            output_dir=config.CHECKPOINT_DIR,
            checkpoint_strategy="adaptive",
        )

        print("[EGX] Trainer initialized with production-grade features")
        print(f"[EGX] Config: {self._egx_config}")

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        """Run a single train or validation epoch with detailed progress tracking."""
        self.model.train(training)
        total_loss = 0.0
        num_batches = len(loader)
        total_samples = len(loader.dataset)
        samples_processed = 0
        batch_times = []

        phase_name = "Training" if training else "Validation"
        print(f"\n  [EGX] {phase_name} Phase ({num_batches} batches, {total_samples} samples)")

        ctx = torch.enable_grad() if training else torch.no_grad()
        start_time = time.time()

        with ctx:
            for batch_idx, (images, _) in enumerate(loader, 1):
                batch_start = time.time()

                images = images.to(self.device, non_blocking=True)
                batch_size = images.size(0)
                samples_processed += batch_size

                x_hat, _ = self.model(images)
                loss = self.loss_fn(x_hat, images)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Calculate batch metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # Calculate progress metrics
                elapsed_time = time.time() - start_time
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = num_batches - batch_idx
                eta_seconds = remaining_batches * avg_batch_time

                # Show detailed batch progress (less frequent for validation)
                print_interval = 5 if training else 10  # Print every 5 training batches, every 10 validation batches
                if batch_idx % print_interval == 0 or batch_idx == num_batches:
                    progress_pct = (batch_idx / num_batches) * 100
                    if training:
                        print(f"    [EGX] Batch {batch_idx:2d}/{num_batches} "
                              f"[{progress_pct:5.1f}%] | "
                              f"Loss: {batch_loss:.6f} | "
                              f"Images: {samples_processed}/{total_samples} | "
                              f"Time: {batch_time:.3f}s | "
                              f"ETA: {eta_seconds/60:.1f}m")
                    else:
                        print(f"    [EGX] Val Batch {batch_idx:2d}/{num_batches} "
                              f"[{progress_pct:5.1f}%] | "
                              f"Loss: {batch_loss:.6f}")

        avg_loss = total_loss / num_batches
        total_time = time.time() - start_time

        if training:
            print(f"  [EGX] Training Complete - Avg Loss: {avg_loss:.6f} | "
                  f"Total Time: {total_time:.2f}s | "
                  f"Avg Batch Time: {sum(batch_times)/len(batch_times):.3f}s")
        else:
            print(f"  [EGX] Validation Complete - Avg Loss: {avg_loss:.6f} | "
                  f"Total Time: {total_time:.2f}s")

        return avg_loss

    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full EGX-orchestrated training loop.
        """
        # ── Build EGX Callback Stack ──
        callbacks = [
            self._autoencoder_log_cb,
            self._checkpoint_cb,
            self._lr_scheduler_cb,
            NaNDetectionCallback(halt_on_nan=False, max_nan_count=10),
        ]

        from egx.api.callbacks import CallbackHandler, EarlyStoppingCallback as EGXES
        cb_handler = CallbackHandler(callbacks)

        # Add EGX early stopping
        es_cb = EGXES(
            patience=config.EARLY_STOP_PATIENCE,
            min_delta=1e-4,
            metric_name="eval_loss",
            greater_is_better=False,
        )
        cb_handler.add(es_cb)

        # ── Fire: on_train_begin ──
        trainer_ref = self
        trainer_ref._model = self.model
        cb_handler.fire("on_train_begin", trainer=trainer_ref)

        # Calculate dataset statistics
        train_samples = len(self.train_loader.dataset)
        val_samples = len(self.val_loader.dataset)
        train_batches = len(self.train_loader)
        val_batches = len(self.val_loader)

        print("=" * 80)
        print("  EGX POWERED PULMONARY ANOMALY DETECTION - TRAINING SESSION")
        print("=" * 80)
        print(f"  Device: {self.device}")
        print(f"  Loss Function: {config.LOSS_FUNCTION}")
        print(f"  Total Epochs: {config.EPOCHS}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
        print(f"  Weight Decay: {config.WEIGHT_DECAY}")
        print(f"  Early Stopping Patience: {config.EARLY_STOP_PATIENCE}")
        print()
        print("  DATASET STATISTICS:")
        print(f"    Training: {train_samples} samples ({train_batches} batches)")
        print(f"    Validation: {val_samples} samples ({val_batches} batches)")
        print(f"    Samples per batch: {config.BATCH_SIZE}")
        print()
        print("  MODEL INFO:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        print()

        print("=" * 80)

        t0 = time.time()

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            
            cb_handler.fire("on_epoch_begin", trainer=trainer_ref, epoch=epoch)

            # Training Phase
            train_loss = self._run_epoch(self.train_loader, training=True)

            # Validation Phase
            val_loss = self._run_epoch(self.val_loader, training=False)

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # Record History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            # EGX Epoch Metrics
            epoch_metrics = {
                "train_loss_epoch": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch": epoch + 1,
            }

            # Detailed epoch summary
            best_train_loss = min(self.history["train_loss"]) if self.history["train_loss"] else float('inf')
            best_val_loss = min(self.history["val_loss"]) if self.history["val_loss"] else float('inf')

            print(f"\n{'='*80}")
            print(f"[EGX] Epoch [{epoch + 1:03d}/{config.EPOCHS}] - Total Time: {epoch_time:.2f}s")
            print(f"{'='*80}")
            print(f"  Training Loss: {train_loss:.6f} (avg per batch)")
            print(f"  Validation Loss: {val_loss:.6f} (avg per batch)")
            print(f"  Learning Rate: {lr:.2e}")
            print(f"  Best Training Loss: {best_train_loss:.6f}")
            print(f"  Best Validation Loss: {best_val_loss:.6f}")
            print(f"  Progress: {epoch + 1}/{config.EPOCHS} epochs ({(epoch + 1)/config.EPOCHS*100:.1f}%)")
            print(f"  Samples Processed: {train_samples + val_samples:,} total")
            print(f"  Batches Processed: {train_batches + val_batches} total")
            print(f"{'='*80}")

            cb_handler.fire(
                "on_epoch_end", trainer=trainer_ref,
                epoch=epoch, metrics=epoch_metrics,
            )

            # Fire evaluate end (checkpoint + scheduler)
            eval_metrics = {"eval_loss": val_loss, "val_loss": val_loss}
            cb_handler.fire(
                "on_evaluate_end", trainer=trainer_ref, metrics=eval_metrics,
            )

            cb_handler.fire(
                "on_step_end", trainer=trainer_ref,
                step=epoch, loss=train_loss, lr=lr,
            )

            # Early stopping check
            if es_cb.should_stop:
                print(f"\n[EGX] Early stopping triggered at epoch {epoch + 1}!")
                print(f"[EGX] Best validation loss: {es_cb.best_value:.6f}")
                print(f"[EGX] No improvement for {config.EARLY_STOP_PATIENCE} epochs")
                break

        # ── Fire: on_train_end ──
        best_val = self._checkpoint_cb.best_score or min(self.history["val_loss"])
        result = {
            "final_loss": self.history["train_loss"][-1],
            "best_val_loss": best_val,
            "duration_s": time.time() - t0,
            "epochs_completed": len(self.history["train_loss"]),
        }
        cb_handler.fire("on_train_end", trainer=trainer_ref, result=result)

        total_time = result['duration_s']
        epochs_completed = result['epochs_completed']

        print("\n" + "=" * 80)
        print("  EGX TRAINING COMPLETED")
        print("=" * 80)
        print(f"  Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"  Epochs completed: {epochs_completed}/{config.EPOCHS}")
        print(f"  Best validation loss: {best_val:.6f}")
        print(f"  Final training loss: {self.history['train_loss'][-1]:.6f}")
        print(f"  Final validation loss: {self.history['val_loss'][-1]:.6f}")
        print(f"  Average epoch time: {total_time/epochs_completed:.2f} seconds")
        print(f"  Best model saved to: {config.BEST_MODEL_PATH}")
        print("=" * 80)
        return self.history