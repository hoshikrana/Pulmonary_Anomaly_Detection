"""
src/training/callbacks.py
─────────────────────────
Training callbacks. Each implements on_epoch_end() called by Trainer.

EarlyStopping       — stops when val loss stagnates
ModelCheckpoint     — saves best model; also saves config snapshot
LRSchedulerCallback — reduces LR on plateau
"""

import os
import torch
import torch.nn as nn

import config


class EarlyStopping:
    """Stop when val loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = config.EARLY_STOP_PATIENCE,
                 min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        score = metrics.get("val_loss")
        if score is None:
            return False

        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] {self.counter}/{self.patience} "
                        f"(best={self.best_score:.6f})")
            if self.counter >= self.patience:
                print(f"[EarlyStopping] Triggered at epoch {epoch}.")
                self.stop = True

        return self.stop


class ModelCheckpoint:
    """Save model + config snapshot whenever val loss improves."""

    def __init__(self, save_path: str = config.BEST_MODEL_PATH):
        self.save_path  = save_path
        self.best_score = None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict, model: nn.Module) -> bool:
        score = metrics.get("val_loss")
        if score is None:
            return False

        if self.best_score is None or score < self.best_score:
            prev           = self.best_score
            self.best_score = score

            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_loss":   score,
                "latent_dim": model.latent_dim,
                "metrics":    metrics,
            }, self.save_path)

            # Save config snapshot alongside checkpoint for experiment tracking
            config.save_snapshot()

            print(
                f"[Checkpoint] Epoch {epoch:03d} — "
                f"val_loss {'∞' if prev is None else f'{prev:.6f}'} → {score:.6f}"
            )
            return True
        return False


class LRSchedulerCallback:
    """Reduce LR when val loss plateaus."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 patience: int  = config.LR_SCHEDULER_PATIENCE,
                 factor: float  = config.LR_SCHEDULER_FACTOR,
                 min_lr: float  = 1e-6):
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=patience, min_lr=min_lr
        )
        self._optimizer = optimizer
        self._prev_lr   = self._lr()

    def _lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"]

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return
        self._scheduler.step(val_loss)
        lr = self._lr()
        if lr < self._prev_lr:
            print(f"[LRScheduler] Epoch {epoch:03d} — "
                        f"LR {self._prev_lr:.2e} → {lr:.2e}")
            self._prev_lr = lr
