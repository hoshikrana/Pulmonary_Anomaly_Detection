"""
src/training/trainer.py
───────────────────────
Trainer: owns the epoch loop, calls callbacks, returns history.
No loss definitions, no callback logic, no architecture, no data loading.
"""

import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.utils              import get_logger, get_device
from src.training.loss      import get_loss_fn
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

logger = get_logger(__name__, log_dir=config.OUTPUT_DIR + "/logs")


class Trainer:

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, loss_name: str = None):
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

        self.early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)
        self.checkpoint     = ModelCheckpoint(save_path=config.BEST_MODEL_PATH)
        self.lr_scheduler   = LRSchedulerCallback(self.optimizer)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": [],
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        ctx   = torch.enable_grad() if training else torch.no_grad()
        desc  = "  Train" if training else "  Val  "

        with ctx:
            for images, _ in tqdm(loader, desc=desc, leave=False, ncols=80):
                images = images.to(self.device, non_blocking=True)
                x_hat, _ = self.model(images)
                loss = self.loss_fn(x_hat, images)   # always returns tensor

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total += loss.item()

        return total / len(loader)

    def fit(self) -> Dict[str, List[float]]:
        logger.info("=" * 55)
        logger.info("  Training started")
        logger.info(f"  Device: {self.device}  |  Loss: {config.LOSS_FUNCTION}")
        logger.info(f"  Epochs: {config.EPOCHS}  |  Batch: {config.BATCH_SIZE}")
        logger.info("=" * 55)

        t0 = time.time()

        for epoch in range(1, config.EPOCHS + 1):
            t1         = time.time()
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss   = self._run_epoch(self.val_loader,   training=False)
            lr         = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            metrics = {"val_loss": val_loss, "train_loss": train_loss, "lr": lr}

            logger.info(
                f"Epoch [{epoch:03d}/{config.EPOCHS}]  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={lr:.2e}  ({time.time()-t1:.1f}s)"
            )

            self.lr_scheduler.on_epoch_end(epoch, metrics)
            self.checkpoint.on_epoch_end(epoch, metrics, self.model)

            if self.early_stopping.on_epoch_end(epoch, metrics):
                logger.info("Early stopping triggered.")
                break

        logger.info(
            f"Training complete. {(time.time()-t0)/60:.1f} min. "
            f"Best val loss: {self.checkpoint.best_score:.6f}"
        )
        return self.history
