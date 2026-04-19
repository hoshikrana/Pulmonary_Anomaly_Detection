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

import config
from src.utils              import get_device
from src.training.loss      import get_loss_fn
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


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
        total_loss = 0.0
        num_batches = len(loader)
        total_samples = len(loader.dataset)
        samples_processed = 0
        batch_times = []

        phase_name = "Training" if training else "Validation"
        print(f"\n  {phase_name} Phase ({num_batches} batches, {total_samples} samples)")

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
                        print(f"    Batch {batch_idx:2d}/{num_batches} "
                              f"[{progress_pct:5.1f}%] | "
                              f"Loss: {batch_loss:.6f} | "
                              f"Images: {samples_processed}/{total_samples} | "
                              f"Time: {batch_time:.3f}s | "
                              f"ETA: {eta_seconds/60:.1f}m")
                    else:
                        print(f"    Val Batch {batch_idx:2d}/{num_batches} "
                              f"[{progress_pct:5.1f}%] | "
                              f"Loss: {batch_loss:.6f}")

        avg_loss = total_loss / num_batches
        total_time = time.time() - start_time

        if training:
            print(f"  Training Complete - Avg Loss: {avg_loss:.6f} | "
                  f"Total Time: {total_time:.2f}s | "
                  f"Avg Batch Time: {sum(batch_times)/len(batch_times):.3f}s")
        else:
            print(f"  Validation Complete - Avg Loss: {avg_loss:.6f} | "
                  f"Total Time: {total_time:.2f}s")

        return avg_loss

    def fit(self) -> Dict[str, List[float]]:
        # Calculate dataset statistics
        train_samples = len(self.train_loader.dataset)
        val_samples = len(self.val_loader.dataset)
        train_batches = len(self.train_loader)
        val_batches = len(self.val_loader)

        print("=" * 80)
        print("  PULMONARY ANOMALY DETECTION - TRAINING SESSION")
        print("=" * 80)
        print(f"  Device: {self.device}")
        print(f"  Loss Function: {config.LOSS_FUNCTION}")
        print(f"  Total Epochs: {config.EPOCHS}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Learning Rate: {config.LEARNING_RATE}")
        print(f"  Weight Decay: {config.WEIGHT_DECAY}")
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
        print("=" * 80)

        t0 = time.time()

        for epoch in range(1, config.EPOCHS + 1):
            epoch_start = time.time()

            # Training phase
            train_loss = self._run_epoch(self.train_loader, training=True)

            # Validation phase
            val_loss = self._run_epoch(self.val_loader, training=False)

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            metrics = {"val_loss": val_loss, "train_loss": train_loss, "lr": lr}

            # Detailed epoch summary
            best_train_loss = min(self.history["train_loss"]) if self.history["train_loss"] else float('inf')
            best_val_loss = min(self.history["val_loss"]) if self.history["val_loss"] else float('inf')

            print(f"\n{'='*80}")
            print(f"Epoch [{epoch:03d}/{config.EPOCHS}] - Total Time: {epoch_time:.2f}s")
            print(f"{'='*80}")
            print(f"  Training Loss: {train_loss:.6f} (avg per batch)")
            print(f"  Validation Loss: {val_loss:.6f} (avg per batch)")
            print(f"  Learning Rate: {lr:.2e}")
            print(f"  Best Training Loss: {best_train_loss:.6f}")
            print(f"  Best Validation Loss: {best_val_loss:.6f}")
            print(f"  Progress: {epoch}/{config.EPOCHS} epochs ({epoch/config.EPOCHS*100:.1f}%)")
            print(f"  Samples Processed: {train_samples + val_samples:,} total")
            print(f"  Batches Processed: {train_batches + val_batches} total")
            print(f"{'='*80}")

            self.lr_scheduler.on_epoch_end(epoch, metrics)
            self.checkpoint.on_epoch_end(epoch, metrics, self.model)

            if self.early_stopping.on_epoch_end(epoch, metrics):
                print(f"\n🚨 EARLY STOPPING TRIGGERED at epoch {epoch}!")
                print(f"   Best validation loss: {self.early_stopping.best_score:.6f}")
                print(f"   No improvement for {self.early_stopping.patience} epochs")
                break

        total_time = time.time() - t0
        epochs_completed = len(self.history["train_loss"])
        best_val_loss = self.checkpoint.best_score

        print("\n" + "=" * 80)
        print("  TRAINING COMPLETED")
        print("=" * 80)
        print(f"  Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"  Epochs completed: {epochs_completed}/{config.EPOCHS}")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Final training loss: {self.history['train_loss'][-1]:.6f}")
        print(f"  Final validation loss: {self.history['val_loss'][-1]:.6f}")
        print(f"  Average epoch time: {total_time/epochs_completed:.2f} seconds")
        print(f"  Best model saved to: {config.BEST_MODEL_PATH}")
        print("=" * 80)

        return self.history
