"""
scripts/train_egx.py — EGX-powered training entry point.

Replaces the vanilla training loop with EGX (Elastic Guardian X) for
production-grade training orchestration.

EGX adds:
  ✓ Automatic hardware probing & optimal device selection
  ✓ Mixed precision (FP16/BF16) based on GPU capability
  ✓ Self-healing checkpointing with atomic writes
  ✓ NaN/Inf detection with configurable halt/skip
  ✓ Callback-driven lifecycle hooks
  ✓ Gradient clipping via EGX kernel
  ✓ Early stopping with configurable patience
  ✓ Recovery FSM for OOM and runtime errors

Run:  python scripts/train_egx.py
"""

import config
from src.utils      import set_seed, print_device_info, get_logger
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.training   import EGXAutoencoderTrainer
from src.evaluation import plot_training_curves

logger = get_logger(__name__, log_dir=config.OUTPUT_DIR + "/logs")


def main():
    set_seed(config.SEED)
    config.print_config()
    print_device_info()

    logger.info("[EGX] Loading data...")
    train_loader, val_loader, _ = get_dataloaders()

    logger.info("[EGX] Building model...")
    model        = ConvAutoencoder(latent_dim=config.LATENT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[EGX] Parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")

    logger.info("[EGX] Starting EGX-powered training...")
    trainer = EGXAutoencoderTrainer(model, train_loader, val_loader)
    history = trainer.fit()

    # Use shared visualiser — same outputs as vanilla training
    plot_training_curves(history)
    logger.info(f"[EGX] Best model saved: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
