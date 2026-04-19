"""
scripts/train.py — training entry point.

Supports two modes:
  python scripts/train.py          → vanilla Trainer
  python scripts/train.py --egx    → EGX-powered Trainer (recommended)

The EGX mode adds: automatic hardware probing, mixed precision,
self-healing checkpoints, NaN detection, recovery FSM, and
production-grade callback-driven logging.
"""

import sys

import config
from src.utils      import set_seed, print_device_info, get_logger
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.evaluation import plot_training_curves

logger = get_logger(__name__, log_dir=config.OUTPUT_DIR + "/logs")


def main():
    use_egx = "--egx" in sys.argv

    set_seed(config.SEED)
    config.print_config()
    print_device_info()

    logger.info("Loading data...")
    train_loader, val_loader, _ = get_dataloaders()

    logger.info("Building model...")
    model        = ConvAutoencoder(latent_dim=config.LATENT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,} (~{total_params*4/1e6:.1f} MB)")

    if use_egx:
        logger.info("Training with EGX (Elastic Guardian X)...")
        from src.training import EGXAutoencoderTrainer
        trainer = EGXAutoencoderTrainer(model, train_loader, val_loader)
    else:
        logger.info("Training with vanilla Trainer...")
        from src.training import Trainer
        trainer = Trainer(model, train_loader, val_loader)

    history = trainer.fit()

    # Use shared visualiser — no duplicate implementation
    plot_training_curves(history)
    logger.info(f"Best model: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
