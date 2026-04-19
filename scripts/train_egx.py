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

import os
import sys

# Ensure the project root is in the Python path before importing config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from src.utils      import set_seed, print_device_info
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.training   import EGXAutoencoderTrainer
from src.evaluation import plot_training_curves


def main():
    print("\n" + "=" * 70)
    print("  PULMONARY ANOMALY DETECTION - EGX TRAINING SCRIPT")
    print("=" * 70)
    print("  Powered by Elastic Guardian X (EGX) Framework")
    print("  Features: Hardware optimization, Mixed precision, Self-healing")
    print(f"  Random Seed: {config.SEED}")
    print("=" * 70)

    set_seed(config.SEED)
    config.print_config()
    print_device_info()

    print("\n[1/4] Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders()

    print("\n[2/4] Building autoencoder model...")
    model = ConvAutoencoder(latent_dim=config.LATENT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters (~{total_params * 4 / 1e6:.1f} MB)")

    print("\n[3/4] Initializing EGX-powered trainer...")
    trainer = EGXAutoencoderTrainer(model, train_loader, val_loader)
    print("✓ EGX trainer ready with production-grade features")

    print("\n[4/4] Starting EGX-powered training...")
    history = trainer.fit()

    print("\n[5/5] Generating training curves...")
    plot_training_curves(history)
    print(f"✓ Training curves saved to {config.OUTPUT_DIR}")
    print(f"✓ Best model saved to {config.BEST_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("  EGX TRAINING SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
