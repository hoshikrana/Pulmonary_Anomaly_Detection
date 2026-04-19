"""
scripts/train.py — training entry point.

Supports two modes:
  python scripts/train.py          → vanilla Trainer
  python scripts/train.py --egx    → EGX-powered Trainer (recommended)

The EGX mode adds: automatic hardware probing, mixed precision,
self-healing checkpoints, NaN detection, recovery FSM, and
production-grade callback-driven logging.

Retraining: If best_model.pth exists, training continues from saved weights.
"""

import sys
import os
from datetime import datetime

# Ensure the project root is in the Python path before importing config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from src.utils      import set_seed, print_device_info
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.evaluation import plot_training_curves


class TerminalLogger:
    """Logs all terminal output to both stdout and a file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main():
    use_egx = "--egx" in sys.argv

    # Setup terminal logging
    log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = TerminalLogger(log_file)
    sys.stdout = logger

    print("\n" + "=" * 70)
    print("  PULMONARY ANOMALY DETECTION - TRAINING SCRIPT")
    print("=" * 70)
    print(f"  Mode: {'EGX Powered' if use_egx else 'Vanilla'} Training")
    print(f"  Random Seed: {config.SEED}")
    print("=" * 70)
    print(f"  Terminal output saved to: {log_file}")
    print("=" * 70)

    set_seed(config.SEED)
    config.print_config()
    print_device_info()

    print("\n[1/4] Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders()

    print("\n[2/4] Building autoencoder model...")
    model = ConvAutoencoder(latent_dim=config.LATENT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters (~{total_params*4/1e6:.1f} MB)")

    # Load best model if it exists (for retraining)
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"✓ Loading best model from {config.BEST_MODEL_PATH}")
        import torch
        checkpoint = torch.load(config.BEST_MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("✓ Model weights loaded successfully - continuing training from saved checkpoint")
        print(f"✓ Previous best validation loss: {checkpoint.get('val_loss', 'N/A')}")
    else:
        print("✓ Starting training from scratch (no saved model found)")

    if use_egx:
        print("\n[3/4] Initializing EGX-powered trainer...")
        from src.training import EGXAutoencoderTrainer
        if EGXAutoencoderTrainer is None:
            print("[ERROR] EGX framework not installed. Falling back to vanilla trainer.")
            print("[TIP] Install EGX: pip install -e <EGX_PATH>")
            from src.training import Trainer
            trainer = Trainer(model, train_loader, val_loader)
            print("✓ Vanilla trainer ready")
        else:
            trainer = EGXAutoencoderTrainer(model, train_loader, val_loader)
            print("✓ EGX trainer ready with production-grade features")
    else:
        print("\n[3/4] Initializing vanilla trainer...")
        from src.training import Trainer
        trainer = Trainer(model, train_loader, val_loader)
        print("✓ Vanilla trainer ready")

    print("\n[4/4] Starting training...")
    history = trainer.fit()

    print("\n[5/5] Generating training curves...")
    plot_training_curves(history)
    print(f"✓ Training curves saved to {config.OUTPUT_DIR}")
    print(f"✓ Best model saved to {config.BEST_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("  TRAINING SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"✓ Terminal log saved to: {log_file}")
    print("=" * 70)
    
    # Close the logger
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # Restore stdout and close logger on error
        if hasattr(sys.stdout, 'terminal'):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
