# =============================================================
# Pulmonary Anomaly Detection in Chest Radiographs
# config.py — Central configuration. Pure data, no side effects.
# =============================================================
# Rule: importing config must never trigger heavy operations.
# Device detection lives in src/utils/device.py.
# =============================================================

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "chest_xray")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

TRAIN_NORMAL    = os.path.join(DATA_DIR, "train", "NORMAL")
TRAIN_PNEUMONIA = os.path.join(DATA_DIR, "train", "PNEUMONIA")
TEST_NORMAL     = os.path.join(DATA_DIR, "test",  "NORMAL")
TEST_PNEUMONIA  = os.path.join(DATA_DIR, "test",  "PNEUMONIA")
VAL_NORMAL      = os.path.join(DATA_DIR, "val",   "NORMAL")
VAL_PNEUMONIA   = os.path.join(DATA_DIR, "val",   "PNEUMONIA")

BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pth")
THRESHOLDS_PATH     = os.path.join(OUTPUT_DIR,     "thresholds.json")
CONFIG_SNAPSHOT_PATH = os.path.join(OUTPUT_DIR,    "config_snapshot.json")

# ------------------------------------------------------------------
# Image settings
# ------------------------------------------------------------------
IMG_SIZE     = 128
IMG_CHANNELS = 1
IMG_MEAN     = (0.5,)
IMG_STD      = (0.5,)

# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------
LATENT_DIM = 128

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
BATCH_SIZE   = 32
NUM_WORKERS  = 2
PIN_MEMORY   = True

LEARNING_RATE           = 0.1
WEIGHT_DECAY            = 0.1
EPOCHS                  = 100
EARLY_STOP_PATIENCE     = 8
LR_SCHEDULER_PATIENCE   = 4
LR_SCHEDULER_FACTOR     = 0.5

VAL_SPLIT    = 0.2
SEED         = 42

# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
LOSS_FUNCTION = "combined"   # "mse" | "ssim" | "combined"

# ------------------------------------------------------------------
# Upload limit (single source of truth — used by both Flask and validators)
# ------------------------------------------------------------------
MAX_UPLOAD_BYTES = 16 * 1024 * 1024   # 16 MB

# ------------------------------------------------------------------
# Ensure output directories exist on import
# ------------------------------------------------------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)


def print_config() -> None:
    """Print all config values at training start."""
    print("\n" + "=" * 55)
    print("  Pulmonary Anomaly Detection — Configuration")
    print("=" * 55)
    print(f"  Image size   : {IMG_SIZE}x{IMG_SIZE} ({IMG_CHANNELS} channel)")
    print(f"  Latent dim   : {LATENT_DIM}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Loss         : {LOSS_FUNCTION}")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Checkpoints  : {CHECKPOINT_DIR}")
    print(f"  Outputs      : {OUTPUT_DIR}")
    print("=" * 55 + "\n")


def save_snapshot() -> None:
    """Save a JSON snapshot of all scalar config values for experiment tracking."""
    import json
    snapshot = {
        k: v for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, str, tuple, bool))
    }
    with open(CONFIG_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
