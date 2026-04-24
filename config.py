# =============================================================
# Pulmonary Anomaly Detection in Chest Radiographs
# config.py — HIGH-RES + SMALLER LATENT DIM
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
# Image settings — ORIGINAL SIZE PRESERVED
# ------------------------------------------------------------------
IMG_SIZE     = 512
IMG_CHANNELS = 1
IMG_MEAN     = (0.5,)
IMG_STD      = (0.5,)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
LATENT_DIM   = 256            # ← Reduced as you asked (better compression)
DROPOUT_RATE = 0.25

# ------------------------------------------------------------------
# Training (GPU ready)
# ------------------------------------------------------------------
BATCH_SIZE   = 16            # ← Reduced to 4 to fit in 4GB GPU VRAM
NUM_WORKERS  = 4
PIN_MEMORY   = True
GRAD_ACCUM_STEPS = 4            # gradient accumulation (effective batch = BATCH_SIZE × 4)

LEARNING_RATE           = 0.0001
WEIGHT_DECAY            = 1e-4
EPOCHS                  = 30
EARLY_STOP_PATIENCE     = 8
LR_SCHEDULER_PATIENCE   = 5
LR_SCHEDULER_FACTOR     = 0.5

VAL_SPLIT    = 0.2
SEED         = 42

# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
LOSS_FUNCTION = "combined"
LOSS_ALPHA       = 0.7          # weight for MSE vs SSIM  (1-α goes to SSIM)
KL_BETA_MAX      = 0.0005       # peak KL weight after warmup
KL_WARMUP_EPOCHS = 5            # epochs to linearly ramp β from 0 → KL_BETA_MAX

# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

def print_config() -> None:
    print("\n" + "=" * 55)
    print("  Pulmonary Anomaly Detection — Configuration")
    print("=" * 55)
    print(f"  Image size   : {IMG_SIZE}x{IMG_SIZE} ({IMG_CHANNELS} channel)")
    print(f"  Latent dim   : {LATENT_DIM}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay : {WEIGHT_DECAY}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Loss         : {LOSS_FUNCTION}")
    print(f"  Dropout      : {DROPOUT_RATE}")
    print("=" * 55 + "\n")

def save_snapshot() -> None:
    import json
    snapshot = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, str, tuple, bool))}
    with open(CONFIG_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)