"""src/training — training pipeline package."""

from .loss      import MSEReconstructionLoss, SSIMLoss, CombinedLoss, get_loss_fn
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from .trainer   import Trainer

try:
    from .egx_trainer import EGXAutoencoderTrainer
    _egx_available = True
except ImportError:
    _egx_available = False
    EGXAutoencoderTrainer = None

__all__ = [
    "MSEReconstructionLoss",
    "SSIMLoss",
    "CombinedLoss",
    "get_loss_fn",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerCallback",
    "Trainer",
    "EGXAutoencoderTrainer",
]