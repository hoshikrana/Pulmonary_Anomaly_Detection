"""
src/utils/seed.py
─────────────────
Reproducibility helper.

Calling set_seed() before anything else guarantees that every run
with the same seed produces identical results — required for a
research paper to be reproducible.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix all random number generators used by PyTorch, NumPy,
    and Python's built-in random module.

    Args:
        seed: Integer seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU setups
    torch.backends.cudnn.deterministic = True  # deterministic conv ops
    torch.backends.cudnn.benchmark = False     # disable auto-tuner