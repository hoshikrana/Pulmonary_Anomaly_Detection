"""
src/data/dataset.py
───────────────────
PyTorch Dataset classes. No transforms, no DataLoaders defined here.

NormalOnlyDataset   — for unsupervised autoencoder training (NORMAL only)
AnomalyEvalDataset  — for evaluation against labelled test data
"""

import os
from collections import Counter
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _collect(folder: str) -> List[str]:
    """Return sorted list of valid image paths from a folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Folder not found: {folder}\n"
            "Check config.py paths and confirm the dataset is downloaded."
        )
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ])


def _open(path: str) -> Image.Image:
    """Open image safely — returns blank on failure."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[Warning] Cannot open {path}: {e}. Using blank image.")
        return Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))


class NormalOnlyDataset(Dataset):
    """
    Loads images from the NORMAL folder only.

    The autoencoder trains exclusively on healthy radiographs.
    The reconstruction manifold fits normal anatomy tightly.
    At test time pathological features produce elevated MSE — our anomaly score.
    """

    def __init__(self, folder_path: str, transform=None):
        self.transform   = transform
        self.image_paths = _collect(folder_path)
        if not self.image_paths:
            raise ValueError(f"No valid images found in: {folder_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = _open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, 0   # label 0 = normal


class AnomalyEvalDataset(Dataset):
    """
    Loads images from BOTH normal and anomaly folders for evaluation.
    Never used during training.
    """

    def __init__(self, normal_path: str, anomaly_path: str, transform=None):
        self.transform = transform
        self.samples   = (
            [(p, 0) for p in _collect(normal_path)] +
            [(p, 1) for p in _collect(anomaly_path)]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = _open(path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict:
        c = Counter(label for _, label in self.samples)
        return {"normal": c[0], "anomaly": c[1]}