"""
src/data/dataloader.py
──────────────────────
DataLoader factory. Single responsibility: wrap Datasets into DataLoaders.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

import config
from src.data.dataset    import NormalOnlyDataset, AnomalyEvalDataset
from src.data.transforms import get_train_transform, get_eval_transform


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Full training set (only normal images)
    full_train = NormalOnlyDataset(
        config.TRAIN_NORMAL,
        transform=get_train_transform()          # ← augmented + normalized
    )

    # Split into train / val
    val_size   = int(config.VAL_SPLIT * len(full_train))
    train_size = len(full_train) - val_size

    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    # Test set (normal + anomaly for evaluation)
    test_ds = AnomalyEvalDataset(
        config.TEST_NORMAL,
        config.TEST_PNEUMONIA,
        transform=get_eval_transform(),          # ← deterministic
    )

    def _loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=shuffle,
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)
    test_loader  = _loader(test_ds,  shuffle=False)

    counts = test_ds.class_counts()

    print("\n" + "=" * 60)
    print("  DATA LOADER CONFIGURATION")
    print("=" * 60)
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Number of Workers: {config.NUM_WORKERS}")
    print(f"  Pin Memory: {config.PIN_MEMORY}")
    print()
    print("  DATASET SPLITS:")
    print(f"    Training Set: {train_size} normal images")
    print(f"      → {len(train_loader)} batches of {config.BATCH_SIZE} images each")
    print(f"      → {train_size % config.BATCH_SIZE} images in last batch" if train_size % config.BATCH_SIZE != 0 else f"      → All batches full ({config.BATCH_SIZE} images)")
    print()
    print(f"    Validation Set: {val_size} normal images")
    print(f"      → {len(val_loader)} batches of {config.BATCH_SIZE} images each")
    print(f"      → {val_size % config.BATCH_SIZE} images in last batch" if val_size % config.BATCH_SIZE != 0 else f"      → All batches full ({config.BATCH_SIZE} images)")
    print()
    print(f"    Test Set: {counts['normal']} normal + {counts['anomaly']} anomaly images")
    print(f"      → {len(test_loader)} batches of {config.BATCH_SIZE} images each")
    total_test = counts['normal'] + counts['anomaly']
    print(f"      → {total_test % config.BATCH_SIZE} images in last batch" if total_test % config.BATCH_SIZE != 0 else f"      → All batches full ({config.BATCH_SIZE} images)")
    print(f"      → Class distribution: {counts['normal']/total_test*100:.1f}% normal, {counts['anomaly']/total_test*100:.1f}% anomaly")
    print("=" * 60)

    return train_loader, val_loader, test_loader