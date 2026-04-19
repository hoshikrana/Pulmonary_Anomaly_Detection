"""
src/utils/device.py
───────────────────
Device detection and memory reporting.

Centralising device logic means no `torch.device(...)` calls
scattered throughout the codebase — every module imports from here.
"""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info() -> None:
    """Log device details — useful at the start of train.py."""
    device = get_device()
    print(f"[Device] Using: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = props.total_memory / 1e9
        print(f"[Device] GPU    : {props.name}")
        print(f"[Device] Memory : {used:.2f} GB used / {total:.1f} GB total")
    else:
        print("[Device] No GPU found - running on CPU.")
        print("[Device] Tip: Enable GPU in Colab -> Runtime -> Change runtime type -> T4 GPU")


def move_to_device(obj, device: torch.device = None):
    """Move a tensor or nn.Module to the target device."""
    if device is None:
        device = get_device()
    return obj.to(device)