"""
app/services/image_processor.py
────────────────────────────────
Single responsibility: bytes → tensor  and  tensor → base64 PNG.
No model code, no Flask code.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image

import config
from src.data.transforms import get_eval_transform, denormalize


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes to a PIL RGB image."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Apply eval transform and add batch dim. Returns (1, 1, H, W)."""
    return get_eval_transform()(pil_image).unsqueeze(0)


def bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convenience: bytes → (1, 1, IMG_SIZE, IMG_SIZE) tensor."""
    return pil_to_tensor(bytes_to_pil(image_bytes))


def tensor_to_b64_png(tensor: torch.Tensor) -> str:
    """
    Convert single-image tensor (any leading dims) to base64 PNG string.
    Embeds as: <img src="data:image/png;base64,...">
    """
    t   = tensor
    while t.dim() > 2:
        t = t.squeeze(0)
    arr    = (denormalize(t).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf    = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_b64_png(pil_image: Image.Image) -> str:
    """Convert any PIL image to base64 PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
