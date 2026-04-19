"""
app/services/inference.py
─────────────────────────
InferenceService — the only place in the web app that touches the model.

Loads the trained model once at startup (singleton).
Loads thresholds from thresholds.json (written by evaluate.py) — never hardcoded.
Runs prediction: bytes → score → heatmap → JSON-serialisable dict.
"""

import io
import json
import base64
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from src.model   import ConvAutoencoder
from src.utils   import get_device, get_logger
from app.services.image_processor import (
    bytes_to_pil, pil_to_tensor, tensor_to_b64_png, pil_to_b64_png,
)

logger = get_logger(__name__)

# ── Default thresholds (overwritten by load from disk) ────────────
_DEFAULT_THRESHOLD_NORMAL  = 0.015
_DEFAULT_THRESHOLD_ANOMALY = 0.040


def _load_thresholds() -> tuple:
    """Load data-driven thresholds from thresholds.json (written by evaluate.py)."""
    if not config.THRESHOLDS_PATH or not __import__("os").path.exists(config.THRESHOLDS_PATH):
        logger.warning(
            "thresholds.json not found. Using default placeholders. "
            "Run scripts/evaluate.py after training."
        )
        return _DEFAULT_THRESHOLD_NORMAL, _DEFAULT_THRESHOLD_ANOMALY

    with open(config.THRESHOLDS_PATH) as f:
        t = json.load(f)
    normal  = float(t.get("threshold_normal",  _DEFAULT_THRESHOLD_NORMAL))
    anomaly = float(t.get("threshold_anomaly", _DEFAULT_THRESHOLD_ANOMALY))
    logger.info(f"Thresholds loaded — normal={normal:.6f}, anomaly={anomaly:.6f}")
    return normal, anomaly


class InferenceService:
    """
    Loads the model once; serves every predict() call.

    Thresholds are loaded from disk — not hardcoded — so re-running
    evaluate.py automatically updates inference behaviour.
    """

    def __init__(self):
        self._model:   Optional[ConvAutoencoder] = None
        self._device = get_device()
        self._ready  = False
        self._threshold_normal, self._threshold_anomaly = _load_thresholds()
        self._load_model()

    def _load_model(self) -> None:
        import os
        if not os.path.exists(config.BEST_MODEL_PATH):
            logger.warning(
                f"No checkpoint at {config.BEST_MODEL_PATH}. "
                "Run scripts/train.py first."
            )
            return
        try:
            self._model = ConvAutoencoder.load(config.BEST_MODEL_PATH,
                                               device=self._device)
            self._ready = True
            logger.info(f"Model loaded from {config.BEST_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, image_bytes: bytes) -> dict:
        """
        Full inference pipeline.

        bytes → PIL → tensor → forward → score → heatmap → dict
        PIL is decoded once and reused — no double-decode.

        Returns dict with: label, score, raw_mse, confidence,
                           heatmap_b64, reconstruction_b64, original_b64
        """
        if not self._ready:
            raise RuntimeError("Model not loaded. Run scripts/train.py first.")

        # Decode image once — reuse PIL for heatmap overlay
        pil_original = bytes_to_pil(image_bytes)
        tensor       = pil_to_tensor(pil_original).to(self._device)

        with torch.no_grad():
            x_hat, _ = self._model(tensor)

        # Per-pixel squared error (H, W)
        error_map = ((tensor - x_hat) ** 2).squeeze()
        raw_mse   = error_map.mean().item()

        score             = self._normalise_score(raw_mse)
        label, confidence = self._classify(raw_mse)
        heatmap_b64       = self._make_heatmap(error_map, pil_original)
        reconstruction_b64 = tensor_to_b64_png(x_hat)
        original_b64       = pil_to_b64_png(
            pil_original.resize((config.IMG_SIZE, config.IMG_SIZE)).convert("L")
        )

        return {
            "label":              label,
            "score":              round(score,   4),
            "raw_mse":            round(raw_mse, 6),
            "confidence":         confidence,
            "heatmap_b64":        heatmap_b64,
            "reconstruction_b64": reconstruction_b64,
            "original_b64":       original_b64,
        }

    def _normalise_score(self, raw_mse: float) -> float:
        """Map raw MSE to [0, 1] using data-driven thresholds."""
        span = self._threshold_anomaly - self._threshold_normal
        if span <= 0:
            return 1.0 if raw_mse >= self._threshold_anomaly else 0.0
        return float(np.clip(
            (raw_mse - self._threshold_normal) / span, 0.0, 1.0
        ))

    def _classify(self, raw_mse: float) -> tuple:
        if raw_mse < self._threshold_normal:
            return "Normal", "High"
        mid = (self._threshold_normal + self._threshold_anomaly) / 2
        if raw_mse < mid:
            return "Anomaly Detected", "Low"
        if raw_mse < self._threshold_anomaly:
            return "Anomaly Detected", "Medium"
        return "Anomaly Detected", "High"

    def _make_heatmap(self, error_map: torch.Tensor,
                      pil_original: Image.Image) -> str:
        """
        Blend hot colormap over the original X-ray.
        Pure NumPy — no OpenCV dependency.
        """
        err  = error_map.cpu().numpy()
        err  = (err - err.min()) / (err.max() - err.min() + 1e-8)
        u8   = (err * 255).astype(np.uint8)

        # Hot colormap: black→red→yellow→white
        r = np.clip(u8 * 3,       0, 255).astype(np.uint8)
        g = np.clip(u8 * 3 - 255, 0, 255).astype(np.uint8)
        b = np.clip(u8 * 3 - 510, 0, 255).astype(np.uint8)
        heat = np.stack([r, g, b], axis=-1)

        # Resize original to model input size, convert to RGB
        orig_gray = np.array(
            pil_original.resize((config.IMG_SIZE, config.IMG_SIZE)).convert("L")
        )
        orig_rgb  = np.stack([orig_gray] * 3, axis=-1)

        blended = (0.6 * orig_rgb + 0.4 * heat).astype(np.uint8)
        return pil_to_b64_png(Image.fromarray(blended, mode="RGB"))


# ── Thread-safe singleton ─────────────────────────────────────────

_instance:      Optional[InferenceService] = None
_instance_lock  = threading.Lock()


def get_inference_service() -> InferenceService:
    """Return the global singleton. Thread-safe creation."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:   # double-checked locking
                _instance = InferenceService()
    return _instance
