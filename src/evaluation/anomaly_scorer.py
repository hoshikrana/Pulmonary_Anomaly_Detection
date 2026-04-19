"""
src/evaluation/anomaly_scorer.py
─────────────────────────────────
Runs trained model over a DataLoader, returns per-image scores + labels.
No metrics, no plots, no thresholds here.

Reference:
  Baur et al. (2019). "Deep Autoencoding Models for Unsupervised
  Anomaly Segmentation in Brain MR Images." MICCAI Workshop.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.model import ConvAutoencoder
from src.utils import get_device, get_logger

logger = get_logger(__name__)


class AnomalyScorer:
    """
    Scores every image in a DataLoader using reconstruction error.

    Args:
        model  : Trained ConvAutoencoder in eval mode.
        device : Defaults to config device.
    """

    def __init__(self, model: ConvAutoencoder, device: torch.device = None):
        self.device = device or get_device()
        self.model  = model.eval().to(self.device)

    @torch.no_grad()
    def score_loader(self, loader: DataLoader,
                     desc: str = "Scoring") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-image MSE for every image in loader.

        Returns:
            scores : (N,) float32 — reconstruction errors (anomaly scores)
            labels : (N,) int32   — 0 normal, 1 anomaly
        """
        all_scores, all_labels = [], []

        for images, labels in tqdm(loader, desc=f"  {desc}", ncols=80, leave=False):
            images = images.to(self.device, non_blocking=True)
            x_hat, _ = self.model(images)
            mse = ((images - x_hat) ** 2).mean(dim=[1, 2, 3])
            all_scores.append(mse.cpu().numpy())
            all_labels.append(labels.numpy())

        scores = np.concatenate(all_scores).astype(np.float32)
        labels = np.concatenate(all_labels).astype(np.int32)

        normal_mean  = scores[labels == 0].mean() if (labels == 0).any() else float("nan")
        anomaly_mean = scores[labels == 1].mean() if (labels == 1).any() else float("nan")
        logger.info(f"{desc}: {len(scores)} images | "
                    f"normal mean={normal_mean:.5f} | anomaly mean={anomaly_mean:.5f}")

        return scores, labels

    @torch.no_grad()
    def score_single(self, image_tensor: torch.Tensor
                     ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Score one image tensor.

        Returns:
            mse_score : scalar float anomaly score
            error_map : (H, W) per-pixel squared error
            x_hat     : (1, C, H, W) reconstruction
        """
        image_tensor = image_tensor.to(self.device)
        x_hat, _     = self.model(image_tensor)
        error_map    = ((image_tensor - x_hat) ** 2).squeeze()
        return error_map.mean().item(), error_map, x_hat

    @torch.no_grad()
    def extract_latent_vectors(self, loader: DataLoader
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract z vectors for every image — used by t-SNE / PCA plots.

        Returns:
            vectors : (N, latent_dim) float32
            labels  : (N,) int32
        """
        all_vectors, all_labels = [], []

        for images, labels in tqdm(loader, desc="  Latents", ncols=80, leave=False):
            images = images.to(self.device, non_blocking=True)
            all_vectors.append(self.model.encode(images).cpu().numpy())
            all_labels.append(labels.numpy())

        vectors = np.concatenate(all_vectors).astype(np.float32)
        labels  = np.concatenate(all_labels).astype(np.int32)
        logger.info(f"Extracted {vectors.shape[0]} latent vectors (dim={vectors.shape[1]})")
        return vectors, labels
