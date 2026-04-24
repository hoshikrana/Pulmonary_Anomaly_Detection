"""
src/model/autoencoder.py
─────────────────────────
ConvAutoencoder: assembles Encoder + Decoder.

forward() returns: x_hat, z, mu, logvar
  - x_hat : reconstruction (B, 1, 512, 512)
  - z      : sampled latent (B, LATENT_DIM)
  - mu     : latent mean    (B, LATENT_DIM)  — for KL loss
  - logvar : log-variance   (B, LATENT_DIM)  — for KL loss

encode() returns only z (mu at eval time) — used by anomaly_scorer.
reconstruction_error() returns per-image MSE — the anomaly score.
"""

import os
import torch
import torch.nn as nn

import config
from src.model.encoder import Encoder
from src.model.decoder import Decoder


class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(latent_dim)
        self.decoder    = Decoder(latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming for Conv/ConvTranspose, Xavier for Linear, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:   x     (B, 1, 512, 512)
        Returns x_hat (B, 1, 512, 512)
                z     (B, LATENT_DIM)
                mu    (B, LATENT_DIM)
                logvar(B, LATENT_DIM)
        """
        z, mu, logvar, skips = self.encoder(x)
        x_hat = self.decoder(z, skips)
        return x_hat, z, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns deterministic latent code (mu) for t-SNE / PCA analysis.
        Called by anomaly_scorer.extract_latent_vectors().
        """
        _, mu, _, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-image MSE anomaly score. Shape: (B,)
        Uses mu (no sampling noise) for a stable, deterministic score.
        """
        was_training = self.training
        self.eval()
        x_hat, _, _, _ = self.forward(x)
        errors = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
        if was_training:
            self.train()
        return errors

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "latent_dim": self.latent_dim,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str,
             device: torch.device = None) -> "ConvAutoencoder":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(latent_dim=ckpt["latent_dim"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()