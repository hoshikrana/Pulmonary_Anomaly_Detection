"""
src/model/autoencoder.py
────────────────────────
Assembles Encoder + Decoder into ConvAutoencoder.
Handles weight initialisation, forward pass, checkpoint save/load.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn

import config
from src.model.encoder import Encoder
from src.model.decoder import Decoder


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for unsupervised pulmonary anomaly detection.

    Train on normal X-rays only → minimise reconstruction loss.
    Inference → high reconstruction error == anomaly.
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(latent_dim)
        self.decoder    = Decoder(latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming normal for Conv (optimal for LeakyReLU).
        Xavier normal for Linear (optimal for unbounded output).
        BatchNorm: weight=1, bias=0.
        """
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat : (B, 1, IMG_SIZE, IMG_SIZE) reconstruction
            z     : (B, latent_dim) latent vector
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image MSE anomaly scores. Shape: (B,)."""
        was_training = self.training
        self.eval()
        x_hat, _ = self.forward(x)
        errors   = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
        if was_training:
            self.train()
        return errors

    # ── Checkpoint I/O ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model checkpoint to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "latent_dim": self.latent_dim,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "ConvAutoencoder":
        """Load checkpoint. Returns model in eval mode."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False required because we save a dict with metadata
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(latent_dim=checkpoint["latent_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
