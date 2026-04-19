"""
src/model/encoder.py
────────────────────
Encoder: image → latent vector z.

EncoderBlock  — Conv2d(stride=2) + BatchNorm + LeakyReLU
Encoder       — stack of EncoderBlock + linear projection to z

Design notes:
- Stride-2 conv instead of MaxPool: learned downsampling (Springenberg 2015).
- LeakyReLU(0.2): prevents dead neurons in deep encoders.
- NO activation on the FC output: unconstrained z gives a richer latent
  space. A ReLU here would zero half the dimensions and hurt AUC-ROC.
"""

import torch
import torch.nn as nn
import config


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    (B, 1, 128, 128) → (B, latent_dim)

    Spatial:  128 → 64 → 32 → 16 → 8
    Channels:   1 → 32 → 64 → 128 → 256
    Flatten: 256×8×8 = 16384 → Linear → latent_dim
    """

    FLAT_SIZE = 256 * 8 * 8

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            EncoderBlock(1,   32),
            EncoderBlock(32,  64),
            EncoderBlock(64,  128),
            EncoderBlock(128, 256),
        )
        self.flatten = nn.Flatten()
        # No activation after FC — unconstrained z space is richer.
        self.fc = nn.Linear(self.FLAT_SIZE, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z
