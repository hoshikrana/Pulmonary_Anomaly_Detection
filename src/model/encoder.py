"""
src/model/encoder.py
────────────────────
Supports 512x512 input with AdaptiveAvgPool so latent dim stays fixed.
"""

import torch
import torch.nn as nn
import config

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=config.DROPOUT_RATE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Encoder(nn.Module):
    """
    Variable size input → fixed latent_dim
    """
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            EncoderBlock(1,   32),
            EncoderBlock(32,  64),
            EncoderBlock(64,  128),
            EncoderBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))          # ← key for any input size
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pool(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z