"""
src/model/decoder.py
────────────────────
Decoder: latent vector z → reconstructed image.

DecoderBlock  — ConvTranspose2d(stride=2) + BatchNorm + ReLU
Decoder       — linear projection + stack of DecoderBlock (final: Tanh)

Tanh output matches normalised input range [-1, 1].
"""

import torch
import torch.nn as nn
import config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_final: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if not is_final:
            layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers.append(nn.Tanh())   # output ∈ [-1, 1]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """
    (B, latent_dim) → (B, 1, 128, 128)

    Linear → Reshape(256,8,8) → 8→16→32→64→128 spatial
    Channels: 256→128→64→32→1, final Tanh
    """

    RESHAPE_CH  = 256
    RESHAPE_SP  = 8
    FLAT_SIZE   = 256 * 8 * 8

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.FLAT_SIZE),
            nn.ReLU(inplace=True),
        )
        self.deconv_layers = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64,  32),
            DecoderBlock(32,  1, is_final=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.RESHAPE_CH, self.RESHAPE_SP, self.RESHAPE_SP)
        return self.deconv_layers(x)
