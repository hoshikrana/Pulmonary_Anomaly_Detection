"""
src/model/encoder.py
─────────────────────
U-Net Encoder with residual blocks, SE attention, and VAE-style dual heads.

Spatial flow (512×512 grayscale input):
  block1: 512→256  ch: 1   → 64
  block2: 256→128  ch: 64  → 128
  block3: 128→64   ch: 128 → 256
  block4:  64→32   ch: 256 → 512
  block5:  32→16   ch: 512 → 512
  block6:  16→8    ch: 512 → 512
  pool:     8→8    AdaptiveAvgPool2d
  fc_mu / fc_logvar: 512*8*8 → LATENT_DIM

Returns: z, mu, logvar, [s1, s2, s3, s4, s5, s6]
"""

import torch
import torch.nn as nn
import config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    α = sigmoid(W2·ReLU(W1·AvgPool(x)))
    out = x * α   (channel-wise rescaling)
    Learns which channels carry lung texture information.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ResEncoderBlock(nn.Module):
    """
    Residual encoder block with SE attention.
    Main:     Conv(4,s=2) → BN → LReLU → Dropout → Conv(3,s=1) → BN → SE
    Shortcut: Conv(1,s=2) → BN
    Output:   LReLU(main + shortcut)
    Residual path ensures clean gradient flow through all 6 blocks.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        dr = config.DROPOUT_RATE
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dr),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.main(x)) + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.block1 = ResEncoderBlock(1,   64)
        self.block2 = ResEncoderBlock(64,  128)
        self.block3 = ResEncoderBlock(128, 256)
        self.block4 = ResEncoderBlock(256, 512)
        self.block5 = ResEncoderBlock(512, 512)
        self.block6 = ResEncoderBlock(512, 512)

        self.pool    = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        flat = 512 * 8 * 8  # 32768
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
          train: z = mu + eps*sigma,  eps ~ N(0,I)
          eval:  z = mu              (deterministic — stable anomaly score)
        Allows gradient to flow through the sampling operation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x: torch.Tensor):
        s1 = self.block1(x)   # (B,  64, 256, 256)
        s2 = self.block2(s1)  # (B, 128, 128, 128)
        s3 = self.block3(s2)  # (B, 256,  64,  64)
        s4 = self.block4(s3)  # (B, 512,  32,  32)
        s5 = self.block5(s4)  # (B, 512,  16,  16)
        s6 = self.block6(s5)  # (B, 512,   8,   8)

        h      = self.flatten(self.pool(s6))             # (B, 32768)
        mu     = self.fc_mu(h)                           # (B, LATENT_DIM)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10) # stability clamp
        z      = self.reparameterize(mu, logvar)

        return z, mu, logvar, [s1, s2, s3, s4, s5, s6]