"""
src/training/loss.py
────────────────────
Reconstruction loss functions.

MSEReconstructionLoss — pixel-wise MSE
SSIMLoss              — structural similarity (Wang et al. 2004)
CombinedLoss          — α·MSE + (1-α)·SSIM, always returns a tensor
get_loss_fn()         — factory

Design fix: CombinedLoss now returns a plain tensor (not a dict).
Trainer calls it uniformly regardless of loss type.
Loss component logging moved to trainer.py as a debug step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MSEReconstructionLoss(nn.Module):
    """Mean Squared Error: L = (1/N) Σ (x - x̂)²"""

    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss(reduction="mean")

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._mse(x_hat, x)


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5,
                     channels: int = 1) -> torch.Tensor:
    coords    = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g         = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g         = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


class SSIMLoss(nn.Module):
    """
    SSIM loss: L = 1 - SSIM(x, x̂).
    Measures luminance, contrast, and structure simultaneously.
    Reference: Wang et al. (2004), IEEE TIP 13(4).
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 channels: int = 1):
        super().__init__()
        self.channels    = channels
        self.kernel_size = kernel_size
        self.register_buffer("kernel", _gaussian_kernel(kernel_size, sigma, channels))

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.kernel_size // 2
        mu_x   = F.conv2d(x,   self.kernel, padding=pad, groups=self.channels)
        mu_y   = F.conv2d(y,   self.kernel, padding=pad, groups=self.channels)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x  = F.conv2d(x*x, self.kernel, padding=pad, groups=self.channels) - mu_x2
        sig_y  = F.conv2d(y*y, self.kernel, padding=pad, groups=self.channels) - mu_y2
        sig_xy = F.conv2d(x*y, self.kernel, padding=pad, groups=self.channels) - mu_xy
        ssim   = ((2*mu_xy + C1) * (2*sig_xy + C2)) / \
                 ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
        return ssim.mean()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Shift [-1,1] → [0,1] before SSIM computation
        return 1.0 - self._ssim((x + 1) / 2, (x_hat + 1) / 2)


class CombinedLoss(nn.Module):
    """
    L = α·MSE + (1-α)·SSIM_loss

    Always returns a plain tensor for uniform handling in Trainer.
    α=0.5 balances pixel fidelity with perceptual quality.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.alpha     = alpha
        self.mse_loss  = MSEReconstructionLoss()
        self.ssim_loss = SSIMLoss(channels=config.IMG_CHANNELS)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse_loss(x_hat, x) + \
               (1.0 - self.alpha) * self.ssim_loss(x_hat, x)


def get_loss_fn(name: str = None) -> nn.Module:
    """Factory: return the configured loss function."""
    name     = (name or config.LOSS_FUNCTION).lower()
    registry = {"mse": MSEReconstructionLoss, "ssim": SSIMLoss, "combined": CombinedLoss}
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Choose: {list(registry)}")
    return registry[name]()
