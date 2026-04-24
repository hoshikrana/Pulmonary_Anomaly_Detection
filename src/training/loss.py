"""
src/training/loss.py
─────────────────────
Loss functions for U-Net VAE-style autoencoder.

CombinedLoss = α·MSE + (1-α)·SSIM + β·KL

Components:
───────────────────────────────────────────────────────────────────────
1. MSE (Mean Squared Error)
   L_mse = mean( (x - x̂)² )
   Forces pixel-level accuracy. Good for overall brightness/structure.

2. SSIM (Structural Similarity Index)
   SSIM(x,x̂) = (2μxμx̂+C1)(2σxy+C2) / ((μx²+μx̂²+C1)(σx²+σx̂²+C2))
   L_ssim = 1 - SSIM
   Captures luminance, contrast, and structure simultaneously.
   Critical for learning lung textures vs just global brightness.

3. KL Divergence (VAE regularization)
   KL = -½ Σ(1 + logvar - μ² - exp(logvar))
   Regularizes latent space → N(0,I).
   Makes normal images cluster tightly; anomalies deviate measurably.
   β is annealed from 0 → KL_BETA_MAX over warmup epochs to prevent
   posterior collapse (model ignoring the latent code entirely).
───────────────────────────────────────────────────────────────────────

get_loss_fn() factory returns a CombinedLoss instance for compatibility
with the existing Trainer/EGXTrainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── SSIM Loss ────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5,
                     channels: int = 1) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.outer(g)
    return k2d.expand(channels, 1, size, size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM(x, x̂). Shift [-1,1]→[0,1] before computing."""

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 channels: int = 1):
        super().__init__()
        self.channels    = channels
        self.kernel_size = kernel_size
        self.register_buffer("kernel",
                             _gaussian_kernel(kernel_size, sigma, channels))

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.kernel_size // 2
        k      = self.kernel.to(x.device)

        mu_x   = F.conv2d(x,   k, padding=pad, groups=self.channels)
        mu_y   = F.conv2d(y,   k, padding=pad, groups=self.channels)
        mu_x2  = mu_x ** 2;  mu_y2 = mu_y ** 2;  mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x*x, k, padding=pad, groups=self.channels) - mu_x2
        sig_y  = F.conv2d(y*y, k, padding=pad, groups=self.channels) - mu_y2
        sig_xy = F.conv2d(x*y, k, padding=pad, groups=self.channels) - mu_xy

        num  = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den  = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        return (num / (den + 1e-8)).mean()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_01    = (x     + 1) / 2
        x_hat01 = (x_hat + 1) / 2
        return 1.0 - self._ssim(x_01, x_hat01)


# ── KL Divergence ─────────────────────────────────────────────────────────────

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(μ,σ²) || N(0,I)) = -½ Σ(1 + logvar - μ² - exp(logvar))
    Averaged over batch and latent dimensions.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L = α·MSE + (1-α)·SSIM + β·KL

    β is annealed: 0 → KL_BETA_MAX over warmup_epochs.
    Call set_epoch(epoch) at the start of each epoch.

    forward() signature is flexible:
      - (x_hat, x)             → KL term skipped (β=0 or no mu/logvar)
      - (x_hat, x, mu, logvar) → full loss with KL
    Returns a plain tensor (compatible with existing Trainer).
    """

    def __init__(self, alpha: float = None):
        super().__init__()
        self.alpha    = alpha if alpha is not None else config.LOSS_ALPHA
        self.beta_max = config.KL_BETA_MAX
        self.warmup   = config.KL_WARMUP_EPOCHS
        self.beta     = 0.0   # updated each epoch via set_epoch()

        self.mse_fn  = nn.MSELoss()
        self.ssim_fn = SSIMLoss(channels=config.IMG_CHANNELS)

    def set_epoch(self, epoch: int) -> None:
        """
        KL annealing: ramp β linearly from 0 → beta_max over warmup epochs.
        Prevents posterior collapse at the start of training.
        """
        self.beta = self.beta_max * min(epoch / max(self.warmup, 1), 1.0)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                mu: torch.Tensor = None,
                logvar: torch.Tensor = None) -> torch.Tensor:

        l_mse  = self.mse_fn(x_hat, x)
        l_ssim = self.ssim_fn(x_hat, x)
        recon  = self.alpha * l_mse + (1.0 - self.alpha) * l_ssim

        if mu is not None and logvar is not None and self.beta > 0:
            l_kl  = kl_divergence(mu, logvar)
            return recon + self.beta * l_kl

        return recon

    def forward_verbose(self, x_hat: torch.Tensor, x: torch.Tensor,
                        mu: torch.Tensor = None,
                        logvar: torch.Tensor = None) -> dict:
        """
        Returns dict of all loss components for logging.
        Used by EGXTrainer for per-batch/epoch prints.
        """
        l_mse  = self.mse_fn(x_hat, x)
        l_ssim = self.ssim_fn(x_hat, x)
        recon  = self.alpha * l_mse + (1.0 - self.alpha) * l_ssim
        l_kl   = kl_divergence(mu, logvar) if (mu is not None and logvar is not None) \
                 else torch.tensor(0.0)
        total  = recon + self.beta * l_kl

        return {
            "loss":   total,
            "mse":    l_mse.item(),
            "ssim":   l_ssim.item(),
            "kl":     l_kl.item() if isinstance(l_kl, torch.Tensor) else l_kl,
            "recon":  recon.item(),
            "beta":   self.beta,
        }


def get_loss_fn(name: str = None) -> CombinedLoss:
    """
    Factory — always returns CombinedLoss.
    Kept for backward compatibility with Trainer / EGXTrainer.
    """
    return CombinedLoss()