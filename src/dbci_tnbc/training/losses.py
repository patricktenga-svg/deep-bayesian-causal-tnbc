"""Loss functions."""
import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, kl_weight: float = 0.1):
    """Standard VAE loss (reconstruction + KL)."""
    recon = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl_weight * kl


def total_loss(pcr_loss, vae_m, vae_t, vae_weight: float = 0.1):
    """Total combined loss."""
    return pcr_loss + vae_weight * (vae_m + vae_t)