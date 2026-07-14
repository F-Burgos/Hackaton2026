from __future__ import annotations

import torch
from torch import nn


class SpectrumAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 739),
        )

    def forward(self, flux: torch.Tensor, spectrum_mask: torch.Tensor) -> torch.Tensor:
        x = torch.stack([flux * spectrum_mask, spectrum_mask], dim=1)
        latent = self.encoder(x)
        return self.decoder(latent)


class ImageAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 9, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, image: torch.Tensor, image_channel_mask: torch.Tensor) -> torch.Tensor:
        masked_image = image * image_channel_mask[:, :, None, None]
        latent = self.encoder(masked_image)
        return self.decoder(latent)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    expanded_mask = torch.broadcast_to(mask, target.shape)
    squared_error = (prediction - target).pow(2) * expanded_mask
    return squared_error.sum() / expanded_mask.sum().clamp_min(eps)
