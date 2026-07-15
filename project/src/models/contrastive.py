from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, image: torch.Tensor, image_channel_mask: torch.Tensor) -> torch.Tensor:
        masked_image = _masked_standardize_image(image, image_channel_mask)
        return self.net(masked_image)


class SpectrumEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, flux: torch.Tensor, spectrum_mask: torch.Tensor) -> torch.Tensor:
        masked_flux = _masked_standardize_sequence(flux, spectrum_mask)
        x = torch.stack([masked_flux, spectrum_mask], dim=1)
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, projection_dim: int = 128) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.spectrum_encoder = SpectrumEncoder(embedding_dim=embedding_dim)
        self.image_projection = _projection_head(embedding_dim, projection_dim)
        self.spectrum_projection = _projection_head(embedding_dim, projection_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(batch["img"], batch["img_channel_mask"])
        spectrum_features = self.spectrum_encoder(
            batch["flux_lambda_normalized"], batch["mask_spectra"]
        )
        image_embedding = F.normalize(self.image_projection(image_features), dim=-1)
        spectrum_embedding = F.normalize(self.spectrum_projection(spectrum_features), dim=-1)
        return {
            "image_embedding": image_embedding,
            "spectrum_embedding": spectrum_embedding,
            "image_features": image_features,
            "spectrum_features": spectrum_features,
        }


def _projection_head(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.GELU(),
        nn.Linear(input_dim, output_dim),
    )


def _masked_standardize_image(
    image: torch.Tensor,
    channel_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = channel_mask[:, :, None, None].to(dtype=image.dtype)
    masked = image * valid
    count = valid.sum(dim=(1, 2, 3), keepdim=True) * image.shape[-1] * image.shape[-2]
    count = count.clamp_min(1.0)
    mean = masked.sum(dim=(1, 2, 3), keepdim=True) / count
    variance = (((masked - mean) * valid) ** 2).sum(dim=(1, 2, 3), keepdim=True) / count
    return ((image - mean) / torch.sqrt(variance + eps)) * valid


def _masked_standardize_sequence(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = mask.to(dtype=values.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    masked = values * valid
    mean = masked.sum(dim=1, keepdim=True) / count
    variance = (((values - mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
    return ((values - mean) / torch.sqrt(variance + eps)) * valid
