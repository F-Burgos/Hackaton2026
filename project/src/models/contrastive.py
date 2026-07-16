from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        embedding_dim: int = 128,
        encoder_width: float = 1.0,
        encoder_variant: str = "simple",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = _scaled_channels((32, 64, 128), encoder_width)
        if encoder_variant == "simple":
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(c1),
                nn.GELU(),
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.GELU(),
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c3),
                nn.GELU(),
            ]
        elif encoder_variant == "residual":
            layers = [
                _image_downsample_block(in_channels, c1, kernel_size=5),
                _ResidualImageBlock(c1, dropout=dropout),
                _image_downsample_block(c1, c2),
                _ResidualImageBlock(c2, dropout=dropout),
                _image_downsample_block(c2, c3),
                _ResidualImageBlock(c3, dropout=dropout),
            ]
        else:
            raise ValueError(
                f"Unknown encoder_variant {encoder_variant!r}; expected 'simple' or 'residual'"
            )
        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, embedding_dim),
        )

    def forward(self, image: torch.Tensor, image_channel_mask: torch.Tensor) -> torch.Tensor:
        masked_image = _masked_standardize_image(image, image_channel_mask)
        return self.net(masked_image)


class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_width: float = 1.0,
        encoder_variant: str = "simple",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = _scaled_channels((32, 64, 128), encoder_width)
        if encoder_variant == "simple":
            layers: list[nn.Module] = [
                nn.Conv1d(2, c1, kernel_size=9, padding=4),
                nn.BatchNorm1d(c1),
                nn.GELU(),
                nn.Conv1d(c1, c2, kernel_size=9, stride=2, padding=4),
                nn.BatchNorm1d(c2),
                nn.GELU(),
                nn.Conv1d(c2, c3, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(c3),
                nn.GELU(),
            ]
        elif encoder_variant == "residual":
            layers = [
                _spectrum_block(2, c1, kernel_size=9, stride=1),
                _ResidualSpectrumBlock(c1, dropout=dropout),
                _spectrum_block(c1, c2, kernel_size=9, stride=2),
                _ResidualSpectrumBlock(c2, dropout=dropout),
                _spectrum_block(c2, c3, kernel_size=7, stride=2),
                _ResidualSpectrumBlock(c3, dropout=dropout),
            ]
        else:
            raise ValueError(
                f"Unknown encoder_variant {encoder_variant!r}; expected 'simple' or 'residual'"
            )
        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, embedding_dim),
        )

    def forward(self, flux: torch.Tensor, spectrum_mask: torch.Tensor) -> torch.Tensor:
        masked_flux = _masked_standardize_sequence(flux, spectrum_mask)
        x = torch.stack([masked_flux, spectrum_mask], dim=1)
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        encoder_width: float = 1.0,
        encoder_variant: str = "simple",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            encoder_width=encoder_width,
            encoder_variant=encoder_variant,
            dropout=dropout,
        )
        self.spectrum_encoder = SpectrumEncoder(
            embedding_dim=embedding_dim,
            encoder_width=encoder_width,
            encoder_variant=encoder_variant,
            dropout=dropout,
        )
        self.image_projection = _projection_head(embedding_dim, projection_dim, dropout=dropout)
        self.spectrum_projection = _projection_head(embedding_dim, projection_dim, dropout=dropout)

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


class _ResidualImageBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.net(values))


class _ResidualSpectrumBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.net(values))


def _image_downsample_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


def _spectrum_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )


def _projection_head(input_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(input_dim, output_dim),
    )


def _scaled_channels(channels: tuple[int, ...], width: float) -> tuple[int, ...]:
    if width <= 0:
        raise ValueError(f"encoder_width must be positive, got {width}")
    return tuple(max(1, int(round(channel * width))) for channel in channels)


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
