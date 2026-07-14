from __future__ import annotations

import torch

from project.src.models.ssl import ImageAutoencoder, SpectrumAutoencoder, masked_mse_loss


def test_spectrum_autoencoder_forward_and_loss() -> None:
    flux = torch.randn(4, 739)
    mask = torch.ones(4, 739)
    model = SpectrumAutoencoder(latent_dim=16)
    prediction = model(flux, mask)

    assert prediction.shape == flux.shape
    loss = masked_mse_loss(prediction, flux, mask)
    assert torch.isfinite(loss)
    loss.backward()


def test_image_autoencoder_forward_and_loss() -> None:
    image = torch.randn(4, 9, 64, 64)
    mask = torch.ones(4, 9)
    model = ImageAutoencoder(latent_dim=16)
    prediction = model(image, mask)

    assert prediction.shape == image.shape
    loss = masked_mse_loss(prediction, image, mask)
    assert torch.isfinite(loss)
    loss.backward()
