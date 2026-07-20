from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.models.contrastive import ContrastiveModel
from project.src.models.contrastive import _masked_standardize_sequence
from project.src.models.losses import info_nce_loss, symmetric_clip_loss, symmetric_info_nce_loss


def test_contrastive_forward_and_loss() -> None:
    batch = {
        "img": torch.randn(4, 9, 64, 64),
        "img_channel_mask": torch.ones(4, 9),
        "flux_lambda_normalized": torch.randn(4, 739),
        "mask_spectra": torch.ones(4, 739),
    }
    model = ContrastiveModel(embedding_dim=32, projection_dim=16)
    outputs = model(batch)

    assert outputs["image_embedding"].shape == (4, 16)
    assert outputs["spectrum_embedding"].shape == (4, 16)
    loss = symmetric_clip_loss(outputs["image_embedding"], outputs["spectrum_embedding"])
    assert torch.isfinite(loss)
    loss.backward()


def test_residual_contrastive_forward_and_loss() -> None:
    batch = {
        "img": torch.randn(2, 9, 64, 64),
        "img_channel_mask": torch.ones(2, 9),
        "flux_lambda_normalized": torch.randn(2, 739),
        "mask_spectra": torch.ones(2, 739),
    }
    model = ContrastiveModel(
        embedding_dim=24,
        projection_dim=12,
        encoder_width=0.5,
        encoder_variant="residual",
        dropout=0.1,
    )
    outputs = model(batch)

    assert outputs["image_embedding"].shape == (2, 12)
    assert outputs["spectrum_embedding"].shape == (2, 12)
    loss = symmetric_clip_loss(outputs["image_embedding"], outputs["spectrum_embedding"])
    assert torch.isfinite(loss)


def test_masked_sequence_standardization_ignores_invalid_values() -> None:
    values = torch.tensor([[1.0, 2.0, 1000.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    normalized = _masked_standardize_sequence(values, mask)

    assert normalized[0, 2] == 0.0
    assert torch.isclose(normalized[mask.bool()].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isfinite(normalized).all()


def test_encoder_width_must_be_positive() -> None:
    with pytest.raises(ValueError, match="encoder_width"):
        ContrastiveModel(encoder_width=0.0)


def test_symmetric_clip_loss_aliases_symmetric_info_nce() -> None:
    image_embedding = torch.eye(4)
    spectrum_embedding = torch.eye(4)

    old_name = symmetric_clip_loss(image_embedding, spectrum_embedding)
    explicit_name = symmetric_info_nce_loss(image_embedding, spectrum_embedding)

    assert torch.isclose(old_name, explicit_name)


def test_info_nce_loss_is_directional_cross_entropy() -> None:
    query_embedding = torch.eye(3)
    target_embedding = torch.eye(3)

    loss = info_nce_loss(query_embedding, target_embedding, temperature=0.07)

    assert torch.isfinite(loss)
    assert loss < 1e-4


def test_info_nce_loss_accepts_trainable_temperature() -> None:
    query_embedding = torch.eye(3)
    target_embedding = torch.eye(3)
    log_temperature = torch.nn.Parameter(torch.tensor(0.15).log())

    loss = info_nce_loss(
        query_embedding,
        target_embedding,
        temperature=log_temperature.exp(),
    )
    loss.backward()

    assert log_temperature.grad is not None
    assert torch.isfinite(log_temperature.grad)
