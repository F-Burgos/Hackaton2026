from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.models.contrastive import ContrastiveModel
from project.src.models.contrastive import _masked_standardize_sequence
from project.src.models.losses import symmetric_clip_loss


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


def test_masked_sequence_standardization_ignores_invalid_values() -> None:
    values = torch.tensor([[1.0, 2.0, 1000.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    normalized = _masked_standardize_sequence(values, mask)

    assert normalized[0, 2] == 0.0
    assert torch.isclose(normalized[mask.bool()].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isfinite(normalized).all()
