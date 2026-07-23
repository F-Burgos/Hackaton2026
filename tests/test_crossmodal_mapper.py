from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.models.crossmodal import CrossModalMapper, crossmodal_mapping_loss


def test_crossmodal_mapper_outputs_normalized_embeddings() -> None:
    mapper = CrossModalMapper(embedding_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
    image = torch.randn(4, 8)
    spectrum = torch.randn(4, 8)

    outputs = mapper(image, spectrum)

    assert outputs["pred_spectrum_embedding"].shape == (4, 8)
    assert outputs["pred_image_embedding"].shape == (4, 8)
    assert torch.allclose(
        outputs["pred_spectrum_embedding"].norm(dim=1),
        torch.ones(4),
        atol=1e-6,
    )


def test_crossmodal_mapping_loss_is_differentiable() -> None:
    mapper = CrossModalMapper(embedding_dim=8, hidden_dim=16)
    image = torch.randn(4, 8)
    spectrum = torch.randn(4, 8)
    outputs = mapper(image, spectrum)

    loss = crossmodal_mapping_loss(
        outputs["pred_spectrum_embedding"],
        spectrum,
        outputs["pred_image_embedding"],
        image,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in mapper.parameters())
