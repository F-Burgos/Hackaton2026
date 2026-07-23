from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CrossModalMapper(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_to_spectrum = _mlp(embedding_dim, hidden_dim, embedding_dim, num_layers, dropout)
        self.spectrum_to_image = _mlp(embedding_dim, hidden_dim, embedding_dim, num_layers, dropout)

    def forward(
        self,
        image_embedding: torch.Tensor,
        spectrum_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted_spectrum = F.normalize(self.image_to_spectrum(image_embedding), dim=-1)
        predicted_image = F.normalize(self.spectrum_to_image(spectrum_embedding), dim=-1)
        return {
            "pred_spectrum_embedding": predicted_spectrum,
            "pred_image_embedding": predicted_image,
            "predicted_spectrum_embedding": predicted_spectrum,
            "predicted_image_embedding": predicted_image,
        }


def crossmodal_mapping_loss(
    predicted_spectrum_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    predicted_image_embedding: torch.Tensor,
    image_embedding: torch.Tensor,
) -> torch.Tensor:
    i2s = 1.0 - F.cosine_similarity(predicted_spectrum_embedding, spectrum_embedding, dim=-1)
    s2i = 1.0 - F.cosine_similarity(predicted_image_embedding, image_embedding, dim=-1)
    return 0.5 * (i2s.mean() + s2i.mean())


def crossmodal_scores(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    predicted_spectrum_embedding: torch.Tensor,
    predicted_image_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pair_distance = 1.0 - F.cosine_similarity(image_embedding, spectrum_embedding, dim=-1)
    i2s_mapping_distance = 1.0 - F.cosine_similarity(
        predicted_spectrum_embedding,
        spectrum_embedding,
        dim=-1,
    )
    s2i_mapping_distance = 1.0 - F.cosine_similarity(
        predicted_image_embedding,
        image_embedding,
        dim=-1,
    )
    return {
        "pair_distance": pair_distance,
        "i2s_mapping_distance": i2s_mapping_distance,
        "s2i_mapping_distance": s2i_mapping_distance,
        "crossmodal_mapping_distance": 0.5 * (i2s_mapping_distance + s2i_mapping_distance),
    }


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
