from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_loss(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = _paired_logits(query_embedding, target_embedding, temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def symmetric_info_nce_loss(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = _paired_logits(image_embedding, spectrum_embedding, temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_to_spectrum = F.cross_entropy(logits, labels)
    spectrum_to_image = F.cross_entropy(logits.T, labels)
    return 0.5 * (image_to_spectrum + spectrum_to_image)


def symmetric_clip_loss(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    return symmetric_info_nce_loss(image_embedding, spectrum_embedding, temperature)


def _paired_logits(
    query_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query_embedding.shape != target_embedding.shape:
        raise ValueError(
            "Embeddings must have the same shape, got "
            f"{query_embedding.shape} and {target_embedding.shape}"
        )
    if query_embedding.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {query_embedding.ndim}D")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return query_embedding @ target_embedding.T / temperature


@torch.no_grad()
def retrieval_at_1(image_embedding: torch.Tensor, spectrum_embedding: torch.Tensor) -> float:
    logits = image_embedding @ spectrum_embedding.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())
