from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_clip_loss(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if image_embedding.shape != spectrum_embedding.shape:
        raise ValueError(
            "Embeddings must have the same shape, got "
            f"{image_embedding.shape} and {spectrum_embedding.shape}"
        )
    if image_embedding.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {image_embedding.ndim}D")
    logits = image_embedding @ spectrum_embedding.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_to_spectrum = F.cross_entropy(logits, labels)
    spectrum_to_image = F.cross_entropy(logits.T, labels)
    return 0.5 * (image_to_spectrum + spectrum_to_image)


@torch.no_grad()
def retrieval_at_1(image_embedding: torch.Tensor, spectrum_embedding: torch.Tensor) -> float:
    logits = image_embedding @ spectrum_embedding.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())
