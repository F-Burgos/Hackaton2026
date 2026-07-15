from __future__ import annotations

import torch


@torch.no_grad()
def recall_at_k(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if query_embeddings.shape != target_embeddings.shape:
        raise ValueError(
            "Expected paired embeddings with matching shapes, got "
            f"{query_embeddings.shape} and {target_embeddings.shape}"
        )
    if query_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {query_embeddings.ndim}D")
    logits = query_embeddings @ target_embeddings.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    max_k = min(max(k_values), logits.shape[1])
    topk = logits.topk(k=max_k, dim=1).indices
    metrics: dict[str, float] = {}
    for k in k_values:
        effective_k = min(k, max_k)
        hits = (topk[:, :effective_k] == labels[:, None]).any(dim=1)
        metrics[f"recall@{k}"] = float(hits.float().mean().item())
    return metrics


@torch.no_grad()
def embedding_diagnostics(embeddings: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_norm": float(embeddings.norm(dim=1).mean().item()),
        f"{prefix}_mean_std": float(embeddings.std(dim=0).mean().item()),
    }


@torch.no_grad()
def pair_similarity_diagnostics(
    image_embeddings: torch.Tensor,
    spectrum_embeddings: torch.Tensor,
) -> dict[str, float]:
    if image_embeddings.shape != spectrum_embeddings.shape:
        raise ValueError(
            "Expected paired embeddings with matching shapes, got "
            f"{image_embeddings.shape} and {spectrum_embeddings.shape}"
        )
    if image_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {image_embeddings.ndim}D")

    similarity = image_embeddings @ spectrum_embeddings.T
    positive = similarity.diag()
    metrics = {
        "positive_cosine_mean": float(positive.mean().item()),
        "positive_cosine_std": float(positive.std(unbiased=False).item()),
    }
    if similarity.shape[0] > 1:
        diagonal = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        negative = similarity[~diagonal]
        negative_mean = negative.mean()
        metrics.update(
            {
                "negative_cosine_mean": float(negative_mean.item()),
                "negative_cosine_std": float(negative.std(unbiased=False).item()),
                "positive_negative_margin": float((positive.mean() - negative_mean).item()),
            }
        )
    else:
        metrics.update(
            {
                "negative_cosine_mean": float("nan"),
                "negative_cosine_std": float("nan"),
                "positive_negative_margin": float("nan"),
            }
        )
    return metrics
