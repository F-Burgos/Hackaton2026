from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.evaluation.retrieval import (  # noqa: E402
    pair_similarity_diagnostics,
    ranking_metrics,
    recall_at_k,
)


def test_recall_at_k_perfect_alignment() -> None:
    embeddings = torch.eye(4)
    metrics = recall_at_k(embeddings, embeddings, k_values=(1, 2))

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0


def test_pair_similarity_diagnostics_detects_positive_margin() -> None:
    embeddings = torch.eye(4)
    metrics = pair_similarity_diagnostics(embeddings, embeddings)

    assert metrics["positive_cosine_mean"] == 1.0
    assert metrics["negative_cosine_mean"] == 0.0
    assert metrics["positive_negative_margin"] == 1.0


def test_ranking_metrics_perfect_alignment() -> None:
    embeddings = torch.eye(4)
    metrics = ranking_metrics(embeddings, embeddings)

    assert metrics["mean_rank"] == 1.0
    assert metrics["median_rank"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["rank_percentile_median"] == 1.0


def test_ranking_metrics_counts_correct_pair_rank() -> None:
    query = torch.eye(3)
    target = torch.eye(3)[[1, 0, 2]]
    metrics = ranking_metrics(query, target)

    assert metrics["median_rank"] == 2.0
    assert metrics["mean_rank"] == pytest.approx(5 / 3)
    assert metrics["mrr"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
