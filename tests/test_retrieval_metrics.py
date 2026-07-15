from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.evaluation.retrieval import recall_at_k  # noqa: E402


def test_recall_at_k_perfect_alignment() -> None:
    embeddings = torch.eye(4)
    metrics = recall_at_k(embeddings, embeddings, k_values=(1, 2))

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0
