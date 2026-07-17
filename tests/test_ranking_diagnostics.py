from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from project.src.evaluation.ranking_diagnostics import (  # noqa: E402
    RankingDiagnosticsConfig,
    run_ranking_diagnostics,
)


def test_ranking_diagnostics_writes_metrics(tmp_path) -> None:
    embeddings_path = tmp_path / "embeddings.npz"
    output_path = tmp_path / "ranking.json"
    embeddings = np.eye(4, dtype=np.float32)
    np.savez_compressed(
        embeddings_path,
        object_id=np.asarray([f"obj_{idx}" for idx in range(4)]),
        image_embedding=embeddings,
        spectrum_embedding=embeddings,
    )

    metrics = run_ranking_diagnostics(
        RankingDiagnosticsConfig(
            embeddings_path=str(embeddings_path),
            output_path=str(output_path),
        )
    )

    loaded = json.loads(output_path.read_text())
    assert metrics["i2s_median_rank"] == 1.0
    assert loaded["s2i_mrr"] == 1.0
