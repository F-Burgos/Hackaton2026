from __future__ import annotations

import json

import numpy as np

from project.src.evaluation.embedding_diagnostics import (
    EmbeddingDiagnosticsConfig,
    run_embedding_diagnostics,
)


def test_embedding_diagnostics_writes_expected_files(tmp_path) -> None:
    embeddings_path = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "diagnostics"
    object_ids = np.asarray([f"hst_field_{idx}" for idx in range(8)])
    image = np.eye(8, 4, dtype=np.float32)
    spectrum = image.copy()
    np.savez_compressed(
        embeddings_path,
        object_id=object_ids,
        image_embedding=image,
        spectrum_embedding=spectrum,
    )

    diagnostics = run_embedding_diagnostics(
        EmbeddingDiagnosticsConfig(
            embeddings_path=str(embeddings_path),
            output_dir=str(output_dir),
            n_neighbors=3,
            make_plot=False,
        )
    )

    assert diagnostics["n_objects"] == 8
    assert (output_dir / "diagnostics.json").exists()
    assert (output_dir / "pca_projection.csv").exists()
    loaded = json.loads((output_dir / "diagnostics.json").read_text())
    assert loaded["prefix_counts"] == {"hst_field": 8}
