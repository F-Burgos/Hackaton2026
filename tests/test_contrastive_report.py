from __future__ import annotations

import json

import numpy as np

from project.src.evaluation.contrastive_report import (
    ContrastiveReportConfig,
    write_contrastive_report,
)


def test_contrastive_report_includes_best_checkpoint_section(tmp_path) -> None:
    run_dir = tmp_path / "run"
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True)

    _write_json(
        run_dir / "summary.json",
        {
            "epoch": 2,
            "n_train": 16,
            "n_val": 8,
            "train_loss": 1.0,
            "val_loss": 1.2,
            "val_i2s_recall@1": 0.1,
            "val_i2s_recall@5": 0.2,
            "val_i2s_recall@10": 0.3,
            "val_s2i_recall@1": 0.1,
            "val_s2i_recall@5": 0.2,
            "val_s2i_recall@10": 0.3,
            "best_epoch": 1,
            "best_train_loss": 0.9,
            "best_val_loss": 1.1,
            "best_val_i2s_recall@1": 0.4,
            "best_val_i2s_recall@5": 0.5,
            "best_val_i2s_recall@10": 0.6,
            "best_val_s2i_recall@1": 0.4,
            "best_val_s2i_recall@5": 0.5,
            "best_val_s2i_recall@10": 0.6,
        },
    )
    _write_json(
        export_dir / "metrics.json",
        {
            "checkpoint_path": "best.pt",
            "split": "val",
            "n_objects": 4,
            "i2s_recall@1": 0.25,
            "i2s_recall@5": 1.0,
            "i2s_recall@10": 1.0,
            "s2i_recall@1": 0.25,
            "s2i_recall@5": 1.0,
            "s2i_recall@10": 1.0,
        },
    )
    embeddings = np.eye(4, dtype=np.float32)
    np.savez_compressed(
        export_dir / "embeddings.npz",
        object_id=np.asarray([f"hst_field_{idx}" for idx in range(4)]),
        image_embedding=embeddings,
        spectrum_embedding=embeddings,
    )

    report_path = write_contrastive_report(
        ContrastiveReportConfig(run_dir=str(run_dir), export_dir=str(export_dir))
    )

    report = report_path.read_text(encoding="utf-8")
    assert "### Best Validation Checkpoint" in report
    assert "- best_epoch: `1`" in report
    assert "- best_val_loss: `1.100000`" in report


def _write_json(path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
