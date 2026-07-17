from __future__ import annotations

import json

from project.scripts.summarize_contrastive_runs import summarize_runs, write_csv, write_markdown


def test_summarize_runs_with_export_metrics(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    export_dir = run_dir / "export_val"
    export_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "best_n_train": 16,
                "best_n_val": 8,
                "best_val_loss": 1.5,
                "best_val_positive_negative_margin": 0.2,
                "best_train_contrastive_accumulation_steps": 4.0,
                "best_train_effective_contrastive_batch_size": 128.0,
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "metrics.json").write_text(
        json.dumps(
            {
                "split": "val",
                "n_objects": 8,
                "i2s_recall@1": 0.125,
                "s2i_recall@1": 0.25,
                "i2s_median_rank": 3.0,
                "s2i_median_rank": 4.0,
                "positive_negative_margin": 0.2,
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_runs(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run"] == "run_a"
    assert rows[0]["export"] == "export_val"
    assert rows[0]["effective_batch_size"] == 128.0
    assert rows[0]["export_i2s_median_rank"] == 3.0


def test_summary_writers_create_files(tmp_path) -> None:
    rows = [{"run": "run_a", "export": "export_val", "best_val_loss": 1.5}]

    write_csv(rows, tmp_path / "summary.csv")
    write_markdown(rows, tmp_path / "summary.md")

    assert "best_val_loss" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "run_a" in (tmp_path / "summary.md").read_text(encoding="utf-8")
