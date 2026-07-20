from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SUMMARY_COLUMNS = [
    "run",
    "export",
    "split",
    "n_train",
    "n_val",
    "n_export",
    "best_epoch",
    "stopped_early",
    "contrastive_loss",
    "temperature",
    "temperature_trainable",
    "accumulation_steps",
    "effective_batch_size",
    "best_val_loss",
    "best_val_margin",
    "best_val_i2s_median_rank",
    "best_val_s2i_median_rank",
    "export_i2s_recall@1",
    "export_s2i_recall@1",
    "export_i2s_median_rank",
    "export_s2i_median_rank",
    "export_margin",
]


MARKDOWN_COLUMNS = [
    "run",
    "export",
    "n_train",
    "n_val",
    "n_export",
    "accumulation_steps",
    "effective_batch_size",
    "best_epoch",
    "contrastive_loss",
    "temperature",
    "temperature_trainable",
    "best_val_loss",
    "best_val_margin",
    "export_i2s_recall@1",
    "export_s2i_recall@1",
    "export_i2s_median_rank",
    "export_s2i_median_rank",
    "export_margin",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize versioned contrastive run metrics.")
    parser.add_argument("--results-root", default="project/results/contrastive")
    parser.add_argument("--output-csv", default="project/reports/contrastive_run_summary.csv")
    parser.add_argument("--output-md", default="project/reports/contrastive_run_summary.md")
    args = parser.parse_args()

    rows = summarize_runs(Path(args.results_root))
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    print(f"wrote {len(rows)} rows")


def summarize_runs(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        run_dir = summary_path.parent
        summary = _read_json(summary_path)
        export_paths = sorted(run_dir.glob("export_*/metrics.json"))
        if not export_paths:
            rows.append(_row_from_metrics(results_root, run_dir, summary, None, None))
            continue
        for export_path in export_paths:
            export_metrics = _read_json(export_path)
            rows.append(
                _row_from_metrics(
                    results_root,
                    run_dir,
                    summary,
                    export_path.parent.name,
                    export_metrics,
                )
            )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Contrastive Run Summary",
        "",
        "Tabla generada desde `summary.json` y `export_*/metrics.json` versionados.",
        "Los checkpoints, embeddings `.npz` y logs crudos no se incluyen.",
        "",
        "|" + "|".join(MARKDOWN_COLUMNS) + "|",
        "|" + "|".join("---" for _ in MARKDOWN_COLUMNS) + "|",
    ]
    for row in rows:
        values = [_format_markdown_value(row.get(column, "")) for column in MARKDOWN_COLUMNS]
        lines.append("|" + "|".join(values) + "|")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _row_from_metrics(
    results_root: Path,
    run_dir: Path,
    summary: dict[str, object],
    export_name: str | None,
    export_metrics: dict[str, object] | None,
) -> dict[str, object]:
    metrics = export_metrics or {}
    return {
        "run": str(run_dir.relative_to(results_root)),
        "export": export_name or "",
        "split": metrics.get("split", ""),
        "n_train": summary.get("best_n_train", summary.get("n_train", "")),
        "n_val": summary.get("best_n_val", summary.get("n_val", "")),
        "n_export": metrics.get("n_objects", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "stopped_early": summary.get("stopped_early", ""),
        "contrastive_loss": summary.get("best_contrastive_loss", summary.get("contrastive_loss", "")),
        "temperature": summary.get("best_temperature", summary.get("temperature", "")),
        "temperature_trainable": summary.get(
            "best_temperature_trainable",
            summary.get("temperature_trainable", False),
        ),
        "accumulation_steps": summary.get(
            "best_train_contrastive_accumulation_steps",
            summary.get("train_contrastive_accumulation_steps", 1),
        ),
        "effective_batch_size": summary.get(
            "best_train_effective_contrastive_batch_size",
            summary.get("train_effective_contrastive_batch_size", ""),
        ),
        "best_val_loss": summary.get("best_val_loss", ""),
        "best_val_margin": summary.get("best_val_positive_negative_margin", ""),
        "best_val_i2s_median_rank": summary.get("best_val_i2s_median_rank", ""),
        "best_val_s2i_median_rank": summary.get("best_val_s2i_median_rank", ""),
        "export_i2s_recall@1": metrics.get("i2s_recall@1", ""),
        "export_s2i_recall@1": metrics.get("s2i_recall@1", ""),
        "export_i2s_median_rank": metrics.get("i2s_median_rank", ""),
        "export_s2i_median_rank": metrics.get("s2i_median_rank", ""),
        "export_margin": metrics.get("positive_negative_margin", ""),
    }


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
