from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContrastiveReportConfig:
    run_dir: str
    export_dir: str
    output_path: str | None = None


def write_contrastive_report(config: ContrastiveReportConfig) -> Path:
    run_dir = Path(config.run_dir)
    export_dir = Path(config.export_dir)
    output_path = Path(config.output_path) if config.output_path else run_dir / "report.md"

    summary = _read_json(run_dir / "summary.json")
    export_metrics = _read_json(export_dir / "metrics.json")
    embeddings = np.load(export_dir / "embeddings.npz")
    image = embeddings["image_embedding"]
    spectrum = embeddings["spectrum_embedding"]
    object_ids = embeddings["object_id"]

    pair_cosine = np.sum(image * spectrum, axis=1)
    pair_distance = 1.0 - pair_cosine
    prefixes = _prefix_counts(object_ids)

    lines = [
        "# Contrastive Run Report",
        "",
        "## Inputs",
        "",
        f"- run dir: `{run_dir}`",
        f"- export dir: `{export_dir}`",
        f"- checkpoint: `{export_metrics.get('checkpoint_path', 'unknown')}`",
        f"- split: `{export_metrics.get('split', 'unknown')}`",
        f"- objects exported: `{export_metrics.get('n_objects', len(object_ids))}`",
        "",
        "## Training Summary",
        "",
        "### Final Epoch",
        "",
        _metric_line(summary, "epoch"),
        _metric_line(summary, "learning_rate"),
        _metric_line(summary, "contrastive_loss"),
        _metric_line(summary, "temperature"),
        _metric_line(summary, "stopped_early"),
        _metric_line(summary, "n_train"),
        _metric_line(summary, "n_val"),
        _metric_line(summary, "train_contrastive_accumulation_steps"),
        _metric_line(summary, "train_effective_contrastive_batch_size"),
        _metric_line(summary, "train_optimizer_steps"),
        _metric_line(summary, "train_loss"),
        _metric_line(summary, "train_grad_norm"),
        _metric_line(summary, "val_loss"),
        _metric_line(summary, "val_i2s_recall@1"),
        _metric_line(summary, "val_i2s_recall@5"),
        _metric_line(summary, "val_i2s_recall@10"),
        _metric_line(summary, "val_i2s_median_rank"),
        _metric_line(summary, "val_i2s_mrr"),
        _metric_line(summary, "val_s2i_recall@1"),
        _metric_line(summary, "val_s2i_recall@5"),
        _metric_line(summary, "val_s2i_recall@10"),
        _metric_line(summary, "val_s2i_median_rank"),
        _metric_line(summary, "val_s2i_mrr"),
        "",
        "### Best Validation Checkpoint",
        "",
        _metric_line(summary, "best_epoch"),
        _metric_line(summary, "best_learning_rate"),
        _metric_line(summary, "best_contrastive_loss"),
        _metric_line(summary, "best_temperature"),
        _metric_line(summary, "best_train_contrastive_accumulation_steps"),
        _metric_line(summary, "best_train_effective_contrastive_batch_size"),
        _metric_line(summary, "best_train_optimizer_steps"),
        _metric_line(summary, "best_train_loss"),
        _metric_line(summary, "best_train_grad_norm"),
        _metric_line(summary, "best_val_loss"),
        _metric_line(summary, "best_val_i2s_recall@1"),
        _metric_line(summary, "best_val_i2s_recall@5"),
        _metric_line(summary, "best_val_i2s_recall@10"),
        _metric_line(summary, "best_val_i2s_median_rank"),
        _metric_line(summary, "best_val_i2s_mrr"),
        _metric_line(summary, "best_val_s2i_recall@1"),
        _metric_line(summary, "best_val_s2i_recall@5"),
        _metric_line(summary, "best_val_s2i_recall@10"),
        _metric_line(summary, "best_val_s2i_median_rank"),
        _metric_line(summary, "best_val_s2i_mrr"),
        "",
        "## Export Retrieval",
        "",
        _metric_line(export_metrics, "i2s_recall@1"),
        _metric_line(export_metrics, "i2s_recall@5"),
        _metric_line(export_metrics, "i2s_recall@10"),
        _metric_line(export_metrics, "i2s_median_rank"),
        _metric_line(export_metrics, "i2s_mrr"),
        _metric_line(export_metrics, "i2s_rank_percentile_median"),
        _metric_line(export_metrics, "s2i_recall@1"),
        _metric_line(export_metrics, "s2i_recall@5"),
        _metric_line(export_metrics, "s2i_recall@10"),
        _metric_line(export_metrics, "s2i_median_rank"),
        _metric_line(export_metrics, "s2i_mrr"),
        _metric_line(export_metrics, "s2i_rank_percentile_median"),
        "",
        "## Embedding Diagnostics",
        "",
        f"- image mean norm: `{_fmt(export_metrics.get('image_mean_norm'))}`",
        f"- spectrum mean norm: `{_fmt(export_metrics.get('spectrum_mean_norm'))}`",
        f"- image mean std: `{_fmt(export_metrics.get('image_mean_std'))}`",
        f"- spectrum mean std: `{_fmt(export_metrics.get('spectrum_mean_std'))}`",
        f"- positive cosine mean: `{_fmt(export_metrics.get('positive_cosine_mean'))}`",
        f"- negative cosine mean: `{_fmt(export_metrics.get('negative_cosine_mean'))}`",
        f"- positive-negative margin: `{_fmt(export_metrics.get('positive_negative_margin'))}`",
        f"- pair cosine mean: `{pair_cosine.mean():.6f}`",
        f"- pair cosine std: `{pair_cosine.std():.6f}`",
        f"- pair distance p50: `{np.quantile(pair_distance, 0.50):.6f}`",
        f"- pair distance p95: `{np.quantile(pair_distance, 0.95):.6f}`",
        "",
        "## Export Prefix Counts",
        "",
        "| Prefix | Objects |",
        "|---|---:|",
    ]
    for prefix, count in prefixes:
        lines.append(f"| `{prefix}` | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report describes representation/retrieval behavior only.",
            "- It does not implement or evaluate the anomaly downstream.",
            "- Low debug-run recall is expected for tiny one-epoch validation runs.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return output_path


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_line(metrics: dict[str, object], key: str) -> str:
    return f"- {key}: `{_fmt(metrics.get(key))}`"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "missing"
    return str(value)


def _prefix_counts(object_ids: np.ndarray) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for value in object_ids:
        object_id = str(value)
        parts = object_id.split("_")
        if len(parts) >= 2:
            prefix = "_".join(parts[:2])
        else:
            prefix = object_id
        counts[prefix] = counts.get(prefix, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
