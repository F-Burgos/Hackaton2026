from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from project.src.evaluation.retrieval import ranking_metrics, recall_at_k


@dataclass(frozen=True)
class RankingDiagnosticsConfig:
    embeddings_path: str
    output_path: str


def run_ranking_diagnostics(config: RankingDiagnosticsConfig) -> dict[str, float | int | str]:
    data = np.load(config.embeddings_path)
    image = torch.from_numpy(data["image_embedding"].astype(np.float32))
    spectrum = torch.from_numpy(data["spectrum_embedding"].astype(np.float32))
    metrics: dict[str, float | int | str] = {
        "embeddings_path": str(config.embeddings_path),
        "n_objects": int(image.shape[0]),
        **{f"i2s_{key}": value for key, value in recall_at_k(image, spectrum).items()},
        **{f"s2i_{key}": value for key, value in recall_at_k(spectrum, image).items()},
        **{f"i2s_{key}": value for key, value in ranking_metrics(image, spectrum).items()},
        **{f"s2i_{key}": value for key, value in ranking_metrics(spectrum, image).items()},
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        " ".join(
            [
                f"n={metrics['n_objects']}",
                f"i2s_med_rank={metrics['i2s_median_rank']:.1f}",
                f"s2i_med_rank={metrics['s2i_median_rank']:.1f}",
                f"i2s_mrr={metrics['i2s_mrr']:.6f}",
                f"s2i_mrr={metrics['s2i_mrr']:.6f}",
            ]
        ),
        flush=True,
    )
    return metrics
