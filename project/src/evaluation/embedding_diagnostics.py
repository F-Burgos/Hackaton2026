from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class EmbeddingDiagnosticsConfig:
    embeddings_path: str
    output_dir: str
    n_neighbors: int = 10
    make_plot: bool = True


def run_embedding_diagnostics(config: EmbeddingDiagnosticsConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(config.embeddings_path)
    object_ids = data["object_id"].astype(str)
    image = data["image_embedding"].astype(np.float32)
    spectrum = data["spectrum_embedding"].astype(np.float32)
    fused = _normalize(0.5 * (image + spectrum))
    pair_distance = 1.0 - np.sum(image * spectrum, axis=1)

    pca = PCA(n_components=2, random_state=0)
    projection = pca.fit_transform(fused)
    k = min(config.n_neighbors, max(1, len(object_ids) - 1))
    neighbor_distances = _knn_distances(fused, k=k)
    prefixes = np.asarray([_prefix(object_id) for object_id in object_ids])

    diagnostics: dict[str, object] = {
        "embeddings_path": str(config.embeddings_path),
        "n_objects": int(len(object_ids)),
        "embedding_dim": int(fused.shape[1]),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "pair_distance": _summary(pair_distance),
        "knn_mean_distance": _summary(neighbor_distances.mean(axis=1)),
        "knn_kth_distance": _summary(neighbor_distances[:, -1]),
        "prefix_counts": _prefix_counts(prefixes),
    }

    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_projection_csv(
        output_dir / "pca_projection.csv",
        object_ids=object_ids,
        prefixes=prefixes,
        projection=projection,
        pair_distance=pair_distance,
        knn_mean_distance=neighbor_distances.mean(axis=1),
        knn_kth_distance=neighbor_distances[:, -1],
    )
    if config.make_plot:
        _write_pca_plot(output_dir / "pca_projection.png", projection, prefixes)

    print(
        " ".join(
            [
                f"n={diagnostics['n_objects']}",
                f"dim={diagnostics['embedding_dim']}",
                f"pair_p50={diagnostics['pair_distance']['p50']:.6f}",
                f"knn_p50={diagnostics['knn_mean_distance']['p50']:.6f}",
            ]
        ),
        flush=True,
    )
    return diagnostics


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, eps)


def _knn_distances(values: np.ndarray, k: int) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    neighbors.fit(values)
    distances, _indices = neighbors.kneighbors(values)
    return distances[:, 1:]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _prefix(object_id: str) -> str:
    parts = object_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return object_id


def _prefix_counts(prefixes: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(prefixes, return_counts=True)
    return {str(prefix): int(count) for prefix, count in zip(unique, counts)}


def _write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_projection_csv(
    path: Path,
    object_ids: np.ndarray,
    prefixes: np.ndarray,
    projection: np.ndarray,
    pair_distance: np.ndarray,
    knn_mean_distance: np.ndarray,
    knn_kth_distance: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "object_id",
                "prefix",
                "pc1",
                "pc2",
                "pair_distance",
                "knn_mean_distance",
                "knn_kth_distance",
            ]
        )
        for row in zip(
            object_ids,
            prefixes,
            projection[:, 0],
            projection[:, 1],
            pair_distance,
            knn_mean_distance,
            knn_kth_distance,
        ):
            writer.writerow(row)


def _write_pca_plot(path: Path, projection: np.ndarray, prefixes: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    unique_prefixes = sorted(set(prefixes.tolist()))
    for prefix in unique_prefixes:
        mask = prefixes == prefix
        ax.scatter(projection[mask, 0], projection[mask, 1], s=12, alpha=0.75, label=prefix)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Contrastive fused embeddings")
    if len(unique_prefixes) <= 12:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
