from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from project.src.data.hdf5_index import Hdf5KeyIndex
from project.src.data.partitions import load_fold, load_test
from project.src.data.paths import DataPaths
from project.src.data.torch_datasets import TorchMultimodalPairDataset, multimodal_collate
from project.src.downstream.anomalies import (
    AnomalySpec,
    apply_anomalies_to_batch,
    build_anomaly_manifest,
)
from project.src.models.contrastive import ContrastiveModel
from project.src.models.crossmodal import CrossModalMapper


@dataclass(frozen=True)
class DownstreamAnomalyConfig:
    contrastive_checkpoint_path: str
    mapper_checkpoint_path: str | None = None
    data_root: str = "hackaton"
    fold: int = 0
    batch_size: int = 64
    num_workers: int = 0
    max_samples: int | None = None
    anomaly_fraction: float = 0.10
    anomaly_seed: int = 42
    anomaly_modality: str = "alternate"
    image_anomaly_kind: str = "bright_patch"
    spectrum_anomaly_kind: str = "spike"
    anomaly_strength: float = 5.0
    threshold_calibration_split: str = "val"
    threshold_fpr: float = 0.05
    device: str = "auto"
    output_dir: str = "project/results/downstream/anomaly_eval"


def evaluate_downstream_anomalies(
    config: DownstreamAnomalyConfig,
) -> dict[str, float | int | str]:
    device = _resolve_device(config.device)
    contrastive_model, projection_dim = _load_contrastive_model(
        config.contrastive_checkpoint_path,
        device,
    )
    mapper = _load_mapper(config.mapper_checkpoint_path, projection_dim, device)

    paths = DataPaths.from_root(config.data_root)
    index = Hdf5KeyIndex.from_paths(paths)
    calibration_ids = _select_split(
        paths,
        index.paired_keys,
        split=config.threshold_calibration_split,
        fold=config.fold,
    )
    object_ids = load_test(paths.partitions_dir).filter(index.paired_keys).all
    if config.max_samples is not None:
        object_ids = object_ids[: config.max_samples]
    manifest = build_anomaly_manifest(
        object_ids=object_ids,
        fraction=config.anomaly_fraction,
        seed=config.anomaly_seed,
        modality=config.anomaly_modality,  # type: ignore[arg-type]
        image_kind=config.image_anomaly_kind,  # type: ignore[arg-type]
        spectrum_kind=config.spectrum_anomaly_kind,  # type: ignore[arg-type]
        strength=config.anomaly_strength,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_rows = _score_split(
        object_ids=calibration_ids,
        paths=paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        contrastive_model=contrastive_model,
        mapper=mapper,
        device=device,
        manifest={},
    )
    thresholds = _thresholds_from_nominal(calibration_rows, config.threshold_fpr)
    rows = _score_split(
        object_ids=object_ids,
        paths=paths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        contrastive_model=contrastive_model,
        mapper=mapper,
        device=device,
        manifest=manifest,
    )
    _add_threshold_predictions(rows, thresholds)

    metrics = _classification_metrics(rows)
    metrics.update(_threshold_metrics(rows, thresholds))
    metrics.update(
        {
            "n_objects": len(rows),
            "n_anomalies": sum(int(row["label"]) for row in rows),
            "anomaly_fraction": config.anomaly_fraction,
            "anomaly_seed": config.anomaly_seed,
            "anomaly_modality": config.anomaly_modality,
            "image_anomaly_kind": config.image_anomaly_kind,
            "spectrum_anomaly_kind": config.spectrum_anomaly_kind,
            "anomaly_strength": config.anomaly_strength,
            "threshold_calibration_split": config.threshold_calibration_split,
            "threshold_fpr": config.threshold_fpr,
            "contrastive_checkpoint_path": config.contrastive_checkpoint_path,
            "mapper_checkpoint_path": config.mapper_checkpoint_path or "",
        }
    )
    _write_rows(output_dir / "scores.csv", rows)
    _write_rows(output_dir / "calibration_scores.csv", calibration_rows)
    _write_json(output_dir / "thresholds.json", thresholds)
    _write_json(output_dir / "manifest.json", {key: spec.to_dict() for key, spec in manifest.items()})
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "config.json", asdict(config))
    print(_format_metrics(metrics), flush=True)
    return metrics


def _score_split(
    object_ids: tuple[str, ...],
    paths: DataPaths,
    batch_size: int,
    num_workers: int,
    contrastive_model: ContrastiveModel,
    mapper: CrossModalMapper | None,
    device: torch.device,
    manifest: dict[str, AnomalySpec],
) -> list[dict[str, str | float | int]]:
    dataset = TorchMultimodalPairDataset(object_ids, paths.images_h5, paths.spectra_h5)
    rows: list[dict[str, str | float | int]] = []
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=multimodal_collate,
        )
        with torch.no_grad():
            for batch in loader:
                anomalous_batch = apply_anomalies_to_batch(batch, manifest)
                tensor_batch = _to_device(anomalous_batch, device)
                outputs = contrastive_model(tensor_batch)
                scores = _score_batch(
                    outputs["image_embedding"],
                    outputs["spectrum_embedding"],
                    mapper,
                )
                object_id_list = anomalous_batch["object_id"]
                if not isinstance(object_id_list, list):
                    raise TypeError("object_id must be a list")
                for row, object_id in enumerate(object_id_list):
                    spec = manifest.get(str(object_id))
                    rows.append(
                        {
                            "object_id": str(object_id),
                            "label": int(spec is not None),
                            "modality": "" if spec is None else spec.modality,
                            "kind": "" if spec is None else spec.kind,
                            "strength": "" if spec is None else spec.strength,
                            **{key: float(value[row].item()) for key, value in scores.items()},
                        }
                    )
    finally:
        dataset.close()
    return rows


def _score_batch(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    mapper: CrossModalMapper | None,
) -> dict[str, torch.Tensor]:
    pair_distance = 1.0 - torch.sum(image_embedding * spectrum_embedding, dim=1)
    scores = {"pair_cosine_distance": pair_distance}
    if mapper is None:
        scores["combined_score"] = pair_distance
        return scores
    outputs = mapper(image_embedding, spectrum_embedding)
    i2s = 1.0 - torch.sum(outputs["pred_spectrum_embedding"] * spectrum_embedding, dim=1)
    s2i = 1.0 - torch.sum(outputs["pred_image_embedding"] * image_embedding, dim=1)
    scores.update(
        {
            "image_to_spectrum_mapping_error": i2s,
            "spectrum_to_image_mapping_error": s2i,
            "crossmodal_mapping_mean_error": 0.5 * (i2s + s2i),
            "combined_score": 0.5 * pair_distance + 0.25 * i2s + 0.25 * s2i,
        }
    )
    return scores


def _classification_metrics(rows: list[dict[str, str | float | int]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows])
    if labels.min(initial=0) == labels.max(initial=0):
        return {}
    score_names = [
        key
        for key in rows[0]
        if key.endswith("score") or key.endswith("distance") or key.endswith("error")
    ]
    metrics: dict[str, float] = {}
    for score_name in score_names:
        scores = np.asarray([float(row[score_name]) for row in rows])
        metrics[f"{score_name}_auroc"] = float(roc_auc_score(labels, scores))
        metrics[f"{score_name}_auprc"] = float(average_precision_score(labels, scores))
        metrics[f"{score_name}_recall_at_1pct_fpr"] = _recall_at_fpr(labels, scores, 0.01)
        metrics[f"{score_name}_recall_at_5pct_fpr"] = _recall_at_fpr(labels, scores, 0.05)
    return metrics


def _thresholds_from_nominal(
    calibration_rows: list[dict[str, str | float | int]],
    target_fpr: float,
) -> dict[str, float]:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"threshold_fpr must be in (0, 1), got {target_fpr}")
    thresholds: dict[str, float] = {}
    for score_name in _score_names(calibration_rows):
        scores = np.asarray([float(row[score_name]) for row in calibration_rows])
        thresholds[score_name] = float(np.quantile(scores, 1.0 - target_fpr))
    return thresholds


def _add_threshold_predictions(
    rows: list[dict[str, str | float | int]],
    thresholds: dict[str, float],
) -> None:
    for row in rows:
        for score_name, threshold in thresholds.items():
            row[f"{score_name}_threshold"] = threshold
            row[f"{score_name}_is_anomaly"] = int(float(row[score_name]) > threshold)


def _threshold_metrics(
    rows: list[dict[str, str | float | int]],
    thresholds: dict[str, float],
) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows])
    metrics: dict[str, float] = {}
    for score_name, threshold in thresholds.items():
        predictions = np.asarray([int(row[f"{score_name}_is_anomaly"]) for row in rows])
        tp = float(((predictions == 1) & (labels == 1)).sum())
        fp = float(((predictions == 1) & (labels == 0)).sum())
        tn = float(((predictions == 0) & (labels == 0)).sum())
        fn = float(((predictions == 0) & (labels == 1)).sum())
        metrics[f"{score_name}_threshold"] = threshold
        metrics[f"{score_name}_threshold_precision"] = _safe_divide(tp, tp + fp)
        metrics[f"{score_name}_threshold_recall"] = _safe_divide(tp, tp + fn)
        metrics[f"{score_name}_threshold_fpr"] = _safe_divide(fp, fp + tn)
        metrics[f"{score_name}_threshold_predicted_fraction"] = _safe_divide(
            tp + fp,
            len(rows),
        )
    return metrics


def _score_names(rows: list[dict[str, str | float | int]]) -> list[str]:
    if not rows:
        return []
    return [
        key
        for key in rows[0]
        if key.endswith("score") or key.endswith("distance") or key.endswith("error")
    ]


def _recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        return float("nan")
    threshold = np.quantile(normal_scores, 1.0 - max_fpr)
    return float((anomaly_scores >= threshold).mean())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _select_split(
    paths: DataPaths,
    paired_keys: frozenset[str],
    split: str,
    fold: int,
) -> tuple[str, ...]:
    if split == "test":
        return load_test(paths.partitions_dir).filter(paired_keys).all
    partition = load_fold(paths.partitions_dir, fold=fold)
    if split == "train":
        return partition.train.filter(paired_keys).all
    if split == "val":
        return partition.val.filter(paired_keys).all
    raise ValueError(f"Unknown split {split!r}; expected train, val, or test")


def _load_contrastive_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ContrastiveModel, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", {})
    projection_dim = int(model_config.get("projection_dim", 128))
    model = ContrastiveModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        projection_dim=projection_dim,
        encoder_width=float(model_config.get("encoder_width", 1.0)),
        encoder_variant=str(model_config.get("encoder_variant", "simple")),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, projection_dim


def _load_mapper(
    checkpoint_path: str | None,
    embedding_dim: int,
    device: torch.device,
) -> CrossModalMapper | None:
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    mapper = CrossModalMapper(
        embedding_dim=int(checkpoint.get("embedding_dim", embedding_dim)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    mapper.load_state_dict(checkpoint["mapper_state_dict"])
    mapper.eval()
    return mapper


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_rows(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_metrics(metrics: dict[str, float | int | str]) -> str:
    return (
        f"n={metrics['n_objects']} anomalies={metrics['n_anomalies']} "
        f"pair_auroc={metrics.get('pair_cosine_distance_auroc', float('nan')):.3f} "
        f"combined_auroc={metrics.get('combined_score_auroc', float('nan')):.3f}"
    )
