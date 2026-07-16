from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from project.src.data.hdf5_index import Hdf5KeyIndex
from project.src.data.partitions import load_fold, load_test
from project.src.data.paths import DataPaths
from project.src.data.torch_datasets import TorchMultimodalPairDataset, multimodal_collate
from project.src.evaluation.data_quality import batch_quality_metrics, mean_metric_records
from project.src.evaluation.retrieval import (
    embedding_diagnostics,
    pair_similarity_diagnostics,
    recall_at_k,
)
from project.src.models.contrastive import ContrastiveModel


@dataclass(frozen=True)
class ContrastiveExportConfig:
    checkpoint_path: str
    data_root: str = "hackaton"
    split: str = "val"
    fold: int = 0
    batch_size: int = 64
    num_workers: int = 0
    max_samples: int | None = None
    device: str = "auto"
    output_dir: str = "project/results/contrastive/export"


def export_contrastive_embeddings(config: ContrastiveExportConfig) -> dict[str, float | int | str]:
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", {})
    model = ContrastiveModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        projection_dim=int(model_config.get("projection_dim", 128)),
        encoder_width=float(model_config.get("encoder_width", 1.0)),
        encoder_variant=str(model_config.get("encoder_variant", "simple")),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    paths = DataPaths.from_root(config.data_root)
    index = Hdf5KeyIndex.from_paths(paths)
    object_ids = _select_split(paths, index.paired_keys, config.split, config.fold)
    if config.max_samples is not None:
        object_ids = object_ids[: config.max_samples]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TorchMultimodalPairDataset(object_ids, paths.images_h5, paths.spectra_h5)
    try:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=multimodal_collate,
        )
        all_ids: list[str] = []
        image_embeddings: list[torch.Tensor] = []
        spectrum_embeddings: list[torch.Tensor] = []
        quality_records: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                all_ids.extend(str(object_id) for object_id in batch["object_id"])
                tensor_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                outputs = model(tensor_batch)
                image_embeddings.append(outputs["image_embedding"].detach().cpu())
                spectrum_embeddings.append(outputs["spectrum_embedding"].detach().cpu())
                quality_records.append(batch_quality_metrics(tensor_batch))
    finally:
        dataset.close()

    image = torch.cat(image_embeddings)
    spectrum = torch.cat(spectrum_embeddings)
    metrics: dict[str, float | int | str] = {
        "checkpoint_path": str(config.checkpoint_path),
        "split": config.split,
        "fold": config.fold,
        "n_objects": len(all_ids),
        **{f"i2s_{key}": value for key, value in recall_at_k(image, spectrum).items()},
        **{f"s2i_{key}": value for key, value in recall_at_k(spectrum, image).items()},
        **embedding_diagnostics(image, "image"),
        **embedding_diagnostics(spectrum, "spectrum"),
        **pair_similarity_diagnostics(image, spectrum),
        **mean_metric_records(quality_records),
    }

    np.savez_compressed(
        output_dir / "embeddings.npz",
        object_id=np.asarray(all_ids),
        image_embedding=image.numpy(),
        spectrum_embedding=spectrum.numpy(),
    )
    _write_json(output_dir / "metrics.json", metrics)
    print(_format_metrics(metrics), flush=True)
    return metrics


def _select_split(
    paths: DataPaths,
    paired_keys: frozenset[str],
    split: str,
    fold: int,
) -> tuple[str, ...]:
    if split == "test":
        return load_test(paths.partitions_dir).filter(paired_keys).all
    partition = load_fold(paths.partitions_dir, fold)
    if split == "train":
        return partition.train.filter(paired_keys).all
    if split == "val":
        return partition.val.filter(paired_keys).all
    raise ValueError(f"Unknown split {split!r}; expected train, val, or test")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_json(path: Path, record: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_metrics(metrics: dict[str, float | int | str]) -> str:
    return (
        f"split={metrics['split']} n={metrics['n_objects']} "
        f"i2s_r1={metrics['i2s_recall@1']:.3f} "
        f"s2i_r1={metrics['s2i_recall@1']:.3f}"
    )
