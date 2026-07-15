from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from project.src.data.hdf5_index import Hdf5KeyIndex
from project.src.data.partitions import load_fold
from project.src.data.paths import DataPaths
from project.src.data.torch_datasets import TorchMultimodalPairDataset, multimodal_collate
from project.src.evaluation.data_quality import batch_quality_metrics, mean_metric_records
from project.src.evaluation.retrieval import embedding_diagnostics, recall_at_k
from project.src.models.contrastive import ContrastiveModel
from project.src.models.losses import symmetric_clip_loss


@dataclass(frozen=True)
class ContrastiveTrainConfig:
    data_root: str = "hackaton"
    fold: int = 0
    batch_size: int = 32
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    projection_dim: int = 128
    temperature: float = 0.07
    seed: int = 42
    device: str = "auto"
    output_dir: str = "project/results/contrastive"
    save_checkpoint: bool = True


def run_contrastive_training(config: ContrastiveTrainConfig) -> dict[str, float | int | str]:
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    paths = DataPaths.from_root(config.data_root)
    index = Hdf5KeyIndex.from_paths(paths)
    fold = load_fold(paths.partitions_dir, config.fold)
    train_ids = _limit(fold.train.filter(index.paired_keys).all, config.max_train_samples)
    val_ids = _limit(fold.val.filter(index.paired_keys).all, config.max_val_samples)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TorchMultimodalPairDataset(train_ids, paths.images_h5, paths.spectra_h5)
    val_dataset = TorchMultimodalPairDataset(val_ids, paths.images_h5, paths.spectra_h5)
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=multimodal_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=multimodal_collate,
        )

        model = ContrastiveModel(
            embedding_dim=config.embedding_dim,
            projection_dim=config.projection_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        best_val_loss = float("inf")
        latest_metrics: dict[str, float | int | str] = {}
        history: list[dict[str, float | int | str]] = []
        for epoch in range(1, config.epochs + 1):
            started = time()
            train_metrics = _train_epoch(model, train_loader, optimizer, config, device)
            val_metrics = _evaluate(model, val_loader, config, device)
            latest_metrics = {
                "epoch": epoch,
                "seconds": round(time() - started, 3),
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(latest_metrics)
            _append_jsonl(output_dir / "metrics.jsonl", latest_metrics)
            if config.save_checkpoint and val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "metrics": latest_metrics,
                    },
                    output_dir / "best.pt",
                )
            print(_format_metrics(latest_metrics), flush=True)

        _write_json(output_dir / "summary.json", latest_metrics)
        return latest_metrics
    finally:
        train_dataset.close()
        val_dataset.close()


def _train_epoch(
    model: ContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ContrastiveTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    for batch in loader:
        tensor_batch = _to_device(batch, device)
        outputs = model(tensor_batch)
        loss = symmetric_clip_loss(
            outputs["image_embedding"],
            outputs["spectrum_embedding"],
            temperature=config.temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return {"loss": _mean(losses)}


@torch.no_grad()
def _evaluate(
    model: ContrastiveModel,
    loader: DataLoader,
    config: ContrastiveTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    image_embeddings: list[torch.Tensor] = []
    spectrum_embeddings: list[torch.Tensor] = []
    quality_records: list[dict[str, float]] = []
    for batch in loader:
        tensor_batch = _to_device(batch, device)
        outputs = model(tensor_batch)
        loss = symmetric_clip_loss(
            outputs["image_embedding"],
            outputs["spectrum_embedding"],
            temperature=config.temperature,
        )
        losses.append(float(loss.item()))
        image_embeddings.append(outputs["image_embedding"].detach().cpu())
        spectrum_embeddings.append(outputs["spectrum_embedding"].detach().cpu())
        quality_records.append(batch_quality_metrics(tensor_batch))
    image = torch.cat(image_embeddings)
    spectrum = torch.cat(spectrum_embeddings)
    image_to_spectrum = recall_at_k(image, spectrum)
    spectrum_to_image = recall_at_k(spectrum, image)
    return {
        "loss": _mean(losses),
        **{f"i2s_{key}": value for key, value in image_to_spectrum.items()},
        **{f"s2i_{key}": value for key, value in spectrum_to_image.items()},
        **embedding_diagnostics(image, "image"),
        **embedding_diagnostics(spectrum, "spectrum"),
        **mean_metric_records(quality_records),
    }


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _limit(values: tuple[str, ...], max_items: int | None) -> tuple[str, ...]:
    if max_items is None:
        return values
    return values[:max_items]


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _append_jsonl(path: Path, record: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, record: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_metrics(metrics: dict[str, float | int | str]) -> str:
    fields = [
        f"epoch={metrics['epoch']}",
        f"train_loss={metrics['train_loss']:.4f}",
        f"val_loss={metrics['val_loss']:.4f}",
        f"i2s_r1={metrics['val_i2s_recall@1']:.3f}",
        f"s2i_r1={metrics['val_s2i_recall@1']:.3f}",
        f"seconds={metrics['seconds']}",
    ]
    return " ".join(fields)
