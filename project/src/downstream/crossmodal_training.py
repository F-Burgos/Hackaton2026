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
from project.src.models.contrastive import ContrastiveModel
from project.src.models.crossmodal import CrossModalMapper, crossmodal_mapping_loss


@dataclass(frozen=True)
class CrossModalTrainConfig:
    contrastive_checkpoint_path: str
    data_root: str = "hackaton"
    fold: int = 0
    batch_size: int = 64
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 0.0
    seed: int = 42
    device: str = "auto"
    output_dir: str = "project/results/downstream/crossmodal_mapper"


def train_crossmodal_mapper(config: CrossModalTrainConfig) -> dict[str, float | int | str]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    contrastive_model, embedding_dim = _load_contrastive_model(
        config.contrastive_checkpoint_path,
        device,
    )
    mapper = CrossModalMapper(
        embedding_dim=embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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

        best_val_loss = float("inf")
        best_metrics: dict[str, float | int | str] = {}
        latest_metrics: dict[str, float | int | str] = {}
        epochs_without_improvement = 0
        for epoch in range(1, config.epochs + 1):
            started = time()
            train_metrics = _run_epoch(
                contrastive_model,
                mapper,
                train_loader,
                config,
                device,
                optimizer,
            )
            val_metrics = _run_epoch(contrastive_model, mapper, val_loader, config, device, None)
            latest_metrics = {
                "epoch": epoch,
                "seconds": round(time() - started, 3),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "stopped_early": False,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            if _is_improvement(float(val_metrics["loss"]), best_val_loss, config):
                best_val_loss = float(val_metrics["loss"])
                epochs_without_improvement = 0
                best_metrics = latest_metrics.copy()
                torch.save(
                    {
                        "mapper_state_dict": mapper.state_dict(),
                        "config": asdict(config),
                        "metrics": latest_metrics,
                        "embedding_dim": embedding_dim,
                    },
                    output_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1
            latest_metrics["stopped_early"] = _should_stop_early(
                epochs_without_improvement,
                config,
            )
            _append_jsonl(output_dir / "metrics.jsonl", latest_metrics)
            print(_format_metrics(latest_metrics), flush=True)
            if latest_metrics["stopped_early"]:
                print(
                    f"early_stop epoch={epoch} patience={config.early_stopping_patience}",
                    flush=True,
                )
                break
        summary = _with_best_metrics(latest_metrics, best_metrics)
        _write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        train_dataset.close()
        val_dataset.close()


def _run_epoch(
    contrastive_model: ContrastiveModel,
    mapper: CrossModalMapper,
    loader: DataLoader,
    config: CrossModalTrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    mapper.train(training)
    losses: list[float] = []
    image_to_spectrum_errors: list[torch.Tensor] = []
    spectrum_to_image_errors: list[torch.Tensor] = []
    for batch in loader:
        tensor_batch = _to_device(batch, device)
        with torch.no_grad():
            embeddings = contrastive_model(tensor_batch)
            image_embedding = embeddings["image_embedding"]
            spectrum_embedding = embeddings["spectrum_embedding"]
        outputs = mapper(image_embedding, spectrum_embedding)
        loss = crossmodal_mapping_loss(
            outputs["pred_spectrum_embedding"],
            spectrum_embedding,
            outputs["pred_image_embedding"],
            image_embedding,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().item()))
        with torch.no_grad():
            image_to_spectrum_errors.append(
                1.0
                - torch.sum(outputs["pred_spectrum_embedding"] * spectrum_embedding, dim=1).detach()
            )
            spectrum_to_image_errors.append(
                1.0
                - torch.sum(outputs["pred_image_embedding"] * image_embedding, dim=1).detach()
            )

    i2s = torch.cat(image_to_spectrum_errors)
    s2i = torch.cat(spectrum_to_image_errors)
    return {
        "loss": _mean(losses),
        "i2s_error_mean": float(i2s.mean().item()),
        "s2i_error_mean": float(s2i.mean().item()),
        "combined_error_mean": float((0.5 * (i2s + s2i)).mean().item()),
    }


def _load_contrastive_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ContrastiveModel, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", {})
    embedding_dim = int(model_config.get("embedding_dim", 128))
    projection_dim = int(model_config.get("projection_dim", 128))
    model = ContrastiveModel(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        encoder_width=float(model_config.get("encoder_width", 1.0)),
        encoder_variant=str(model_config.get("encoder_variant", "simple")),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, projection_dim


def _validate_config(config: CrossModalTrainConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.num_layers < 1:
        raise ValueError("num_layers must be >= 1")


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


def _is_improvement(
    val_loss: float,
    best_val_loss: float,
    config: CrossModalTrainConfig,
) -> bool:
    return val_loss < best_val_loss - config.early_stopping_min_delta


def _should_stop_early(
    epochs_without_improvement: int,
    config: CrossModalTrainConfig,
) -> bool:
    if config.early_stopping_patience is None:
        return False
    return epochs_without_improvement >= config.early_stopping_patience


def _append_jsonl(path: Path, record: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, record: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _with_best_metrics(
    latest_metrics: dict[str, float | int | str],
    best_metrics: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    if not best_metrics:
        return latest_metrics
    summary = latest_metrics.copy()
    for key, value in best_metrics.items():
        summary[f"best_{key}"] = value
    return summary


def _format_metrics(metrics: dict[str, float | int | str]) -> str:
    return (
        f"epoch={metrics['epoch']} "
        f"train_loss={metrics['train_loss']:.4f} "
        f"val_loss={metrics['val_loss']:.4f} "
        f"seconds={metrics['seconds']}"
    )
