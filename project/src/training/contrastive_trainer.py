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
from project.src.evaluation.retrieval import (
    embedding_diagnostics,
    pair_similarity_diagnostics,
    ranking_metrics,
    recall_at_k,
)
from project.src.models.contrastive import ContrastiveModel
from project.src.models.losses import info_nce_loss, symmetric_info_nce_loss


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
    contrastive_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    lr_scheduler: str = "none"
    min_learning_rate: float = 0.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    contrastive_loss: str = "symmetric_info_nce"
    embedding_dim: int = 128
    projection_dim: int = 128
    encoder_width: float = 1.0
    encoder_variant: str = "simple"
    dropout: float = 0.0
    temperature: float = 0.07
    seed: int = 42
    device: str = "auto"
    output_dir: str = "project/results/contrastive"
    save_checkpoint: bool = True


def run_contrastive_training(config: ContrastiveTrainConfig) -> dict[str, float | int | str]:
    _validate_config(config)
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
            encoder_width=config.encoder_width,
            encoder_variant=config.encoder_variant,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, config)

        best_val_loss = float("inf")
        best_metrics: dict[str, float | int | str] = {}
        latest_metrics: dict[str, float | int | str] = {}
        history: list[dict[str, float | int | str]] = []
        epochs_without_improvement = 0
        for epoch in range(1, config.epochs + 1):
            started = time()
            train_metrics = _train_epoch(model, train_loader, optimizer, config, device)
            val_metrics = _evaluate(model, val_loader, config, device)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            latest_metrics = {
                "epoch": epoch,
                "seconds": round(time() - started, 3),
                "learning_rate": learning_rate,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "contrastive_loss": config.contrastive_loss,
                "temperature": config.temperature,
                "stopped_early": False,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(latest_metrics)
            if _is_improvement(float(val_metrics["loss"]), best_val_loss, config):
                best_val_loss = float(val_metrics["loss"])
                epochs_without_improvement = 0
                best_metrics = latest_metrics.copy()
                if config.save_checkpoint:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": asdict(config),
                            "metrics": latest_metrics,
                        },
                        output_dir / "best.pt",
                    )
            else:
                epochs_without_improvement += 1
            should_stop_early = _should_stop_early(epochs_without_improvement, config)
            latest_metrics["stopped_early"] = should_stop_early
            _append_jsonl(output_dir / "metrics.jsonl", latest_metrics)
            print(_format_metrics(latest_metrics), flush=True)
            if scheduler is not None:
                scheduler.step()
            if should_stop_early:
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


def _train_epoch(
    model: ContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ContrastiveTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []
    image_embedding_chunks: list[torch.Tensor] = []
    spectrum_embedding_chunks: list[torch.Tensor] = []
    optimizer_steps = 0
    for batch in loader:
        tensor_batch = _to_device(batch, device)
        outputs = model(tensor_batch)
        image_embedding_chunks.append(outputs["image_embedding"])
        spectrum_embedding_chunks.append(outputs["spectrum_embedding"])
        if len(image_embedding_chunks) < config.contrastive_accumulation_steps:
            continue
        loss, grad_norm = _optimize_contrastive_group(
            model,
            optimizer,
            config,
            image_embedding_chunks,
            spectrum_embedding_chunks,
        )
        optimizer_steps += 1
        image_embedding_chunks = []
        spectrum_embedding_chunks = []
        if grad_norm is not None:
            grad_norms.append(grad_norm)
        losses.append(float(loss.item()))
    if image_embedding_chunks:
        loss, grad_norm = _optimize_contrastive_group(
            model,
            optimizer,
            config,
            image_embedding_chunks,
            spectrum_embedding_chunks,
        )
        optimizer_steps += 1
        if grad_norm is not None:
            grad_norms.append(grad_norm)
        losses.append(float(loss.item()))
    metrics = {
        "loss": _mean(losses),
        "contrastive_accumulation_steps": float(config.contrastive_accumulation_steps),
        "effective_contrastive_batch_size": float(
            config.batch_size * config.contrastive_accumulation_steps
        ),
        "optimizer_steps": float(optimizer_steps),
    }
    if grad_norms:
        metrics["grad_norm"] = _mean(grad_norms)
    return metrics


def _optimize_contrastive_group(
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    config: ContrastiveTrainConfig,
    image_embedding_chunks: list[torch.Tensor],
    spectrum_embedding_chunks: list[torch.Tensor],
) -> tuple[torch.Tensor, float | None]:
    image_embedding = torch.cat(image_embedding_chunks, dim=0)
    spectrum_embedding = torch.cat(spectrum_embedding_chunks, dim=0)
    loss = _contrastive_loss(
        image_embedding,
        spectrum_embedding,
        config,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = None
    if config.gradient_clip_norm is not None:
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip_norm,
        )
        grad_norm = float(clipped_grad_norm.item())
    optimizer.step()
    return loss.detach(), grad_norm


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
        loss = _contrastive_loss(
            outputs["image_embedding"],
            outputs["spectrum_embedding"],
            config,
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
        **{f"i2s_{key}": value for key, value in ranking_metrics(image, spectrum).items()},
        **{f"s2i_{key}": value for key, value in ranking_metrics(spectrum, image).items()},
        **embedding_diagnostics(image, "image"),
        **embedding_diagnostics(spectrum, "spectrum"),
        **pair_similarity_diagnostics(image, spectrum),
        **mean_metric_records(quality_records),
    }


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_config(config: ContrastiveTrainConfig) -> None:
    if config.contrastive_accumulation_steps < 1:
        raise ValueError("contrastive_accumulation_steps must be >= 1")
    if config.contrastive_loss not in {
        "symmetric_info_nce",
        "image_to_spectrum_info_nce",
        "spectrum_to_image_info_nce",
    }:
        raise ValueError(
            "contrastive_loss must be one of: symmetric_info_nce, "
            "image_to_spectrum_info_nce, spectrum_to_image_info_nce"
        )


def _contrastive_loss(
    image_embedding: torch.Tensor,
    spectrum_embedding: torch.Tensor,
    config: ContrastiveTrainConfig,
) -> torch.Tensor:
    if config.contrastive_loss == "symmetric_info_nce":
        return symmetric_info_nce_loss(
            image_embedding,
            spectrum_embedding,
            temperature=config.temperature,
        )
    if config.contrastive_loss == "image_to_spectrum_info_nce":
        return info_nce_loss(
            image_embedding,
            spectrum_embedding,
            temperature=config.temperature,
        )
    if config.contrastive_loss == "spectrum_to_image_info_nce":
        return info_nce_loss(
            spectrum_embedding,
            image_embedding,
            temperature=config.temperature,
        )
    raise ValueError(f"Unknown contrastive_loss {config.contrastive_loss!r}")


def _limit(values: tuple[str, ...], max_items: int | None) -> tuple[str, ...]:
    if max_items is None:
        return values
    return values[:max_items]


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ContrastiveTrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.min_learning_rate,
        )
    raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}; expected 'none' or 'cosine'")


def _is_improvement(
    val_loss: float,
    best_val_loss: float,
    config: ContrastiveTrainConfig,
) -> bool:
    return val_loss < best_val_loss - config.early_stopping_min_delta


def _should_stop_early(
    epochs_without_improvement: int,
    config: ContrastiveTrainConfig,
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
    fields = [
        f"epoch={metrics['epoch']}",
        f"train_loss={metrics['train_loss']:.4f}",
        f"val_loss={metrics['val_loss']:.4f}",
        f"i2s_r1={metrics['val_i2s_recall@1']:.3f}",
        f"s2i_r1={metrics['val_s2i_recall@1']:.3f}",
        f"seconds={metrics['seconds']}",
    ]
    return " ".join(fields)
