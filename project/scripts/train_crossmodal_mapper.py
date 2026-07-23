from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.downstream.crossmodal_training import (  # noqa: E402
    CrossModalTrainConfig,
    train_crossmodal_mapper,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bidirectional crossmodal mappers.")
    parser.add_argument("--config", default="project/configs/downstream/crossmodal.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    train_cfg = CrossModalTrainConfig(
        contrastive_checkpoint_path=str(cfg.contrastive.checkpoint_path),
        data_root=str(cfg.data.root),
        fold=int(cfg.data.fold),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_train_samples=_optional_int(cfg.train.max_train_samples),
        max_val_samples=_optional_int(cfg.train.max_val_samples),
        epochs=int(cfg.train.epochs),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        hidden_dim=int(cfg.mapper.hidden_dim),
        num_layers=int(cfg.mapper.num_layers),
        dropout=float(cfg.mapper.dropout),
        early_stopping_patience=_optional_int(cfg.train.early_stopping_patience),
        early_stopping_min_delta=float(cfg.train.early_stopping_min_delta),
        seed=int(cfg.run.seed),
        device=str(cfg.resources.device),
        output_dir=str(cfg.outputs.crossmodal_dir),
    )
    train_crossmodal_mapper(train_cfg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
