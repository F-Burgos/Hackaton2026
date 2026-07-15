from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.training.contrastive_trainer import (  # noqa: E402
    ContrastiveTrainConfig,
    run_contrastive_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the base image-spectrum contrastive model.")
    parser.add_argument("--config", default="project/configs/default.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.epochs=2 data.batch_size=16",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    train_cfg = ContrastiveTrainConfig(
        data_root=str(cfg.data.root),
        fold=int(cfg.data.fold),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_train_samples=_optional_int(cfg.train.max_train_samples),
        max_val_samples=_optional_int(cfg.train.max_val_samples),
        epochs=int(cfg.train.epochs),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        embedding_dim=int(cfg.model.embedding_dim),
        projection_dim=int(cfg.model.projection_dim),
        temperature=float(cfg.train.temperature),
        seed=int(cfg.run.seed),
        device=str(cfg.resources.device),
        output_dir=str(cfg.outputs.contrastive_dir),
        save_checkpoint=bool(cfg.train.save_checkpoint),
    )
    run_contrastive_training(train_cfg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
