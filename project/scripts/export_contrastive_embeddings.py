from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.evaluation.contrastive_export import (  # noqa: E402
    ContrastiveExportConfig,
    export_contrastive_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export embeddings from a contrastive checkpoint.")
    parser.add_argument("--config", default="project/configs/default.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. eval.split=test eval.max_samples=128",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    export_cfg = ContrastiveExportConfig(
        checkpoint_path=str(cfg.eval.checkpoint_path),
        data_root=str(cfg.data.root),
        split=str(cfg.eval.split),
        fold=int(cfg.data.fold),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_samples=_optional_int(cfg.eval.max_samples),
        device=str(cfg.resources.device),
        output_dir=str(cfg.eval.output_dir),
    )
    export_contrastive_embeddings(export_cfg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
