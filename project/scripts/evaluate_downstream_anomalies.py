from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.downstream.anomaly_evaluation import (  # noqa: E402
    DownstreamAnomalyConfig,
    evaluate_downstream_anomalies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic downstream anomalies.")
    parser.add_argument("--config", default="project/configs/downstream/crossmodal.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    eval_cfg = DownstreamAnomalyConfig(
        contrastive_checkpoint_path=str(cfg.contrastive.checkpoint_path),
        mapper_checkpoint_path=_optional_str(cfg.eval.mapper_checkpoint_path),
        data_root=str(cfg.data.root),
        fold=int(cfg.data.fold),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_samples=_optional_int(cfg.eval.max_samples),
        anomaly_fraction=float(cfg.anomaly.fraction),
        anomaly_seed=int(cfg.anomaly.seed),
        anomaly_modality=str(cfg.anomaly.modality),
        image_anomaly_kind=str(cfg.anomaly.image_kind),
        spectrum_anomaly_kind=str(cfg.anomaly.spectrum_kind),
        anomaly_strength=float(cfg.anomaly.strength),
        threshold_calibration_split=str(cfg.threshold.calibration_split),
        threshold_fpr=float(cfg.threshold.fpr),
        device=str(cfg.resources.device),
        output_dir=str(cfg.outputs.anomaly_dir),
    )
    evaluate_downstream_anomalies(eval_cfg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return str(value)


if __name__ == "__main__":
    main()
