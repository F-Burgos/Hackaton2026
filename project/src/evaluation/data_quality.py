from __future__ import annotations

import torch


@torch.no_grad()
def batch_quality_metrics(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    image_mask = batch["img_channel_mask"].float()
    spectrum_mask = batch["mask_spectra"].float()
    return {
        "image_channel_valid_fraction": float(image_mask.mean().item()),
        "spectrum_valid_fraction": float(spectrum_mask.mean().item()),
    }


def mean_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {key: float(sum(record[key] for record in records) / len(records)) for key in keys}
