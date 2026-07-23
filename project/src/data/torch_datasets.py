from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from project.src.data.datasets import MultimodalPairDataset


class TorchMultimodalPairDataset(MultimodalPairDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        return {
            "object_id": sample["object_id"],
            "img": torch.from_numpy(np.asarray(sample["img"], dtype=np.float32)),
            "img_channel_mask": torch.from_numpy(
                np.asarray(sample["img_channel_mask"], dtype=np.float32)
            ),
            "flux_lambda_normalized": torch.from_numpy(
                np.asarray(sample["flux_lambda_normalized"], dtype=np.float32)
            ),
            "mask_spectra": torch.from_numpy(np.asarray(sample["mask_spectra"], dtype=np.float32)),
            "wave": torch.from_numpy(np.asarray(sample["wave"], dtype=np.float32)),
        }


def multimodal_collate(
    samples: Sequence[dict[str, torch.Tensor | str]],
    max_image_channels: int = 9,
) -> dict[str, torch.Tensor | list[str]]:
    object_ids: list[str] = []
    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    fluxes: list[torch.Tensor] = []
    spectrum_masks: list[torch.Tensor] = []
    waves: list[torch.Tensor] = []
    is_anomaly: list[bool] = []
    anomaly_modalities: list[str] = []
    anomaly_kinds: list[str] = []
    has_anomaly_fields = any("is_anomaly" in sample for sample in samples)

    for sample in samples:
        object_ids.append(str(sample["object_id"]))
        if has_anomaly_fields:
            is_anomaly.append(bool(sample.get("is_anomaly", False)))
            anomaly_modalities.append(str(sample.get("anomaly_modality", "none")))
            anomaly_kinds.append(str(sample.get("anomaly_kind", "none")))

        image = _expect_tensor(sample["img"])
        image_mask = _expect_tensor(sample["img_channel_mask"])
        if image.shape[0] > max_image_channels:
            raise ValueError(f"Image has {image.shape[0]} channels, max is {max_image_channels}")

        padded_image = torch.zeros((max_image_channels, 64, 64), dtype=torch.float32)
        padded_image[: image.shape[0]] = image
        padded_mask = torch.zeros((max_image_channels,), dtype=torch.float32)
        padded_mask[: image_mask.shape[0]] = image_mask

        images.append(padded_image)
        image_masks.append(padded_mask)
        fluxes.append(_expect_tensor(sample["flux_lambda_normalized"]).float())
        spectrum_masks.append(_expect_tensor(sample["mask_spectra"]).float())
        waves.append(_expect_tensor(sample["wave"]).float())

    batch: dict[str, torch.Tensor | list[str]] = {
        "object_id": object_ids,
        "img": torch.stack(images),
        "img_channel_mask": torch.stack(image_masks),
        "flux_lambda_normalized": torch.stack(fluxes),
        "mask_spectra": torch.stack(spectrum_masks),
        "wave": torch.stack(waves),
    }
    if has_anomaly_fields:
        batch["is_anomaly"] = torch.tensor(is_anomaly, dtype=torch.bool)
        batch["anomaly_modality"] = anomaly_modalities
        batch["anomaly_kind"] = anomaly_kinds
    return batch


def _expect_tensor(value: torch.Tensor | str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(value)!r}")
    return value
