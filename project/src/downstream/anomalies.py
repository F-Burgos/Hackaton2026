from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch


ImageAnomalyKind = Literal["bright_patch", "dark_patch", "noise_patch", "channel_dropout"]
SpectrumAnomalyKind = Literal["spike", "segment_shift", "segment_noise", "continuum_tilt"]
ModalityName = Literal["image", "spectrum"]


@dataclass(frozen=True)
class AnomalySpec:
    object_id: str
    modality: ModalityName
    kind: str
    strength: float
    seed: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def build_anomaly_manifest(
    object_ids: list[str] | tuple[str, ...],
    fraction: float = 0.10,
    seed: int = 42,
    modality: Literal["image", "spectrum", "alternate", "random"] = "alternate",
    image_kind: ImageAnomalyKind = "bright_patch",
    spectrum_kind: SpectrumAnomalyKind = "spike",
    strength: float = 5.0,
) -> dict[str, AnomalySpec]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    ids = list(object_ids)
    n_anomalies = int(round(len(ids) * fraction))
    selected = rng.choice(len(ids), size=n_anomalies, replace=False) if n_anomalies else []
    manifest: dict[str, AnomalySpec] = {}
    for rank, index in enumerate(selected):
        object_id = ids[int(index)]
        selected_modality = _select_modality(modality, rank, rng)
        kind = image_kind if selected_modality == "image" else spectrum_kind
        manifest[object_id] = AnomalySpec(
            object_id=object_id,
            modality=selected_modality,
            kind=kind,
            strength=strength,
            seed=seed + rank + 1,
        )
    return manifest


def apply_anomalies_to_batch(
    batch: dict[str, torch.Tensor | list[str]],
    manifest: dict[str, AnomalySpec],
) -> dict[str, torch.Tensor | list[str]]:
    output: dict[str, torch.Tensor | list[str]] = {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in batch.items()
    }
    object_ids = output.get("object_id")
    if not isinstance(object_ids, list):
        raise TypeError("batch must contain object_id as a list")

    for row, object_id in enumerate(object_ids):
        spec = manifest.get(str(object_id))
        if spec is None:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(spec.seed)
        if spec.modality == "image":
            image = _expect_tensor(output["img"])[row]
            image_mask = _expect_tensor(output["img_channel_mask"])[row]
            _apply_image_anomaly(image, image_mask, spec, generator)
        elif spec.modality == "spectrum":
            flux = _expect_tensor(output["flux_lambda_normalized"])[row]
            mask = _expect_tensor(output["mask_spectra"])[row]
            _apply_spectrum_anomaly(flux, mask, spec, generator)
        else:
            raise ValueError(f"Unknown modality {spec.modality!r}")
    return output


def _apply_image_anomaly(
    image: torch.Tensor,
    channel_mask: torch.Tensor,
    spec: AnomalySpec,
    generator: torch.Generator,
) -> None:
    valid_channels = torch.where(channel_mask > 0)[0]
    if len(valid_channels) == 0:
        return
    channel = valid_channels[
        int(torch.randint(len(valid_channels), (1,), generator=generator).item())
    ]
    patch_size = int(torch.randint(6, 17, (1,), generator=generator).item())
    y0 = int(torch.randint(0, image.shape[-2] - patch_size + 1, (1,), generator=generator).item())
    x0 = int(torch.randint(0, image.shape[-1] - patch_size + 1, (1,), generator=generator).item())
    patch = image[channel, y0 : y0 + patch_size, x0 : x0 + patch_size]
    scale = image[channel].std(unbiased=False).clamp_min(1e-3)

    if spec.kind == "bright_patch":
        patch.add_(spec.strength * scale)
    elif spec.kind == "dark_patch":
        patch.sub_(spec.strength * scale)
    elif spec.kind == "noise_patch":
        noise = torch.randn(patch.shape, generator=generator, dtype=patch.dtype)
        patch.add_(noise.to(patch.device) * spec.strength * scale)
    elif spec.kind == "channel_dropout":
        image[channel].zero_()
    else:
        raise ValueError(f"Unknown image anomaly kind {spec.kind!r}")


def _apply_spectrum_anomaly(
    flux: torch.Tensor,
    mask: torch.Tensor,
    spec: AnomalySpec,
    generator: torch.Generator,
) -> None:
    valid = torch.where(mask > 0)[0]
    if len(valid) == 0:
        return
    scale = flux[valid].std(unbiased=False).clamp_min(1e-3)
    if spec.kind == "spike":
        width = min(int(torch.randint(1, 5, (1,), generator=generator).item()), len(valid))
        start = int(torch.randint(0, len(valid) - width + 1, (1,), generator=generator).item())
        indexes = valid[start : start + width]
        sign = -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
        flux[indexes] += sign * spec.strength * scale
    elif spec.kind in {"segment_shift", "segment_noise"}:
        width = min(int(torch.randint(24, 96, (1,), generator=generator).item()), len(valid))
        start = int(torch.randint(0, len(valid) - width + 1, (1,), generator=generator).item())
        indexes = valid[start : start + width]
        if spec.kind == "segment_shift":
            flux[indexes] += spec.strength * scale
        else:
            noise = torch.randn(indexes.shape, generator=generator, dtype=flux.dtype)
            flux[indexes] += noise.to(flux.device) * spec.strength * scale
    elif spec.kind == "continuum_tilt":
        centered = torch.linspace(-1.0, 1.0, len(valid), dtype=flux.dtype, device=flux.device)
        flux[valid] += centered * spec.strength * scale
    else:
        raise ValueError(f"Unknown spectrum anomaly kind {spec.kind!r}")


def _select_modality(
    modality: Literal["image", "spectrum", "alternate", "random"],
    rank: int,
    rng: np.random.Generator,
) -> ModalityName:
    if modality == "image":
        return "image"
    if modality == "spectrum":
        return "spectrum"
    if modality == "alternate":
        return "image" if rank % 2 == 0 else "spectrum"
    if modality == "random":
        return "image" if bool(rng.integers(0, 2)) else "spectrum"
    raise ValueError(f"Unknown anomaly modality mode {modality!r}")


def _expect_tensor(value: torch.Tensor | list[str]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(value)!r}")
    return value
