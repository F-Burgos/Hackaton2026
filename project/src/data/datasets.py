from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


class MultimodalPairDataset:
    """Lazy HDF5-backed dataset for paired image-spectrum objects."""

    def __init__(
        self,
        object_ids: Sequence[str],
        images_h5: str | Path,
        spectra_h5: str | Path,
    ) -> None:
        self.object_ids = tuple(object_ids)
        self.images_h5 = Path(images_h5)
        self.spectra_h5 = Path(spectra_h5)
        self._images_handle: h5py.File | None = None
        self._spectra_handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        object_id = self.object_ids[index]
        image_group = self._images[object_id]
        spectrum_group = self._spectra[object_id]
        return {
            "object_id": object_id,
            "img": image_group["img"][...],
            "img_channel_mask": image_group["img_channel_mask"][...],
            "flux_lambda_normalized": spectrum_group["flux_lambda_normalized"][...],
            "mask_spectra": spectrum_group["mask_spectra"][...],
            "wave": spectrum_group["wave"][...],
        }

    def close(self) -> None:
        for handle in [self._images_handle, self._spectra_handle]:
            if handle is not None:
                handle.close()
        self._images_handle = None
        self._spectra_handle = None

    def __enter__(self) -> "MultimodalPairDataset":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def _images(self) -> h5py.File:
        if self._images_handle is None:
            self._images_handle = h5py.File(self.images_h5, "r")
        return self._images_handle

    @property
    def _spectra(self) -> h5py.File:
        if self._spectra_handle is None:
            self._spectra_handle = h5py.File(self.spectra_h5, "r")
        return self._spectra_handle


def validate_sample(sample: dict[str, np.ndarray | str]) -> None:
    for name in ["img", "img_channel_mask", "flux_lambda_normalized", "mask_spectra", "wave"]:
        value = sample[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} should be a numpy array, got {type(value)!r}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    img = sample["img"]
    img_mask = sample["img_channel_mask"]
    flux = sample["flux_lambda_normalized"]
    spec_mask = sample["mask_spectra"]
    wave = sample["wave"]
    if img.ndim != 3 or img.shape[1:] != (64, 64):
        raise ValueError(f"Unexpected image shape: {img.shape}")
    if img_mask.shape != (img.shape[0],):
        raise ValueError(f"Image mask shape {img_mask.shape} does not match image channels {img.shape[0]}")
    if flux.shape != (739,) or spec_mask.shape != (739,) or wave.shape != (739,):
        raise ValueError(
            "Unexpected spectrum shapes: "
            f"flux={flux.shape}, mask={spec_mask.shape}, wave={wave.shape}"
        )
