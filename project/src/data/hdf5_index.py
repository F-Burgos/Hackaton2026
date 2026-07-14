from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from project.src.data.paths import DataPaths


@dataclass(frozen=True)
class Hdf5KeyIndex:
    image_keys: frozenset[str]
    spectrum_keys: frozenset[str]

    @property
    def paired_keys(self) -> frozenset[str]:
        return self.image_keys & self.spectrum_keys

    @property
    def spectrum_only_keys(self) -> frozenset[str]:
        return self.spectrum_keys - self.image_keys

    @property
    def image_only_keys(self) -> frozenset[str]:
        return self.image_keys - self.spectrum_keys

    @classmethod
    def from_paths(cls, paths: DataPaths) -> "Hdf5KeyIndex":
        paths.validate()
        return cls(
            image_keys=frozenset(_read_top_level_keys(paths.images_h5)),
            spectrum_keys=frozenset(_read_top_level_keys(paths.spectra_h5)),
        )


def _read_top_level_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return list(handle.keys())


def summarize_hdf5_schema(paths: DataPaths, sample_keys: int = 3) -> dict[str, object]:
    paths.validate()
    summary: dict[str, object] = {}
    for label, path in [("images", paths.images_h5), ("spectra", paths.spectra_h5)]:
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            sample = {}
            for key in keys[:sample_keys]:
                group = handle[key]
                sample[key] = {
                    name: {
                        "shape": tuple(dataset.shape),
                        "dtype": str(dataset.dtype),
                        "compression": dataset.compression,
                    }
                    for name, dataset in group.items()
                }
            summary[label] = {
                "path": str(path),
                "n_objects": len(keys),
                "first_keys": keys[:sample_keys],
                "sample_schema": sample,
            }
    return summary
