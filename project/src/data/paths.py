from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path
    images_h5: Path
    spectra_h5: Path
    partitions_dir: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path = "hackaton",
        images_h5: str = "images_reduced.h5",
        spectra_h5: str = "spectra.h5",
        partitions_dir: str = "partitions/v1/ssl",
    ) -> "DataPaths":
        root_path = Path(root)
        return cls(
            root=root_path,
            images_h5=root_path / images_h5,
            spectra_h5=root_path / spectra_h5,
            partitions_dir=root_path / partitions_dir,
        )

    def validate(self) -> None:
        missing = [
            path
            for path in [self.root, self.images_h5, self.spectra_h5, self.partitions_dir]
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing expected dataset paths:\n{formatted}")
