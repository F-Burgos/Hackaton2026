from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


SplitName = Literal["train", "val", "test"]
FoldName = Literal["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]


@dataclass(frozen=True)
class SplitIds:
    hst: tuple[str, ...]
    jwst: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return self.hst + self.jwst

    def filter(self, available: set[str] | frozenset[str]) -> "SplitIds":
        return SplitIds(
            hst=tuple(object_id for object_id in self.hst if object_id in available),
            jwst=tuple(object_id for object_id in self.jwst if object_id in available),
        )


@dataclass(frozen=True)
class FoldPartition:
    fold: int
    train: SplitIds
    val: SplitIds


def load_fold(partitions_dir: str | Path, fold: int) -> FoldPartition:
    if fold < 0 or fold > 4:
        raise ValueError(f"Expected fold in [0, 4], got {fold}")
    path = Path(partitions_dir) / f"fold_{fold}.json"
    data = _load_json(path)
    return FoldPartition(
        fold=int(data["fold"]),
        train=_split_ids(data["train"]),
        val=_split_ids(data["val"]),
    )


def load_test(partitions_dir: str | Path) -> SplitIds:
    data = _load_json(Path(partitions_dir) / "test.json")
    return _split_ids(data["test"])


def filtered_fold_counts(partitions_dir: str | Path, available: set[str] | frozenset[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fold in range(5):
        partition = load_fold(partitions_dir, fold)
        counts[f"fold_{fold}_train"] = len(partition.train.filter(available).all)
        counts[f"fold_{fold}_val"] = len(partition.val.filter(available).all)
    counts["test"] = len(load_test(partitions_dir).filter(available).all)
    return counts


def assert_no_overlap(*splits: Iterable[str]) -> None:
    seen: set[str] = set()
    for split in splits:
        current = set(split)
        overlap = seen & current
        if overlap:
            example = sorted(overlap)[:5]
            raise ValueError(f"Partition overlap detected, examples: {example}")
        seen.update(current)


def _split_ids(data: dict[str, list[str]]) -> SplitIds:
    return SplitIds(hst=tuple(data.get("hst", [])), jwst=tuple(data.get("jwst", [])))


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
