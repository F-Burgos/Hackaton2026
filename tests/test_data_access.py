from __future__ import annotations

from pathlib import Path

import pytest

from project.src.data.datasets import MultimodalPairDataset, validate_sample
from project.src.data.hdf5_index import Hdf5KeyIndex
from project.src.data.partitions import assert_no_overlap, filtered_fold_counts, load_fold, load_test
from project.src.data.paths import DataPaths


DATA_ROOT = Path("hackaton")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="local dataset mirror is not available")
def test_hdf5_key_counts_match_report() -> None:
    paths = DataPaths.from_root(DATA_ROOT)
    index = Hdf5KeyIndex.from_paths(paths)

    assert len(index.image_keys) == 64_365
    assert len(index.spectrum_keys) == 69_351
    assert len(index.paired_keys) == 64_365
    assert len(index.spectrum_only_keys) == 4_986
    assert len(index.image_only_keys) == 0


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="local dataset mirror is not available")
def test_filtered_partition_counts_match_report() -> None:
    paths = DataPaths.from_root(DATA_ROOT)
    index = Hdf5KeyIndex.from_paths(paths)
    counts = filtered_fold_counts(paths.partitions_dir, index.paired_keys)

    assert counts["fold_0_train"] == 46_213
    assert counts["fold_0_val"] == 11_566
    assert counts["test"] == 6_586


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="local dataset mirror is not available")
def test_filtered_splits_do_not_overlap() -> None:
    paths = DataPaths.from_root(DATA_ROOT)
    index = Hdf5KeyIndex.from_paths(paths)
    fold = load_fold(paths.partitions_dir, 0)
    train = fold.train.filter(index.paired_keys).all
    val = fold.val.filter(index.paired_keys).all
    test = load_test(paths.partitions_dir).filter(index.paired_keys).all

    assert_no_overlap(train, val, test)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="local dataset mirror is not available")
def test_multimodal_pair_dataset_sample_is_valid() -> None:
    paths = DataPaths.from_root(DATA_ROOT)
    index = Hdf5KeyIndex.from_paths(paths)
    first_ids = tuple(sorted(index.paired_keys)[:2])

    with MultimodalPairDataset(first_ids, paths.images_h5, paths.spectra_h5) as dataset:
        assert len(dataset) == 2
        sample = dataset[0]
        validate_sample(sample)
        assert sample["object_id"] == first_ids[0]
