from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.data.datasets import MultimodalPairDataset, validate_sample
from project.src.data.hdf5_index import Hdf5KeyIndex, summarize_hdf5_schema
from project.src.data.partitions import assert_no_overlap, filtered_fold_counts, load_fold, load_test
from project.src.data.paths import DataPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run inspection of Hackaton2026 data.")
    parser.add_argument("--data-root", default="hackaton")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()

    paths = DataPaths.from_root(args.data_root)
    index = Hdf5KeyIndex.from_paths(paths)
    paired_keys = index.paired_keys

    print("== HDF5 index ==")
    print(f"images: {len(index.image_keys)}")
    print(f"spectra: {len(index.spectrum_keys)}")
    print(f"paired: {len(paired_keys)}")
    print(f"spectrum_only: {len(index.spectrum_only_keys)}")
    print(f"image_only: {len(index.image_only_keys)}")

    print("\n== Schema sample ==")
    schema = summarize_hdf5_schema(paths, sample_keys=1)
    for label, info in schema.items():
        print(f"{label}: {info['n_objects']} objects, first={info['first_keys']}")
        print(info["sample_schema"])

    print("\n== Filtered partitions ==")
    counts = filtered_fold_counts(paths.partitions_dir, paired_keys)
    for key, value in counts.items():
        print(f"{key}: {value}")

    fold = load_fold(paths.partitions_dir, args.fold)
    train = fold.train.filter(paired_keys).all
    val = fold.val.filter(paired_keys).all
    test = load_test(paths.partitions_dir).filter(paired_keys).all
    assert_no_overlap(train, val, test)
    print(f"\nfold_{args.fold} overlap check: OK")

    print("\n== Dataset sample ==")
    sample_ids = train[: args.sample_size]
    with MultimodalPairDataset(sample_ids, paths.images_h5, paths.spectra_h5) as dataset:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            validate_sample(sample)
            print(
                sample["object_id"],
                "img",
                sample["img"].shape,
                "channels_valid",
                int(sample["img_channel_mask"].sum()),
                "spec_valid",
                int(sample["mask_spectra"].sum()),
            )

    print("\ndry-run OK")


if __name__ == "__main__":
    main()
