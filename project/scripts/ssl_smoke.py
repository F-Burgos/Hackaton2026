from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.data.hdf5_index import Hdf5KeyIndex
from project.src.data.partitions import load_fold
from project.src.data.paths import DataPaths
from project.src.data.torch_datasets import TorchMultimodalPairDataset, multimodal_collate
from project.src.models.ssl import ImageAutoencoder, SpectrumAutoencoder, masked_mse_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke test for SSL reconstruction paths.")
    parser.add_argument("--data-root", default="hackaton")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modality", choices=["spectrum", "image"], default="spectrum")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _seed_everything(args.seed)
    paths = DataPaths.from_root(args.data_root)
    index = Hdf5KeyIndex.from_paths(paths)
    fold = load_fold(paths.partitions_dir, args.fold)
    train_ids = fold.train.filter(index.paired_keys).all[: args.subset_size]

    dataset = TorchMultimodalPairDataset(train_ids, paths.images_h5, paths.spectra_h5)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multimodal_collate,
    )

    if args.modality == "spectrum":
        model = SpectrumAutoencoder(latent_dim=64)
    else:
        model = ImageAutoencoder(latent_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"ssl modality: {args.modality}")
    print(f"smoke train objects: {len(train_ids)}")
    for step, batch in enumerate(loader, start=1):
        if args.modality == "spectrum":
            prediction = model(batch["flux_lambda_normalized"], batch["mask_spectra"])
            loss = masked_mse_loss(
                prediction,
                batch["flux_lambda_normalized"],
                batch["mask_spectra"],
            )
        else:
            prediction = model(batch["img"], batch["img_channel_mask"])
            loss = masked_mse_loss(prediction, batch["img"], batch["img_channel_mask"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}")
        if step >= args.steps:
            break

    dataset.close()
    print("ssl smoke OK")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
