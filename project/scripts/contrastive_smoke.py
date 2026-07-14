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
from project.src.models.contrastive import ContrastiveModel
from project.src.models.losses import retrieval_at_1, symmetric_clip_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke test for the base contrastive path.")
    parser.add_argument("--data-root", default="hackaton")
    parser.add_argument("--fold", type=int, default=0)
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
    model = ContrastiveModel(embedding_dim=64, projection_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"smoke train objects: {len(train_ids)}")
    for step, batch in enumerate(loader, start=1):
        tensor_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        outputs = model(tensor_batch)
        loss = symmetric_clip_loss(outputs["image_embedding"], outputs["spectrum_embedding"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        r1 = retrieval_at_1(outputs["image_embedding"], outputs["spectrum_embedding"])
        print(f"step={step} loss={loss.item():.4f} batch_recall@1={r1:.3f}")
        if step >= args.steps:
            break

    dataset.close()
    print("contrastive smoke OK")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
