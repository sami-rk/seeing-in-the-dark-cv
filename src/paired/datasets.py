"""Paired LoLI-Street dataset for Part B (AE / VAE / cGAN).

Pairs low-light inputs with their well-exposed targets. For iteration speed,
prototype on a deterministic seeded subset at reduced resolution, then train
the final pass on the full split by passing subset=None and size=512.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_roots import loli_root

REPO_ROOT = Path(__file__).resolve().parents[2]
LOLI_ROOT = loli_root()


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.permute(2, 0, 1)


class PairedLoLI(Dataset):
    def __init__(self, split: str = "train", size: int = 256, subset: int | None = 6000,
                 seed: int = 0, tanh_range: bool = False):
        low_dir = LOLI_ROOT / ("Train" if split == "train" else "Val") / "low"
        high_dir = LOLI_ROOT / ("Train" if split == "train" else "Val") / "high"
        names = sorted(p.name for p in low_dir.iterdir() if p.is_file())
        assert all((high_dir / n).exists() for n in names), "low/high filename mismatch"
        if subset is not None and len(names) > subset:
            rng = random.Random(seed)
            names = sorted(rng.sample(names, subset))
        self.names = names
        self.low_dir, self.high_dir = low_dir, high_dir
        self.size = size
        self.tanh_range = tanh_range

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        low = _load_rgb(self.low_dir / self.names[i], self.size)
        high = _load_rgb(self.high_dir / self.names[i], self.size)
        if self.tanh_range:
            low, high = low * 2 - 1, high * 2 - 1
        return low, high


def paired_loader(split: str = "train", batch: int = 16, size: int = 256,
                  subset: int | None = 6000, shuffle: bool = True, **kwargs) -> DataLoader:
    ds = PairedLoLI(split=split, size=size, subset=subset, **kwargs)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    loader = paired_loader("train", batch=4, size=256, subset=100)
    low, high = next(iter(loader))
    print("batches:", len(loader), "low:", tuple(low.shape), low.min().item(), low.max().item())
    val = PairedLoLI(split="val", size=256, subset=None)
    print("full val pairs:", len(val))
