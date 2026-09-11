"""Part C: CycleGAN losses and training.

Domains: A = ExDark low-light images, B = LoLI-Street well-exposed images,
sampled UNPAIRED (independent shuffles -- no low/high correspondence is used).

Losses (standard CycleGAN):
  adversarial: MSE(G_A2B(a) vs 1) + MSE(G_B2A(b) vs 1), and the reverse for D
  cycle:       L1(F(G(a)) - a) + L1(G(F(b)) - b), weight lambda_cycle=10
  identity:    L1(G(b) - b) + L1(F(a) - a), weight lambda_id=5 (keeps tint)

Run a smoke epoch:

    <venv>/bin/python src/cyclegan/train_cyclegan.py --epochs 1 --subset 32
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclegan.discriminators import FakePool, PatchDiscriminator
from cyclegan.generators import ResNetGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
EXDARK_IMGS = REPO_ROOT / "ExDarkDataset" / "Dataset" / "Dataset"
LOLI_HIGH = REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Train" / "high"

LAMBDA_CYCLE = 10.0
LAMBDA_ID = 5.0


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    import numpy as np

    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.permute(2, 0, 1) * 2 - 1


class UnpairedDataset(Dataset):
    """Domain A (ExDark) and B (LoLI high) sampled independently."""

    def __init__(self, size: int = 256, subset: int | None = 2000, seed: int = 0):
        a = sorted(p.name for p in EXDARK_IMGS.iterdir() if p.is_file())
        b = sorted(p.name for p in LOLI_HIGH.iterdir() if p.is_file())
        rng = random.Random(seed)
        if subset is not None:
            a = sorted(rng.sample(a, min(subset, len(a))))
            b = sorted(rng.sample(b, min(subset, len(b))))
        self.a_names, self.b_names = a, b
        self.size = size
        self.n = max(len(a), len(b))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        a = _load_rgb(EXDARK_IMGS / self.a_names[i % len(self.a_names)], self.size)
        b = _load_rgb(LOLI_HIGH / self.b_names[i % len(self.b_names)], self.size)
        return a, b


def train_cyclegan(epochs: int = 20, subset: int = 2000, size: int = 256, batch: int = 1,
                   lr: float = 2e-4, device: str = "cuda", out: Path | None = None) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g_a2b, g_b2a = ResNetGenerator().to(dev), ResNetGenerator().to(dev)
    d_a, d_b = PatchDiscriminator().to(dev), PatchDiscriminator().to(dev)
    opt_g = torch.optim.Adam(list(g_a2b.parameters()) + list(g_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=lr, betas=(0.5, 0.999))
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    pool_a, pool_b = FakePool(), FakePool()
    loader = DataLoader(UnpairedDataset(size, subset), batch_size=batch, shuffle=True, num_workers=4)
    history = {"g": [], "d": []}
    for ep in range(epochs):
        tot_g = tot_d = n = 0
        for a, b in loader:
            a, b = a.to(dev), b.to(dev)
            fake_b, fake_a = g_a2b(a), g_b2a(b)
            rec_a, rec_b = g_b2a(fake_b), g_a2b(fake_a)
            opt_g.zero_grad()
            loss_g = (mse(d_b(fake_b), torch.ones_like(d_b(fake_b)))
                      + mse(d_a(fake_a), torch.ones_like(d_a(fake_a)))
                      + LAMBDA_CYCLE * (l1(rec_a, a) + l1(rec_b, b))
                      + LAMBDA_ID * (l1(g_a2b(b), b) + l1(g_b2a(a), a)))
            loss_g.backward()
            opt_g.step()
            opt_d.zero_grad()
            loss_d = (mse(d_a(pool_a.query(fake_a.detach())), torch.zeros_like(d_a(fake_a)))
                      + mse(d_a(a), torch.ones_like(d_a(a)))
                      + mse(d_b(pool_b.query(fake_b.detach())), torch.zeros_like(d_b(fake_b)))
                      + mse(d_b(b), torch.ones_like(d_b(b)))) / 2
            # NOTE: ones_like/zeros_like reuse the live prediction shapes; the pool
            # returns the same batch size, so shapes always match.
            loss_d.backward()
            opt_d.step()
            tot_g += loss_g.item() * len(a)
            tot_d += loss_d.item() * len(a)
            n += len(a)
        history["g"].append(tot_g / n)
        history["d"].append(tot_d / n)
        print(f"epoch {ep + 1}/{epochs} G={tot_g / n:.4f} D={tot_d / n:.4f}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"g_a2b": g_a2b.state_dict(), "g_b2a": g_b2a.state_dict()}, out)
        print(f"saved to {out}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN ExDark <-> LoLI-Street high.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--subset", type=int, default=2000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    train_cyclegan(args.epochs, args.subset, args.size, args.batch, out=args.out)


if __name__ == "__main__":
    main()
