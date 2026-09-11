"""Part B.1: convolutional autoencoder baseline (L1 reconstruction loss).

Encoder compresses 256x256x3 to 16x16x512; the decoder mirrors it back with a
sigmoid output in [0,1]. No skip connections, no stochasticity -- this is the
floor every later generative model must beat.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired.datasets import PairedLoLI


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(epochs: int = 10, subset: int = 6000, size: int = 256, batch: int = 16,
             lr: float = 2e-4, device: str = "cuda", out: Path | None = None) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ConvAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    train_loader = DataLoader(PairedLoLI("train", size, subset), batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(PairedLoLI("val", size, min(subset, 1000)), batch_size=batch,
                            shuffle=False, num_workers=4, pin_memory=True)
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for low, high in train_loader:
            low, high = low.to(dev), high.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(low), high)
            loss.backward()
            opt.step()
            tot += loss.item() * len(low)
            n += len(low)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(dev), high.to(dev)
                vtot += loss_fn(model(low), high).item() * len(low)
                vn += len(low)
        history["train"].append(tot / n)
        history["val"].append(vtot / vn)
        print(f"epoch {ep + 1}/{epochs} train_L1={tot / n:.4f} val_L1={vtot / vn:.4f}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        print(f"saved to {out}")
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train convolutional AE on LoLI-Street pairs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--subset", type=int, default=6000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = ConvAE().to(dev).eval()
        with torch.no_grad():
            y = m(torch.zeros(2, 3, 256, 256, device=dev))
        print("forward:", tuple(y.shape), float(y.min()), float(y.max()))
        train_ae(epochs=1, subset=200, batch=8)
    else:
        history = train_ae(args.epochs, args.subset, args.size, args.batch, args.lr, out=args.out)
        if args.history:
            args.history.parent.mkdir(parents=True, exist_ok=True)
            args.history.write_text(__import__("json").dumps(history))
            print(f"history saved to {args.history}")
