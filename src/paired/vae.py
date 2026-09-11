"""Part B.2: variational autoencoder (same skeleton as B.1 + KL term).

The encoder predicts a diagonal Gaussian (mu, logvar) per bottleneck
activation; the reparameterization trick keeps sampling differentiable.
Loss = L1 reconstruction + beta * KL(N(mu,logvar) || N(0,1)).
Question for the master table: does the VAE's typically smoother output help
or hurt detection specifically?
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired.datasets import PairedLoLI


class ConvVAE(nn.Module):
    def __init__(self, latent_ch: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.to_mu = nn.Conv2d(512, latent_ch, 1)
        self.to_logvar = nn.Conv2d(512, latent_ch, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(recon, target, mu, logvar, beta: float = 1e-3):
    recon_loss = F.l1_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def train_vae(epochs: int = 10, subset: int = 6000, size: int = 256, batch: int = 16,
              lr: float = 2e-4, beta: float = 1e-3, device: str = "cuda",
              out: Path | None = None) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ConvVAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
            recon, mu, logvar = model(low)
            loss, _, _ = vae_loss(recon, high, mu, logvar, beta)
            loss.backward()
            opt.step()
            tot += loss.item() * len(low)
            n += len(low)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(dev), high.to(dev)
                recon, mu, logvar = model(low)
                loss, _, _ = vae_loss(recon, high, mu, logvar, beta)
                vtot += loss.item() * len(low)
                vn += len(low)
        history["train"].append(tot / n)
        history["val"].append(vtot / vn)
        print(f"epoch {ep + 1}/{epochs} train={tot / n:.4f} val={vtot / vn:.4f}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        print(f"saved to {out}")
    return history


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ConvVAE().to(dev).eval()
    with torch.no_grad():
        y, mu, lv = m(torch.zeros(2, 3, 256, 256, device=dev))
    print("forward:", tuple(y.shape), tuple(mu.shape), float(y.min()), float(y.max()))
    train_vae(epochs=1, subset=200, batch=8)
