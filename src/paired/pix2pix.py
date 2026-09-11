"""Part B.3: conditional GAN, pix2pix-style.

U-Net generator with skip connections (tanh output in [-1,1]) plus a 70x70
PatchGAN discriminator. Loss = adversarial (BCE) + lambda_L1 * L1, exactly the
pix2pix recipe. Target on LoLI-Street val: PSNR > 23, SSIM > 0.6.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired.datasets import PairedLoLI

LAMBDA_L1 = 100.0


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = _Down(3, 64, norm=False)
        self.d2 = _Down(64, 128)
        self.d3 = _Down(128, 256)
        self.d4 = _Down(256, 512)
        self.d5 = _Down(512, 512)
        self.d6 = _Down(512, 512)
        self.d7 = _Down(512, 512)
        self.d8 = _Down(512, 512, norm=False)
        self.u1 = _Up(512, 512, dropout=True)
        self.u2 = _Up(1024, 512, dropout=True)
        self.u3 = _Up(1024, 512, dropout=True)
        self.u4 = _Up(1024, 512)
        self.u5 = _Up(1024, 256)
        self.u6 = _Up(512, 128)
        self.u7 = _Up(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        e5 = self.d5(e4)
        e6 = self.d6(e5)
        e7 = self.d7(e6)
        z = self.d8(e7)
        u = self.u1(z, e7)
        u = self.u2(u, e6)
        u = self.u3(u, e5)
        u = self.u4(u, e4)
        u = self.u5(u, e3)
        u = self.u6(u, e2)
        u = self.u7(u, e1)
        return self.final(u)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _Down(6, 64, norm=False),
            _Down(64, 128),
            _Down(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, low, high):
        return self.net(torch.cat([low, high], dim=1))


def train_pix2pix(epochs: int = 20, subset: int = 6000, size: int = 256, batch: int = 8,
                  lr: float = 2e-4, lambda_l1: float = LAMBDA_L1, device: str = "cuda",
                  out: Path | None = None) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    gen, disc = UNetGenerator().to(dev), PatchDiscriminator().to(dev)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    loader = DataLoader(PairedLoLI("train", size, subset, tanh_range=True), batch_size=batch,
                        shuffle=True, num_workers=4, pin_memory=True)
    history = {"g": [], "d": []}
    for ep in range(epochs):
        tot_g = tot_d = n = 0
        for low, high in loader:
            low, high = low.to(dev), high.to(dev)
            fake = gen(low)
            opt_d.zero_grad()
            loss_d = (bce(disc(low, high), torch.ones_like(disc(low, high)))
                      + bce(disc(low, fake.detach()), torch.zeros_like(disc(low, fake)))) / 2
            loss_d.backward()
            opt_d.step()
            opt_g.zero_grad()
            pred = disc(low, fake)
            loss_g = bce(pred, torch.ones_like(pred)) + lambda_l1 * l1(fake, high)
            loss_g.backward()
            opt_g.step()
            tot_g += loss_g.item() * len(low)
            tot_d += loss_d.item() * len(low)
            n += len(low)
        history["g"].append(tot_g / n)
        history["d"].append(tot_d / n)
        print(f"epoch {ep + 1}/{epochs} G={tot_g / n:.4f} D={tot_d / n:.4f}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict()}, out)
        print(f"saved to {out}")
    return history


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g, d = UNetGenerator().to(dev).eval(), PatchDiscriminator().to(dev).eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 256, 256, device=dev)
        f = g(x)
        print("gen:", tuple(f.shape), float(f.min()), float(f.max()), "| disc:", tuple(d(x, f).shape))
    train_pix2pix(epochs=1, subset=200, batch=4)
