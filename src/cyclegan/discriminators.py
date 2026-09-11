"""Part C: PatchGAN discriminators for CycleGAN, plus the fake-image pool.

Same 70x70 receptive field as the pix2pix discriminator, but with InstanceNorm
(CycleGAN trains on batch size 1, where BatchNorm statistics are useless).
The FakePool keeps a history of generated images and samples from it, which
dampens discriminator oscillation -- straight from the original paper.
"""
import random

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = 64
        for i in range(1, n_layers + 1):
            out_ch = min(ch * 2, 512)
            stride = 1 if i == n_layers else 2
            layers += [nn.Conv2d(ch, out_ch, 4, stride, 1, bias=False),
                       nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, True)]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FakePool:
    """History buffer of past fakes; returns a mix of old and new images."""

    def __init__(self, size: int = 50):
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        out = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img.detach().clone())
                out.append(img)
            elif random.random() < 0.5:
                out.append(img)
            else:
                idx = random.randrange(len(self.images))
                old = self.images[idx].clone()
                self.images[idx] = img.detach().clone()
                out.append(old)
        return torch.cat(out)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = PatchDiscriminator().to(dev).eval()
    with torch.no_grad():
        y = d(torch.zeros(2, 3, 256, 256, device=dev))
    print("forward:", tuple(y.shape))
    pool = FakePool(4)
    q = pool.query(torch.zeros(2, 3, 8, 8))
    print("pool:", tuple(q.shape), "stored:", len(pool.images))
