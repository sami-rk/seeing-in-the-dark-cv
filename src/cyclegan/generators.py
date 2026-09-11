"""Part C: ResNet generators for CycleGAN.

Two mappings are learned: G: low-light -> well-lit and F: well-lit -> low-light.
Each is a fully convolutional ResNet (Johnson et al.): downsample, N residual
blocks at the bottleneck, upsample, tanh output. InstanceNorm throughout, as
in the original CycleGAN paper (batch statistics are meaningless for the
unpaired single-image batches used here).
"""
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, n_blocks: int = 9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ]
        layers += [ResidualBlock(256) for _ in range(n_blocks)]
        layers += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = ResNetGenerator(n_blocks=2).to(dev).eval()
    with torch.no_grad():
        y = g(torch.zeros(1, 3, 256, 256, device=dev))
    print("forward:", tuple(y.shape), float(y.min()), float(y.max()))
    print("params:", sum(p.numel() for p in ResNetGenerator().parameters()))
