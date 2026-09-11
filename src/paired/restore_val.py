"""Restore LoLI-Street val with a trained Part B model and score it.

Handles all three Part B models (ae [0,1], vae [0,1], pix2pix [-1,1] tanh):

    <venv>/bin/python src/paired/restore_val.py --model ae --weights checkpoints/ae.pt \\
        --out restored_ae/loli_val --metrics results/cells/ae_loli.json

Models are fully convolutional, so inference runs at native 512 (no resize
artifacts) even though training used 256px prototypes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.psnr_ssim import all_metrics
from paired.autoencoder import ConvAE
from paired.pix2pix import UNetGenerator
from paired.vae import ConvVAE


def load_model(name: str, weights: Path, dev: torch.device) -> tuple[torch.nn.Module, bool]:
    if name == "ae":
        model = ConvAE()
        model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True))
        return model, False
    if name == "vae":
        model = ConvVAE()
        model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True))
        return model, False
    if name == "pix2pix":
        model = UNetGenerator()
        model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True)["gen"])
        return model, True
    raise ValueError(f"unknown model {name!r}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Restore LoLI val with a Part B model.")
    parser.add_argument("--model", required=True, choices=["ae", "vae", "pix2pix"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tanh_range = load_model(args.model, args.weights, dev)
    model.to(dev).eval()

    from paired.datasets import PairedLoLI

    loader = DataLoader(PairedLoLI("val", args.size, subset=None, tanh_range=tanh_range),
                        batch_size=args.batch, shuffle=False, num_workers=4)
    names = PairedLoLI("val", args.size, subset=None).names
    args.out.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list[float]] = {}
    idx = 0
    for low, high in loader:
        out = model(low.to(dev))
        if isinstance(out, tuple):  # VAE returns (recon, mu, logvar)
            out = out[0]
        out = out.clamp(0, 1)
        if tanh_range:
            out = (out + 1) / 2
            high = (high + 1) / 2
        for j in range(len(low)):
            arr = (out[j].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(args.out / (Path(names[idx]).stem + ".png"))
            ref = (high[j].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            for k, v in all_metrics(arr, ref).items():
                if np.isfinite(v):
                    acc.setdefault(k, []).append(v)
            idx += 1
    summary = {k: float(sum(v) / len(v)) for k, v in acc.items()}
    summary["n"] = idx
    print(json.dumps(summary, indent=2))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
