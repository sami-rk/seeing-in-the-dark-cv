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


def load_cyclegan(weights: Path, dev: torch.device):
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cyclegan.generators import ResNetGenerator

    model = ResNetGenerator()
    model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True)["g_a2b"])
    return model


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
    if name == "cyclegan":
        return load_cyclegan(weights, dev), True
    raise ValueError(f"unknown model {name!r}")


def _to_input_tensor(path: Path, tanh_range: bool) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t * 2 - 1 if tanh_range else t


@torch.no_grad()
def restore_exdark_native(model_name: str, weights: Path, img_dir: Path, out_dir: Path) -> int:
    """Restore ExDark val images at native resolution (boxes stay aligned).

    Pix2pix's U-Net downsamples 256x (8 strides of 2). Native ExDark frames
    have arbitrary sizes, so we pad each input to the next multiple of 256,
    run the model, then crop the output back to the original size.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tanh_range = load_model(model_name, weights, dev)
    model.to(dev).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        orig = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig.size
        t = _to_input_tensor(img_path, tanh_range).unsqueeze(0).to(dev)
        _, _, h, w = t.shape
        pad_h = (256 - h % 256) % 256
        pad_w = (256 - w % 256) % 256
        if pad_h or pad_w:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(t)
        if isinstance(out, tuple):
            out = out[0]
        out = out.clamp(-1, 1) if tanh_range else out.clamp(0, 1)
        if tanh_range:
            out = (out + 1) / 2
        # Crop back to original size if we padded for the U-Net
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        arr = (out[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(out_dir / (img_path.stem + ".jpg"), quality=95)
        n += 1
    print(f"restored {n} ExDark images with {model_name}")
    return n


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Restore LoLI val with a Part B model.")
    parser.add_argument("--model", required=True, choices=["ae", "vae", "pix2pix"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataset", choices=["loli", "exdark"], default="loli")
    parser.add_argument("--img-dir", type=Path, default=None,
                        help="ExDark image dir (with --dataset exdark).")
    args = parser.parse_args()

    if args.dataset == "exdark":
        assert args.img_dir, "--img-dir is required with --dataset exdark"
        restore_exdark_native(args.model, args.weights, args.img_dir, args.out)
        return

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
            Image.fromarray(arr).save(args.out / (Path(names[idx]).stem + ".jpg"), quality=95)
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
