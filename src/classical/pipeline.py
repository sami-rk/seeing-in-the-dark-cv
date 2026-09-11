"""Classical pipeline: fixed ordered restoration + full-val scoring.

Candidates (each a fixed sequence of the step modules):

    raw, equalize, equalize_freq, gamma_freq, ssr_freq, clahe_gamma_bilateral

Run a comparison over LoLI-Street val (default: full 3000 pairs):

    <venv>/bin/python src/classical/pipeline.py --limit 50
    <venv>/bin/python src/classical/pipeline.py --out results/classical.json   # full val

Restore one folder (e.g. ExDark) with the frozen pipeline:

    <venv>/bin/python src/classical/pipeline.py --restore-only <imgdir> <outdir>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))from classical.clahe_gamma import adaptive_gamma, apply_clahe, equalize_hist_luminance  # noqa: E402
from classical.denoise_frequency import denoise_frequency  # noqa: E402
from classical.denoise_spatial import denoise_bilateral  # noqa: E402
from classical.retinex import single_scale_retinex  # noqa: E402
from metrics.psnr_ssim import all_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LOLI_VAL = REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"

# FROZEN pipeline (chosen on full-val numbers, see results/classical.json).
# equalize_freq: global equalization (best exposure match on LoLI pairs) +
# frequency low-pass (best edge-preserving denoiser in step comparisons).
FROZEN = "equalize_freq"


def restore(img, mode: str = FROZEN):
    if mode == "raw":
        return img
    if mode == "equalize":
        return equalize_hist_luminance(img)
    if mode == "equalize_freq":
        return denoise_frequency(equalize_hist_luminance(img), 0.25)
    if mode == "gamma_freq":
        return denoise_frequency(adaptive_gamma(img), 0.25)
    if mode == "ssr_freq":
        return denoise_frequency(single_scale_retinex(img), 0.25)
    if mode == "clahe_gamma_bilateral":
        return denoise_bilateral(apply_clahe(adaptive_gamma(img)))
    raise ValueError(f"unknown mode {mode!r}")


CANDIDATES = ["raw", "equalize", "equalize_freq", "gamma_freq", "ssr_freq", "clahe_gamma_bilateral"]


def compare(limit: int | None = None) -> dict:
    names = sorted((LOLI_VAL / "low").glob("*.jpg"))
    if limit:
        names = names[:limit]
    results: dict[str, dict] = {}
    for mode in CANDIDATES:
        acc: dict[str, list[float]] = {}
        for name in names:
            out = restore(cv2.imread(str(name)), mode)
            high = cv2.imread(str(LOLI_VAL / "high" / name.name))
            for k, v in all_metrics(out, high).items():
                acc.setdefault(k, []).append(v)
        summary = {}
        for k, v in acc.items():
            finite = [x for x in v if np.isfinite(x)]
            summary[k] = float(sum(finite) / len(finite)) if finite else float("inf")
        summary["n_pairs"] = len(names)
        summary["n_inf_psnr"] = len(acc["psnr"]) - len([x for x in acc["psnr"] if np.isfinite(x)])
        results[mode] = summary
        print(f"{mode:22s} " + " ".join(f"{k}={v:.3f}" for k, v in results[mode].items() if k != "n_inf_psnr"), flush=True)
    return results


def restore_folder(src_dir: Path, dst_dir: Path, mode: str = FROZEN) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(src_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(dst_dir / (img_path.stem + ".png")), restore(img, mode))
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classical candidates or restore a folder.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--restore-only", nargs=2, metavar=("SRCDIR", "DSTDIR"), default=None)
    parser.add_argument("--mode", default=FROZEN)
    args = parser.parse_args()

    if args.restore_only:
        n = restore_folder(Path(args.restore_only[0]), Path(args.restore_only[1]), args.mode)
        print(f"restored {n} images with mode={args.mode}")
        return
    results = compare(args.limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
