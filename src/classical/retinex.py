"""Classical step 2: Single-Scale Retinex (SSR), hand-implemented.

Model: observed image = illumination x reflectance,  I = L . R.
In the log domain the product becomes a sum, so reflectance (the intrinsic,
illumination-invariant part) is recovered by subtraction:

    log R = log I - log L,   with  L ~= GaussianBlur(I, sigma)

No library call implements this: only cv2.GaussianBlur (a plain convolution)
is used for the illumination estimate. Output gain is set by percentile
clipping so one scene's tuning transfers to the next.
"""
import cv2
import numpy as np


def single_scale_retinex(
    bgr: np.ndarray,
    sigma: float = 80.0,
    low_clip: float = 1.0,
    high_clip: float = 99.0,
) -> np.ndarray:
    """SSR per channel: R_c = log(I_c + 1) - log(Blur(I_c, sigma) + 1)."""
    img = bgr.astype(np.float64) + 1.0
    illumination = np.stack(
        [cv2.GaussianBlur(img[:, :, c], (0, 0), sigma) for c in range(3)], axis=-1
    )
    log_r = np.log(img) - np.log(illumination + 1.0)
    out = np.zeros_like(log_r)
    for c in range(3):
        lo, hi = np.percentile(log_r[:, :, c], (low_clip, high_clip))
        span = max(hi - lo, 1e-6)
        out[:, :, c] = np.clip((log_r[:, :, c] - lo) / span * 255.0, 0, 255)
    return out.astype(np.uint8)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from metrics.psnr_ssim import all_metrics

    root = Path(__file__).resolve().parents[2] / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"
    for name in sorted((root / "low").glob("*.jpg"))[:5]:
        low = cv2.imread(str(name))
        high = cv2.imread(str(root / "high" / name.name))
        print(f"== {name.name}")
        for label, out in (("raw", low), ("ssr-s80", single_scale_retinex(low)),
                           ("ssr-s30", single_scale_retinex(low, sigma=30.0))):
            m = all_metrics(out, high)
            print(f"  {label:8s} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f}")
