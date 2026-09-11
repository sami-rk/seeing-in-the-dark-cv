"""Classical step 1: luminance-channel equalization and adaptive gamma.

Why the luminance channel: equalizing R/G/B independently rescales each color
distribution on its own, which shifts hues (whites turn pink/green). Working on
the Y channel of YCrCb (or L of LAB) changes brightness only and preserves hue.
"""
import cv2
import numpy as np

TARGET_MEAN = 0.5
GAMMA_CLIP = (0.2, 3.0)


def _to_ycrcb(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb, ycrcb[:, :, 0]


def equalize_hist_luminance(bgr: np.ndarray) -> np.ndarray:
    """Global histogram equalization on Y (baseline; often harsh)."""
    ycrcb, y = _to_ycrcb(bgr)
    ycrcb[:, :, 0] = cv2.equalizeHist(y)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on Y: local contrast without blowing out bright regions."""
    ycrcb, y = _to_ycrcb(bgr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    ycrcb[:, :, 0] = clahe.apply(y)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def estimate_gamma(bgr: np.ndarray, target: float = TARGET_MEAN) -> float:
    """Per-image gamma from mean luminance: gamma = ln(target) / ln(mean).

    Dark images (mean < target) get gamma < 1, which brightens them; the
    exponent adapts to each frame instead of using one fixed constant.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray)) / 255.0
    mean = min(max(mean, 1e-3), 1.0 - 1e-6)
    gamma = float(np.log(target) / np.log(mean))
    return float(min(max(gamma, GAMMA_CLIP[0]), GAMMA_CLIP[1]))


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, lut)


def adaptive_gamma(bgr: np.ndarray, target: float = TARGET_MEAN) -> np.ndarray:
    return apply_gamma(bgr, estimate_gamma(bgr, target))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
    from metrics.psnr_ssim import all_metrics

    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"
    for name in sorted((root / "low").glob("*.jpg"))[:5]:
        low = cv2.imread(str(name))
        high = cv2.imread(str(root / "high" / name.name))
        print(f"== {name.name} (gamma={estimate_gamma(low):.2f})")
        for label, out in (("raw", low), ("equalizeHist", equalize_hist_luminance(low)),
                           ("clahe", apply_clahe(low)), ("gamma", adaptive_gamma(low))):
            m = all_metrics(out, high)
            print(f"  {label:12s} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f}")
