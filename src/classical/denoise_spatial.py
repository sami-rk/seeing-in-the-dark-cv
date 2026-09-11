"""Classical step 3: spatial denoising compared by edge preservation.

Brightening amplifies sensor noise, so the denoiser that follows matters.
Each filter is scored not just by PSNR/SSIM but by how well it keeps edges:
the MSE between Sobel gradient magnitudes of the denoised output and the
clean reference (lower is better).
"""
import cv2
import numpy as np


def denoise_mean(bgr: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.blur(bgr, (k, k))


def denoise_median(bgr: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.medianBlur(bgr, k)


def denoise_gaussian(bgr: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(bgr, (k, k), sigma)


def denoise_bilateral(bgr: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)


def edge_mse(denoised: np.ndarray, reference: np.ndarray) -> float:
    """MSE of Sobel gradient magnitudes (grayscale) vs the clean reference."""
    g_d = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_r = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mag_d = np.hypot(cv2.Sobel(g_d, cv2.CV_64F, 1, 0), cv2.Sobel(g_d, cv2.CV_64F, 0, 1))
    mag_r = np.hypot(cv2.Sobel(g_r, cv2.CV_64F, 1, 0), cv2.Sobel(g_r, cv2.CV_64F, 0, 1))
    return float(np.mean((mag_d - mag_r) ** 2))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from classical.clahe_gamma import adaptive_gamma
    from metrics.psnr_ssim import all_metrics

    root = Path(__file__).resolve().parents[2] / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"
    names = sorted((root / "low").glob("*.jpg"))[:5]
    scores: dict[str, list] = {}
    for name in names:
        bright = adaptive_gamma(cv2.imread(str(name)))
        high = cv2.imread(str(root / "high" / name.name))
        for label, out in (("mean", denoise_mean(bright)), ("median", denoise_median(bright)),
                           ("gaussian", denoise_gaussian(bright)), ("bilateral", denoise_bilateral(bright))):
            m = all_metrics(out, high)
            scores.setdefault(label, []).append((m["psnr"], m["ssim"], edge_mse(out, high)))
    for label, vals in scores.items():
        arr = np.mean(vals, axis=0)
        print(f"{label:9s} PSNR={arr[0]:.2f} SSIM={arr[1]:.3f} edgeMSE={arr[2]:.1f}")
