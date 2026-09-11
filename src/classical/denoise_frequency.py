"""Classical step 4: frequency-domain denoising (2D FFT + Gaussian low-pass).

The filter is designed directly in the frequency domain: a Gaussian mask
centered on the DC component keeps low frequencies (scene structure) and rolls
off high frequencies (noise). Compared head-to-head with the spatial filters
of step 3 on the same brightened frames.
"""
import cv2
import numpy as np


def gaussian_lowpass_mask(h: int, w: int, sigma_ratio: float = 0.15) -> np.ndarray:
    """Gaussian mask, peak 1 at DC, width proportional to image size."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    sigma = sigma_ratio * min(h, w)
    return np.exp(-dist2 / (2 * sigma * sigma))


def denoise_frequency(bgr: np.ndarray, sigma_ratio: float = 0.15) -> np.ndarray:
    img = bgr.astype(np.float64)
    h, w = img.shape[:2]
    mask = gaussian_lowpass_mask(h, w, sigma_ratio)
    out = np.zeros_like(img)
    for c in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(img[:, :, c]))
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * mask)))
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from classical.clahe_gamma import adaptive_gamma
    from classical.denoise_spatial import denoise_bilateral, denoise_gaussian, edge_mse
    from metrics.psnr_ssim import all_metrics

    root = Path(__file__).resolve().parents[2] / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"
    names = sorted((root / "low").glob("*.jpg"))[:5]
    scores: dict[str, list] = {}
    for name in names:
        bright = adaptive_gamma(cv2.imread(str(name)))
        high = cv2.imread(str(root / "high" / name.name))
        for label, out in (("gaussian", denoise_gaussian(bright)), ("bilateral", denoise_bilateral(bright)),
                           ("freq-0.15", denoise_frequency(bright)), ("freq-0.25", denoise_frequency(bright, 0.25))):
            m = all_metrics(out, high)
            scores.setdefault(label, []).append((m["psnr"], m["ssim"], edge_mse(out, high)))
    for label, vals in scores.items():
        arr = np.mean(vals, axis=0)
        print(f"{label:9s} PSNR={arr[0]:.2f} SSIM={arr[1]:.3f} edgeMSE={arr[2]:.1f}")
