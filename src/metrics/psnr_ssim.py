"""Restoration quality metrics: MSE, MAE, PSNR, SSIM.

All functions take uint8 images (H,W) or (H,W,C) and assume the pair is
already aligned and equally sized. SSIM follows Wang et al. (Gaussian window
11x11, sigma 1.5, K1=0.01, K2=0.03), averaged over channels for color input.
"""
import cv2
import numpy as np

_MAX = 255.0


def _as_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float64)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = _as_float(a) - _as_float(b)
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(_as_float(a) - _as_float(b))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return float(20 * np.log10(_MAX / np.sqrt(err)))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _as_float(a), _as_float(b)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    cov_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    c1, c2 = (0.01 * _MAX) ** 2, (0.03 * _MAX) ** 2
    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    if a.ndim == 2:
        return _ssim_gray(a, b)
    return float(np.mean([_ssim_gray(a[..., c], np.asarray(b)[..., c]) for c in range(a.shape[2])]))


def all_metrics(restored: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "mse": mse(restored, reference),
        "mae": mae(restored, reference),
        "psnr": psnr(restored, reference),
        "ssim": ssim(restored, reference),
    }
