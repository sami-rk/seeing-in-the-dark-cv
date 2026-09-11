"""Classical step 5: morphological cleanup of brightening artifacts.

Aggressive brightening (e.g. global equalization) leaves speckle and ragged
edges. A small opening removes isolated bright specks; a closing fills pinhole
gaps. The structuring element below is defined by hand (3x3 square), and the
operation runs per channel so no color information leaks across bands.
"""
import cv2
import numpy as np

KERNEL = np.ones((3, 3), dtype=np.uint8)

_OPS = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}


def morphological_cleanup(bgr: np.ndarray, op: str = "open") -> np.ndarray:
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}, got {op!r}")
    return np.stack(
        [cv2.morphologyEx(bgr[:, :, c], _OPS[op], KERNEL) for c in range(3)], axis=-1
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from classical.clahe_gamma import equalize_hist_luminance
    from metrics.psnr_ssim import all_metrics

    root = Path(__file__).resolve().parents[2] / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset" / "Val"
    names = sorted((root / "low").glob("*.jpg"))[:5]
    scores: dict[str, list] = {}
    for name in names:
        bright = equalize_hist_luminance(cv2.imread(str(name)))
        high = cv2.imread(str(root / "high" / name.name))
        for label, out in (("bright", bright), ("+open", morphological_cleanup(bright, "open")),
                           ("+close", morphological_cleanup(bright, "close"))):
            m = all_metrics(out, high)
            scores.setdefault(label, []).append((m["psnr"], m["ssim"]))
    for label, vals in scores.items():
        arr = np.mean(vals, axis=0)
        print(f"{label:8s} PSNR={arr[0]:.2f} SSIM={arr[1]:.3f}")
