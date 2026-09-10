"""Dataset inventory for LoLI-Street and ExDark.

Scans the local (gitignored) dataset folders and writes small JSON manifests
to data_manifests/. Run from the repo root:

    <venv>/bin/python src/yolo_eval/inventory.py

Both datasets must already be present locally; nothing is downloaded.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOLI_ROOT = REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset"
EXDARK_ROOT = REPO_ROOT / "ExDarkDataset"
MANIFEST_DIR = REPO_ROOT / "data_manifests"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _sample_sizes(files, n=20):
    from PIL import Image

    sizes = []
    for path in files[:n]:
        try:
            with Image.open(path) as im:
                sizes.append({"file": path.name, "size": list(im.size), "mode": im.mode})
        except Exception as exc:  # noqa: BLE001 - inventory must not crash on one file
            sizes.append({"file": path.name, "error": str(exc)})
    return sizes


def _parse_yolo_labels(label_dir: Path):
    """Return (n_files, n_empty, class_histogram, errors)."""
    hist: Counter = Counter()
    n_empty = 0
    errors = []
    files = sorted(label_dir.glob("*.txt"))
    for path in files:
        try:
            lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": str(exc)})
            continue
        if not lines:
            n_empty += 1
        for parts in lines:
            hist[parts[0]] += 1
    return len(files), n_empty, dict(sorted(hist.items(), key=lambda kv: int(kv[0]))), errors[:10]


def inventory_loli(root: Path) -> dict:
    out: dict = {"root": str(root), "exists": root.exists(), "splits": {}}
    if not root.exists():
        return out
    for split in ("Train", "Val"):
        high = sorted(p for p in (root / split / "high").iterdir() if _is_image(p)) if (root / split / "high").exists() else []
        low = sorted(p for p in (root / split / "low").iterdir() if _is_image(p)) if (root / split / "low").exists() else []
        high_stems = {p.stem for p in high}
        low_stems = {p.stem for p in low}
        out["splits"][split] = {
            "n_high": len(high),
            "n_low": len(low),
            "paired": high_stems == low_stems,
            "only_in_high": sorted(high_stems - low_stems)[:10],
            "only_in_low": sorted(low_stems - high_stems)[:10],
            "sample_high": [p.name for p in high[:3]],
            "sample_sizes_high": _sample_sizes(high),
            "sample_sizes_low": _sample_sizes(low),
        }
    test_dir = root / "Test"
    test_files = sorted(p for p in test_dir.iterdir() if _is_image(p)) if test_dir.exists() else []
    out["test"] = {"n": len(test_files), "sample": [p.name for p in test_files[:3]], "has_labels": False}
    yolo = root / "YOLO Annotations"
    out["yolo"] = {}
    for key, label_dir in (
        ("train_high", yolo / "Train" / "high" / "Labels"),
        ("train_low", yolo / "Train" / "low" / "Labels"),
        ("val_high", yolo / "Val" / "YOLO Annotations (high)" / "Labels"),
        ("val_low", yolo / "Val" / "YOLO Annotations (low)" / "Labels"),
    ):
        if label_dir.exists():
            n_files, n_empty, hist, errors = _parse_yolo_labels(label_dir)
            out["yolo"][key] = {"n_files": n_files, "n_empty": n_empty, "class_hist": hist, "errors": errors}
        else:
            out["yolo"][key] = {"missing": str(label_dir)}
    classes_file = yolo / "Train" / "high" / "Classes" / "class_names.txt"
    if classes_file.exists():
        out["class_names"] = [ln.strip().split(": ", 1)[-1] for ln in classes_file.read_text().splitlines() if ln.strip()]
    return out


def inventory_exdark(root: Path) -> dict:
    out: dict = {"root": str(root), "exists": root.exists()}
    if not root.exists():
        return out
    img_dir = root / "Dataset" / "Dataset"
    images = sorted(p for p in img_dir.iterdir() if _is_image(p)) if img_dir.exists() else []
    ext_hist = Counter(p.suffix for p in images)
    out["images"] = {
        "n": len(images),
        "by_ext": dict(sorted(ext_hist.items())),
        "sample": [p.name for p in images[:3]],
        "sample_sizes": _sample_sizes(images),
    }
    csv_path = root / "annotations.csv"
    out["csv"] = {"path": str(csv_path), "exists": csv_path.exists()}
    if csv_path.exists():
        rows = 0
        class_hist: Counter = Counter()
        csv_images: set = set()
        bad_coords = 0
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                class_hist[row["class"]] += 1
                csv_images.add(row["image_path"])
                if not (int(row["x_br"]) > int(row["x_tl"]) and int(row["y_br"]) > int(row["y_tl"])):
                    bad_coords += 1
        on_disk = {p.name for p in images}
        out["csv"].update(
            {
                "n_boxes": rows,
                "n_unique_images": len(csv_images),
                "class_hist": dict(sorted(class_hist.items(), key=lambda kv: int(kv[0]))),
                "bad_coord_rows": bad_coords,
                "in_csv_missing_on_disk": sorted(csv_images - on_disk)[:10],
                "on_disk_missing_in_csv": sorted(on_disk - csv_images)[:10],
                "n_in_csv_missing_on_disk": len(csv_images - on_disk),
                "n_on_disk_missing_in_csv": len(on_disk - csv_images),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory LoLI-Street and ExDark datasets.")
    parser.add_argument("--loli", type=Path, default=LOLI_ROOT)
    parser.add_argument("--exdark", type=Path, default=EXDARK_ROOT)
    parser.add_argument("--out", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    loli = inventory_loli(args.loli)
    exdark = inventory_exdark(args.exdark)
    (args.out / "loli_inventory.json").write_text(json.dumps(loli, indent=2))
    (args.out / "exdark_inventory.json").write_text(json.dumps(exdark, indent=2))
    print(f"LoLI-Street splits: { {k: (v['n_high'], v['n_low'], v['paired']) for k, v in loli['splits'].items()} }")
    print(f"LoLI-Street test: {loli['test']['n']} images, labels: {loli['test']['has_labels']}")
    print(f"ExDark images: {exdark['images']['n']}, boxes: {exdark['csv'].get('n_boxes')}")


if __name__ == "__main__":
    main()
