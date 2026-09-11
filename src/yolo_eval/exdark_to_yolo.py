"""Convert ExDark annotations.csv to Ultralytics YOLO label format.

ExDark-native class ids 0-11 are kept as-is (see data_manifests/overlap_map.json
for the id-to-name mapping and the COCO equivalents used at eval time).

Layout (all under a local, gitignored build dir; images are symlinked, never copied):

    <out>/images/{train,val}/<name>  -> symlink to ExDarkDataset/Dataset/Dataset/<name>
    <out>/labels/{train,val}/<stem>.txt
    <out>/exdark.yaml

Split is a deterministic 90/10 filename split (seed 42). Run from repo root:

    <venv>/bin/python src/yolo_eval/exdark_to_yolo.py [--out exdark_yolo]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_roots import exdark_root

REPO_ROOT = Path(__file__).resolve().parents[2]
EXDARK_ROOT = exdark_root()
OVERLAP_MAP = REPO_ROOT / "data_manifests" / "overlap_map.json"

VAL_FRACTION = 0.10
SEED = 42


def _resolve_image(name: str, img_dir: Path) -> Path | None:
    direct = img_dir / name
    if direct.exists():
        return direct
    lowered = name.lower()
    for cand in img_dir.iterdir():
        if cand.name.lower() == lowered:
            return cand
    return None


def _to_yolo(x_tl, y_tl, x_br, y_br, img_w, img_h):
    cx = (x_tl + x_br) / 2 / img_w
    cy = (y_tl + y_br) / 2 / img_h
    w = (x_br - x_tl) / img_w
    h = (y_br - y_tl) / img_h
    if w <= 0 or h <= 0:
        return None
    return (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ExDark CSV annotations to YOLO format.")
    parser.add_argument("--exdark", type=Path, default=EXDARK_ROOT)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "exdark_yolo")
    args = parser.parse_args()

    img_dir = args.exdark / "Dataset" / "Dataset"
    boxes: dict[str, list[str]] = {}
    skipped = 0
    with (args.exdark / "annotations.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            yolo = _to_yolo(
                float(row["x_tl"]), float(row["y_tl"]), float(row["x_br"]), float(row["y_br"]),
                float(row["image_width"]), float(row["image_height"]),
            )
            if yolo is None:
                skipped += 1
                continue
            cx, cy, w, h = yolo
            boxes.setdefault(row["image_path"], []).append(f"{row['class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    names = sorted({p.name for p in img_dir.iterdir() if p.is_file()})
    rng = random.Random(SEED)
    shuffled = sorted(names)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    splits = {"val": set(shuffled[:n_val]), "train": set(shuffled[n_val:])}

    names_map = json.loads(OVERLAP_MAP.read_text())["exdark_csv_id_to_name"]
    class_names = [names_map[str(i)] for i in range(12)]

    missing_images: list[str] = []
    for split, split_names in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, split_names in splits.items():
        for name in sorted(split_names):
            src = _resolve_image(name, img_dir)
            if src is None:
                missing_images.append(name)
                continue
            link = args.out / "images" / split / src.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src.resolve())
            label = args.out / "labels" / split / (src.stem + ".txt")
            rows = boxes.get(name, boxes.get(src.name, []))
            label.write_text("\n".join(rows) + "\n" if rows else "")

    (args.out / "exdark.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names: {json.dumps({i: n for i, n in enumerate(class_names)})}\n"
    )
    n_boxes = sum(len(v) for v in boxes.values())
    summary = {
        "images": len(names),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "boxes": n_boxes,
        "skipped_degenerate": skipped,
        "missing_images": missing_images,
        "class_hist": dict(sorted(Counter(l.split()[0] for v in boxes.values() for l in v).items(), key=lambda kv: int(kv[0]))),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
