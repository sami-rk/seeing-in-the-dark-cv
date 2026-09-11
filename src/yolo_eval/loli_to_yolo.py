"""Build a YOLO layout for LoLI-Street (low-light inputs, low-split labels).

LoLI-Street labels already use COCO 0-79 ids, so no remapping is needed.
Images and labels are symlinked, never copied:

    <out>/images/{train,val}/<name>  -> LoLI Train/low or Val/low
    <out>/labels/{train,val}/<stem>.txt
    <out>/loli.yaml

Run from repo root:

    <venv>/bin/python src/yolo_eval/loli_to_yolo.py [--out loli_yolo]
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOLI_ROOT = REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset"

def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO layout for LoLI-Street low images.")
    parser.add_argument("--loli", type=Path, default=LOLI_ROOT)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "loli_yolo")
    args = parser.parse_args()

    label_dirs = {
        "train": args.loli / "YOLO Annotations" / "Train" / "low" / "Labels",
        "val": args.loli / "YOLO Annotations" / "Val" / "YOLO Annotations (low)" / "Labels",
    }
    img_dirs = {"train": args.loli / "Train" / "low", "val": args.loli / "Val" / "low"}
    summary = {}
    for split in ("train", "val"):
        img_out = args.out / "images" / split
        lbl_out = args.out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_img = n_lbl = 0
        for img in sorted(img_dirs[split].iterdir()):
            if not img.is_file():
                continue
            _link(img, img_out / img.name)
            n_img += 1
            src_lbl = label_dirs[split] / (img.stem + ".txt")
            if src_lbl.exists():
                _link(src_lbl, lbl_out / src_lbl.name)
                n_lbl += 1
        summary[split] = {"images": n_img, "labels": n_lbl}

    classes_file = args.loli / "YOLO Annotations" / "Train" / "high" / "Classes" / "class_names.txt"
    names = [ln.strip().split(": ", 1)[-1] for ln in classes_file.read_text().splitlines() if ln.strip()]
    (args.out / "loli.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names: {json.dumps({i: n for i, n in enumerate(names)})}\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
