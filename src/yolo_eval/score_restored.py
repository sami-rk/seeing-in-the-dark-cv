"""Score restored images with a YOLO detector (one master-table cell).

Builds a throwaway YOLO layout (symlinks only) under results/eval_layouts/ and
runs the shared evaluate() on it:

    <venv>/bin/python src/yolo_eval/score_restored.py \\
        --images restored_classical/loli_val --labels loli_yolo/labels/val \\
        --ref-yaml loli_yolo/loli.yaml --name classical_loli --save results/cells/classical_loli.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_eval.evaluate import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_layout(images: Path, labels: Path, ref_yaml: Path, name: str,
                 include: set[str] | None = None) -> Path:
    import yaml

    root = REPO_ROOT / "results" / "eval_layouts" / name
    img_out, lbl_out = root / "images", root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = 0
    for img in sorted(images.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        if include is not None and img.stem not in include:
            continue
        link = img_out / img.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img.resolve())
        n_img += 1
        src_lbl = labels / (img.stem + ".txt")
        if src_lbl.exists():
            dst = lbl_out / src_lbl.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src_lbl.resolve())
            n_lbl += 1
    ref = yaml.safe_load(ref_yaml.read_text())
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images\nval: images\nnames: {ref['names']}\n"
    )
    print(f"layout {name}: {n_img} images, {n_lbl} labels")
    return root / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one restored set with YOLO.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ref-yaml", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--include", type=Path, default=None,
                        help="Text file with image stems (one per line) to include.")
    args = parser.parse_args()

    include = None
    if args.include:
        include = {ln.strip() for ln in args.include.read_text().splitlines() if ln.strip()}
    yaml_path = build_layout(args.images, args.labels, args.ref_yaml, args.name, include)
    import json

    result = evaluate(args.weights, str(yaml_path), "val", 0, args.imgsz)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2))
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
