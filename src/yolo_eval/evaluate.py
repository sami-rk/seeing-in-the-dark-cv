"""Single YOLO evaluation harness for every (dataset, restoration, detector) combo.

Usage (zero-shot COCO detector on ExDark val, remapping GT 0-11 to COCO ids):

    <venv>/bin/python src/yolo_eval/evaluate.py --weights yolov8n.pt \\
        --data exdark_yolo/exdark.yaml --remap-coco --save results/baselines/exdark_zero_shot.json

Metrics come straight from ``model.val()``: mAP@0.5, mAP@0.5:0.95, precision,
recall, plus per-class AP@0.5. Nothing is reimplemented.
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_overlap_map(path: Path | None = None) -> dict:
    path = path or REPO_ROOT / "data_manifests" / "overlap_map.json"
    return json.loads(Path(path).read_text())


def remap_exdark_labels_to_coco(build_dir: Path, overlap: dict) -> Path:
    """Write a shadow val set whose GT ids are COCO ids; return its yaml path.

    Ultralytics derives each label path from its image path (``images/`` ->
    ``labels/``), so both trees are mirrored: images are symlinked, labels
    are rewritten with COCO ids.
    """
    build_dir = Path(build_dir)
    name_to_coco = overlap["exdark_name_to_coco_id"]
    id_to_name = overlap["exdark_csv_id_to_name"]
    remap = {int(k): name_to_coco[v] for k, v in id_to_name.items()}
    img_src = build_dir / "images" / "val"
    img_dst = build_dir / "images" / "val_coco"
    lbl_src = build_dir / "labels" / "val"
    lbl_dst = build_dir / "labels" / "val_coco"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img in sorted(img_src.iterdir()):
        link = img_dst / img.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img.resolve())
        src = lbl_src / (img.stem + ".txt")
        lines = []
        if src.exists():
            for ln in src.read_text().splitlines():
                parts = ln.split()
                if not parts:
                    continue
                lines.append(f"{remap[int(parts[0])]} {' '.join(parts[1:])}")
        (lbl_dst / (img.stem + ".txt")).write_text("\n".join(lines) + "\n" if lines else "")
    base = (build_dir / "exdark.yaml").read_text().splitlines()
    yaml_path = build_dir / "exdark_coco.yaml"
    yaml_path.write_text(
        "\n".join(ln if not ln.startswith("val:") else "val: images/val_coco" for ln in base) + "\n"
    )
    return yaml_path


def evaluate(weights: str, data: str, split: str = "val", device=0, imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data, split=split, verbose=False, device=device, imgsz=imgsz, plots=False, save_json=False)
    box = metrics.box
    # NOTE: box.ap / box.ap50 pair with box.ap_class_index (seen classes only).
    # box.maps is length nc with unseen classes filled by the mean -- do NOT zip it.
    per_class = {
        str(int(c)): {"ap50": float(a50), "ap": float(a)}
        for c, a50, a in zip(box.ap_class_index, box.ap50, box.ap)
    }
    return {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }


def _count_split_images(data_yaml: str, split: str) -> int:
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    split_dir = Path(cfg["path"]) / cfg[split]
    return sum(1 for p in split_dir.iterdir() if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one (weights, dataset) combo.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--remap-coco", action="store_true", help="Remap ExDark-native GT ids to COCO ids first.")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    data = args.data
    if args.remap_coco:
        data = str(remap_exdark_labels_to_coco(Path(args.data).parent, load_overlap_map()))
    result = evaluate(args.weights, data, args.split, args.device, args.imgsz)
    result["n_images"] = _count_split_images(data, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2))
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
