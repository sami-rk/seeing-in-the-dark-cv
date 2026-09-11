"""Fine-tune YOLOv8n on low-light labels (M5).

    #1 LoLI-Street (COCO ids, street-adapted detector):
    <venv>/bin/python src/yolo_eval/finetune.py --data loli_yolo/loli.yaml --name loli_ft --epochs 20

    #2 ExDark (native 0-11 ids, real-low-light-adapted detector):
    <venv>/bin/python src/yolo_eval/finetune.py --data exdark_yolo/exdark.yaml --name exdark_ft --epochs 30

Weights land in runs/detect/<name>/weights/best.pt (gitignored). Every run is
resumed from COCO-pretrained yolov8n.pt unless --weights points elsewhere.
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on a low-light dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, seed=args.seed, name=args.name, verbose=True,
                plots=True, save=True, exist_ok=False)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
