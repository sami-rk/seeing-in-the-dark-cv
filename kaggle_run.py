"""One-shot runner: executes configs/kaggle.yaml end to end on a fresh GPU box.

On Kaggle (GPU on), three cells do everything:

    !git clone https://github.com/sami-rk/seeing-in-the-dark-cv.git
    %cd seeing-in-the-dark-cv
    !pip install -q -r requirements-kaggle.txt

    !python kaggle_run.py            # ~8h on a T4, logs stream to the console

Then download /kaggle/working/seeing-in-the-dark-cv/results (Save Version ->
Output) and send back: results/master_table.json, results/cells/*.json,
checkpoints/*.pt, reports/figures/*.png, results/stage_log.jsonl.

Resume after an interruption: !python kaggle_run.py --skip <done,stages>.
Re-run one piece: !python kaggle_run.py --only <stage>.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
RESULTS = REPO / "results"
CELLS = RESULTS / "cells"
STAGE_LOG = RESULTS / "stage_log.jsonl"

PAIRED = ["ae", "vae", "pix2pix"]

STAGES = ["inventory", "layouts", "baselines", "classical", "ae", "vae", "pix2pix",
          "cyclegan", "finetune", "adapted_cells", "master"]


def log_stage(name: str, status: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with STAGE_LOG.open("a") as fh:
        fh.write(json.dumps({"stage": name, "status": status, "t": time.strftime("%FT%T")}) + "\n")


def run(cmd: list[str], stage: str) -> None:
    print(f"\n===== [{stage}] $ {' '.join(str(c) for c in cmd)} =====", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=REPO)
    if r.returncode != 0:
        log_stage(stage, f"FAILED rc={r.returncode}")
        raise SystemExit(f"stage {stage} failed with rc={r.returncode} -- fix, then resume with --skip")
    log_stage(stage, "ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole project end to end.")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "kaggle.yaml")
    parser.add_argument("--only", default=None, help="Comma-separated stages to run.")
    parser.add_argument("--skip", default=None, help="Comma-separated stages to skip.")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    only = set(args.only.split(",")) if args.only else set(STAGES)
    skip = set(args.skip.split(",")) if args.skip else set()
    active = [s for s in STAGES if s in only and s not in skip]
    print("stages:", active, flush=True)

    PY = sys.executable
    dev = str(cfg.get("device", 0))
    size = cfg.get("train_size", 256)
    imgsz = cfg.get("yolo_imgsz", 640)
    CELLS.mkdir(parents=True, exist_ok=True)

    if "inventory" in active:
        run([PY, "src/yolo_eval/inventory.py"], "inventory")

    if "layouts" in active:
        run([PY, "src/yolo_eval/exdark_to_yolo.py"], "layouts")
        run([PY, "src/yolo_eval/loli_to_yolo.py"], "layouts")
        stems = sorted(p.stem for p in (REPO / "exdark_yolo" / "images" / "val").iterdir())
        (RESULTS / "exdark_val_stems.txt").write_text("\n".join(stems) + "\n")
        print(f"wrote {len(stems)} val stems", flush=True)

    if "baselines" in active:
        run([PY, "src/yolo_eval/evaluate.py", "--weights", "yolov8n.pt",
             "--data", "exdark_yolo/exdark.yaml", "--remap-coco", "--device", dev,
             "--save", "results/cells/none_exdark_map.json"], "baselines")
        run([PY, "src/yolo_eval/evaluate.py", "--weights", "yolov8n.pt",
             "--data", "loli_yolo/loli.yaml", "--device", dev,
             "--save", "results/cells/none_loli_map.json"], "baselines")

    if "classical" in active:
        c = cfg["stages"]["classical"]
        if c.get("full_val_compare"):
            run([PY, "src/classical/pipeline.py", "--out", "results/classical.json"], "classical")
        if c.get("restore_loli_val"):
            import sys as _s
            _s.path.insert(0, str(SRC))
            from classical.pipeline import FROZEN
            from data_roots import loli_root
            run([PY, "src/classical/pipeline.py", "--mode", FROZEN, "--restore-only",
                 str(loli_root() / "Val" / "low"), "restored_classical/loli_val"], "classical")
        if c.get("restore_exdark"):
            from data_roots import exdark_root
            run([PY, "src/classical/pipeline.py", "--mode", FROZEN, "--restore-only",
                 str(exdark_root() / "Dataset" / "Dataset"), "restored_classical/exdark"], "classical")
        run([PY, "src/yolo_eval/score_restored.py", "--images", "restored_classical/loli_val",
             "--labels", "loli_yolo/labels/val", "--ref-yaml", "loli_yolo/loli.yaml",
             "--name", "classical_loli", "--save", "results/cells/classical_loli_map.json"], "classical")
        run([PY, "src/yolo_eval/score_restored.py", "--images", "restored_classical/exdark",
             "--labels", "exdark_yolo/labels/val_coco", "--ref-yaml", "loli_yolo/loli.yaml",
             "--include", "results/exdark_val_stems.txt",
             "--name", "classical_exdark", "--save", "results/cells/classical_exdark_map.json"], "classical")

    for method in PAIRED:
        if method not in active:
            continue
        p = cfg["stages"]["paired"][method]
        subset = [str(p["subset"])] if p.get("subset") else ["-1"]
        run([PY, f"src/paired/{'autoencoder' if method == 'ae' else method}.py",
             "--epochs", str(p["epochs"]), "--subset", *subset, "--size", str(size),
             "--batch", str(p.get("batch", 16 if method != "pix2pix" else 8)),
             "--out", f"checkpoints/{method}.pt",
             "--history", f"results/paired/{method}_history.json"], method)
        run([PY, "src/paired/restore_val.py", "--model", method,
             "--weights", f"checkpoints/{method}.pt",
             "--out", f"restored_{method}/loli_val",
             "--metrics", f"results/cells/{method}_loli.json"], method)
        run([PY, "src/yolo_eval/score_restored.py", "--images", f"restored_{method}/loli_val",
             "--labels", "loli_yolo/labels/val", "--ref-yaml", "loli_yolo/loli.yaml",
             "--name", f"{method}_loli", "--save", f"results/cells/{method}_loli_map.json"], method)
        run([PY, "src/paired/restore_val.py", "--model", method,
             "--weights", f"checkpoints/{method}.pt", "--dataset", "exdark",
             "--img-dir", "exdark_yolo/images/val",
             "--out", f"restored_{method}/exdark_val"], method)
        run([PY, "src/yolo_eval/score_restored.py", "--images", f"restored_{method}/exdark_val",
             "--labels", "exdark_yolo/labels/val_coco", "--ref-yaml", "loli_yolo/loli.yaml",
             "--name", f"{method}_exdark", "--save", f"results/cells/{method}_exdark_map.json"], method)

    if "cyclegan" in active:
        g = cfg["stages"]["cyclegan"]
        run([PY, "src/cyclegan/train_cyclegan.py", "--epochs", str(g["epochs"]),
             "--subset", str(g["subset"]), "--size", str(size), "--batch", str(g.get("batch", 1)),
             "--out", "checkpoints/cyclegan.pt",
             "--history", "results/paired/cyclegan_history.json"], "cyclegan")
        run([PY, "src/paired/restore_val.py", "--model", "cyclegan",
             "--weights", "checkpoints/cyclegan.pt", "--dataset", "exdark",
             "--img-dir", "exdark_yolo/images/val",
             "--out", "restored_cyclegan/exdark_val"], "cyclegan")
        run([PY, "src/yolo_eval/score_restored.py", "--images", "restored_cyclegan/exdark_val",
             "--labels", "exdark_yolo/labels/val_coco", "--ref-yaml", "loli_yolo/loli.yaml",
             "--name", "cyclegan_exdark",
             "--save", "results/cells/cyclegan_exdark_map.json"], "cyclegan")

    if "finetune" in active:
        f = cfg["stages"]["finetune"]
        run([PY, "src/yolo_eval/finetune.py", "--data", "loli_yolo/loli.yaml",
             "--name", "loli_ft", "--epochs", str(f["loli"]["epochs"]),
             "--batch", str(f["loli"].get("batch", 32)), "--imgsz", str(imgsz),
             "--device", dev], "finetune")
        run([PY, "src/yolo_eval/finetune.py", "--data", "exdark_yolo/exdark.yaml",
             "--name", "exdark_ft", "--epochs", str(f["exdark"]["epochs"]),
             "--batch", str(f["exdark"].get("batch", 32)), "--imgsz", str(imgsz),
             "--device", dev], "finetune")

    if "adapted_cells" in active:
        ft_loli = "runs/detect/loli_ft/weights/best.pt"
        ft_exdark = "runs/detect/exdark_ft/weights/best.pt"
        loli_src = {"none": "loli_yolo/images/val", **{m: f"restored_{m}/loli_val" for m in LOLI_METHODS}}
        for m, imgdir in loli_src.items():
            run([PY, "src/yolo_eval/score_restored.py", "--images", imgdir,
                 "--labels", "loli_yolo/labels/val", "--ref-yaml", "loli_yolo/loli.yaml",
                 "--weights", ft_loli, "--name", f"{m}_loli_lolift",
                 "--save", f"results/cells/{m}_loli_lolift_map.json"], "adapted_cells")
        ex_src = {"none": "exdark_yolo/images/val", "classical": "restored_classical/exdark",
                  **{m: f"restored_{m}/exdark_val" for m in ["ae", "vae", "pix2pix", "cyclegan"]}}
        for m, imgdir in ex_src.items():
            extra = ["--include", "results/exdark_val_stems.txt"] if m in ("none", "classical") else []
            labels = "exdark_yolo/labels/val_coco"
            run([PY, "src/yolo_eval/score_restored.py", "--images", imgdir,
                 "--labels", labels, "--ref-yaml", "loli_yolo/loli.yaml",
                 "--weights", ft_exdark, "--name", f"{m}_exdark_exdarkft", *extra,
                 "--save", f"results/cells/{m}_exdark_exdarkft_map.json"], "adapted_cells")

    if "master" in active:
        run([PY, "kaggle_collect.py"], "master")

    print("\nALL STAGES DONE:", active, flush=True)


if __name__ == "__main__":
    main()
