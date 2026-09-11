"""Assemble the master comparison table (proposal section 6) from cell JSONs.

Reads results/cells/*.json + results/classical.json, writes
results/master_table.json, prints the markdown table to stdout (copy it back
for the report), and saves sample grids to reports/figures/.
"""
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent
CELLS = REPO / "results" / "cells"
sys.path.insert(0, str(REPO / "src"))

ROWS = ["none", "classical", "ae", "vae", "pix2pix", "cyclegan"]
ROW_LABEL = {"none": "no restoration (baseline)", "classical": "classical (Part A)",
             "ae": "AE (B.1)", "vae": "VAE (B.2)", "pix2pix": "cGAN (B.3)",
             "cyclegan": "CycleGAN (Part C, ExDark only)"}


def _load(name: str):
    p = CELLS / name
    return json.loads(p.read_text()) if p.exists() else None


def _map(cell: dict | None):
    if not cell:
        return None
    return {"map50": round(cell["map50"], 3), "map50_95": round(cell["map50_95"], 3)}


def build_table() -> dict:
    from classical.pipeline import FROZEN

    classical_full = json.loads((REPO / "results" / "classical.json").read_text())
    table = {"rows": [], "frozen_classical": FROZEN}
    for m in ROWS:
        row: dict = {"method": m, "label": ROW_LABEL[m], "loli": {}, "exdark": {}}
        if m in ("classical", "ae", "vae", "pix2pix"):
            src = classical_full[FROZEN] if m == "classical" else _load(f"{m}_loli.json")
            if src:
                row["loli"]["restoration"] = {k: round(src[k], 3 if k != "psnr" else 2)
                                              for k in ("psnr", "ssim", "mse", "mae") if k in src}
        if m != "cyclegan":
            row["loli"]["zero_shot"] = _map(_load(f"{m}_loli_map.json"))
            row["loli"]["lolift"] = _map(_load(f"{m}_loli_lolift_map.json"))
        row["exdark"]["zero_shot"] = _map(_load(f"{m}_exdark_map.json"))
        row["exdark"]["exdarkft"] = _map(_load(f"{m}_exdark_exdarkft_map.json"))
        table["rows"].append(row)
    return table


def print_markdown(table: dict) -> None:
    def cell50(d):
        return f"{d['map50']:.3f}" if d else "N/A"

    print("\n| method | LoLI PSNR/SSIM | LoLI mAP zero | LoLI mAP lolift | ExDark mAP zero | ExDark mAP exdarkft |")
    print("|---|---|---|---|---|---|")
    for r in table["rows"]:
        rest = r["loli"].get("restoration", {})
        rs = f"{rest.get('psnr', 'N/A')}/{rest.get('ssim', 'N/A')}" if rest else ("—" if r["method"] == "none" else "N/A")
        print(f"| {r['label']} | {rs} | {cell50(r['loli'].get('zero_shot'))} | "
              f"{cell50(r['loli'].get('lolift'))} | {cell50(r['exdark'].get('zero_shot'))} | "
              f"{cell50(r['exdark'].get('exdarkft'))} |")


def _thumb(path: Path, h: int = 256) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
        return im.resize((round(im.width * h / im.height), h))
    except Exception:
        return Image.new("RGB", (h, h), (40, 40, 40))


def save_grids() -> None:
    from data_roots import exdark_root, loli_root

    out = REPO / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    loli_val = loli_root() / "Val"
    stems = sorted(p.stem for p in (loli_val / "low").glob("*.jpg"))[:3]
    cols = [("low", lambda s: loli_val / "low" / f"{s}.jpg"),
            ("high", lambda s: loli_val / "high" / f"{s}.jpg"),
            ("classical", lambda s: REPO / "restored_classical" / "loli_val" / f"{s}.png"),
            ("ae", lambda s: REPO / "restored_ae" / "loli_val" / f"{s}.png"),
            ("vae", lambda s: REPO / "restored_vae" / "loli_val" / f"{s}.png"),
            ("pix2pix", lambda s: REPO / "restored_pix2pix" / "loli_val" / f"{s}.png")]
    rows = [[_thumb(fn(s)) for _, fn in cols] for s in stems]
    w = max(sum(im.width for im in r) for r in rows)
    canvas = Image.new("RGB", (w, 256 * len(rows)), (20, 20, 20))
    for i, r in enumerate(rows):
        x = 0
        for im in r:
            canvas.paste(im, (x, i * 256))
            x += im.width
    canvas.save(out / "grid_loli.png")
    print("saved reports/figures/grid_loli.png", flush=True)

    ex_stems = (REPO / "results" / "exdark_val_stems.txt").read_text().split()
    ex_stems = ex_stems[:3] if len(ex_stems) >= 3 else ex_stems
    ecol = [("raw", lambda s: REPO / "exdark_yolo" / "images" / "val" / f"{s}.jpg"),
            ("classical", lambda s: REPO / "restored_classical" / "exdark" / f"{s}.png"),
            ("ae", lambda s: REPO / "restored_ae" / "exdark_val" / f"{s}.jpg"),
            ("vae", lambda s: REPO / "restored_vae" / "exdark_val" / f"{s}.jpg"),
            ("pix2pix", lambda s: REPO / "restored_pix2pix" / "exdark_val" / f"{s}.jpg"),
            ("cyclegan", lambda s: REPO / "restored_cyclegan" / "exdark_val" / f"{s}.jpg")]
    # Restored ExDark keeps original filenames (any extension): resolve loosely.
    def resolve(fn, s):
        p = fn(s)
        if p.exists():
            return p
        for ext in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG"):
            if p.with_suffix(ext).exists():
                return p.with_suffix(ext)
        return p

    ex_rows = [[_thumb(resolve(fn, s)) for _, fn in ecol] for s in ex_stems]
    if ex_rows:
        w = max(sum(im.width for im in r) for r in ex_rows)
        canvas = Image.new("RGB", (w, 256 * len(ex_rows)), (20, 20, 20))
        for i, r in enumerate(ex_rows):
            x = 0
            for im in r:
                canvas.paste(im, (x, i * 256))
                x += im.width
        canvas.save(out / "grid_exdark.png")
        print("saved reports/figures/grid_exdark.png", flush=True)


def main() -> None:
    table = build_table()
    (REPO / "results" / "master_table.json").write_text(json.dumps(table, indent=2))
    print_markdown(table)
    save_grids()


if __name__ == "__main__":
    main()
