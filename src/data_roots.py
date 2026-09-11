"""Dataset root resolution: env vars > auto-discovery > repo-local defaults.

Environment (used on Kaggle/Colab, where datasets mount elsewhere):

    LOLI_ROOT   -> dir containing Train/Val/Test  (e.g. .../LoLI-Street Dataset)
    EXDARK_ROOT -> dir containing annotations.csv (e.g. .../ExDarkDataset)

Without env vars, the loaders search the repo root and /kaggle/input for those
markers, so the Kaggle datasets work with zero code changes as long as their
internal layout matches the local copy.
"""
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_BASES = [REPO_ROOT, Path("/kaggle/input")]


def _discover(marker: tuple[str, ...], extra: tuple[str, ...] = ()) -> Path | None:
    """Depth-limited recursive search of SEARCH_BASES for a dir holding marker.

    Returns the shallowest match; prefers one that also holds extra.
    """
    for base in SEARCH_BASES:
        if not base.exists():
            continue
        candidates: list[Path] = []
        stack: list[tuple[Path, int]] = [(base, 0)]
        while stack:
            cand, depth = stack.pop()
            try:
                children = sorted(p for p in cand.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                continue
            if all((cand / m).exists() for m in marker):
                candidates.append(cand)
            if depth < 4:
                stack.extend((c, depth + 1) for c in children)
        if candidates:
            candidates.sort(key=lambda p: (len(p.parts), str(p)))
            for cand in candidates:
                if all((cand / m).exists() for m in extra):
                    return cand
            return candidates[0]
    return None


def _local_loli() -> Path:
    return REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset"


def _local_exdark() -> Path:
    return REPO_ROOT / "ExDarkDataset"


@lru_cache(maxsize=None)
def loli_root() -> Path:
    if os.environ.get("LOLI_ROOT"):
        return Path(os.environ["LOLI_ROOT"])
    found = _discover(("Train", "Val"), extra=("Train/low", "Val/low"))
    if found:
        return found
    return _local_loli()


@lru_cache(maxsize=None)
def exdark_root() -> Path:
    if os.environ.get("EXDARK_ROOT"):
        return Path(os.environ["EXDARK_ROOT"])
    found = _discover(("annotations.csv",))
    if found:
        return found
    return _local_exdark()


def describe() -> dict:
    """Resolved roots + a shallow listing of /kaggle/input for diagnostics."""
    info: dict = {"loli_root": str(loli_root()), "exdark_root": str(exdark_root())}
    kaggle_in = Path("/kaggle/input")
    tree: list[str] = []
    if kaggle_in.exists():
        for lvl1 in sorted(p for p in kaggle_in.iterdir()):
            tree.append(lvl1.name + ("/" if lvl1.is_dir() else ""))
            if lvl1.is_dir():
                try:
                    kids = sorted(p.name + ("/" if p.is_dir() else "") for p in lvl1.iterdir())
                except OSError:
                    kids = ["<unreadable>"]
                tree.extend(f"  {k}" for k in kids[:15])
                if len(kids) > 15:
                    tree.append(f"  ... ({len(kids) - 15} more)")
    info["kaggle_input_tree"] = tree
    return info
