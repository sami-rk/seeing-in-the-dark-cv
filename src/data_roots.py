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


def _discover(marker: tuple[str, ...]) -> Path | None:
    for base in SEARCH_BASES:
        if not base.exists():
            continue
        for cand in [base, *sorted(p for p in base.iterdir() if p.is_dir())]:
            if all((cand / m).exists() for m in marker):
                for inner in sorted(p for p in cand.iterdir() if p.is_dir()):
                    if all((inner / m).exists() for m in marker):
                        return inner
                return cand
    return None


def _local_loli() -> Path:
    return REPO_ROOT / "LoLI-Street: Low-Light Image Enhancement of Street" / "LoLI-Street Dataset"


def _local_exdark() -> Path:
    return REPO_ROOT / "ExDarkDataset"


@lru_cache(maxsize=None)
def loli_root() -> Path:
    if os.environ.get("LOLI_ROOT"):
        return Path(os.environ["LOLI_ROOT"])
    found = _discover(("Train", "Val"))
    if found and (found / "Train" / "low").exists():
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
