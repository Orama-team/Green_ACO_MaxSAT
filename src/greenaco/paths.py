"""
paths.py
=======
Canonical filesystem layout for the project.

Every path used anywhere in the pipeline is derived here, so that moving or
copying the repository never requires editing code. All directories are created
on demand by :func:`ensure_dirs`.
"""

from __future__ import annotations

from pathlib import Path

# Repository root: .../<repo>/src/greenaco/paths.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
INSTANCES_DIR = DATA / "mse24"
COMPARISON_DIR = DATA / "comparison"
CONFIGS_DIR = ROOT / "configs"
RESULTS = ROOT / "results"
SHIPPED = RESULTS / "shipped"
CHECKPOINTS = RESULTS / "checkpoints"
FIGURES = ROOT / "figures"
FIGS = SHIPPED
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist."""
    for d in (RESULTS, SHIPPED, CHECKPOINTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def manifest_path() -> Path:
    return DATA / "manifest.csv"


def subset_suffix(subset: str) -> str:
    """File-name suffix identifying an instance subset.

    The benchmark artifacts carry the subset in their name so that results from
    different scopes never silently overwrite one another.
    """
    return "all54" if subset == "all" else subset