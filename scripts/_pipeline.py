"""
pipeline.py
===========
Stage definitions and shared CLI plumbing for the experiment scripts.

Each stage writes exactly one or two CSV files into ``results/`` and nothing
else. Figures are produced by a separate pass that only ever reads those CSVs,
so regenerating a figure never re-runs an experiment and a re-run never
silently changes a figure.

Stages, in dependency order:

  prepare     verify the manifest and the instance files
  profile     measure each operator per instance and regime
  finetune    derive the hyper-parameters (or reuse the tuned ones)
  final       the main sweep: instances x budgets x repeats
  ablation    remove one component at a time
  comparison  run the GA/ACO methods for the comparison table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Allow `python scripts/xxx.py` to import the package without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from greenaco import paths  # noqa: E402
from greenaco.data import CORE_INSTANCES  # noqa: E402

STAGES = ["prepare", "profile", "finetune", "final", "ablation", "comparison"]

# Instances above this clause count run serially: they are the slowest and the
# most memory-hungry, and running several at once risks exhausting RAM.
SIZE_THRESHOLD_CLAUSES = 400_000

# Default repeat count for the main sweep.
DEFAULT_RUNS = 10

BUDGETS = [400.0, 1000.0, 2000.0]


def base_parser(description: str, subset_default: str = "all") -> argparse.ArgumentParser:
    """Argument parser shared by every stage script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--subset", default=subset_default,
        help="Instance subset: 'all' (the full benchmark), 'core' (the "
             f"{len(CORE_INSTANCES)} original comparison instances), or a "
             "comma-separated list of benchmark names.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed; fixed runs are reproducible.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers (default: min(4, cpus-1)).")
    parser.add_argument("--region", default="DZ",
                        help="Grid region code for the CO2 intensity.")
    parser.add_argument("--budgets", default=None,
                        help="Comma-separated energy budgets in Joules "
                             "(default: 400,1000,2000).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: results/).")
    return parser


def selected_budgets(raw: str | None) -> List[float]:
    if not raw:
        return list(BUDGETS)
    return [float(x) for x in raw.split(",") if x.strip()]


def results_dir(out_dir: str | None = None) -> Path:
    return Path(out_dir) if out_dir else paths.RESULTS


def subset_tag(subset: str) -> str:
    """Short tag appended to artifact names so scopes never collide."""
    return paths.subset_suffix(subset)


def energy_meter(region: str = "DZ"):
    from greenaco.energy import EnergyMeter

    return EnergyMeter(region=region)


def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def write_csv(df, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    df.to_csv(target, index=False)
    print(f"  -> wrote {target.name} ({len(df)} rows)")
    return target