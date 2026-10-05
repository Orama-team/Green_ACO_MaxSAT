#!/usr/bin/env python3
"""
run_profiling.py
================
Measure each operator in isolation, per instance and per regime.

Writes ``operator_profiles_<subset>.csv`` with, for every
(instance, operator, regime), the mean and standard deviation of the energy
cost and of the quality gain.

This stage must run before the main sweep: the profiles seed the scheduler's
cost priors, and without them the first operator calls are chosen blind, which
lets a single expensive call overrun the budget several times over.

Usage:
    python scripts/run_profiling.py --subset core
    python scripts/run_profiling.py --steps 200 --n-jobs 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import (SIZE_THRESHOLD_CLAUSES, banner, base_parser,  # noqa: E402
                       energy_meter, results_dir, write_csv)
from greenaco.config import OPERATOR_POOL, REGIME_EXPLOITATION  # noqa: E402
from greenaco.data import load_benchmark  # noqa: E402
from greenaco.profile import PROFILE_COLUMNS, profile_operator  # noqa: E402
from greenaco.runner import CheckpointStore, run_tasks  # noqa: E402


def main() -> int:
    parser = base_parser("Profile the operator pool.")
    parser.add_argument("--steps", type=int, default=200,
                        help="Operator calls per (instance, operator, regime).")
    args = parser.parse_args()

    banner("Operator profiling")
    bench = load_benchmark(args.subset)
    print(f"  subset {args.subset}: {len(bench)} instances x "
          f"{len(OPERATOR_POOL)} operators x {len(REGIME_EXPLOITATION)} regimes "
          f"x {args.steps} steps")

    # A live CodeCarbon tracker cannot be pickled to a worker, so the meter is
    # created lazily inside whichever process runs the task and cached there.
    _meters = {}

    def _meter_for(region):
        if region not in _meters:
            _meters[region] = energy_meter(region)
        return _meters[region]

    def _task(item):
        b, op, regime = item
        return profile_operator(b, op, regime, meter=_meter_for(args.region),
                                n_steps=args.steps, seed=args.seed)

    tasks = [(b, op, regime)
             for b in bench
             for op in OPERATOR_POOL
             for regime in REGIME_EXPLOITATION]

    store = CheckpointStore(f"profile_{args.subset}")
    rows = run_tasks(
        tasks, _task, store,
        n_jobs=args.n_jobs,
        size_threshold=SIZE_THRESHOLD_CLAUSES,
        task_key=lambda t: (t[0].benchmark, t[1], t[2]),
    )

    if not rows:
        print("  no profiling rows produced")
        return 1

    df = pd.DataFrame(rows)
    for col in PROFILE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PROFILE_COLUMNS + [c for c in df.columns if c not in PROFILE_COLUMNS]]

    out = results_dir(args.out_dir)
    tag = "all50" if args.subset == "all" else args.subset
    write_csv(df, out, f"operator_profiles_{tag}.csv")

    print("\n  Mean cost per operator (Joules):")
    summary = df.groupby(["operator", "regime"])["mean_cost_j"].mean().round(2)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())