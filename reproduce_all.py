#!/usr/bin/env python3
"""
reproduce_all.py
================
One command to reproduce the study.

Runs the stages in dependency order and then builds the figures::

    prepare -> profile -> finetune -> final -> ablation -> comparison
            -> figures -> verify

Each stage is a separate script and can equally be run on its own; this is the
convenience wrapper, not the only way in.

Two ways to use it:

  --use-shipped   skip the experiments and build figures from the CSVs shipped
                  in results/shipped/. Fast, and the right default for anyone
                  who just wants the figures.

  (default)      run the experiments. The full 50-instance sweep is a long job;
                  it is resumable, so an interrupted run continues where it
                  stopped. Use --subset core for a much shorter run, and
                  --stages to run part of the pipeline.

Examples:
    python reproduce_all.py --use-shipped
    python reproduce_all.py --subset core --stages profile,final
    python reproduce_all.py --subset all --stages prepare,profile,finetune,final
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

STAGES = ["prepare", "profile", "finetune", "final", "ablation", "comparison"]


def run(label: str, argv: list) -> int:
    print(f"\n{'#' * 68}\n#  {label}\n{'#' * 68}")
    t0 = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"\n-- {label}: {status} in {time.time() - t0:.1f}s")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="all", choices=["all", "core"])
    parser.add_argument("--stages", default="all",
                        help=f"Comma-separated subset of {STAGES}, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers. Recorded in MANIFEST.json "
                             "because contention affects measured energy.")
    parser.add_argument("--runs", type=int, default=10,
                        help="Repeats per (instance, budget) in the sweep.")
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--steps", type=int, default=200,
                        help="Operator calls per profile cell.")
    parser.add_argument("--use-shipped", action="store_true",
                        help="Only build figures from the shipped CSVs.")
    args = parser.parse_args()

    if args.use_shipped:
        return run("Figures (shipped results)",
                   ["scripts/make_figures.py", "--use-shipped",
                    "--subset", args.subset])

    chosen = STAGES if args.stages == "all" else [
        s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in STAGES]
    if unknown:
        print(f"Unknown stage(s): {unknown}. Valid: {STAGES}")
        return 2

    common = ["--subset", args.subset, "--seed", str(args.seed)]
    if args.n_jobs:
        common += ["--n-jobs", str(args.n_jobs)]
    if args.budgets:
        common += ["--budgets", args.budgets]

    scripts = {
        "prepare": ["scripts/prepare_data.py"],
        "profile": ["scripts/run_profiling.py", "--steps", str(args.steps)],
        "finetune": ["scripts/run_finetuning.py", "--mode", "frozen"],
        "final": ["scripts/run_final.py", "--runs", str(args.runs)],
        "ablation": ["scripts/run_ablation.py"],
        "comparison": ["scripts/run_comparison.py"],
    }

    failures = []
    for stage in chosen:
        argv = [*scripts[stage], *common]
        if run(f"stage: {stage}", argv) != 0:
            failures.append(stage)

    tag = "all50" if args.subset == "all" else args.subset
    if run(f"Figures ({tag})",
           ["scripts/make_figures.py", "--subset", args.subset]) != 0:
        failures.append("figures")

    run("Verify", ["scripts/verify_results.py", "--subset", args.subset])

    print(f"\n{'=' * 68}")
    if failures:
        print(f"  finished with failures in: {', '.join(failures)}")
        return 1
    print("  all stages completed")
    print("  results -> results/    figures -> figures/")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())