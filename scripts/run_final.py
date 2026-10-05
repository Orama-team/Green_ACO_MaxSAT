#!/usr/bin/env python3
"""
run_final.py
============
The main sweep: every instance x every budget x N repeats.

Writes two files:

  final_runs_<subset>.csv      one row per individual run
  final_summary_<subset>.csv   mean and standard deviation per (instance, budget)

Each run starts from the tuned hyper-parameters and is seeded, so a given
(instance, budget, seed) reproduces the same solution. Energy is measured, so
the Joule columns reflect the machine; quality and the resulting rankings are
seed-reproducible.

Operator profiles from the profiling stage seed the scheduler's cost priors.
Without them the first calls are chosen blind and can overrun the budget badly,
so this stage reads ``operator_profiles_<subset>.csv`` when present and warns
if it does not.

Usage:
    python scripts/run_final.py --subset core --runs 10
    python scripts/run_final.py --subset core --budgets 400
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import (DEFAULT_RUNS, SIZE_THRESHOLD_CLAUSES,  # noqa: E402
                       banner, base_parser, energy_meter, results_dir,
                       selected_budgets, write_csv)
from greenaco.config import GreenACOConfig, TUNED_PARAMS  # noqa: E402
from greenaco.data import load_benchmark  # noqa: E402
from greenaco.profile import load_profiles  # noqa: E402
from greenaco.runner import (CheckpointStore, machine_info,  # noqa: E402
                             run_tasks, write_manifest)
from greenaco.solver import GreenACOSolver, solve_instance  # noqa: E402

SUMMARY_METRICS = ["qualite_pct", "energie_joules", "score_per_joule",
                   "co2_micrograms"]


def load_params(path: Path) -> dict:
    """Tuned parameters, from ``configs/best_params.json`` if present."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return dict(TUNED_PARAMS)


def main() -> int:
    parser = base_parser("Run the main Green ACO sweep.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Repeats per (instance, budget).")
    parser.add_argument("--params", default=None,
                        help="Tuned-parameter JSON "
                             "(default: configs/best_params.json).")
    parser.add_argument("--no-profiles", action="store_true",
                        help="Ignore operator profiles.")
    args = parser.parse_args()

    banner("Final sweep")
    out = results_dir(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "all54" if args.subset == "all" else args.subset
    budgets = selected_budgets(args.budgets)

    bench = load_benchmark(args.subset)
    print(f"  subset {args.subset}: {len(bench)} instances x "
          f"{len(budgets)} budgets x {args.runs} runs")

    params_path = (Path(args.params) if args.params
                   else Path("configs/best_params.json"))
    params = load_params(params_path)
    print(f"  parameters   : {params_path}"
          f"{'' if params_path.exists() else ' (absent, tuned defaults)'}")

    # Operator profiles seed the scheduler's cost priors.
    profiles = {}
    profile_file = out / f"operator_profiles_{tag}.csv"
    if args.no_profiles:
        print("  profiles     : disabled by --no-profiles")
    elif profile_file.exists():
        rows = pd.read_csv(profile_file).to_dict("records")
        profiles = load_profiles(rows, [b.benchmark for b in bench])
        covered = sum(1 for v in profiles.values() if v)
        print(f"  profiles     : {profile_file.name} "
              f"({covered}/{len(bench)} covered)")
    else:
        print(f"  [warn] {profile_file.name} not found: operator selection "
              f"starts from uninformed priors. Run run_profiling.py first.")

    # One meter per process; a live tracker cannot cross a process boundary.
    _meters = {}

    def _meter_for(region):
        if region not in _meters:
            _meters[region] = energy_meter(region)
        return _meters[region]

    def _task(item):
        b, budget, run_id = item
        cfg = GreenACOConfig.from_dict({
            **params, "budget_j": budget, "seed": args.seed + run_id,
        })
        solver = GreenACOSolver(cfg, _meter_for(args.region),
                                profiles=profiles.get(b.benchmark, {}))
        stats = solve_instance(solver, b)
        stats["run"] = run_id
        stats["budget_j"] = budget
        stats["region"] = args.region
        return stats

    tasks = [(b, budget, run_id)
             for b in bench
             for budget in budgets
             for run_id in range(args.runs)]

    store = CheckpointStore(f"final_{args.subset}")
    rows = run_tasks(
        tasks, _task, store,
        n_jobs=args.n_jobs,
        size_threshold=SIZE_THRESHOLD_CLAUSES,
        task_key=lambda t: (t[0].benchmark, int(t[1]), t[2]),
    )

    rows = [r for r in rows if r and "__error__" not in r]
    if not rows:
        print("  no results produced")
        return 1

    df = pd.DataFrame(rows)
    df = df.drop(columns=["operateurs_usage", "operateurs_energie_j"],
                 errors="ignore")
    df = df.sort_values(["instance", "budget_j", "run"]).reset_index(drop=True)
    write_csv(df, out, f"final_runs_{tag}.csv")

    summary = (df.groupby(["instance", "budget_j"])
                 .agg(n_runs=("run", "count"),
                      **{f"mean_{c}": (c, "mean") for c in SUMMARY_METRICS},
                      **{f"std_{c}": (c, "std") for c in SUMMARY_METRICS})
                 .reset_index()
                 .sort_values(["instance", "budget_j"]))
    write_csv(summary, out, f"final_summary_{tag}.csv")

    write_manifest(out, {
        "stage": "final",
        "subset": args.subset,
        "runs": args.runs,
        "budgets": budgets,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "params": params,
        "region": args.region,
        "machine": machine_info(),
        "energy": "measured via CodeCarbon",
    })

    print("\n  Mean quality by budget:")
    print(summary.groupby("budget_j")["mean_qualite_pct"].mean()
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())