#!/usr/bin/env python3
"""
run_comparison.py
=================
Run the comparison methods and assemble the comparison table.

Every method is measured by this repository, on this machine, in this session,
with the same energy meter and green metrics. That is the point: mixing freshly
computed Green ACO numbers with historical rows would compare across machines
and measurement regimes.

Seven methods qualify -- three GA and four ACO variants, each of which produced
usable results on the comparison instances.

Each method carries its calibrated parameters into the output, so a table can
be traced back to the settings that produced it.

Usage:
    python scripts/run_comparison.py --subset core
    python scripts/run_comparison.py --subset core --timeout 600
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import (SIZE_THRESHOLD_CLAUSES, banner, base_parser,  # noqa: E402
                       energy_meter, results_dir, write_csv)
from greenaco.comparison.harness import run_method  # noqa: E402
from greenaco.comparison.registry import (FAMILY,  # noqa: E402
                                          METHOD_ORDER)
from greenaco.config import GreenACOConfig, TUNED_PARAMS  # noqa: E402
from greenaco.data import load_benchmark  # noqa: E402
from greenaco.metrics import rank_methods  # noqa: E402
from greenaco.profile import load_profiles  # noqa: E402
from greenaco.runner import (CheckpointStore, machine_info,  # noqa: E402
                             run_tasks, write_manifest)
from greenaco.solver import GreenACOSolver, solve_instance  # noqa: E402

COMPARISON_METHODS = [m for m in METHOD_ORDER if m != "Green ACO"]

def main() -> int:
    parser = base_parser("Run the comparison methods.")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Per (method, instance) wall-clock limit, seconds.")
    args = parser.parse_args()

    banner("Method comparison")
    out = results_dir(args.out_dir)
    tag = "all50" if args.subset == "all" else args.subset

    bench = load_benchmark(args.subset)
    print(f"  subset {args.subset}: {len(bench)} instances x "
          f"{len(COMPARISON_METHODS)} methods")

    profile_file = out / f"operator_profiles_{tag}.csv"
    profiles = {}
    if profile_file.exists():
        rows = pd.read_csv(profile_file).to_dict("records")
        profiles = load_profiles(rows, [b.benchmark for b in bench])

    _meters = {}

    def _meter_for(region):
        if region not in _meters:
            _meters[region] = energy_meter(region)
        return _meters[region]

    def _task(item):
        method, b = item
        row = run_method(b, method, meter=_meter_for(args.region),
                         seed=args.seed, timeout_s=args.timeout)
        row["family"] = FAMILY[method]
        row["region"] = args.region
        return row

    tasks = [(m, b) for m in COMPARISON_METHODS for b in bench]
    store = CheckpointStore(f"comparison_{args.subset}")
    rows = run_tasks(tasks, _task, store, n_jobs=args.n_jobs,
                     size_threshold=SIZE_THRESHOLD_CLAUSES,
                     task_key=lambda t: (t[0], t[1].benchmark))
    rows = [r for r in rows if r and "__error__" not in r]
    if not rows:
        print("  no comparison results produced")
        return 1

    df = pd.DataFrame(rows)
    df["family"] = df["method"].map(FAMILY)

    # Green ACO on the same instances, so the table has a like-for-like row
    # measured here rather than imported from another machine.
    budget = 1000.0
    print(f"  adding Green ACO at {budget:.0f} J for a like-for-like row ...")

    def _green_task(b):
        cfg = GreenACOConfig.from_dict({**TUNED_PARAMS, "budget_j": budget,
                                        "seed": args.seed})
        solver = GreenACOSolver(cfg, _meter_for(args.region),
                                profiles=profiles.get(b.benchmark, {}))
        stats = solve_instance(solver, b)
        return {"method": "Green ACO", "family": FAMILY["Green ACO"],
                "instance": b.benchmark, "n_vars": b.n_vars,
                "n_clauses": b.n_clauses,
                "qualite_solution": stats["qualite_solution"],
                "qualite_pct": stats["qualite_pct"],
                "energie_joules": stats["energie_joules"],
                "score_per_joule": stats["score_per_joule"],
                "co2_micrograms": stats["co2_micrograms"],
                "temps_exec": stats["temps_exec"], "timeout": False,
                "budget_j": budget, "region": args.region}

    gstore = CheckpointStore(f"comparison_green_{args.subset}")
    grows = run_tasks(list(bench), _green_task, gstore, n_jobs=args.n_jobs,
                      size_threshold=SIZE_THRESHOLD_CLAUSES,
                      task_key=lambda t: (t.benchmark,))
    grows = [g for g in grows if g and "__error__" not in g]

    full = pd.concat([df, pd.DataFrame(grows)], ignore_index=True)
    write_csv(full.sort_values(["method", "instance"]), out,
              f"comparison_runs_{tag}.csv")

    valid = full[full["qualite_pct"].notna() & (~full["timeout"])]
    if not valid.empty:
        summary = rank_methods(valid, group_col="method")
        write_csv(summary, out, f"comparison_ranking_{tag}.csv")
        print("\n  Ranking by quality per Joule:")
        print(summary[["rang_global", "method", "qualite_moy_pct",
                       "score_per_joule", "co2_moy_ug"]].to_string(index=False))

    write_manifest(out, {
        "stage": "comparison", "subset": args.subset,
        "methods": COMPARISON_METHODS,
        "timeout_s": args.timeout, "green_aco_budget_j": budget,
        "seed": args.seed, "n_jobs": args.n_jobs, "machine": machine_info(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
