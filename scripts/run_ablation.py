#!/usr/bin/env python3
"""
run_ablation.py
===============
Remove one component at a time and measure what it costs.

Two independent axes, because they isolate different things:

  operators   which operator matters?  (pool composition)
  mechanisms  which green mechanism matters?  (sparse update, profiling,
              overrun guard, frugal skip)

They are written to separate files because they are not comparable row-wise:
the first varies the pool with every mechanism on, the second fixes the pool
and varies one mechanism.

Usage:
    python scripts/run_ablation.py --subset core
    python scripts/run_ablation.py --subset core --axis operators
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import (SIZE_THRESHOLD_CLAUSES, banner, base_parser,  # noqa: E402
                       energy_meter, results_dir, selected_budgets, write_csv)
from greenaco.config import (ABLATION_GREEN_FLAGS,  # noqa: E402
                             ABLATION_OPERATOR_CONFIGS, GreenACOConfig,
                             TUNED_PARAMS)
from greenaco.data import load_benchmark  # noqa: E402
from greenaco.profile import load_profiles  # noqa: E402
from greenaco.runner import CheckpointStore, run_tasks  # noqa: E402
from greenaco.solver import GreenACOSolver, solve_instance  # noqa: E402

METRICS = ["qualite_pct", "energie_joules", "score_per_joule",
           "co2_micrograms"]


def load_params(args) -> dict:
    path = Path(args.params) if args.params else Path("configs/best_params.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return dict(TUNED_PARAMS)


def build_configs(axis: str):
    configs = []
    if axis in ("operators", "both"):
        for name, ops in ABLATION_OPERATOR_CONFIGS.items():
            configs.append(("operators", name, {"active_operators": list(ops)}))
    if axis in ("mechanisms", "both"):
        for name, flags in ABLATION_GREEN_FLAGS.items():
            configs.append(("mechanisms", name, dict(flags)))
    return configs


def main() -> int:
    parser = base_parser("Ablation study.")
    parser.add_argument("--axis", choices=["operators", "mechanisms", "both"],
                        default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--params", default=None)
    args = parser.parse_args()

    banner("Ablation study")
    out = results_dir(args.out_dir)
    tag = "all54" if args.subset == "all" else args.subset
    budgets = selected_budgets(args.budgets)

    bench = load_benchmark(args.subset)
    params = load_params(args)
    configs = build_configs(args.axis)
    print(f"  subset {args.subset}: {len(bench)} instances, "
          f"{len(configs)} configurations, budgets {budgets}")

    profile_file = out / f"operator_profiles_{tag}.csv"
    profiles = {}
    if profile_file.exists():
        rows = pd.read_csv(profile_file).to_dict("records")
        profiles = load_profiles(rows, [b.benchmark for b in bench])
        print(f"  profiles     : {profile_file.name}")
    else:
        print(f"  [warn] {profile_file.name} absent: the 'mechanisms' axis "
              f"cannot separate the effect of profiling priors.")

    _meters = {}

    def _meter_for(region):
        if region not in _meters:
            _meters[region] = energy_meter(region)
        return _meters[region]

    def _task(item):
        axis, cfg_name, flags, b, budget, run_id = item
        cfg = GreenACOConfig.from_dict({
            **params, **flags, "budget_j": budget, "seed": args.seed + run_id,
        })
        prof = profiles.get(b.benchmark, {}) if cfg.use_profiling else {}
        solver = GreenACOSolver(cfg, _meter_for(args.region), profiles=prof)
        stats = solve_instance(solver, b)
        stats.update({"axis": axis, "config": cfg_name, "budget_j": budget,
                      "run": run_id})
        return stats

    tasks = [(axis, name, flags, b, budget, run_id)
             for axis, name, flags in configs
             for b in bench for budget in budgets
             for run_id in range(args.runs)]

    store = CheckpointStore(f"ablation_{args.subset}")
    rows = run_tasks(tasks, _task, store, n_jobs=args.n_jobs,
                     size_threshold=SIZE_THRESHOLD_CLAUSES,
                     task_key=lambda t: (t[0], t[1], t[3].benchmark,
                                         int(t[4]), t[5]))
    rows = [r for r in rows if r and "__error__" not in r]
    if not rows:
        print("  no ablation results produced")
        return 1

    df = pd.DataFrame(rows)
    df = df.drop(columns=["operateurs_usage", "operateurs_energie_j"],
                 errors="ignore")

    for axis, filename in (("operators", f"ablation_operators_{tag}.csv"),
                           ("mechanisms", f"ablation_mechanisms_{tag}.csv")):
        sub = df[df["axis"] == axis]
        if sub.empty:
            continue
        agg = (sub.groupby(["config", "instance", "budget_j"])
                  .agg(n_runs=("run", "count"),
                       **{f"mean_{c}": (c, "mean") for c in METRICS})
                  .reset_index())
        write_csv(agg.sort_values(["config", "instance", "budget_j"]),
                  out, filename)

    print("\n  Mean quality per Joule by configuration:")
    print(df.groupby(["axis", "config"])["score_per_joule"]
          .mean().round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())