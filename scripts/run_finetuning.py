#!/usr/bin/env python3
"""
run_finetuning.py
=================
Derive, or reuse, the solver's hyper-parameters.

  frozen  (default) Read the tuned parameters and record them. Reproduces the
          configuration the reported results were produced with, at no cost.

  full    Re-derive them from scratch: an Optuna TPE search over the declared
          space, scored by mean quality per Joule across the tuning instances
          and budgets. Seeded end to end, including the sampler.

`full` searches, so it may land on different parameters than `frozen`; Optuna
results are also version-sensitive, hence the pin in pyproject.toml. What was
actually used is recorded in results/shipped/finetune_d_best_params.csv.

Usage:
    python scripts/run_finetuning.py --mode frozen
    python scripts/run_finetuning.py --mode full --trials 10 --subset core
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import (banner, base_parser, energy_meter,  # noqa: E402
                       results_dir, selected_budgets, write_csv)
from greenaco.config import (OPTUNA_SEARCH_SPACE, GreenACOConfig,  # noqa: E402
                             TUNED_PARAMS)
from greenaco.data import load_benchmark  # noqa: E402
from greenaco.profile import load_profiles  # noqa: E402
from greenaco.runner import machine_info, write_manifest  # noqa: E402
from greenaco.solver import GreenACOSolver, solve_instance  # noqa: E402

INTEGER_PARAMS = {"stagnation_window", "best_stagnation_limit"}


def frozen_params() -> dict:
    path = Path("configs/best_params.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return dict(TUNED_PARAMS)


def main() -> int:
    parser = base_parser("Tune or reuse the Green ACO hyper-parameters.")
    parser.add_argument("--mode", choices=["frozen", "full"], default="frozen")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    banner(f"Hyper-parameters ({args.mode})")
    out = results_dir(args.out_dir)
    tag = "all50" if args.subset == "all" else args.subset
    bench = load_benchmark(args.subset)
    budgets = selected_budgets(args.budgets)

    if args.mode == "frozen":
        params = frozen_params()
        write_csv(pd.DataFrame([{"source": "frozen", **params}]), out,
                  f"finetune_d_best_params_{tag}.csv")
        Path("configs").mkdir(exist_ok=True)
        with open("configs/best_params.json", "w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2)
        print(f"  parameters: {json.dumps(params)}")
        print("  -> wrote configs/best_params.json")
        return 0

    try:
        import optuna
    except ImportError:
        print("  [FAIL] optuna required for --mode full: pip install optuna")
        return 1

    print(f"  Optuna TPE: {args.trials} trials, seed {args.seed}")
    print(f"  objective : mean quality per Joule over {len(bench)} instances "
          f"x {len(budgets)} budgets")

    profile_file = out / f"operator_profiles_{tag}.csv"
    profiles = {}
    if profile_file.exists():
        rows = pd.read_csv(profile_file).to_dict("records")
        profiles = load_profiles(rows, [b.benchmark for b in bench])

    meter = energy_meter(args.region)

    def _objective(trial):
        params = {}
        for name, (lo, hi) in OPTUNA_SEARCH_SPACE.items():
            if name in INTEGER_PARAMS:
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        scores = []
        for b in bench:
            for budget in budgets:
                cfg = GreenACOConfig.from_dict({**params, "budget_j": budget,
                                                "seed": args.seed})
                solver = GreenACOSolver(
                    cfg, meter, profiles=profiles.get(b.benchmark, {}))
                scores.append(
                    solve_instance(solver, b)["score_per_joule"])
        return sum(scores) / len(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(_objective, n_trials=args.trials)

    trials = study.trials_dataframe().drop(
        columns=[c for c in study.trials_dataframe().columns
                 if c.startswith("datetime")], errors="ignore")
    write_csv(trials, out, f"finetune_a_optuna_{tag}.csv")

    best = dict(TUNED_PARAMS)
    best.update(study.best_params)
    write_csv(pd.DataFrame([{"source": "optuna", **best}]), out,
              f"finetune_d_best_params_{tag}.csv")
    print(f"\n  best objective : {study.best_value:.6f}")
    print(f"  best parameters: {json.dumps(study.best_params)}")

    if args.write_config:
        with open("configs/best_params.json", "w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
        print("  -> wrote configs/best_params.json")

    write_manifest(out, {
        "stage": "finetune", "mode": "full", "subset": args.subset,
        "trials": args.trials, "seed": args.seed,
        "optuna": getattr(optuna, "__version__", "unknown"),
        "best_value": study.best_value, "machine": machine_info()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())