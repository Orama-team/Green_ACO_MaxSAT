"""
run_best_params.py
===================
Runs Green ACO v4 with the best (Optuna-tuned, or empirical TP5
default) hyperparameters across the 3 budgets [400, 1000, 2000] J on
every instance in `cnf_dir`, writing `best_params_results.csv` — the
file consumed by notebook sections 4.1/4.2/5.1
("Comparaison Green ACO v4 vs Baseline", convergence curves, final
ranking table).

Usage:
    python examples/run_best_params.py path/to/cnf_dir/
    python examples/run_best_params.py path/to/cnf_dir/ --optuna-csv optuna_results.csv
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from method.benchmark import run_best_params
from method.config import DEFAULT_BEST_PARAMS
from method.parser import parse_dimacs_cnf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf_dir")
    ap.add_argument("--optuna-csv", default=None,
                     help="If provided, use the best trial's params instead of the "
                          "empirical TP5 defaults")
    ap.add_argument("--out", default="best_params_results.csv")
    args = ap.parse_args()

    cnf_files = sorted(glob.glob(os.path.join(args.cnf_dir, "*.cnf")))
    if not cnf_files:
        raise SystemExit(f"No .cnf files found in {args.cnf_dir}")
    instances = [parse_dimacs_cnf(p) for p in cnf_files]

    best_params = dict(DEFAULT_BEST_PARAMS)
    if args.optuna_csv and os.path.exists(args.optuna_csv):
        df_optuna = pd.read_csv(args.optuna_csv)
        best_trial = df_optuna.loc[df_optuna["value"].idxmax()]
        param_cols = [c for c in df_optuna.columns if c.startswith("params_")]
        best_params = {c.replace("params_", ""): best_trial[c] for c in param_cols}
        print(f"✔ using Optuna best trial (value={best_trial['value']:.4f})")
    else:
        print("⚠ no optuna_results.csv provided — using empirical TP5 defaults")

    df = run_best_params(instances, best_params=best_params, out_csv=args.out)
    print(f"✔ wrote {args.out} ({len(df)} rows)")
    print(df.groupby("instance")[["qualite_pct", "score_per_joule"]].mean())


if __name__ == "__main__":
    main()
