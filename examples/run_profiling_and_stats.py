"""
run_profiling_and_stats.py
===========================
Reproduces notebook sections 1.3 ("Tableau des opérateurs") and 1.4
("Tests Statistiques"): profiles every operator on every instance in
both regimes, writes `operator_profiles.csv`, then runs the
Kruskal-Wallis / Mann-Whitney battery on the resulting Δf.

Usage:
    python examples/run_profiling_and_stats.py path/to/cnf_dir/
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from method.benchmark import run_profiling
from method.config import OPERATOR_POOL
from method.metrics import run_statistical_tests
from method.parser import parse_dimacs_cnf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf_dir")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--out", default="operator_profiles.csv")
    args = ap.parse_args()

    cnf_files = sorted(glob.glob(os.path.join(args.cnf_dir, "*.cnf")))
    if not cnf_files:
        raise SystemExit(f"No .cnf files found in {args.cnf_dir}")

    instances = [parse_dimacs_cnf(p) for p in cnf_files]
    print(f"Profiling {len(instances)} instance(s) x {len(OPERATOR_POOL)} operators "
          f"x 2 regimes x {args.n_runs} runs ...")

    df_profiles = run_profiling(instances, n_runs=args.n_runs, out_csv=args.out)
    print(f"✔ wrote {args.out} ({len(df_profiles)} rows)")

    tests = run_statistical_tests(df_profiles, operator_pool=OPERATOR_POOL)
    kw = tests["kruskal_wallis"]
    print(f"\nKruskal-Wallis: H={kw['H']:.4f}, p={kw['p_value']:.4f} "
          f"({'significant' if kw['significant'] else 'not significant'})")
    for r in tests["mann_whitney_pairwise"]:
        sig = "✓ sig." if r["significant"] else "  n.s."
        print(f"  {r['op1']:<24s} vs {r['op2']:<24s}: U={r['U']:.1f} p={r['p_value']:.4f} {sig}")


if __name__ == "__main__":
    main()
