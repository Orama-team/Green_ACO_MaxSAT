"""
run_ablation_study.py
======================
Reproduces notebook section 2.4 ("Étude d'ablation"): runs every
ablation configuration (full_v4, no_walksat, no_focused_vns,
no_greedy_restart, no_sparse_ph, no_caching, no_profiling,
no_overrun_fix) across the 3 budgets [400, 1000, 2000] J, and writes
`ablation_results.csv` for the notebook's visualization cell (Cellule B).

Usage:
    python examples/run_ablation_study.py path/to/cnf_dir/
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from method.benchmark import run_ablation
from method.parser import parse_dimacs_cnf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf_dir")
    ap.add_argument("--out", default="ablation_results.csv")
    args = ap.parse_args()

    cnf_files = sorted(glob.glob(os.path.join(args.cnf_dir, "*.cnf")))
    if not cnf_files:
        raise SystemExit(f"No .cnf files found in {args.cnf_dir}")

    instances = [parse_dimacs_cnf(p) for p in cnf_files]
    print(f"Running ablation study on {len(instances)} instance(s) "
          f"— this may take a while (as noted in the notebook: "
          f"'Cette cellule est longue')...")

    df = run_ablation(instances, out_csv=args.out)
    print(f"✔ wrote {args.out} ({len(df)} rows, configs: {sorted(df['config'].unique())})")


if __name__ == "__main__":
    main()
