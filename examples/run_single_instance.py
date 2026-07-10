"""
run_single_instance.py
=======================
Minimal example: solve one MAX-SAT instance with Green ACO using the
default (empirical TP5) hyperparameters, and print a result summary
equivalent to one row of `best_params_results.csv` (notebook section 4).

Usage:
    python examples/run_single_instance.py path/to/instance.cnf --budget 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from method.config import DEFAULT_BEST_PARAMS, GreenACOConfig
from method.parser import parse_dimacs_cnf
from method.solver import GreenACOSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf_path")
    ap.add_argument("--budget", type=float, default=1000.0)
    ap.add_argument("--regime", choices=["easy", "hard"], default="easy")
    args = ap.parse_args()

    instance = parse_dimacs_cnf(args.cnf_path)
    print(f"Loaded {instance.name}: {instance.n_vars} vars, {instance.n_clauses} clauses")

    q0 = 0.5 if args.regime == "easy" else 0.95
    cfg = GreenACOConfig.from_dict({**DEFAULT_BEST_PARAMS, "budget_j": args.budget, "q0": q0})

    solver = GreenACOSolver(cfg)
    result = solver.solve(instance)

    print("\n── Result ──────────────────────────────")
    print(f"  quality        : {result.qualite_pct:.2f}%  "
          f"({result.best_n_satisfied}/{result.n_clauses} clauses)")
    print(f"  iterations     : {result.nb_iterations}  (stopped: {result.stopped_reason})")
    print(f"  energy         : {result.energie_joules:.2f} J")
    print(f"  score/joule    : {result.score_per_joule:.4f} %/J")
    print(f"  CO2 footprint  : {result.co2_micrograms:.1f} µgCO2eq")
    print(f"  wall time      : {result.wall_time_s:.2f} s")


if __name__ == "__main__":
    main()
