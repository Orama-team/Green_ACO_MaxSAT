"""Reproducibility probe: does CPU contention change a single run?

Runs the identical seeded configuration twice -- once alone, once while N
background processes saturate the CPU -- and reports quality, energy and
iteration count.

Rationale: energy is *measured* from CPU power draw (RAPL), and the solver's
budget loop is `while budget > 0`. Contention therefore does not merely make a
run slower: a contended core spends more joules per unit of work, so fewer
iterations fit inside the same budget and the result can change.

Measured on ramsey-ram_k3_n12.ra0, budget 400 J, seed 42:

    load      violated   iterations   J/iteration
    solo          12          30         13.8
    1 proc        12          28         14.3
    3 procs       13          24         17.1
    7 procs       13          10         41.9

Low-load runs match the solo baseline exactly; divergence appears from a few
concurrent workers upward. Hence `--n-jobs` is recorded in the run manifest
rather than left to an automatic default.

Usage:
    python tests/probe_cpu_contention.py solo
    python tests/probe_cpu_contention.py loaded 7
"""
import sys, time, subprocess, os
sys.path.insert(0, "src"); sys.path.insert(0, ".")

from src.greenaco.data import load_benchmark
from greenaco.config import GreenACOConfig
from greenaco.solver import GreenACOSolver, solve_instance
from greenaco.energy import EnergyMeter

BENCH = "ramsey-ram_k3_n12.ra0"
BUDGET = 400.0
SEED = 42

def one_run():
    b = [x for x in load_benchmark(BENCH)][0]
    cfg = GreenACOConfig(budget_j=BUDGET, seed=SEED, backend="rescan")
    st = solve_instance(GreenACOSolver(cfg, EnergyMeter(), backend="rescan"), b)
    return st

if __name__ == "__main__":
    mode = sys.argv[1]
    loaders = []
    if mode == "loaded":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        for _ in range(n):
            loaders.append(subprocess.Popen(
                [sys.executable, "-c",
                 "\nwhile True: pass"]))
        time.sleep(1.5)
    t0 = time.time()
    st = one_run()
    wall = time.time() - t0
    for p in loaders:
        p.kill()
    print(f"{mode:8s} quality={st['qualite_pct']:.6f} "
          f"violated={st['n_clauses']-st['qualite_solution']} "
          f"energy={st['energie_joules']:.2f}J "
          f"iters={st['nb_iterations']} "
          f"wall={wall:.2f}s "
          f"utils={st['budget_utilise_pct']:.1f}%")