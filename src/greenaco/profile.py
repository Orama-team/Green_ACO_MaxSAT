"""
profile.py
==========
Offline profiling of the operator pool, per instance and per regime.

For each instance the three operators are exercised in isolation so their mean
energy cost and mean quality gain can be measured. Profiling happens twice per
instance, under two assignment regimes that differ only in the exploitation
rate of the greedy constructor:

``easy``  (0.50) -- exploratory, many unsatisfied clauses; models the early
                   phase of a run, or the behaviour on a hard instance.
``hard``  (0.95) -- near-greedy, few unsatisfied clauses; models late-phase
                   operation close to the optimum.

The solver blends these two profiles at run time according to the quality of
its initial solution, so a run that starts near-optimal inherits the *hard*
priors (where the expensive operators genuinely are expensive) while a run that
starts poorly inherits the *easy* ones.

Profiling is an input to every other stage, and it should run before them:
without a profile the scheduler has no cost priors, so the first operator calls
are chosen blind and can overrun the budget badly.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Dict, List, Optional, Sequence

from .config import GreenACOConfig, REGIME_EXPLOITATION
from .energy import EnergyMeter
from .operators import build_operator_pool
from .solver import build_eta_matrix, build_greedy_assignment

PROFILE_COLUMNS = [
    "benchmark",
    "n_vars",
    "n_clauses",
    "operator",
    "regime",
    "mean_cost_j",
    "std_cost_j",
    "mean_df",
    "std_df",
    "n_runs",
    "best_known",
]


def profile_operator(bench, operator_name: str, regime: str,
                     meter: Optional[EnergyMeter] = None,
                     n_steps: int = 200,
                     seed: int = 42) -> Dict:
    """Exercise one operator in isolation and summarise its cost and gain.

    Each step calls the operator, measures its energy, and records the quality
    gain it produced. The operator updates the pheromone matrix itself, exactly
    as it would inside a real run.
    """
    meter = meter or EnergyMeter()
    cfg = GreenACOConfig(exploitation_rate=REGIME_EXPLOITATION[regime], seed=seed)

    formula = bench.instance.clauses
    n_vars = bench.n_vars
    n_clauses = bench.n_clauses

    rng = random.Random(seed)
    pheromone = [[cfg.tau0, cfg.tau0] for _ in range(n_vars)]
    eta = build_eta_matrix(formula, n_vars)
    op = build_operator_pool([operator_name])[operator_name]

    asgn = build_greedy_assignment(n_vars, pheromone, eta,
                                   cfg.exploitation_rate, rng)
    best_asgn = dict(asgn)

    costs: List[float] = []
    deltas: List[float] = []
    t0 = time.time()

    for _ in range(n_steps):
        def _invoke(a=asgn):
            return op(formula, n_vars, pheromone, eta, a, best_asgn, n_clauses)

        (new_asgn, delta_f), energy_j = meter.measure(_invoke)
        costs.append(energy_j)
        deltas.append(delta_f)
        asgn = new_asgn

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {
        "benchmark": bench.benchmark,
        "n_vars": n_vars,
        "n_clauses": n_clauses,
        "operator": operator_name,
        "regime": regime,
        "mean_cost_j": round(_mean(costs), 6),
        "std_cost_j": round(_std(costs), 6),
        "mean_df": round(_mean(deltas), 6),
        "std_df": round(_std(deltas), 6),
        "n_runs": n_steps,
        "best_known": bench.best_known,
        "wall_time_s": round(time.time() - t0, 3),
    }


def structural_features(n_vars: int, n_clauses: int) -> Dict[str, float]:
    """The three structural features used for nearest-neighbour matching.

    Matching happens in log space because the raw quantities span five orders of
    magnitude across the benchmark; without the transform the largest instances
    would dominate every distance.
    """
    ratio = n_clauses / n_vars if n_vars else 0.0
    return {"n_vars": float(n_vars), "n_clauses": float(n_clauses),
            "clauses_per_var": ratio}


def log_space_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Squared Euclidean distance between two feature vectors in log space.

    A zero feature would send ``log`` to -inf, so each term falls back to the
    raw value when the feature is zero on either side.
    """
    total = 0.0
    for key in ("n_vars", "n_clauses", "clauses_per_var"):
        x, y = float(a[key]), float(b[key])
        if x > 0 and y > 0:
            total += (math.log(x) - math.log(y)) ** 2
        else:
            total += (x - y) ** 2
    return total


def nearest_reference(target: Dict[str, float],
                      references: Sequence[Dict[str, float]]) -> str:
    """Name of the closest reference instance to ``target``."""
    return min(references,
               key=lambda r: log_space_distance(target, r["features"]))["name"]


def _vars_of(bench: dict) -> int:
    """Variable count from a manifest row (``vars``) or a Benchmark (``n_vars``)."""
    return int(bench.get("n_vars", bench.get("vars", 0)))


def _clauses_of(bench: dict) -> int:
    return int(bench.get("n_clauses", bench.get("clauses", 0)))


def transfer_profiles(rows: List[Dict],
                      benchmarks: Sequence[dict]) -> List[Dict]:
    """Extend measured profiles to unprofiled instances by nearest neighbour.

    Profiling every instance directly is impractical: the measured cost is
    dominated by ``clause_restart_greedy``, which costs minutes per call on the
    largest instances. A small set of instances is therefore measured
    directly, and every other instance inherits the full profile block of its
    closest measured neighbour.

    Transfer is applied per *instance*, not per row, so an instance inherits a
    consistent, complete profile across all operators and both regimes.
    Returned rows carry ``profile_source``, ``proxy_of`` and
    ``match_log_dist`` so the transfer is auditable.
    """
    sizes = {b["benchmark"]: (_vars_of(b), _clauses_of(b)) for b in benchmarks}
    measured_names = {r["benchmark"] for r in rows if r.get("n_runs", 0) > 0}

    references = [{"name": name,
                   "features": structural_features(*sizes[name])}
                  for name in sorted(measured_names) if name in sizes]

    by_benchmark: Dict[str, List[Dict]] = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], []).append(row)

    out: List[Dict] = []
    for name, (n_vars, n_clauses) in sizes.items():
        if name in by_benchmark:
            for row in by_benchmark[name]:
                enriched = dict(row)
                enriched.setdefault("profile_source", "real")
                out.append(enriched)
            continue

        features = structural_features(n_vars, n_clauses)
        donor = nearest_reference(features, references)
        distance = log_space_distance(
            features, next(r["features"] for r in references
                           if r["name"] == donor))
        for row in by_benchmark.get(donor, []):
            copied = dict(row)
            copied["benchmark"] = name
            copied["n_vars"] = n_vars
            copied["n_clauses"] = n_clauses
            copied["n_runs"] = 0
            copied["profile_source"] = "proxy"
            copied["proxy_of"] = donor
            copied["match_log_dist"] = round(distance, 6)
            out.append(copied)
    return out


def profiles_for(benchmark: str, rows: List[Dict]) -> Dict[str, Dict]:
    """Reshape profiling rows into the ``{operator: {regime: stats}}`` form the
    solver consumes, keeping only the requested benchmark."""
    out: Dict[str, Dict] = {}
    for row in rows:
        if row.get("benchmark") != benchmark:
            continue
        op = row.get("operator")
        regime = row.get("regime")
        if not op or not regime:
            continue
        out.setdefault(op, {})[regime] = {
            "mean_cost_j": float(row.get("mean_cost_j", 50.0)),
            "std_cost_j": float(row.get("std_cost_j", 10.0)),
            "mean_df": float(row.get("mean_df", 0.0)),
            "std_df": float(row.get("std_df", 1.0)),
            "n_runs": row.get("n_runs", 0),
        }
    return out


def load_profiles(rows: List[Dict], benchmarks: Optional[List[str]] = None):
    """Build a ``{benchmark: {operator: {regime: stats}}}`` lookup.

    A benchmark with no rows yields an empty entry, and the solver then runs
    without priors rather than failing.
    """
    lookup: Dict[str, Dict] = {}
    for name in (benchmarks or []):
        lookup[name] = profiles_for(name, rows)
    return lookup