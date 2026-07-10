"""
benchmark.py
============
Benchmark harness that produces exactly the CSV artifacts read by the
analysis notebook (`TP5_presentation_finale.ipynb`):

  - operator_profiles.csv   (section 1.3 / 2.3 — per-operator profiling,
                              n_runs=10, 4 instances x 2 regimes)
  - ablation_results.csv    (section 2.4 — ABLATION_CONFIGS x
                              ABLATION_OPT_FLAGS x BUDGETS_JOULES)
  - best_params_results.csv (section 4 — Green ACO v4 with best/default
                              hyperparameters, across BUDGETS_JOULES)
  - optuna_results.csv      (section 2.2 — one row per Optuna trial;
                              produced by `run_optuna_search`, optional,
                              requires the `optuna` package)

Baseline files (`green_metrics_aco_methods.csv`, `green_metrics_ga_methods.csv`,
section 3) are produced by a separate TP4 codebase and are only *read*
by this notebook; they are out of scope for this repository.
"""

import itertools
import os
from typing import Dict, List, Optional

import pandas as pd

from .config import (ABLATION_CONFIGS, ABLATION_OPT_FLAGS, BUDGETS_JOULES,
                      DEFAULT_BEST_PARAMS, OPERATOR_POOL, REGIMES, GreenACOConfig)
from .operators import build_operator_pool
from .parser import CNFInstance, build_var_clause_index
from .pheromone import PheromoneMatrix
from .solver import GreenACOSolver
from .utils import count_satisfied_clauses, make_rng, random_assignment

INSTANCES_ALIASES = {
    "decision-tree-soybean-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree": "soybean",
    "min-fill-MinFill_R0_myciel5": "min-fill",
    "decision-tree-car-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree": "car",
    "decision-tree-vote-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree": "vote",
}


# ── 1. Operator profiling (single-operator runs, per regime) ─────────────
def profile_operator(instance: CNFInstance, operator_name: str, regime: str,
                      seed: int, n_steps: int = 200) -> Dict:
    """Run a single operator in isolation for `n_steps` calls and report
    mean cost / mean delta_f, matching operator_profiles.csv columns."""
    from .energy import EnergyTracker, OperatorCostModel

    rng = make_rng(seed)
    var_clause_index = build_var_clause_index(instance)
    pheromone = PheromoneMatrix(instance.n_vars)
    operator = build_operator_pool([operator_name])[operator_name]
    cost_model = OperatorCostModel()
    energy = EnergyTracker()

    assignment = random_assignment(instance.n_vars, rng)
    costs, dfs = [], []

    for _ in range(n_steps):
        result = operator.apply(instance, assignment, pheromone, var_clause_index, rng)
        cost = cost_model.estimate_cost(operator_name, result.work_units)
        costs.append(cost)
        dfs.append(result.delta_f)
        energy.record(operator_name, cost, result.delta_f)
        pheromone.update(assignment, result.delta_f, rho=0.1, variables=result.touched_vars)

    return {
        "instance": instance.name,
        "regime": regime,
        "operator": operator_name,
        "n_clauses": instance.n_clauses,
        "mean_cost_j": sum(costs) / len(costs),
        "mean_df": sum(dfs) / len(dfs),
        "benchmark": instance.name,
    }


def run_profiling(instances: List[CNFInstance], n_runs: int = 10,
                   operator_pool: List[str] = None,
                   out_csv: str = "operator_profiles.csv") -> pd.DataFrame:
    """Profile every operator on every instance, in both regimes
    (easy: q0=0.5, hard: q0=0.95), n_runs=10 per (instance, operator,
    regime) combination -- matches "n_runs=10, 4 instances x 2 régimes"."""
    operator_pool = operator_pool or OPERATOR_POOL
    rows = []
    for instance, regime, op_name, run_idx in itertools.product(
            instances, REGIMES.keys(), operator_pool, range(n_runs)):
        row = profile_operator(instance, op_name, regime, seed=run_idx)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ── 2. Ablation study ──────────────────────────────────────────────────
def run_ablation(instances: List[CNFInstance],
                  budgets: List[float] = None,
                  out_csv: str = "ablation_results.csv") -> pd.DataFrame:
    """Run every ablation configuration (operator-pool ablations from
    ABLATION_CONFIGS, and feature-flag ablations from
    ABLATION_OPT_FLAGS) across all budgets and instances."""
    budgets = budgets or BUDGETS_JOULES
    rows = []

    all_configs = {}
    for name, ops in ABLATION_CONFIGS.items():
        flags = ABLATION_OPT_FLAGS.get("full_v4")
        all_configs[name] = (ops, flags)
    for name, flags in ABLATION_OPT_FLAGS.items():
        if name not in all_configs:
            all_configs[name] = (OPERATOR_POOL, flags)

    for config_name, (ops, flags) in all_configs.items():
        for instance in instances:
            for budget in budgets:
                cfg = GreenACOConfig.from_dict({**DEFAULT_BEST_PARAMS,
                                                 "budget_j": budget,
                                                 "active_operators": ops,
                                                 **flags})
                solver = GreenACOSolver(cfg)
                result = solver.solve(instance)
                rows.append({
                    "config": config_name,
                    "active_ops": ",".join(ops),
                    "instance": instance.name,
                    "budget_j": budget,
                    "qualite_pct": result.qualite_pct,
                    "energie_joules": result.energie_joules,
                    "score_per_joule": result.score_per_joule,
                    "co2_micrograms": result.co2_micrograms,
                    "nb_iterations": result.nb_iterations,
                    "stopped_reason": result.stopped_reason,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ── 3. Full run with best (Optuna-tuned or default) parameters ───────────
def run_best_params(instances: List[CNFInstance], best_params: Dict = None,
                     budgets: List[float] = None,
                     out_csv: str = "best_params_results.csv") -> pd.DataFrame:
    best_params = best_params or DEFAULT_BEST_PARAMS
    budgets = budgets or BUDGETS_JOULES
    rows = []

    for instance in instances:
        for budget in budgets:
            cfg = GreenACOConfig.from_dict({**best_params, "budget_j": budget})
            solver = GreenACOSolver(cfg)
            result = solver.solve(instance)
            rows.append({
                "instance": instance.name,
                "budget_j": budget,
                "qualite_pct": result.qualite_pct,
                "qualite_solution": result.best_n_satisfied,
                "n_clauses": result.n_clauses,
                "energie_joules": result.energie_joules,
                "score_per_joule": result.score_per_joule,
                "co2_micrograms": result.co2_micrograms,
                "nb_iterations": result.nb_iterations,
                "stopped_reason": result.stopped_reason,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ── 4. Optuna hyperparameter search (section 2.2) ─────────────────────────
def run_optuna_search(instances: List[CNFInstance], n_trials: int = 50,
                       out_csv: str = "optuna_results.csv"):
    """Optuna TPE search maximizing mean score_per_joule, over the
    search space defined in config.OPTUNA_SEARCH_SPACE. Requires
    `pip install optuna`."""
    import optuna
    from .config import OPTUNA_SEARCH_SPACE

    def objective(trial: "optuna.Trial") -> float:
        params = {}
        for name, (lo, hi) in OPTUNA_SEARCH_SPACE.items():
            if name in ("stagnation_window", "best_stagnation_limit"):
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        scores = []
        for instance in instances:
            cfg = GreenACOConfig.from_dict({**params, "budget_j": 1000})
            solver = GreenACOSolver(cfg)
            result = solver.solve(instance)
            scores.append(result.score_per_joule)
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    df = study.trials_dataframe()
    df.to_csv(out_csv, index=False)
    return df, study.best_params
