"""
solver.py
=========
Main Green ACO solver loop for MAX-SAT.

Ties together:
  - PheromoneMatrix (pheromone.py)
  - EIJScheduler (selection.py) — picks the operator maximizing Phi(o)
  - Operator pool (operators.py) — WalkSAT / Focused VNS / Clause Restart Greedy
  - EnergyTracker + OperatorCostModel (energy.py) — Joule accounting
  - Adaptive evaporation (rho_base in normal regime, rho_stagnant during
    stagnation) and early stopping via stagnation_window /
    best_stagnation_limit / plateau_tolerance_pct, as described in
    section 1.2.2 and the ablation study of the notebook.
  - The anti-overrun guard (`max_overrun_factor`): a single operator
    call may not exceed `max_overrun_factor * remaining_budget`
    without being vetoed (falls back to the cheapest operator instead),
    preventing the catastrophic overruns reported for
    `clause_restart_greedy` in the "no_overrun_fix" ablation
    (up to +1210% budget overrun on myciel5/400J).
"""

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import GreenACOConfig
from .energy import EnergyTracker, OperatorCostModel
from .operators import build_operator_pool
from .parser import CNFInstance, build_var_clause_index
from .pheromone import PheromoneMatrix
from .selection import EIJScheduler
from .utils import count_satisfied_clauses, make_rng, quality_pct, random_assignment


@dataclass
class SolveResult:
    instance_name: str
    best_assignment: Dict[int, bool]
    best_n_satisfied: int
    n_clauses: int
    qualite_pct: float
    nb_iterations: int
    stopped_reason: str
    energie_joules: float
    co2_micrograms: float
    score_per_joule: float
    budget_j: float
    scheduler_snapshot: Dict = field(default_factory=dict)
    wall_time_s: float = 0.0


class GreenACOSolver:
    def __init__(self, config: GreenACOConfig, cost_model: Optional[OperatorCostModel] = None,
                 profile_means: Optional[Dict[str, Dict[str, float]]] = None):
        self.config = config
        self.cost_model = cost_model or OperatorCostModel()
        self.profile_means = profile_means

    def solve(self, instance: CNFInstance,
               pheromone_cache_path: Optional[str] = None) -> SolveResult:
        cfg = self.config
        rng = make_rng(cfg.seed)
        t0 = time.time()

        var_clause_index = build_var_clause_index(instance)

        # ── Pheromone init / warm-start (use_pheromone_cache) ──────────
        if cfg.use_pheromone_cache and pheromone_cache_path:
            try:
                pheromone = PheromoneMatrix.load(pheromone_cache_path)
            except FileNotFoundError:
                pheromone = PheromoneMatrix(instance.n_vars)
        else:
            pheromone = PheromoneMatrix(instance.n_vars)

        operators = build_operator_pool(cfg.active_operators)
        scheduler = EIJScheduler(list(operators.keys()), alpha_ewma=cfg.alpha_ewma, rng=rng)
        if cfg.use_profiling and self.profile_means:
            scheduler.seed_profiles(self.profile_means)

        energy = EnergyTracker()
        assignment = random_assignment(instance.n_vars, rng)
        best_assignment = dict(assignment)
        best_n_sat = count_satisfied_clauses(instance, assignment)

        iterations_since_improve = 0
        recent_delta_fs: List[float] = []
        stopped_reason = "budget_exhausted"
        it = 0

        while energy.total_joules < cfg.budget_j:
            it += 1
            remaining = energy.remaining_budget(cfg.budget_j)
            if remaining <= 0:
                stopped_reason = "budget_exhausted"
                break

            op_name = scheduler.select(remaining)
            operator = operators[op_name]

            # Cheap "would-be" cost estimate for the anti-overrun guard,
            # based on the operator's EWMA-estimated mean cost.
            import math
            est_cost = math.exp(scheduler.stats[op_name].mu_lnE)
            if est_cost > cfg.max_overrun_factor * max(remaining, 1e-6):
                # Veto: fall back to the cheapest operator in the pool
                op_name = min(operators.keys(),
                               key=lambda n: math.exp(scheduler.stats[n].mu_lnE))
                operator = operators[op_name]

            trial_assignment = copy.copy(assignment)
            result = operator.apply(instance, trial_assignment, pheromone,
                                     var_clause_index, rng)
            cost_j = self.cost_model.estimate_cost(op_name, result.work_units)

            energy.record(op_name, cost_j, result.delta_f)
            scheduler.update(op_name, result.delta_f, cost_j)

            # Adaptive evaporation rate: rho_stagnant while stagnating
            is_stagnant = iterations_since_improve >= cfg.stagnation_window
            rho = cfg.rho_stagnant if is_stagnant else cfg.rho_base

            variables = result.touched_vars if cfg.use_sparse_update else None
            pheromone.update(trial_assignment, result.delta_f, rho,
                              variables=variables,
                              frugal_epsilon=cfg.stagnation_epsilon * 0.1 if is_stagnant else 0.0)

            assignment = trial_assignment
            n_sat = count_satisfied_clauses(instance, assignment)

            if n_sat > best_n_sat:
                best_n_sat = n_sat
                best_assignment = dict(assignment)
                iterations_since_improve = 0
            else:
                iterations_since_improve += 1

            recent_delta_fs.append(result.delta_f)
            if len(recent_delta_fs) > cfg.stagnation_window:
                recent_delta_fs.pop(0)

            # ── Early stopping conditions ───────────────────────────
            if iterations_since_improve >= cfg.best_stagnation_limit:
                stopped_reason = "best_stagnation_limit"
                break
            if (len(recent_delta_fs) >= cfg.stagnation_window and
                    sum(abs(d) for d in recent_delta_fs) / len(recent_delta_fs)
                    < cfg.stagnation_epsilon):
                mean_quality = quality_pct(best_n_sat, instance.n_clauses)
                if mean_quality > 0 and \
                        (instance.n_clauses - best_n_sat) / instance.n_clauses < cfg.plateau_tolerance_pct:
                    stopped_reason = "plateau_tolerance"
                    break
            if best_n_sat == instance.n_clauses:
                stopped_reason = "optimal_found"
                break

        if cfg.use_pheromone_cache and pheromone_cache_path:
            pheromone.save(pheromone_cache_path)

        qpct = quality_pct(best_n_sat, instance.n_clauses)
        summary = energy.summary(qpct, cfg.budget_j)

        return SolveResult(
            instance_name=instance.name,
            best_assignment=best_assignment,
            best_n_satisfied=best_n_sat,
            n_clauses=instance.n_clauses,
            qualite_pct=qpct,
            nb_iterations=it,
            stopped_reason=stopped_reason,
            energie_joules=summary["energie_joules"],
            co2_micrograms=summary["co2_micrograms"],
            score_per_joule=summary["score_per_joule"],
            budget_j=cfg.budget_j,
            scheduler_snapshot=scheduler.snapshot(),
            wall_time_s=time.time() - t0,
        )
