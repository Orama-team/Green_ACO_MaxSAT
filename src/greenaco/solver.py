"""
solver.py
=========
The Green ACO search loop.

One run proceeds as follows:

1. Build a pheromone matrix and a clause-density heuristic matrix (``eta``).
2. Seed the scheduler's priors from the instance's measured operator profiles,
   blending the *easy* and *hard* regimes in proportion to the quality of the
   initial solution.
3. Construct a greedy starting assignment guided by ``pheromone x eta^2``.
4. Loop while energy budget remains: select an operator (warm-up first, then by
   EI/J), execute it under energy measurement, update the scheduler statistics
   and the pheromone matrix, and check the stopping conditions.

Stopping conditions are an exhausted budget, reaching the target quality, or a
best-plateau: no improvement for ``best_stagnation_limit`` iterations while
within ``plateau_tolerance_pct`` of optimal. Early stopping preserves budget and
is therefore also a direct CO2 saving.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from .config import (FLIP_OPS, GreenACOConfig,
                     REGIME_QUALITY_THRESHOLD)
from .energy import EnergyMeter, compute_green_metrics
from .operators import (RescanBackend, build_operator_pool,
                        update_pheromone, update_pheromone_sparse)
from .scheduler import EIJScheduler
from .wcnf import count_satisfied_clauses


class StagnationTracker:
    """Detects stagnation from the mean quality gain over a sliding window."""

    def __init__(self, window: int = 10, epsilon: float = 0.5):
        self.window = window
        self.epsilon = epsilon
        self._history: List[float] = []

    def update(self, delta_f: float) -> None:
        self._history.append(delta_f)
        if len(self._history) > self.window:
            self._history.pop(0)

    def is_stagnant(self) -> bool:
        if len(self._history) < self.window:
            return False
        return (sum(self._history) / self.window) < self.epsilon

    def reset(self) -> None:
        self._history.clear()


def build_eta_matrix(formula, n_vars: int):
    """Clause-density heuristic: how often each literal value occurs."""
    eta = [[0.0, 0.0] for _ in range(n_vars)]
    for clause in formula:
        for lit in clause:
            i = abs(lit) - 1
            if i < n_vars:
                eta[i][1 if lit > 0 else 0] += 1.0
    return eta


def build_greedy_assignment(n_vars, pheromone, eta, exploitation_rate=0.95,
                            rng: random.Random | None = None):
    """Greedy starting assignment guided by pheromone x eta^2.

    With probability ``exploitation_rate`` the better-scoring value is taken
    directly; otherwise a value is sampled proportionally. The mixing rate is
    what distinguishes the *easy* regime (0.50, exploratory) from the *hard*
    regime (0.95, near-greedy).
    """
    rng = rng or random
    assignment = {}
    for i in range(n_vars):
        var = i + 1
        scores = [(pheromone[i][v] ** 1.0) * ((eta[i][v] + 1e-6) ** 2.0)
                  for v in (0, 1)]
        if rng.random() < exploitation_rate:
            assignment[var] = bool(scores[1] > scores[0])
        else:
            total = scores[0] + scores[1]
            assignment[var] = rng.random() < scores[1] / (total + 1e-12)
    return assignment


def blend_profiles(profiles: Dict[str, Dict], initial_quality: float) -> Dict[str, Dict]:
    """Blend the easy/hard regime profiles for one benchmark.

    The blend weight rises linearly from 0 to 1 as the initial solution quality
    goes from ``REGIME_QUALITY_THRESHOLD`` to 1.0. Instances whose greedy start
    is already near-optimal therefore draw on the *hard* priors, where the
    expensive operators look expensive; poorly-started instances use the *easy*
    priors, where they look affordable.
    """
    thresh = REGIME_QUALITY_THRESHOLD
    blend_hard = max(0.0, min(1.0, (initial_quality - thresh) / (1.0 - thresh)))
    blend_easy = 1.0 - blend_hard

    out: Dict[str, Dict] = {}
    for op_name, prof in profiles.items():
        p_easy = prof.get("easy", {})
        p_hard = prof.get("hard", {})
        out[op_name] = {
            "mean_cost_j": (blend_easy * p_easy.get("mean_cost_j", 50.0)
                            + blend_hard * p_hard.get("mean_cost_j", 50.0)),
            "std_cost_j": (blend_easy * p_easy.get("std_cost_j", 10.0)
                           + blend_hard * p_hard.get("std_cost_j", 10.0)),
            "mean_df": (blend_easy * p_easy.get("mean_df", 0.0)
                        + blend_hard * p_hard.get("mean_df", 0.0)),
            "std_df": (blend_easy * p_easy.get("std_df", 1.0)
                       + blend_hard * p_hard.get("std_df", 1.0)),
        }
    return out

class GreenACOSolver:
    """Runs Green ACO on a single MAX-SAT instance under an energy budget."""

    def __init__(self, config: GreenACOConfig,
                 energy_meter: Optional[EnergyMeter] = None,
                 profiles: Optional[Dict[str, Dict]] = None,
                 backend: Optional[str] = None,
                 verbose: bool = False):
        self.config = config
        self.meter = energy_meter or EnergyMeter()
        self.profiles = profiles or {}
        # An explicit argument wins; otherwise the config decides. Defaulting
        # the argument to a literal would shadow config.backend.
        self.backend_name = backend or config.backend
        self.verbose = verbose

    def _select_operator(self, scheduler: EIJScheduler, budget: float,
                         rng: random.Random) -> str:
        """Warm-up first (one call per operator), then EI/J under a budget guard.

        During warm-up an operator that has never been called is picked at
        random, preferring those whose estimated cost still fits the budget.
        Afterwards the EI/J scheduler decides, restricted to operators that fit.
        """
        cfg = self.config

        uncalled = scheduler.uncalled()
        if uncalled:
            affordable = [o for o in uncalled
                          if scheduler.stats[o].estimated_cost()
                          <= budget * cfg.max_overrun_factor]
            if affordable:
                return rng.choice(affordable)

        eligible = scheduler.eligible(budget, cfg.max_overrun_factor)
        if eligible:
            return scheduler.select(eligible, budget)
        # Nothing fits the remaining budget: take the cheapest operator rather
        # than stalling the search entirely.
        return scheduler.cheapest()

    def solve(self, formula, n_vars: int, benchmark: str = "",
                  seed: Optional[int] = None) -> Dict:
            cfg = self.config
            rng = random.Random(cfg.seed if seed is None else seed)

            t_start = time.perf_counter()
            n_clauses = len(formula)
            budget_total = cfg.budget_j
            target_score = int(n_clauses * cfg.target_quality)

            pheromone = [[cfg.tau0, cfg.tau0] for _ in range(n_vars)]
            eta = build_eta_matrix(formula, n_vars)

            backend = RescanBackend()
            backend.prepare(formula, n_vars)
            pool = build_operator_pool(cfg.active_operators, backend)

            scheduler = EIJScheduler(list(pool.keys()), alpha_ewma=cfg.alpha_ewma,
                                     rng=rng)
            stag_tracker = StagnationTracker(window=cfg.stagnation_window,
                                             epsilon=cfg.stagnation_epsilon)

            current_asgn = build_greedy_assignment(n_vars, pheromone, eta,
                                                  cfg.exploitation_rate, rng)
            best_asgn = dict(current_asgn)
            best_score = count_satisfied_clauses(formula, best_asgn)

            total_energy = 0.0
            budget = budget_total
            n_improving = n_deteriorating = n_neutral = 0
            stagnation_cnt = 0
            stagnation_list: List[int] = []
            op_usage: Dict[str, int] = {}
            op_energy: Dict[str, float] = {}
            ph_done = ph_skipped = 0
            consecutive_no_improve = 0
            stopped_reason = "budget_exhausted"
            iteration = 0

            prior_note = "none"
            if cfg.use_profiling and self.profiles:
                initial_quality = best_score / max(n_clauses, 1)
                scheduler.seed_profiles(
                    blend_profiles(self.profiles, initial_quality))
                prior_note = "blended"

            plateau_tolerance = int(n_clauses * cfg.plateau_tolerance_pct)

            while budget > 0:
                rho_current = (cfg.rho_stagnant if stag_tracker.is_stagnant()
                               else cfg.rho_base)

                op_name = self._select_operator(scheduler, budget, rng)
                op_fn = pool[op_name]
                prev_asgn = dict(current_asgn)

                def _invoke(fn=op_fn, ca=current_asgn, ba=best_asgn):
                    return fn(formula, n_vars, pheromone, eta, ca, ba, n_clauses)

                (new_asgn, delta_f), energy_j = self.meter.measure(_invoke)

                total_energy += energy_j
                budget -= energy_j
                op_usage[op_name] = op_usage.get(op_name, 0) + 1
                op_energy[op_name] = op_energy.get(op_name, 0.0) + energy_j
                scheduler.update(op_name, delta_f, energy_j)

                new_score = count_satisfied_clauses(formula, new_asgn)
                old_score = count_satisfied_clauses(formula, current_asgn)

                stag_tracker.update(delta_f)
                is_stagnant = stag_tracker.is_stagnant()

                # Frugal skip: do not reinforce a solution that is not progressing.
                skip_update = cfg.use_frugal_skip and is_stagnant and delta_f <= 0

                if not skip_update:
                    if cfg.use_sparse_update and op_name in FLIP_OPS:
                        update_pheromone_sparse(
                            pheromone, prev_asgn, new_asgn, new_score, n_clauses,
                            evaporation_rate=rho_current, n_vars=n_vars)
                    else:
                        update_pheromone(pheromone, new_asgn, new_score,
                                         n_clauses, evaporation_rate=rho_current,
                                         n_vars=n_vars)
                    ph_done += 1
                else:
                    ph_skipped += 1

                if new_score > old_score:
                    n_improving += 1
                    stagnation_list.append(stagnation_cnt)
                    stagnation_cnt = 0
                    current_asgn = new_asgn
                elif new_score == old_score:
                    n_neutral += 1
                    stagnation_cnt += 1
                    current_asgn = new_asgn
                else:
                    n_deteriorating += 1
                    stagnation_cnt += 1

                if new_score > best_score:
                    best_score = new_score
                    best_asgn = dict(new_asgn)
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1

                iteration += 1

                if self.verbose:
                    print(f"  iter {iteration:4d} op={op_name:<24s} "
                          f"E={energy_j:8.3f}J df={delta_f:+7.1f} "
                          f"q={new_score}/{n_clauses} budget={budget:.2f}J")

                if best_score >= target_score:
                    stopped_reason = "target_reached"
                    break

                violated = n_clauses - best_score
                if (consecutive_no_improve >= cfg.best_stagnation_limit
                        and violated <= plateau_tolerance):
                    stopped_reason = "best_plateau"
                    break

            stagnation_list.append(stagnation_cnt)
            t_exec = time.perf_counter() - t_start

            stats = {
                "qualite_solution": best_score,
                "qualite_pct": 100.0 * best_score / n_clauses,
                "temps_exec": round(t_exec, 6),
                "energie_joules": round(total_energy, 6),
                "budget_restant_j": round(max(budget, 0.0), 6),
                "budget_utilise_pct": round(
                    100.0 * (budget_total - max(budget, 0.0)) / budget_total, 2),
                "nbr_mouvements_ameliorants": n_improving,
                "nbr_mouvements_deteriorants": n_deteriorating,
                "nbr_mouvements_neutres": n_neutral,
                "stagnation_moyenne": sum(stagnation_list) / len(stagnation_list),
                "nb_iterations": iteration,
                "pheromone_savings_pct": round(
                    100.0 * ph_skipped / max(1, ph_done + ph_skipped), 2),
                "early_stop": stopped_reason != "budget_exhausted",
                "stopped_reason": stopped_reason,
                "consecutive_no_improve_final": consecutive_no_improve,
                "operateurs_usage": dict(op_usage),
                "operateurs_energie_j": {k: round(v, 6) for k, v in op_energy.items()},
                "priors": prior_note,
                "seed": cfg.seed if seed is None else seed,
                "backend": self.backend_name,
            }
            stats.update(compute_green_metrics(best_score, n_clauses, total_energy,
                                               self.meter.region))
            return stats

def solve_instance(solver, bench, seed: int | None = None) -> Dict:
    """Run ``solver`` on a :class:`~greenaco.data.Benchmark` row.

    Thin convenience wrapper that adds the instance metadata to the solver's
    own statistics dictionary.
    """
    stats = solver.solve(bench.instance.clauses, bench.n_vars,
                         benchmark=bench.benchmark, seed=seed)
    stats["instance"] = bench.benchmark
    stats["n_clauses"] = bench.n_clauses
    stats["n_vars"] = bench.n_vars
    stats["best_known"] = bench.best_known
    return stats
