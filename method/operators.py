"""
operators.py
============
The three operators of the Green ACO operator pool (section 1.3 of the
notebook, "Tableau des opérateurs — coût moyen & qualité apportée"):

  - WalkSAT: targets unsatisfied clauses; greedy flip (max satisfied
    clauses) or random flip with probability p_noise.
  - Focused VNS: ranks variables by
        (nb unsatisfied clauses involved) x (pheromone signal)
    and flips the top-k most problematic ones.
  - Clause Restart Greedy: guided partial reconstruction — greedily
    reassigns the variables most involved in unsatisfied clauses.

Every operator returns an `OperatorResult` with the number of "work
units" performed (used by `energy.OperatorCostModel` to estimate the
Joule cost) and the set of variables it touched (used for the sparse
pheromone update).
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .parser import CNFInstance
from .pheromone import PheromoneMatrix
from .utils import clause_is_satisfied, count_satisfied_clauses, unsatisfied_clause_indices


@dataclass
class OperatorResult:
    assignment: Dict[int, bool]
    delta_f: float          # improvement in #satisfied clauses
    work_units: int         # proxy for energy cost estimation
    touched_vars: Set[int] = field(default_factory=set)


class BaseOperator:
    name = "base"

    def apply(self, instance: CNFInstance, assignment: Dict[int, bool],
              pheromone: PheromoneMatrix, var_clause_index: Dict[int, list],
              rng: random.Random) -> OperatorResult:
        raise NotImplementedError


class WalkSAT(BaseOperator):
    name = "walksat"

    def __init__(self, p_noise: float = 0.3, max_flips: int = 1):
        self.p_noise = p_noise
        self.max_flips = max_flips

    def apply(self, instance, assignment, pheromone, var_clause_index, rng):
        before = count_satisfied_clauses(instance, assignment)
        unsat = unsatisfied_clause_indices(instance, assignment)
        touched: Set[int] = set()
        work = 0

        for _ in range(self.max_flips):
            if not unsat:
                break
            actionable_unsat = [
                c_idx for c_idx in unsat
                if len(instance.clauses[c_idx]) > 0
            ]
            if not actionable_unsat:
                break
            c_idx = rng.choice(actionable_unsat)
            clause = instance.clauses[c_idx]
            vars_in_clause = [abs(l) for l in clause]
            work += len(vars_in_clause)

            if rng.random() < self.p_noise:
                var = rng.choice(vars_in_clause)
            else:
                # Greedy: pick the flip that maximizes #satisfied clauses
                best_var, best_gain = vars_in_clause[0], -1
                for v in vars_in_clause:
                    assignment[v] = not assignment[v]
                    gain = sum(1 for (ci, _) in var_clause_index[v]
                               if clause_is_satisfied(instance.clauses[ci], assignment))
                    assignment[v] = not assignment[v]  # revert, apply below
                    if gain > best_gain:
                        best_gain, best_var = gain, v
                var = best_var

            assignment[var] = not assignment[var]
            touched.add(var)
            unsat = unsatisfied_clause_indices(instance, assignment)

        after = count_satisfied_clauses(instance, assignment)
        return OperatorResult(assignment, after - before, max(1, work), touched)


class FocusedVNS(BaseOperator):
    name = "focused_vns"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def apply(self, instance, assignment, pheromone, var_clause_index, rng):
        before = count_satisfied_clauses(instance, assignment)
        unsat_idx = set(unsatisfied_clause_indices(instance, assignment))
        if not unsat_idx:
            return OperatorResult(assignment, 0, 1, set())

        # Rank variables by (nb unsat clauses involved) x (pheromone signal)
        scores = {}
        candidate_vars = set()
        for c_idx in unsat_idx:
            for lit in instance.clauses[c_idx]:
                candidate_vars.add(abs(lit))

        for v in candidate_vars:
            n_unsat_involved = sum(1 for (ci, _) in var_clause_index[v] if ci in unsat_idx)
            pher_signal = pheromone.desirability(v, not assignment[v])
            scores[v] = n_unsat_involved * pher_signal

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_vars = [v for v, _ in ranked[: self.top_k]]
        touched = set(top_vars)
        for v in top_vars:
            assignment[v] = not assignment[v]

        after = count_satisfied_clauses(instance, assignment)
        work = sum(len(var_clause_index[v]) for v in top_vars)
        return OperatorResult(assignment, after - before, max(1, work), touched)


class ClauseRestartGreedy(BaseOperator):
    """Guided partial reconstruction: the most expensive but highest
    potential-gain operator (+1000 to +3700 delta_f per call, 800-3600J,
    per the notebook's conclusions). Requires the anti-overrun guard."""

    name = "clause_restart_greedy"

    def __init__(self, restart_fraction: float = 0.1):
        self.restart_fraction = restart_fraction

    def apply(self, instance, assignment, pheromone, var_clause_index, rng):
        before = count_satisfied_clauses(instance, assignment)
        unsat_idx = unsatisfied_clause_indices(instance, assignment)
        if not unsat_idx:
            return OperatorResult(assignment, 0, 1, set())

        involved_vars = set()
        for c_idx in unsat_idx:
            for lit in instance.clauses[c_idx]:
                involved_vars.add(abs(lit))

        n_restart = max(1, int(len(involved_vars) * self.restart_fraction))
        restart_vars = rng.sample(sorted(involved_vars), min(n_restart, len(involved_vars)))
        touched = set(restart_vars)
        work = 0

        for v in restart_vars:
            work += len(var_clause_index[v])
            # Greedy reassignment guided by pheromone desirability
            score_true = pheromone.desirability(v, True)
            score_false = pheromone.desirability(v, False)
            # Also weight by local satisfaction gain
            gain_true, gain_false = 0, 0
            old = assignment[v]
            assignment[v] = True
            gain_true = sum(1 for (ci, _) in var_clause_index[v]
                             if clause_is_satisfied(instance.clauses[ci], assignment))
            assignment[v] = False
            gain_false = sum(1 for (ci, _) in var_clause_index[v]
                              if clause_is_satisfied(instance.clauses[ci], assignment))
            assignment[v] = old

            combined_true = gain_true + score_true
            combined_false = gain_false + score_false
            assignment[v] = combined_true >= combined_false

        after = count_satisfied_clauses(instance, assignment)
        return OperatorResult(assignment, after - before, max(1, work), touched)


OPERATOR_REGISTRY = {
    "walksat": WalkSAT,
    "focused_vns": FocusedVNS,
    "clause_restart_greedy": ClauseRestartGreedy,
}


def build_operator_pool(names: List[str]) -> Dict[str, BaseOperator]:
    return {name: OPERATOR_REGISTRY[name]() for name in names if name in OPERATOR_REGISTRY}
