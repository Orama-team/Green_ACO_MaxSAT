"""
operators.py
=============
The three operators of the Green ACO pool, plus the pheromone update rules.

Each operator receives the current assignment and returns a *better* one (or
the incumbent, if it cannot improve). Working on a copy and returning the best
candidate keeps the search monotone: an operator can never degrade the
assignment it was handed.

Two interchangeable backends are provided:

``RescanBackend``  (default)
    The reference implementation, ported unchanged from the final research
    version. It rescans the full formula on every call, which is O(m) per
    operator evaluation.

``IndexedBackend``
    Behaviourally identical but consults a precomputed variable->clauses index
    to avoid full rescans. ``tests/test_operators.py`` asserts that both
    backends return identical results under a fixed seed, so switching is safe
    and purely a matter of speed.

The pheromone updates are exact ports of the original routines.
"""

from __future__ import annotations

import random

from .wcnf import (
    count_satisfied_clauses,
    count_unsat_clauses_per_variable,
    get_unsatisfied_clause_indices,
)


# ── Pheromone updates ────────────────────────────────────────────────────────


def update_pheromone(
    pheromone,
    assignment,
    n_satisfied,
    n_clauses,
    evaporation_rate: float = 0.1,
    n_vars: int | None = None,
) -> None:
    """Full O(n) update: evaporate every variable, then reinforce the assigned
    value by the normalised solution quality."""
    n = n_vars or len(pheromone)
    delta = n_satisfied / n_clauses
    for i in range(n):
        pheromone[i][0] *= (1 - evaporation_rate)
        pheromone[i][1] *= (1 - evaporation_rate)
        v = 1 if assignment[i + 1] else 0
        pheromone[i][v] += delta


def update_pheromone_sparse(
    pheromone,
    previous_assignment,
    new_assignment,
    n_satisfied,
    n_clauses,
    evaporation_rate: float = 0.1,
    n_vars: int | None = None,
) -> int:
    """Sparse O(k) reinforcement for flip-based operators.

    Evaporation still sweeps every variable (it is global by nature), but
    reinforcement is restricted to the k variables that actually flipped.
    Returns the number of reinforced variables.
    """
    n = n_vars or len(pheromone)
    delta = n_satisfied / max(n_clauses, 1)

    for i in range(n):
        pheromone[i][0] *= (1 - evaporation_rate)
        pheromone[i][1] *= (1 - evaporation_rate)

    n_updated = 0
    for var in range(1, n + 1):
        if previous_assignment.get(var) != new_assignment.get(var):
            i = var - 1
            v = 1 if new_assignment[var] else 0
            pheromone[i][v] += delta
            n_updated += 1
    return n_updated

# ── Operator implementations ─────────────────────────────────────────────────


def op_walksat(formula, n_vars, pheromone, eta, current_asgn, best_asgn,
               n_clauses, p_noise=0.3, max_flips=10, score=None,
               unsat_fn=None):
    """Target an unsatisfied clause and flip a variable to repair it.

    With probability ``p_noise`` the flip is random, otherwise the variable
    whose flip maximises the number of satisfied clauses is chosen. The best
    assignment seen across the flips is returned, so the operator never worsens
    the assignment it was given.
    """
    score = score or count_satisfied_clauses
    unsat_fn = unsat_fn or get_unsatisfied_clause_indices

    asgn = dict(current_asgn)
    score0 = score(formula, asgn)
    best_a, best_s = asgn, score0

    for _ in range(max_flips):
        unsat = unsat_fn(formula, asgn)
        if not unsat:
            break
        clause = formula[random.choice(unsat)]
        clause_vars = [abs(lit) for lit in clause]

        if random.random() < p_noise:
            flip_var = random.choice(clause_vars)
        else:
            best_var, best_gain = clause_vars[0], -1e9
            for v in clause_vars:
                asgn[v] = not asgn[v]
                s = score(formula, asgn)
                if s > best_gain:
                    best_gain, best_var = s, v
                asgn[v] = not asgn[v]
            flip_var = best_var

        asgn[flip_var] = not asgn[flip_var]
        s = score(formula, asgn)
        if s > best_s:
            best_s, best_a = s, dict(asgn)

    update_pheromone(pheromone, best_a, best_s, n_clauses,
                     evaporation_rate=0.08, n_vars=n_vars)
    return best_a, best_s - score0

def op_focused_vns(formula, n_vars, pheromone, eta, current_asgn, best_asgn,
                   n_clauses, k=5, score=None, unsat_scores_fn=None):
    """Focus on the k variables most implicated in unsatisfied clauses.

    Variables are ranked by (number of unsatisfied clauses they appear in)
    weighted by the pheromone signal for their current value; the best single
    flip among the top-k is kept.
    """
    score = score or count_satisfied_clauses
    unsat_scores_fn = unsat_scores_fn or count_unsat_clauses_per_variable

    asgn = dict(current_asgn)
    score0 = score(formula, asgn)

    unsat_count_per_var = unsat_scores_fn(formula, asgn, n_vars)
    combined = sorted(
        range(1, n_vars + 1),
        key=lambda v: unsat_count_per_var[v]
        * (pheromone[v - 1][1 if asgn[v] else 0] + 1e-6),
        reverse=True,
    )
    top_k_vars = combined[:k]

    best_a, best_s = asgn, score0
    for var in top_k_vars:
        candidate = dict(asgn)
        candidate[var] = not candidate[var]
        s = score(formula, candidate)
        if s > best_s:
            best_s, best_a = s, candidate

    update_pheromone(pheromone, best_a, best_s, n_clauses,
                     evaporation_rate=0.08, n_vars=n_vars)
    return best_a, best_s - score0


def op_clause_restart_greedy(formula, n_vars, pheromone, eta, current_asgn,
                             best_asgn, n_clauses, max_focus_ratio=0.20,
                             score=None, unsat_fn=None, score_delta_fn=None):
    """Guided partial reconstruction of the most problematic region.

    The variables appearing in the most unsatisfied clauses are reassigned
    greedily, one at a time, keeping whichever value scores better. This is the
    highest-gain and by far the most expensive operator in the pool, which is
    why the solver guards it with the overrun budget check.

    ``score_delta_fn(formula, assignment, var, value)`` is an optional hook
    returning the change in satisfied clauses caused by setting ``var`` to
    ``value``. When supplied (by the indexed backend) the per-candidate
    evaluation touches only the clauses mentioning that variable instead of
    rescanning the whole formula. Both paths yield the same assignment; the
    hook exists purely so the cost is not quadratic on large instances.
    """
    score = score or count_satisfied_clauses
    unsat_fn = unsat_fn or get_unsatisfied_clause_indices

    asgn = dict(current_asgn)
    score0 = score(formula, asgn)

    unsat_indices = unsat_fn(formula, asgn)
    if not unsat_indices:
        return asgn, 0.0

    unsat_count = {}
    for ci in unsat_indices:
        for lit in formula[ci]:
            v = abs(lit)
            unsat_count[v] = unsat_count.get(v, 0) + 1

    max_to_rebuild = max(1, int(n_vars * max_focus_ratio))
    focus_vars = sorted(unsat_count, key=unsat_count.get,
                        reverse=True)[:max_to_rebuild]

    if score_delta_fn is None:
        candidate = dict(asgn)
        for var in focus_vars:
            candidate[var] = True
            score_if_true = score(formula, candidate)
            candidate[var] = False
            score_if_false = score(formula, candidate)
            candidate[var] = score_if_true >= score_if_false
        new_score = score(formula, candidate)
    else:
        # Mirror the same decision rule incrementally: try True, try False,
        # and keep True on a tie. `score_delta_fn` leaves the candidate
        # holding the tested value, so each step is a single variable flip.
        candidate = dict(asgn)
        for var in focus_vars:
            d_true = score_delta_fn(formula, candidate, var, True)
            d_false = score_delta_fn(formula, candidate, var, False)
            candidate[var] = d_true >= d_false
        new_score = score(formula, candidate)

    if new_score <= score0:
        return asgn, 0.0

    update_pheromone(pheromone, candidate, new_score, n_clauses,
                     evaporation_rate=0.07, n_vars=n_vars)
    return candidate, new_score - score0

# ── Backends ─────────────────────────────────────────────────────────────────

OPERATOR_FUNCTIONS = {
    "walksat": op_walksat,
    "focused_vns": op_focused_vns,
    "clause_restart_greedy": op_clause_restart_greedy,
}


class RescanBackend:
    """Reference backend: full formula rescans. Default for all runs."""

    name = "rescan"

    def prepare(self, formula, n_vars):
        """No index to build."""

    def score(self, formula, assignment):
        return count_satisfied_clauses(formula, assignment)

    def unsatisfied(self, formula, assignment):
        return get_unsatisfied_clause_indices(formula, assignment)

    def unsat_per_variable(self, formula, assignment, n_vars):
        return count_unsat_clauses_per_variable(formula, assignment, n_vars)

    def pool(self, names):
        return {name: _bind(self, name) for name in names
                if name in OPERATOR_FUNCTIONS}


class IndexedBackend(RescanBackend):
    """Speed-oriented backend, behaviourally identical to the reference.

    The reference recomputes the full score after every candidate move, which
    is O(m) per evaluation. Operators that try many candidates therefore become
    very expensive: ``clause_restart_greedy`` rebuilds a fifth of the variables
    and evaluates each of them twice, so on a 109k-clause instance a single call
    takes minutes.

    This backend answers the same questions more cheaply:

    * ``score_delta`` -- flipping variable ``v`` can only change the status of
      the clauses mentioning ``v``, so the score change is derived from that
      clause list instead of rescanning the formula.
    * ``unsat_per_variable`` -- unsatisfied counts are accumulated from one
      clause scan rather than a rescan per variable.

    Only the arithmetic changes, never the result. ``tests/test_operators.py``
    asserts the two backends agree exactly under a fixed seed, so selecting
    between them is a performance decision, not a behavioural one.
    """

    name = "indexed"

    def __init__(self):
        self._clauses_of_var = None
        self._n_vars = None

    def prepare(self, formula, n_vars):
        """Build the variable -> clauses index once per instance."""
        if self._clauses_of_var is not None and self._n_vars == n_vars:
            return
        clauses_of_var = [[] for _ in range(n_vars + 1)]
        for c_idx, clause in enumerate(formula):
            for lit in clause:
                clauses_of_var[abs(lit)].append(c_idx)
        self._clauses_of_var = clauses_of_var
        self._n_vars = n_vars

    @staticmethod
    def _clause_sat(clause, bits) -> bool:
        for lit in clause:
            val = bits[abs(lit)]
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    def score_delta(self, formula, bits, var, new_value) -> int:
        """Change in satisfied-clause count when ``var`` takes ``new_value``.

        ``bits`` is left holding ``new_value``.
        """
        old_value = bits[var]
        if old_value == new_value:
            return 0
        bits[var] = new_value
        delta = 0
        for c_idx in self._clauses_of_var[var]:
            clause = formula[c_idx]
            now = self._clause_sat(clause, bits)
            bits[var] = old_value
            was = self._clause_sat(clause, bits)
            bits[var] = new_value
            if now and not was:
                delta += 1
            elif was and not now:
                delta -= 1
        return delta

    def unsat_per_variable(self, formula, assignment, n_vars):
        """Count unsatisfied clauses per variable, reusing one clause scan."""
        unsat = set(self.unsatisfied(formula, assignment))
        per_var = [0] * (n_vars + 1)
        for c_idx in unsat:
            for lit in formula[c_idx]:
                per_var[abs(lit)] += 1
        return per_var


def _bind(backend: RescanBackend, name: str):
    """Bind a backend's evaluation helpers into an operator function."""
    fn = OPERATOR_FUNCTIONS[name]
    score = backend.score
    unsat_fn = backend.unsatisfied
    unsat_scores_fn = backend.unsat_per_variable
    score_delta_fn = getattr(backend, "score_delta", None)

    if name == "focused_vns":
        def call(*args, **kwargs):
            kwargs["score"] = score
            kwargs["unsat_scores_fn"] = unsat_scores_fn
            return fn(*args, **kwargs)
    elif name == "clause_restart_greedy" and score_delta_fn is not None:
        def call(*args, **kwargs):
            kwargs["score"] = score
            kwargs["unsat_fn"] = unsat_fn
            kwargs["score_delta_fn"] = score_delta_fn
            return fn(*args, **kwargs)
    else:
        def call(*args, **kwargs):
            kwargs["score"] = score
            kwargs["unsat_fn"] = unsat_fn
            return fn(*args, **kwargs)
    return call


def build_operator_pool(names, backend: RescanBackend | None = None):
    """Instantiate the operator pool for a run."""
    backend = backend or RescanBackend()
    return backend.pool(names)


BACKENDS = {"rescan": RescanBackend, "indexed": IndexedBackend}
