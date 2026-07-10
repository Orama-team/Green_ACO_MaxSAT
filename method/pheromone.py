"""
pheromone.py
============
Pheromone matrix tau[v][b] for Green ACO, where tau[v][b] encodes the
desirability of assigning boolean value b to variable v (section 1.1 of
the notebook).

Implements:
  - Dense init + evaporation with adaptive rate (rho_base / rho_stagnant,
    section 1.2.2 "évaporation adaptative")
  - Sparse update (`use_sparse_update`): only the O(k) variables touched
    by the last operator call are updated, instead of the full O(n)
    matrix, as described in the ablation ("no_sparse_ph").
  - Pheromone caching (`use_pheromone_cache`): warm-start across budgets,
    reusing tau learned on a smaller budget B1 for a larger budget B2
    ("no_caching" ablation).
  - Frugal skip: updates that don't improve on the incumbent by more
    than a tolerance are skipped, saving 17-27% of updates per the
    notebook's conclusions (section 5.2).
"""

import pickle
from typing import Dict, Iterable, Optional


class PheromoneMatrix:
    def __init__(self, n_vars: int, tau_init: float = 1.0,
                 tau_min: float = 0.01, tau_max: float = 10.0):
        self.n_vars = n_vars
        self.tau_min = tau_min
        self.tau_max = tau_max
        # tau[v] = [tau_false, tau_true]
        self.tau: Dict[int, list] = {v: [tau_init, tau_init] for v in range(1, n_vars + 1)}

    # ── Evaporation ────────────────────────────────────────────────────
    def evaporate(self, rho: float, variables: Optional[Iterable[int]] = None):
        """Evaporate pheromone by factor (1 - rho). If `variables` is given,
        only evaporate those (sparse update path)."""
        targets = variables if variables is not None else self.tau.keys()
        for v in targets:
            self.tau[v][0] = max(self.tau_min, self.tau[v][0] * (1.0 - rho))
            self.tau[v][1] = max(self.tau_min, self.tau[v][1] * (1.0 - rho))

    # ── Deposit / reinforcement ────────────────────────────────────────
    def deposit(self, assignment: Dict[int, bool], amount: float,
                variables: Optional[Iterable[int]] = None):
        """Reinforce the pheromone entries consistent with `assignment`.
        If `variables` is provided (sparse update), only touch those."""
        targets = variables if variables is not None else assignment.keys()
        for v in targets:
            b = 1 if assignment[v] else 0
            self.tau[v][b] = min(self.tau_max, self.tau[v][b] + amount)

    def update(self, assignment: Dict[int, bool], delta_f: float, rho: float,
               variables: Optional[Iterable[int]] = None,
               frugal_epsilon: float = 0.0) -> bool:
        """
        Combined evaporate + deposit step, proportional to `delta_f`
        (quality improvement of the move). Returns True if the update
        was actually applied (False if skipped by the frugal-skip rule).
        """
        if delta_f <= frugal_epsilon:
            # Frugal skip: do not reinforce moves that barely progress,
            # preserving pheromone-matrix diversity during stagnation.
            return False
        self.evaporate(rho, variables=variables)
        self.deposit(assignment, amount=delta_f, variables=variables)
        return True

    # ── Desirability lookup used by operators for probabilistic choice ──
    def desirability(self, var: int, value: bool) -> float:
        return self.tau[var][1 if value else 0]

    # ── Warm-start cache (inter-budget reuse) ─────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"n_vars": self.n_vars, "tau": self.tau,
                         "tau_min": self.tau_min, "tau_max": self.tau_max}, f)

    @classmethod
    def load(cls, path: str) -> "PheromoneMatrix":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["n_vars"], tau_min=state["tau_min"], tau_max=state["tau_max"])
        obj.tau = state["tau"]
        return obj
