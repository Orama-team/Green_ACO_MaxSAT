"""
utils.py
========
Small shared utilities: assignment satisfaction checks, RNG helpers,
and logging helpers used across the Green ACO codebase.
"""

import random
from typing import Dict, List, Tuple

from .parser import CNFInstance


def random_assignment(n_vars: int, rng: random.Random) -> Dict[int, bool]:
    return {v: rng.random() < 0.5 for v in range(1, n_vars + 1)}


def clause_is_satisfied(clause: Tuple[int, ...], assignment: Dict[int, bool]) -> bool:
    for lit in clause:
        var = abs(lit)
        val = assignment[var]
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False


def count_satisfied_clauses(instance: CNFInstance, assignment: Dict[int, bool]) -> int:
    return sum(1 for c in instance.clauses if clause_is_satisfied(c, assignment))


def unsatisfied_clause_indices(instance: CNFInstance, assignment: Dict[int, bool]) -> List[int]:
    return [i for i, c in enumerate(instance.clauses) if not clause_is_satisfied(c, assignment)]


def quality_pct(n_satisfied: int, n_clauses: int) -> float:
    """Percentage of clauses satisfied, as reported throughout the notebook
    (`qualite_pct` = mean_df / n_clauses * 100 for the profiling tables)."""
    if n_clauses == 0:
        return 0.0
    return 100.0 * n_satisfied / n_clauses


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)
