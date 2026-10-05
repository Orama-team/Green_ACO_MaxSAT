"""
Unweighted MAX-SAT instance representation and WCNF parsing.

The benchmark corpus (MSE 2024) is distributed as *weighted* MAX-SAT
(``*.wcnf.xz``) containing both hard clauses (``h 1 -2 0``) and weighted soft
clauses (``1 -96 0``). The experimental protocol adopted for this work is
**unweighted** MAX-SAT: every clause is treated with weight 1, so the objective
is simply "minimise the number of violated clauses".

Parsing therefore *discards* both the leading weight token and the ``h`` marker,
keeping only the literals. This matches ``build_pickle2.py::parse_wcnf``, the
script that produced the original experiment pickles, so instances parsed here
are identical to those the original results were computed on.

Supported inputs: plain ``.wcnf``, ``.cnf``, and their ``.xz``-compressed forms.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass, field
from typing import List, Tuple

__all__ = [
    "Instance",
    "parse_wcnf",
    "parse_wcnf_file",
    "count_satisfied_clauses",
    "get_unsatisfied_clause_indices",
    "count_unsat_clauses_per_variable",
]


@dataclass
class Instance:
    """A parsed unweighted MAX-SAT instance.

    Attributes
    ----------
    benchmark:
        Instance name without any file extension.
    n_vars:
        Number of variables, inferred from the largest absolute literal seen
        (what ``build_pickle2.py`` does). This can exceed the value declared
        in the file header for instances with trailing unused variables.
    clauses:
        The formula, as a list of tuples of signed 1-based literals.
    """

    benchmark: str
    n_vars: int
    clauses: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def strip_empty_clauses(self) -> int:
        """Remove zero-length clauses; return how many were removed.

        Empty clauses are trivially unsatisfiable and make the greedy
        constructor and the unsatisfied-clause scanners misbehave.
        """
        before = len(self.clauses)
        self.clauses = [c for c in self.clauses if len(c) > 0]
        return before - len(self.clauses)

    def to_record(self) -> dict:
        """Project to the manifest schema used by the CSV artifacts."""
        return {
            "benchmark": self.benchmark,
            "n_vars": self.n_vars,
            "n_clauses": self.n_clauses,
        }


def _open_text(path):
    """Open a possibly ``.xz``-compressed text file."""
    if str(path).endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_wcnf(path, name: str | None = None) -> Instance:
    """Parse a WCNF/CNF file into an unweighted :class:`Instance`.

    Skips comment lines (``c ...``), the old-style header (``p wcnf n m``), the
    hard-clause marker ``h``, and the leading weight token of a soft clause,
    keeping only the literals up to the terminating ``0``.
    """
    path = str(path)
    clauses: List[Tuple[int, ...]] = []
    max_var = 0

    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            parts = line.split()
            if not parts:
                continue

            # Old-style "p wcnf <nvars> <nclauses>" header.
            if parts[0] == "p":
                continue

            # Drop the hard-clause marker or the soft-clause weight, keeping
            # only the literals.
            tokens = parts[1:]

            try:
                lits = [int(t) for t in tokens]
            except ValueError:
                # Malformed numeric payload: skip rather than abort the parse.
                continue

            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if not lits:
                continue

            local_max = max(abs(lit) for lit in lits)
            if local_max > max_var:
                max_var = local_max

            clauses.append(tuple(lits))

    if name is None:
        base = path.replace("\\", "/").rsplit("/", 1)[-1]
        name = base
        for suffix in (".wcnf.xz", ".wcnf", ".cnf.xz", ".cnf", ".xz"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

    return Instance(benchmark=name, n_vars=max_var, clauses=clauses)
# ── Shared evaluation helpers ────────────────────────────────────────────────
# These mirror the original notebook implementations exactly. They rescan the
# full formula on every call; ``operators.IndexedBackend`` provides an
# equivalent-but-faster variant, verified against these by the test suite.


def count_satisfied_clauses(formula, assignment) -> int:
    """Count clauses satisfied by ``assignment`` (1-based variable keys)."""
    count = 0
    for clause in formula:
        for lit in clause:
            var = abs(lit)
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                count += 1
                break
    return count


def get_unsatisfied_clause_indices(formula, assignment) -> List[int]:
    """Indices of clauses not satisfied by ``assignment``."""
    unsat = []
    for idx, clause in enumerate(formula):
        satisfied = any(
            (lit > 0 and assignment[abs(lit)]) or (lit < 0 and not assignment[abs(lit)])
            for lit in clause
        )
        if not satisfied:
            unsat.append(idx)
    return unsat


def count_unsat_clauses_per_variable(formula, assignment, n_vars) -> List[int]:
    """For each variable (1-based), count unsatisfied clauses it occurs in."""
    per_var = [0] * (n_vars + 1)
    for clause_idx in get_unsatisfied_clause_indices(formula, assignment):
        for lit in formula[clause_idx]:
            per_var[abs(lit)] += 1
    return per_var


def parse_wcnf_file(path: str, benchmark: str | None = None) -> Instance:
    """Convenience wrapper: parse ``path``, optionally forcing the name."""

    return parse_wcnf(path, name=benchmark)


# Aliases preserved from the original notebook module namespace.
_score_assignment = count_satisfied_clauses
_unsatisfied_clauses = get_unsatisfied_clause_indices
_var_unsat_scores = count_unsat_clauses_per_variable