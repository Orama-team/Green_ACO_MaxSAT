"""
cnf.py
======
Minimal DIMACS CNF reader.

The solver core works on weighted MaxSAT in WCNF format (see ``wcnf.py``).
The budget predictor, however, is deliberately separable: it can score
structural features of an instance that will never be solved, and so is
distributed independently of the search machinery. That separation is why
this reader lives apart from the WCNF path rather than inside it.

Only what the predictor needs is provided: a small record type, a reader for
the ``p cnf`` format, and the variable-to-clause index used to measure how
often each variable occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["CNFInstance", "parse_dimacs_cnf", "build_var_clause_index"]


@dataclass(frozen=True)
class CNFInstance:
    """An unweighted CNF formula.

    Attributes:
        name: Identifier, usually the file stem.
        n_vars: Number of variables declared in the ``p`` line.
        n_clauses: Number of clauses. Kept as declared; trust
            ``len(clauses)`` if the two disagree, which malformed files can.
        clauses: Clauses as tuples of signed integer literals, no trailing
            zero.
    """

    name: str
    n_vars: int
    n_clauses: int
    clauses: List[Tuple[int, ...]] = field(default_factory=list)


def parse_dimacs_cnf(path: str, name: str | None = None) -> CNFInstance:
    """Read a DIMACS CNF file.

    Handles the two details that break naive readers: clauses may span several
    lines, and comment lines beginning with ``c`` may appear anywhere.

    Args:
        path: Path to the ``.cnf`` file.
        name: Identifier to record. Defaults to the file stem.

    Returns:
        The parsed :class:`CNFInstance`.

    Raises:
        ValueError: If the file has no ``p cnf`` header.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    declared_vars = declared_clauses = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("p") and line.split()[:2] == ["p", "cnf"]:
            parts = line.split()
            declared_vars, declared_clauses = int(parts[2]), int(parts[3])
            break

    if declared_vars is None:
        raise ValueError(f"no 'p cnf' header found in {path}")

    literals: List[int] = []
    clauses: List[Tuple[int, ...]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("c"):
            continue
        if stripped.startswith("p"):
            continue
        for token in stripped.split():
            value = int(token)
            if value == 0:
                if literals:
                    clauses.append(tuple(literals))
                    literals = []
            else:
                literals.append(value)

    if literals:
        # Trailing clause without its terminating zero.
        clauses.append(tuple(literals))

    return CNFInstance(
        name=name or path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
        n_vars=declared_vars,
        n_clauses=declared_clauses if declared_clauses is not None else len(clauses),
        clauses=clauses,
    )


def build_var_clause_index(instance: CNFInstance) -> Dict[int, List[int]]:
    """Map each variable to the indices of the clauses mentioning it.

    Args:
        instance: The formula to index.

    Returns:
        Variable number to the list of clause indices it occurs in. Variables
        that do not occur are absent, so ``len(index.values())`` counts only
        variables actually present in the formula.
    """
    index: Dict[int, List[int]] = {}
    for position, clause in enumerate(instance.clauses):
        # Deduplicate within the clause: a variable occurs *in* a clause once,
        # however many times the literal is repeated, and counting repeats
        # would inflate the degree features the predictor is built on.
        for var in {abs(literal) for literal in clause}:
            index.setdefault(var, []).append(position)
    return index