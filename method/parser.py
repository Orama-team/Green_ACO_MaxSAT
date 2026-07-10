"""
parser.py
=========
Minimal DIMACS CNF / WCNF parser for MAX-SAT instances, as used to load
the 4 benchmark instances of the notebook (soybean, min-fill/myciel5,
car, vote — all "decision-tree-*" / "min-fill-MinFill_R0_*" encodings).

Format assumed (standard DIMACS CNF):
    c comment lines
    p cnf <n_vars> <n_clauses>
    <lit_1> <lit_2> ... 0
    ...
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CNFInstance:
    name: str
    n_vars: int
    n_clauses: int
    clauses: List[Tuple[int, ...]]  # each clause is a tuple of signed ints

    def clause_lengths(self):
        return [len(c) for c in self.clauses]


def parse_dimacs_cnf(path: str) -> CNFInstance:
    """Parse a .cnf file in DIMACS format into a CNFInstance."""
    n_vars = n_clauses = 0
    clauses: List[Tuple[int, ...]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                # p cnf <n_vars> <n_clauses>
                n_vars = int(parts[2])
                n_clauses = int(parts[3])
                continue
            lits = [int(x) for x in line.split() if x != ""]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if lits:
                clauses.append(tuple(lits))

    import os
    name = os.path.splitext(os.path.basename(path))[0]
    return CNFInstance(name=name, n_vars=n_vars, n_clauses=len(clauses) or n_clauses,
                        clauses=clauses)


def build_var_clause_index(instance: CNFInstance):
    """
    Build a mapping variable -> list of (clause_idx, is_negated) for O(1)
    lookup of which clauses a variable participates in. Used by
    Focused VNS / Clause Restart Greedy to avoid O(n_clauses) scans.
    """
    index = {v: [] for v in range(1, instance.n_vars + 1)}
    for c_idx, clause in enumerate(instance.clauses):
        for lit in clause:
            var = abs(lit)
            index[var].append((c_idx, lit < 0))
    return index
