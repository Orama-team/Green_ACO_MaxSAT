"""
feature_extractor.py
=====================
Extracts structural features from a CNF/MAX-SAT instance, used to
predict the energy budget (Joules) a Green ACO run needs to reach a
target quality/joule regime, without having to run the solver first.

Features are cheap to compute (O(n_clauses)) and intentionally use only
information available from a new instance before solving it. Metadata
columns, solver outputs, quality scores, selected budgets, and labels are
not model inputs.
"""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict

from .cnf_parser import CNFInstance, build_var_clause_index


@dataclass
class InstanceFeatures:
    n_vars: int
    n_clauses: int
    clause_var_ratio: float
    mean_clause_len: float
    std_clause_len: float
    mean_var_degree: float
    std_var_degree: float
    max_var_degree: int
    frac_unit_clauses: float
    frac_binary_clauses: float
    frac_horn_clauses: float
    positive_literal_frac: float

    def to_dict(self) -> Dict:
        return self.__dict__


def extract_features(instance: CNFInstance) -> InstanceFeatures:
    clause_lens = [len(c) for c in instance.clauses]
    var_index = build_var_clause_index(instance)
    degrees = [len(v) for v in var_index.values()]

    n_pos = sum(1 for c in instance.clauses for lit in c if lit > 0)
    n_lits = sum(clause_lens) or 1

    n_horn = sum(1 for c in instance.clauses
                 if sum(1 for lit in c if lit > 0) <= 1)

    return InstanceFeatures(
        n_vars=instance.n_vars,
        n_clauses=instance.n_clauses,
        clause_var_ratio=instance.n_clauses / max(instance.n_vars, 1),
        mean_clause_len=mean(clause_lens) if clause_lens else 0.0,
        std_clause_len=pstdev(clause_lens) if len(clause_lens) > 1 else 0.0,
        mean_var_degree=mean(degrees) if degrees else 0.0,
        std_var_degree=pstdev(degrees) if len(degrees) > 1 else 0.0,
        max_var_degree=max(degrees) if degrees else 0,
        frac_unit_clauses=sum(1 for l in clause_lens if l == 1) / max(len(clause_lens), 1),
        frac_binary_clauses=sum(1 for l in clause_lens if l == 2) / max(len(clause_lens), 1),
        frac_horn_clauses=n_horn / max(instance.n_clauses, 1),
        positive_literal_frac=n_pos / n_lits,
    )


FEATURE_DESCRIPTIONS = {
    "n_vars": "Number of Boolean variables in the instance.",
    "n_clauses": "Number of clauses.",
    "clause_var_ratio": "Clause density: n_clauses / n_vars.",
    "mean_clause_len": "Average number of literals per clause.",
    "std_clause_len": "Spread of clause lengths.",
    "mean_var_degree": "Average number of clauses in which a variable appears.",
    "std_var_degree": "Spread of variable occurrence counts.",
    "max_var_degree": "Largest variable occurrence count.",
    "frac_unit_clauses": "Fraction of clauses with one literal.",
    "frac_binary_clauses": "Fraction of clauses with two literals.",
    "frac_horn_clauses": "Fraction of Horn clauses, with at most one positive literal.",
    "positive_literal_frac": "Fraction of all literals that are positive.",
}

FEATURE_NAMES = list(InstanceFeatures.__dataclass_fields__.keys())
