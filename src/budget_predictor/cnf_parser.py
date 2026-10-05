"""
cnf_parser.py
=============
Re-export of the reader in ``greenaco.cnf``, so that ``budget_predictor`` has
no hard dependency on the internals of the solver package and can be imported
independently, for example to extract features from instances that will never
actually be solved.
"""

from greenaco.cnf import CNFInstance, parse_dimacs_cnf, build_var_clause_index

__all__ = ["CNFInstance", "parse_dimacs_cnf", "build_var_clause_index"]
