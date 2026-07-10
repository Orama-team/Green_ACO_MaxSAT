"""
cnf_parser.py
=============
Thin wrapper around `method.parser` so that `budget_predictor` has no
hard dependency on the internals of the solver package, and can be
distributed / imported independently (e.g. for feature extraction on
instances that will never actually be solved).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from method.parser import CNFInstance, parse_dimacs_cnf, build_var_clause_index  # noqa: E402

__all__ = ["CNFInstance", "parse_dimacs_cnf", "build_var_clause_index"]
