"""
budget_predictor
=================
XGBoost-based predictor that recommends the Green ACO energy budget
(Joules) to allocate to a given MAX-SAT instance, based on cheap
structural features (variables, clauses, degree distribution, ...),
so the [400, 1000, 2000] J grid explored manually in the notebook can
be replaced by a per-instance prediction for new, unseen instances.
"""

from .predictor import BudgetPredictor

__all__ = ["BudgetPredictor"]
