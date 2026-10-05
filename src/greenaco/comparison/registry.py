"""
comparison/registry.py
======================
The set of methods entering the comparison, and how each is invoked.

Seven methods qualify for the comparison: three GA variants and four ACO
variants, each of which produced usable results on the comparison instances.
Green ACO is added by the runner, giving an eight-row comparison table.

Parameters come from the calibration artefacts in ``data/comparison/``. They
are stored as data rather than code so that a re-run records exactly which
settings produced a given table.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .aco import (ACO_SHARED, EXCLUDED, aco_sat, aco_sat_elitist,  # noqa: F401
                  acs_sat, mmas_sat)
from .ga import GA_CLASS_MAP, TUNED_PARAMS as GA_PARAMS, build_ga, make_ga_solver

# Display order of the comparison table.
METHOD_ORDER: List[str] = [
    "AG Classique",
    "AG Adaptatif",
    "AG + KC",
    "AS-SAT",
    "AS-SAT Elitiste",
    "MMAS",
    "ACS",
    "Green ACO",
]

FAMILY = {
    "AG Classique": "GA",
    "AG Adaptatif": "GA",
    "AG + KC": "GA",
    "AS-SAT": "ACO",
    "AS-SAT Elitiste": "ACO",
    "MMAS": "ACO",
    "ACS": "ACO",
    "Green ACO": "ACO + EI/J",
}

# Per-method ACO settings, from the calibration summary.
ACO_METHOD_PARAMS: Dict[str, Dict] = {
    "AS-SAT": {"tau0": 1.0, "Q": 1.0},
    "AS-SAT Elitiste": {"tau0": 0.5, "e": 5.0},
    "MMAS": {"tau_min": 0.05, "tau_max": 1.0, "stagnation_limit": 10},
    "ACS": {"tau0": 1.0, "q0": 0.95, "xi": 0.1, "cl": 5.0},
}

ACO_FUNCTIONS = {
    "AS-SAT": aco_sat,
    "AS-SAT Elitiste": aco_sat_elitist,
    "MMAS": mmas_sat,
    "ACS": acs_sat,
}

GA_METHODS = ["AG Classique", "AG Adaptatif", "AG + KC"]


def clean_params(params: Dict) -> Dict:
    """Replace NaN values with ``None``.

    ``mutation_prob: nan`` in the calibration artefacts means "use the class
    default"; passing a literal NaN would make the mutation test always false
    and silently disable mutation.
    """
    out = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        out[key] = value
    return out


def get_solver(method: str, seed: int = 42, timeout_s: int = 300):
    """Return a ``fn(formula, n_vars) -> (assignment, stats)`` for ``method``."""
    if method in ACO_FUNCTIONS:
        params = dict(ACO_SHARED)
        params.update(ACO_METHOD_PARAMS[method])
        fn = ACO_FUNCTIONS[method]

        def _solver(formula, n_vars):
            import random as _random

            _random.seed(seed)
            return fn(formula, n_vars, **params)

        return _solver

    if method in GA_METHODS:
        ga = build_ga(method, timeout=timeout_s, seed=seed)
        return make_ga_solver(type(ga))

    raise KeyError(f"Unknown comparison method: {method!r}")


def method_params(method: str) -> Dict:
    """The settings a method will run with, for recording in the output."""
    if method in ACO_METHOD_PARAMS:
        return dict(ACO_SHARED, **ACO_METHOD_PARAMS[method])
    if method in GA_METHODS:
        return clean_params(GA_PARAMS[method])
    return {}