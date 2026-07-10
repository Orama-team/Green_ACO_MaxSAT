"""
energy.py
=========
Energy accounting for Green ACO.

Each operator call has a measured/estimated cost in Joules (see
`operator_profiles.csv`, columns `mean_cost_j`). This module tracks
cumulative energy consumption for a run and converts it to an estimated
CO2 footprint, matching the `energie_joules` / `co2_micrograms` /
`score_per_joule` columns used throughout the notebook's comparison
plots (section 4).

Carbon intensity default (grid average, micrograms CO2eq per Joule) is
a rough illustrative constant; override via `EnergyTracker(carbon_intensity=...)`
if you have a measured value for your hardware/grid.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Illustrative average grid carbon intensity: ~475 gCO2eq / kWh
#   1 kWh = 3.6e6 J  =>  475 g / 3.6e6 J = 1.319e-4 g/J = 131.9 microg/J
DEFAULT_CARBON_INTENSITY_UGCO2_PER_J = 131.9


@dataclass
class EnergyTracker:
    carbon_intensity: float = DEFAULT_CARBON_INTENSITY_UGCO2_PER_J
    total_joules: float = 0.0
    per_operator_joules: Dict[str, float] = field(default_factory=dict)
    calls_log: List[Dict] = field(default_factory=list)

    def record(self, operator: str, cost_j: float, delta_f: float):
        self.total_joules += cost_j
        self.per_operator_joules[operator] = self.per_operator_joules.get(operator, 0.0) + cost_j
        self.calls_log.append({"operator": operator, "cost_j": cost_j, "delta_f": delta_f})

    @property
    def co2_micrograms(self) -> float:
        return self.total_joules * self.carbon_intensity

    def score_per_joule(self, quality_pct: float) -> float:
        if self.total_joules <= 0:
            return 0.0
        return quality_pct / self.total_joules

    def remaining_budget(self, budget_j: float) -> float:
        return max(0.0, budget_j - self.total_joules)

    def overrun_ratio(self, budget_j: float) -> float:
        if budget_j <= 0:
            return 0.0
        return self.total_joules / budget_j

    def summary(self, quality_pct: float, budget_j: float) -> Dict:
        return {
            "energie_joules": self.total_joules,
            "co2_micrograms": self.co2_micrograms,
            "score_per_joule": self.score_per_joule(quality_pct),
            "qualite_pct": quality_pct,
            "budget_j": budget_j,
            "overrun_pct": 100.0 * (self.overrun_ratio(budget_j) - 1.0),
        }


class OperatorCostModel:
    """
    Estimates the energy cost (Joules) of a single operator call.

    In the absence of real hardware power measurements, cost is modeled
    as proportional to the amount of work performed (number of variables
    touched / clauses scanned), scaled by an empirical per-operator
    constant fit from `operator_profiles.csv`. This mirrors the
    dual-regime profiling described in section 1.2.2 of the notebook.
    """

    # Fallback constants (Joules per "unit of work") if no profiling CSV
    # is available. Calibrated to roughly reproduce the notebook's
    # reported ranges (clause_restart_greedy: 800-3600J, most expensive).
    DEFAULT_UNIT_COST = {
        "walksat": 0.02,
        "focused_vns": 0.05,
        "clause_restart_greedy": 0.35,
    }

    def __init__(self, unit_cost: Dict[str, float] = None):
        self.unit_cost = unit_cost or dict(self.DEFAULT_UNIT_COST)

    def estimate_cost(self, operator: str, work_units: int) -> float:
        c = self.unit_cost.get(operator, 0.05)
        return max(1e-6, c * work_units)
