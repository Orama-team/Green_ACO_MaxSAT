"""
config.py
=========
Central configuration for Green ACO (MAX-SAT).

The default values below are the empirical TP5 values reported in the
notebook ("valeurs empiriques TP5") for when `optuna_results.csv` is not
available. The `OPTUNA_SEARCH_SPACE` mirrors the search space described
in section 2.2 of the notebook ("Protocole de fine-tuning et meilleurs
paramètres").
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ── Operator pool (section 1.3 of the notebook) ──────────────────────────
OPERATOR_POOL = ["walksat", "focused_vns", "clause_restart_greedy"]

# ── Optuna search space (section 2.2) ─────────────────────────────────────
OPTUNA_SEARCH_SPACE: Dict[str, Tuple[float, float]] = {
    "alpha_ewma": (0.5, 0.99),
    "max_overrun_factor": (1.0, 3.0),
    "stagnation_window": (5, 20),
    "stagnation_epsilon": (0.1, 2.0),
    "best_stagnation_limit": (5, 30),
    "plateau_tolerance_pct": (0.001, 0.02),
    "rho_base": (0.01, 0.3),
    "rho_stagnant": (0.005, 0.1),
}

# ── Empirical defaults (fallback when optuna_results.csv is absent) ──────
DEFAULT_BEST_PARAMS: Dict[str, float] = {
    "alpha_ewma": 0.9,
    "max_overrun_factor": 1.8,
    "stagnation_window": 10,
    "stagnation_epsilon": 0.5,
    "best_stagnation_limit": 15,
    "plateau_tolerance_pct": 0.005,
    "rho_base": 0.1,
    "rho_stagnant": 0.02,
}

# ── Budgets used throughout the ablation / benchmark study ───────────────
BUDGETS_JOULES = [400, 1000, 2000]

# ── Ablation configurations (section 2.4) ─────────────────────────────────
ABLATION_CONFIGS: Dict[str, list] = {
    "full_v4": ["walksat", "focused_vns", "clause_restart_greedy"],
    "no_walksat": ["focused_vns", "clause_restart_greedy"],
    "no_focused_vns": ["walksat", "clause_restart_greedy"],
    "no_greedy_restart": ["walksat", "focused_vns"],
}

ABLATION_OPT_FLAGS: Dict[str, Dict] = {
    # use_sparse_update    -> O(k) pheromone update per iteration instead of O(n)
    # use_pheromone_cache  -> inter-budget warm start (reuse tau learned on B1 for B2)
    # use_profiling        -> calibrated priors from operator_profiles.csv
    # max_overrun_factor   -> anti budget-overrun guard
    "full_v4": {"use_sparse_update": True, "use_pheromone_cache": True,
                "use_profiling": True, "max_overrun_factor": 1.8},
    "no_sparse_ph": {"use_sparse_update": False, "use_pheromone_cache": True,
                      "use_profiling": True, "max_overrun_factor": 1.8},
    "no_caching": {"use_sparse_update": True, "use_pheromone_cache": False,
                   "use_profiling": True, "max_overrun_factor": 1.8},
    "no_profiling": {"use_sparse_update": True, "use_pheromone_cache": True,
                      "use_profiling": False, "max_overrun_factor": 1.8},
    "no_overrun_fix": {"use_sparse_update": True, "use_pheromone_cache": True,
                        "use_profiling": True, "max_overrun_factor": 9999},
}

# ── Green ACO regimes (q0 = ACS pseudo-random-proportional threshold) ────
REGIMES = {
    "easy": {"q0": 0.5},
    "hard": {"q0": 0.95},
}


@dataclass
class GreenACOConfig:
    """Full runtime configuration for a Green ACO run."""

    # Scheduler / EI-per-Joule
    alpha_ewma: float = DEFAULT_BEST_PARAMS["alpha_ewma"]
    max_overrun_factor: float = DEFAULT_BEST_PARAMS["max_overrun_factor"]

    # Stagnation detection & early stopping
    stagnation_window: int = int(DEFAULT_BEST_PARAMS["stagnation_window"])
    stagnation_epsilon: float = DEFAULT_BEST_PARAMS["stagnation_epsilon"]
    best_stagnation_limit: int = int(DEFAULT_BEST_PARAMS["best_stagnation_limit"])
    plateau_tolerance_pct: float = DEFAULT_BEST_PARAMS["plateau_tolerance_pct"]

    # Pheromone evaporation
    rho_base: float = DEFAULT_BEST_PARAMS["rho_base"]
    rho_stagnant: float = DEFAULT_BEST_PARAMS["rho_stagnant"]

    # ACS-style pseudo-random-proportional rule
    q0: float = 0.5  # regime dependent, see REGIMES

    # Operator pool
    active_operators: list = field(default_factory=lambda: list(OPERATOR_POOL))

    # Feature flags (see ABLATION_OPT_FLAGS)
    use_sparse_update: bool = True
    use_pheromone_cache: bool = True
    use_profiling: bool = True

    # Budget (Joules) allocated to the run
    budget_j: float = 1000.0

    # Random seed
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Dict) -> "GreenACOConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
