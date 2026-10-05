"""
config.py
=========
Run configuration and the tuned hyper-parameters.

GreenACOConfig carries every knob the solver exposes. The defaults are the
values obtained by the finetuning stage (``finetune_d_best_params.csv``), not
hand-picked guesses, so a plain run reproduces the reported configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

# Energy budgets (Joules) explored throughout the study.
BUDGETS_JOULES = [400.0, 1000.0, 2000.0]

# Hyper-parameters identified by the finetuning stage. These are the values in
# configs/best_params.yaml and the fallback when no config is supplied.
TUNED_PARAMS: Dict[str, float] = {
    "alpha_ewma": 0.632961,
    "max_overrun_factor": 1.176985,
    "stagnation_window": 8,
    "stagnation_epsilon": 0.185932,
    "best_stagnation_limit": 13,
    "plateau_tolerance_pct": 0.008385,
    "rho_base": 0.250334,
    "rho_stagnant": 0.038892,
}

# Optuna search space used by the `full` finetuning mode.
OPTUNA_SEARCH_SPACE: Dict[str, tuple] = {
    "max_overrun_factor": (1.0, 3.0),
    "stagnation_window": (5, 20),
    "stagnation_epsilon": (0.1, 2.0),
    "best_stagnation_limit": (5, 30),
    "plateau_tolerance_pct": (0.001, 0.02),
    "alpha_ewma": (0.5, 0.99),
    "rho_base": (0.01, 0.3),
    "rho_stagnant": (0.005, 0.1),
}

# Operator pool.
OPERATOR_POOL = ["walksat", "focused_vns", "clause_restart_greedy"]

# Operators implemented as flip moves, eligible for the sparse pheromone update.
FLIP_OPS = {"walksat", "bandit_var_selection"}

# Dual-regime profiling: exploitation rate of the greedy constructor.
REGIME_EXPLOITATION = {"easy": 0.50, "hard": 0.95}

# Fraction of the quality gap to the ideal above which the `hard` priors take
# over. Below this threshold the `easy` priors are used instead.
REGIME_QUALITY_THRESHOLD = 0.70

# Profile rows used when a benchmark has no measured profile.
FALLBACK_PROFILE = {
    "mean_cost_j": 50.0,
    "std_cost_j": 10.0,
    "mean_df": 0.0,
    "std_df": 1.0,
    "n_runs": 3,
}

# Ablation axes. Axis A removes one operator at a time; axis B disables one
# green mechanism at a time while holding the operator pool fixed.
#
# Names match the configurations tabulated in the paper (Table 9) so that a
# generated table can be compared against it directly.
ABLATION_OPERATOR_CONFIGS: Dict[str, list] = {
    "full": list(OPERATOR_POOL),
    "no_walksat": ["focused_vns", "clause_restart_greedy"],
    "no_focused_vns": ["walksat", "clause_restart_greedy"],
    "no_greedy_restart": ["walksat", "focused_vns"],
}

ABLATION_GREEN_FLAGS: Dict[str, dict] = {
    "full": {},
    "no_sparse_ph": {"use_sparse_update": False},
    "no_profiling": {"use_profiling": False},
    "no_overrun_fix": {"max_overrun_factor": 9999.0},
    "no_caching": {"use_pheromone_cache": False},
}


@dataclass
class GreenACOConfig:
    """Full runtime configuration for one Green ACO run."""

    # Energy budget in Joules.
    budget_j: float = 1000.0

    # Seed for every stochastic decision, making a run reproducible.
    seed: int = 42

    # Scheduler / EI-per-Joule.
    alpha_ewma: float = TUNED_PARAMS["alpha_ewma"]
    max_overrun_factor: float = TUNED_PARAMS["max_overrun_factor"]

    # Stagnation detection and early stopping.
    stagnation_window: int = TUNED_PARAMS["stagnation_window"]
    stagnation_epsilon: float = TUNED_PARAMS["stagnation_epsilon"]
    best_stagnation_limit: int = TUNED_PARAMS["best_stagnation_limit"]
    plateau_tolerance_pct: float = TUNED_PARAMS["plateau_tolerance_pct"]

    # Pheromone evaporation.
    rho_base: float = TUNED_PARAMS["rho_base"]
    rho_stagnant: float = TUNED_PARAMS["rho_stagnant"]

    # Initial pheromone level and greedy exploitation rate.
    tau0: float = 0.5
    exploitation_rate: float = 0.95

    # Target quality as a fraction of clauses (1.0 == solve completely).
    target_quality: float = 1.0

    # Evaluation backend. "rescan" is the reference implementation and the default.
    backend: str = "rescan"

    # Mechanism switches, all toggled by the ablation axis B.
    use_sparse_update: bool = True
    use_frugal_skip: bool = True
    use_profiling: bool = True

    # Operator pool for this run.
    active_operators: list = field(default_factory=lambda: list(OPERATOR_POOL))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GreenACOConfig":
        """Build a config from a mapping, ignoring unknown keys."""
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)