"""
selection.py
============
The EI/J (Expected Improvement per Joule) scheduler — the core
contribution of Green ACO (section 1.2.1 of the notebook).

    EI/J(o) = E[delta_f(o)] / E[E(o)]

Statistics are maintained with an EWMA (Exponentially Weighted Moving
Average) with smoothing factor alpha:

    mu_df   <- alpha * mu_df   + (1 - alpha) * delta_f_t
    mu_lnE  <- alpha * mu_lnE  + (1 - alpha) * ln(E_t)

Thompson Sampling injects controlled exploration: instead of using the
mean directly, we sample from the learned distribution:

    delta_f~  ~ N(mu_df, sigma_df^2)
    E~        ~ LogNormal(mu_lnE, sigma_lnE^2)

A budget penalty discourages expensive operators when remaining budget
is low:

    pi(o, B) = B / (B + E[E~(o)])

Final selection score:

    Phi(o) = (delta_f~ / E~) * pi(o, B)   =>   o* = argmax_o Phi(o)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OperatorStats:
    """EWMA-tracked statistics for one operator, per the notebook's
    dual-regime profiling. Can be seeded from `operator_profiles.csv`
    when `use_profiling=True` (see config.ABLATION_OPT_FLAGS)."""

    operator: str
    alpha: float = 0.9
    mu_df: float = 1.0
    var_df: float = 1.0
    mu_lnE: float = 0.0     # ln(Joules)
    var_lnE: float = 0.25
    n_calls: int = 0

    def update(self, delta_f: float, energy_j: float):
        energy_j = max(energy_j, 1e-6)
        ln_e = math.log(energy_j)

        if self.n_calls == 0:
            self.mu_df = delta_f
            self.mu_lnE = ln_e
        else:
            # EWMA mean update
            old_mu_df, old_mu_lnE = self.mu_df, self.mu_lnE
            self.mu_df = self.alpha * self.mu_df + (1 - self.alpha) * delta_f
            self.mu_lnE = self.alpha * self.mu_lnE + (1 - self.alpha) * ln_e
            # EWMA variance update (simple exponential smoothing of squared error)
            self.var_df = self.alpha * self.var_df + (1 - self.alpha) * (delta_f - old_mu_df) ** 2
            self.var_lnE = self.alpha * self.var_lnE + (1 - self.alpha) * (ln_e - old_mu_lnE) ** 2

        self.n_calls += 1

    def seed_from_profile(self, mean_cost_j: float, mean_df: float,
                           std_cost_j: float = None, std_df: float = None):
        """Seed EWMA stats from operator_profiles.csv (profiling prior)."""
        self.mu_df = mean_df
        self.var_df = (std_df or max(0.1 * abs(mean_df), 1e-3)) ** 2
        self.mu_lnE = math.log(max(mean_cost_j, 1e-6))
        self.var_lnE = (std_cost_j / max(mean_cost_j, 1e-6) if std_cost_j else 0.25) ** 2
        self.n_calls = 1


class EIJScheduler:
    """Expected-Improvement-per-Joule operator scheduler with Thompson
    sampling and a budget-aware penalty."""

    def __init__(self, operator_names: List[str], alpha_ewma: float = 0.9,
                 rng: Optional[random.Random] = None):
        self.stats: Dict[str, OperatorStats] = {
            name: OperatorStats(operator=name, alpha=alpha_ewma) for name in operator_names
        }
        self.rng = rng or random.Random()

    def seed_profiles(self, profile_means: Dict[str, Dict[str, float]]):
        """profile_means: {operator: {"mean_cost_j":..., "mean_df":..., ...}}"""
        for name, stat in self.stats.items():
            if name in profile_means:
                p = profile_means[name]
                stat.seed_from_profile(p.get("mean_cost_j", 1.0), p.get("mean_df", 1.0),
                                        p.get("std_cost_j"), p.get("std_df"))

    def _sample_operator(self, stat: OperatorStats):
        sampled_df = self.rng.gauss(stat.mu_df, math.sqrt(max(stat.var_df, 1e-9)))
        sampled_ln_e = self.rng.gauss(stat.mu_lnE, math.sqrt(max(stat.var_lnE, 1e-9)))
        sampled_e = math.exp(sampled_ln_e)
        return sampled_df, max(sampled_e, 1e-6)

    def budget_penalty(self, expected_e: float, remaining_budget: float) -> float:
        return remaining_budget / (remaining_budget + expected_e)

    def select(self, remaining_budget: float) -> str:
        """Returns the name of the operator o* = argmax_o Phi(o)."""
        best_op, best_phi = None, -math.inf
        for name, stat in self.stats.items():
            sampled_df, sampled_e = self._sample_operator(stat)
            expected_e = math.exp(stat.mu_lnE)
            pi = self.budget_penalty(expected_e, remaining_budget)
            phi = (sampled_df / sampled_e) * pi
            if phi > best_phi:
                best_phi, best_op = phi, name
        return best_op

    def update(self, operator: str, delta_f: float, energy_j: float):
        self.stats[operator].update(delta_f, energy_j)

    def snapshot(self) -> Dict[str, Dict]:
        return {
            name: {"mu_df": s.mu_df, "mu_lnE": s.mu_lnE, "n_calls": s.n_calls}
            for name, s in self.stats.items()
        }
