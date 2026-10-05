"""
scheduler.py
============
The EI/J (Expected Improvement per Joule) operator scheduler.

Green ACO does not choose operators round-robin. At each iteration the
scheduler picks the operator with the best ratio of expected quality gain to
expected energy cost, biased against expensive operators when the remaining
budget is small.

For each operator four statistics are tracked with an EWMA, giving a log-normal
model of its energy cost and a normal model of its quality gain::

    mu_df  <- a * mu_df  + (1 - a) * df_t
    mu_lnE <- a * mu_lnE + (1 - a) * ln(E_t)

Selection scores every eligible operator by

    Phi(o) = (df_tilde / E_tilde) * B / (B + E[E(o)])

where the tilded quantities are Thompson samples drawn from the learned
distributions (providing calibrated exploration) and the second factor is the
budget penalty.
"""

from __future__ import annotations

import math
import random
from typing import Dict


class OperatorStats:
    """Per-operator EWMA statistics backing the EI/J scheduler."""

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.mu_df = 0.0     # EWMA of the quality gain
        self.var_df = 1.0    # EWMA variance of the quality gain
        self.mu_lnE = 0.0    # EWMA of log energy (log-normal model)
        self.var_lnE = 1.0   # EWMA variance of log energy
        self.n_calls = 0     # how many times this operator has been called

    def seed_from_profile(self, mean_cost_j: float, std_cost_j: float,
                          mean_df: float, std_df: float,
                          n_profile_runs: int = 0) -> None:
        """Initialise all four parameters from offline profiling statistics.

        The energy side uses the exact log-normal variance implied by the
        coefficient of variation, ``ln(1 + cv^2)``, rather than an arbitrary
        constant, so profiling data yields a calibrated prior. The quality
        side floors the variance to keep Thompson sampling exploratory.

        ``n_profile_runs`` is deliberately *not* credited to ``n_calls``: the
        warm-up phase still calls every operator once, so each operator's live
        statistics supersede its prior.
        """
        safe_mean = max(mean_cost_j, 1e-9)
        self.mu_lnE = math.log(safe_mean)
        if safe_mean > 0 and std_cost_j > 0:
            cv2 = (std_cost_j / safe_mean) ** 2
            self.var_lnE = math.log(1.0 + cv2)
        else:
            self.var_lnE = 0.01

        self.mu_df = mean_df
        self.var_df = max(std_df ** 2, 1e-6)

    def update(self, delta_f: float, energy_j: float) -> None:
        """Fold one observation into the EWMA statistics."""
        a = self.alpha
        lnE = math.log(max(energy_j, 1e-12))
        if self.n_calls == 0:
            self.mu_df, self.mu_lnE = delta_f, lnE
        else:
            prev_mu_df, prev_mu_lnE = self.mu_df, self.mu_lnE
            self.mu_df = a * prev_mu_df + (1 - a) * delta_f
            self.var_df = a * self.var_df + (1 - a) * (delta_f - self.mu_df) ** 2
            self.mu_lnE = a * prev_mu_lnE + (1 - a) * lnE
            self.var_lnE = a * self.var_lnE + (1 - a) * (lnE - self.mu_lnE) ** 2
        self.n_calls += 1

    def sample_eij(self, rng: random.Random | None = None) -> float:
        """Thompson sample of EI/J = df_tilde / E_tilde."""
        rng = rng or random
        df_sample = rng.gauss(self.mu_df, math.sqrt(max(self.var_df, 1e-12)))
        lnE_sample = rng.gauss(self.mu_lnE, math.sqrt(max(self.var_lnE, 1e-12)))
        return df_sample / max(math.exp(lnE_sample), 1e-12)

    def expected_energy(self) -> float:
        """Log-normal mean E[E] = exp(mu_lnE + var_lnE / 2)."""
        return math.exp(self.mu_lnE + self.var_lnE / 2.0)

    def estimated_cost(self) -> float:
        """Point estimate of the next call's cost, used by the budget guard."""
        return math.exp(self.mu_lnE)

    def snapshot(self) -> Dict:
        return {
            "mu_df": self.mu_df,
            "var_df": self.var_df,
            "mu_lnE": self.mu_lnE,
            "var_lnE": self.var_lnE,
            "n_calls": self.n_calls,
        }

class EIJScheduler:
    """Chooses the next operator by maximising EI/J x budget penalty."""

    def __init__(self, operator_names, alpha_ewma: float = 0.9,
                 rng: random.Random | None = None):
        self.stats: Dict[str, OperatorStats] = {
            name: OperatorStats(alpha=alpha_ewma) for name in operator_names
        }
        self.rng = rng or random.Random()

    def seed_profiles(self, profiles: Dict[str, Dict]) -> None:
        """Seed priors from measured profiles, keyed by operator name."""
        for name, stat in self.stats.items():
            prof = profiles.get(name)
            if not prof:
                continue
            stat.seed_from_profile(
                mean_cost_j=prof.get("mean_cost_j", 50.0),
                std_cost_j=prof.get("std_cost_j", 10.0),
                mean_df=prof.get("mean_df", 0.0),
                std_df=prof.get("std_df", 1.0),
            )

    def select(self, eligible: Dict[str, OperatorStats],
               remaining_budget_j: float) -> str:
        """Return the name of the operator maximising Phi.

        ``eligible`` restricts the choice to operators whose estimated cost
        fits the remaining budget; the caller applies that guard.
        """
        best_name, best_phi = None, -1e18
        for name, stat in eligible.items():
            eij = stat.sample_eij(self.rng)
            e_expected = stat.expected_energy()
            pi = (remaining_budget_j / (remaining_budget_j + e_expected)
                  if remaining_budget_j > 0 else 0.0)
            phi = eij * pi
            if phi > best_phi:
                best_phi, best_name = phi, name
        return best_name

    def cheapest(self) -> str:
        """Name of the operator with the lowest estimated energy cost."""
        return min(self.stats, key=lambda o: self.stats[o].estimated_cost())

    def update(self, operator: str, delta_f: float, energy_j: float) -> None:
        self.stats[operator].update(delta_f, energy_j)

    def uncalled(self) -> list:
        return [name for name, s in self.stats.items() if s.n_calls == 0]

    def eligible(self, remaining_budget_j: float,
                 max_overrun_factor: float) -> Dict:
        """Operators whose estimated cost fits the remaining budget."""
        return {
            name: stat
            for name, stat in self.stats.items()
            if stat.estimated_cost() <= remaining_budget_j * max_overrun_factor
        }

    def snapshot(self) -> Dict:
        return {name: stat.snapshot() for name, stat in self.stats.items()}
