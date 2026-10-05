"""The EI/J scheduler: EWMA statistics, Thompson sampling, budget guard."""

from __future__ import annotations

import math
import random

import pytest

from greenaco.scheduler import EIJScheduler, OperatorStats


def test_first_observation_initialises_exactly():
    stats = OperatorStats(alpha=0.9)
    stats.update(delta_f=7.0, energy_j=20.0)
    assert stats.mu_df == 7.0
    assert stats.mu_lnE == pytest.approx(math.log(20.0))
    assert stats.n_calls == 1


def test_ewma_smooths_subsequent_observations():
    stats = OperatorStats(alpha=0.9)
    stats.update(0.0, 10.0)
    stats.update(10.0, 20.0)
    # 0.9 * 0 + 0.1 * 10
    assert stats.mu_df == pytest.approx(1.0)


def test_expected_energy_matches_lognormal_mean():
    stats = OperatorStats()
    stats.mu_lnE, stats.var_lnE = 2.0, 0.5
    assert stats.expected_energy() == pytest.approx(
        math.exp(2.0 + 0.25))


def test_profile_seed_sets_all_four_parameters():
    stats = OperatorStats()
    stats.seed_from_profile(mean_cost_j=50.0, std_cost_j=10.0,
                            mean_df=12.0, std_df=3.0)
    assert stats.mu_lnE == pytest.approx(math.log(50.0))
    # exact log-normal variance from the coefficient of variation
    assert stats.var_lnE == pytest.approx(math.log(1 + (10.0 / 50.0) ** 2))
    assert stats.mu_df == 12.0
    assert stats.var_df == pytest.approx(9.0)


def test_profile_seed_does_not_credit_warmup():
    """Warm-up must still run, so live statistics supersede the prior."""
    stats = OperatorStats()
    stats.seed_from_profile(50.0, 10.0, 12.0, 3.0, n_profile_runs=20)
    assert stats.n_calls == 0


def test_degenerate_profile_stays_explorable():
    stats = OperatorStats()
    stats.seed_from_profile(mean_cost_j=0.0, std_cost_j=0.0,
                            mean_df=0.0, std_df=0.0)
    assert stats.var_lnE > 0
    assert stats.var_df > 0


def test_scheduler_starts_with_every_operator_uncalled():
    sched = EIJScheduler(["a", "b"], rng=random.Random(0))
    assert sorted(sched.uncalled()) == ["a", "b"]


def test_budget_guard_excludes_expensive_operators():
    sched = EIJScheduler(["cheap", "pricey"], rng=random.Random(0))
    sched.stats["cheap"].mu_lnE = math.log(1.0)
    sched.stats["pricey"].mu_lnE = math.log(1000.0)
    eligible = sched.eligible(remaining_budget_j=10.0, max_overrun_factor=1.5)
    assert set(eligible) == {"cheap"}


def test_cheapest_is_chosen_when_nothing_fits():
    sched = EIJScheduler(["cheap", "pricey"], rng=random.Random(0))
    sched.stats["cheap"].mu_lnE = math.log(1.0)
    sched.stats["pricey"].mu_lnE = math.log(1000.0)
    eligible = sched.eligible(remaining_budget_j=0.01, max_overrun_factor=1.0)
    assert eligible == {}
    assert sched.cheapest() == "cheap"


def test_selection_prefers_the_efficient_operator():
    sched = EIJScheduler(["good", "bad"], rng=random.Random(0))
    sched.stats["good"].update(delta_f=50.0, energy_j=5.0)
    sched.stats["good"].update(delta_f=50.0, energy_j=5.0)
    sched.stats["bad"].update(delta_f=1.0, energy_j=900.0)
    sched.stats["bad"].update(delta_f=1.0, energy_j=900.0)
    picks = {sched.select(sched.stats, remaining_budget_j=500.0)
             for _ in range(30)}
    assert picks == {"good"}


def test_sampling_is_reproducible_with_a_seeded_rng():
    a = OperatorStats(); a.update(4.0, 12.0); a.update(5.0, 11.0)
    b = OperatorStats(); b.update(4.0, 12.0); b.update(5.0, 11.0)
    s1 = [a.sample_eij(random.Random(99)) for _ in range(5)]
    s2 = [b.sample_eij(random.Random(99)) for _ in range(5)]
    assert s1 == s2


def test_snapshot_exposes_call_counts():
    sched = EIJScheduler(["a"], rng=random.Random(0))
    sched.update("a", 1.0, 2.0)
    assert sched.snapshot()["a"]["n_calls"] == 1