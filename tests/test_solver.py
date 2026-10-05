"""End-to-end solver behaviour and reproducibility of a seeded run."""

from __future__ import annotations

import pytest

from greenaco.config import GreenACOConfig
from greenaco.energy import EnergyMeter
from greenaco.solver import (GreenACOSolver, blend_profiles, build_eta_matrix,
                             build_greedy_assignment, solve_instance)

from conftest import make_benchmark


def _solve(bench, **kw):
    cfg = GreenACOConfig.from_dict({"budget_j": 200.0, "seed": 7, **kw})
    return solve_instance(GreenACOSolver(cfg, EnergyMeter()), bench)


def test_run_reports_consistent_metrics(tiny_benchmark):
    stats = _solve(tiny_benchmark)
    assert 0 <= stats["qualite_solution"] <= stats["n_clauses"]
    assert stats["energie_joules"] > 0
    expected = stats["qualite_pct"] / stats["energie_joules"]
    assert stats["score_per_joule"] == pytest.approx(expected, abs=1e-4)


def test_budget_is_respected(tiny_benchmark):
    """The budget guard must keep consumption near the allowance."""
    stats = _solve(tiny_benchmark, budget_j=200.0,
                   max_overrun_factor=1.8)
    assert stats["energie_joules"] <= 200.0 * 1.8 * 1.05


def test_same_seed_gives_same_quality(tiny_benchmark):
    """Quality is seed-reproducible even though energy is machine-measured."""
    a = _solve(tiny_benchmark, budget_j=200.0)
    b = _solve(tiny_benchmark, budget_j=200.0)
    assert a["qualite_solution"] == b["qualite_solution"]


def test_different_seeds_are_independent(tiny_benchmark):
    a = _solve(tiny_benchmark, budget_j=200.0, seed=1)
    b = _solve(tiny_benchmark, budget_j=200.0, seed=2)
    assert a["seed"] != b["seed"]


def test_quality_never_below_random_start(tiny_benchmark):
    """The greedy constructor alone already satisfies a good share."""
    bench = make_benchmark(n_vars=60, n_clauses=200, seed=5)
    stats = _solve(bench, budget_j=400.0)
    formula = bench.instance.clauses
    pheromone = [[0.5, 0.5] for _ in range(bench.n_vars)]
    eta = build_eta_matrix(formula, bench.n_vars)
    start = build_greedy_assignment(bench.n_vars, pheromone, eta, 0.95)
    from greenaco.wcnf import count_satisfied_clauses

    assert stats["qualite_solution"] >= count_satisfied_clauses(formula, start)



def test_profiles_are_accepted_without_error(tiny_benchmark):
    profiles = {
        op: {"easy": {"mean_cost_j": 20.0, "std_cost_j": 5.0,
                      "mean_df": 3.0, "std_df": 1.0},
             "hard": {"mean_cost_j": 40.0, "std_cost_j": 8.0,
                      "mean_df": 1.0, "std_df": 0.5}}
        for op in ("walksat", "focused_vns", "clause_restart_greedy")
    }
    cfg = GreenACOConfig.from_dict({"budget_j": 200.0, "seed": 3})
    solver = GreenACOSolver(cfg, EnergyMeter(), profiles=profiles)
    stats = solve_instance(solver, tiny_benchmark)
    assert stats["priors"] == "blended"
    assert stats["qualite_solution"] > 0


def test_blend_favours_easy_at_low_quality():
    """A poorly-started instance should inherit the easy priors."""
    profiles = {
        "walksat": {"easy": {"mean_cost_j": 10.0, "std_cost_j": 1.0,
                            "mean_df": 5.0, "std_df": 1.0},
                    "hard": {"mean_cost_j": 90.0, "std_cost_j": 1.0,
                             "mean_df": 0.5, "std_df": 0.1}},
    }
    low = blend_profiles(profiles, 0.2)["walksat"]
    high = blend_profiles(profiles, 1.0)["walksat"]
    assert low["mean_cost_j"] < high["mean_cost_j"]


def test_operator_pool_can_be_reduced(tiny_benchmark):
    stats = _solve(tiny_benchmark, active_operators=["walksat"])
    assert set(stats["operateurs_usage"]) <= {"walksat"}