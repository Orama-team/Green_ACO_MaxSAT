"""The comparison methods: they solve, and they report consistently."""

from __future__ import annotations

import pytest

from greenaco.comparison.aco import (aco_sat, aco_sat_elitist, acs_sat,
                                     mmas_sat)
from greenaco.comparison.harness import run_method
from greenaco.comparison.registry import (ACO_FUNCTIONS, EXCLUDED, FAMILY,
                                          GA_METHODS, METHOD_ORDER,
                                          clean_params, get_solver)
from greenaco.energy import EnergyMeter
from greenaco.wcnf import count_satisfied_clauses

from conftest import make_benchmark

COMPARISON_METHODS = [m for m in METHOD_ORDER if m != "Green ACO"]


def test_exactly_seven_methods_are_compared():
    assert len(COMPARISON_METHODS) == 7
    assert "Green ACO" in METHOD_ORDER


def test_non_terminating_methods_are_excluded():
    assert set(EXCLUDED) == {"NL-ACO", "FACO"} or "NL-ACO" in EXCLUDED
    for name in EXCLUDED:
        assert name not in COMPARISON_METHODS


def test_every_method_has_a_family():
    for name in METHOD_ORDER:
        assert FAMILY.get(name)


@pytest.mark.parametrize("name", COMPARISON_METHODS)
def test_method_produces_a_valid_solution(name):
    bench = make_benchmark(n_vars=60, n_clauses=150, seed=2)
    row = run_method(bench, name, meter=EnergyMeter(), seed=42, timeout_s=180)
    assert row["qualite_solution"] is not None
    assert 0 <= row["qualite_solution"] <= bench.n_clauses
    # compute_green_metrics rounds to 4 dp for readability, so compare loosely.
    assert row["qualite_pct"] == pytest.approx(
        100.0 * row["qualite_solution"] / bench.n_clauses, abs=1e-3)
    assert row["energie_joules"] > 0


@pytest.mark.parametrize("name", ["AS-SAT", "AS-SAT Elitiste", "MMAS", "ACS"])
def test_aco_quality_matches_an_independent_recount(name, ):
    """The reported quality must match a fresh count of the assignment."""
    bench = make_benchmark(n_vars=60, n_clauses=150, seed=4)
    fn = get_solver(name, seed=42)
    assignment, stats = fn(bench.instance.clauses, bench.n_vars)
    assert count_satisfied_clauses(bench.instance.clauses,
                                   assignment) == stats["qualite_solution"]


def test_unknown_method_is_rejected():
    with pytest.raises(KeyError):
        get_solver("No Such Method")


def test_nan_parameters_are_dropped():
    """A literal NaN mutation rate would silently disable mutation."""
    cleaned = clean_params({"pop_size": 50, "mutation_prob": float("nan")})
    assert "mutation_prob" not in cleaned
    assert cleaned["pop_size"] == 50


def test_ga_parameters_are_recorded():
    row = run_method(make_benchmark(n_vars=40, n_clauses=100, seed=6),
                     "AG Classique", meter=EnergyMeter(), seed=1, timeout_s=120)
    assert row["params"]["pop_size"] == 50
    assert row["params"]["max_gen"] == 500


def test_aco_functions_are_importable():
    assert set(ACO_FUNCTIONS) == {"AS-SAT", "AS-SAT Elitiste", "MMAS", "ACS"}
    assert all(callable(f) for f in
               (aco_sat, aco_sat_elitist, mmas_sat, acs_sat))


def test_ga_methods_are_listed():
    assert GA_METHODS == ["AG Classique", "AG Adaptatif", "AG + KC"]