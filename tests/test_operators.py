"""Operators: they never degrade the assignment, and the two backends agree.

The backend-equivalence test is the important one. ``IndexedBackend`` exists
only as a speed optimisation, so if it ever changed a result the default would
be silently wrong. Both are exercised under the same seed and required to return
identical assignments and identical gains.
"""

from __future__ import annotations

import random

import pytest

from greenaco.operators import (OPERATOR_FUNCTIONS, IndexedBackend,
                                RescanBackend, build_operator_pool,
                                update_pheromone, update_pheromone_sparse)
from greenaco.solver import build_eta_matrix, build_greedy_assignment
from greenaco.wcnf import count_satisfied_clauses

from conftest import make_benchmark

BACKENDS = [RescanBackend, IndexedBackend]
OPERATORS = list(OPERATOR_FUNCTIONS)


def _prepare(backend, bench):
    backend.prepare(bench.instance.clauses, bench.n_vars)
    return build_operator_pool(OPERATORS, backend)


def _run(backend, bench, operator, seed=11):
    """Run one operator from a fresh, deterministic starting state."""
    pool = _prepare(backend, bench)
    formula, n_vars = bench.instance.clauses, bench.n_vars
    pheromone = [[0.5, 0.5] for _ in range(n_vars)]
    eta = build_eta_matrix(formula, n_vars)

    random.seed(seed)  # operators draw from the global RNG
    assignment = build_greedy_assignment(n_vars, pheromone, eta, 0.95,
                                         random.Random(seed))
    before = count_satisfied_clauses(formula, assignment)
    new_assignment, delta = pool[operator](formula, n_vars, pheromone, eta,
                                           assignment, dict(assignment),
                                           bench.n_clauses)
    after = count_satisfied_clauses(formula, new_assignment)
    assert delta == after - before, "operator delta must match the actual change"
    return new_assignment, delta


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_operator_never_degrades(operator, backend_cls, small_benchmark):
    assignment, delta = _run(backend_cls(), small_benchmark, operator)
    assert delta >= 0


@pytest.mark.parametrize("operator", OPERATORS)
def test_backends_agree_exactly(operator, small_benchmark):
    ref_assignment, ref_delta = _run(RescanBackend(), small_benchmark, operator)
    idx_assignment, idx_delta = _run(IndexedBackend(), small_benchmark, operator)
    assert idx_delta == ref_delta
    assert idx_assignment == ref_assignment


@pytest.mark.parametrize("operator", OPERATORS)
def test_assignment_keys_cover_every_variable(operator, small_benchmark):
    assignment, _ = _run(RescanBackend(), small_benchmark, operator)
    assert set(assignment) == set(range(1, small_benchmark.n_vars + 1))


def test_operators_leave_formula_untouched(small_benchmark):
    before = list(small_benchmark.instance.clauses)
    _run(RescanBackend(), small_benchmark, "focused_vns")
    assert small_benchmark.instance.clauses == before


def test_pheromone_update_reinforces_assignment(small_benchmark):
    n = small_benchmark.n_vars
    pheromone = [[1.0, 1.0] for _ in range(n)]
    assignment = {v: True for v in range(1, n + 1)}
    # delta = n_satisfied / n_clauses = 90/100, against 10% evaporation.
    update_pheromone(pheromone, assignment, 90, 100, evaporation_rate=0.1)
    assert pheromone[0][1] == pytest.approx(0.9 + 0.9)  # reinforced
    assert pheromone[0][0] == pytest.approx(0.9)         # only evaporated


def test_sparse_update_only_touches_changed_variables(small_benchmark):
    n = small_benchmark.n_vars
    pheromone = [[1.0, 1.0] for _ in range(n)]
    previous = {v: False for v in range(1, n + 1)}
    updated = dict(previous)
    updated[1] = True  # a single flip

    touched = update_pheromone_sparse(pheromone, previous, updated, 10, 100,
                                      evaporation_rate=0.0, n_vars=n)
    assert touched == 1
    assert pheromone[0][1] > 1.0   # the flipped variable reinforced
    assert pheromone[1][1] == 1.0   # untouched otherwise


def test_pool_contains_only_requested_operators(small_benchmark):
    pool = _prepare(RescanBackend(), small_benchmark)
    assert set(pool) == set(OPERATORS)
    subset = build_operator_pool(["walksat"], RescanBackend())
    assert set(subset) == {"walksat"}


def test_satisfied_formula_yields_no_gain(small_benchmark):
    """With every clause satisfied there is nothing left to improve."""
    bench = make_benchmark(n_vars=10, n_clauses=5, seed=1)
    formula = [(1,), (2,), (3,)]
    bench.instance.clauses = formula
    bench.n_clauses = len(formula)
    pool = _prepare(RescanBackend(), bench)
    assignment = {1: True, 2: True, 3: True}
    for v in range(4, bench.n_vars + 1):
        assignment[v] = False
    pheromone = [[0.5, 0.5] for _ in range(bench.n_vars)]
    eta = build_eta_matrix(formula, bench.n_vars)
    for operator in OPERATORS:
        new_assignment, delta = pool[operator](
            formula, bench.n_vars, pheromone, eta, assignment,
            dict(assignment), len(formula))
        assert delta >= 0