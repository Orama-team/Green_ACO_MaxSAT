"""
Shared fixtures: small synthetic instances with known structure.

Everything runs on generated formulas rather than the benchmark corpus, so the
tests stay fast and independent of the instance files being present.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenaco.data import Benchmark  # noqa: E402
from greenaco.wcnf import Instance  # noqa: E402


def make_formula(n_vars: int, n_clauses: int, seed: int = 0):
    """Random 3-SAT-style formula, deterministic for a given seed."""
    rng = random.Random(seed)
    clauses = []
    for _ in range(n_clauses):
        vs = rng.sample(range(1, n_vars + 1), 3)
        clauses.append(tuple(v if rng.random() < 0.5 else -v for v in vs))
    return clauses


def make_benchmark(n_vars: int = 120, n_clauses: int = 400, seed: int = 0,
                   name: str = "synthetic") -> Benchmark:
    formula = make_formula(n_vars, n_clauses, seed)
    inst = Instance(benchmark=name, n_vars=n_vars, clauses=formula)
    return Benchmark(benchmark=name, domain="synthetic", category="random",
                     file=f"{name}.cnf", n_vars=n_vars,
                     n_clauses=len(formula), best_known=0.0, mandatory=False,
                     instance=inst)


@pytest.fixture
def small_benchmark() -> Benchmark:
    return make_benchmark()


@pytest.fixture
def tiny_benchmark() -> Benchmark:
    return make_benchmark(n_vars=40, n_clauses=120, seed=3)