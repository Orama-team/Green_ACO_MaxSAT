"""Profiling, manifest handling, and the shipped artifacts."""

from __future__ import annotations

import pandas as pd
import pytest

from greenaco import paths
from greenaco.config import REGIME_EXPLOITATION
from greenaco.energy import EnergyMeter
from greenaco.profile import profile_operator, profiles_for, load_profiles

from conftest import make_benchmark

OPERATORS = ["walksat", "focused_vns", "clause_restart_greedy"]


def test_two_regimes_are_defined():
    assert set(REGIME_EXPLOITATION) == {"easy", "hard"}
    assert REGIME_EXPLOITATION["easy"] < REGIME_EXPLOITATION["hard"]


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("regime", ["easy", "hard"])
def test_profile_row_has_required_fields(operator, regime):
    bench = make_benchmark(n_vars=60, n_clauses=200, seed=1)
    row = profile_operator(bench, operator, regime, meter=EnergyMeter(),
                           n_steps=5, seed=42)
    for key in ("benchmark", "operator", "regime", "mean_cost_j",
                "std_cost_j", "mean_df", "std_df", "n_runs"):
        assert key in row
    assert row["mean_cost_j"] > 0
    assert row["n_runs"] == 5


def test_profiles_for_groups_by_operator_and_regime():
    rows = [
        {"benchmark": "b1", "operator": "walksat", "regime": "easy",
         "mean_cost_j": 10.0, "std_cost_j": 1.0, "mean_df": 5.0,
         "std_df": 1.0},
        {"benchmark": "b1", "operator": "walksat", "regime": "hard",
         "mean_cost_j": 20.0, "std_cost_j": 2.0, "mean_df": 1.0,
         "std_df": 0.5},
        {"benchmark": "other", "operator": "walksat", "regime": "easy",
         "mean_cost_j": 99.0, "std_cost_j": 1.0, "mean_df": 9.0,
         "std_df": 1.0},
    ]
    grouped = profiles_for("b1", rows)
    assert set(grouped) == {"walksat"}
    assert grouped["walksat"]["easy"]["mean_cost_j"] == 10.0
    assert grouped["walksat"]["hard"]["mean_df"] == 1.0


def test_load_profiles_keeps_uncovered_benchmarks():
    lookup = load_profiles([], ["a", "b"])
    assert lookup == {"a": {}, "b": {}}


# ── shipped artifacts ────────────────────────────────────────────────────────


SHIPMENT_REQUIRED = [
    "final_runs_all54.csv",
    "final_summary_all54.csv",
    "operator_profiles_all54.csv",
    "comparison_runs_core.csv",
]


@pytest.mark.parametrize("name", SHIPMENT_REQUIRED)
def test_shipped_artifact_exists_and_is_populated(name):
    path = paths.SHIPPED / name
    if not path.exists():
        pytest.skip(f"{name} not shipped in this checkout")
    frame = pd.read_csv(path)
    assert not frame.empty


def test_manifest_is_present_and_well_formed():
    manifest = paths.DATA / "manifest.csv"
    if not manifest.exists():
        pytest.skip("benchmark manifest not present")
    frame = pd.read_csv(manifest)
    assert not frame["benchmark"].duplicated().any()
    assert frame["best_known"].notna().all()
    assert len(frame) == 54
    assert int(frame["mandatory"].sum()) == 4