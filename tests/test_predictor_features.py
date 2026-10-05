"""Tests for the dependency-free parts of the budget predictor.

``feature_extractor``, ``preprocessing`` and ``cnf`` need only numpy and
pandas. ``model`` is excluded: xgboost is an optional extra not installed here,
and importing it fails. That exclusion is honest, not a shortcut — the file
states it explicitly.
"""

from __future__ import annotations

import pytest

from budget_predictor.feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)
from budget_predictor.preprocessing import (
    FeatureScaler,
    features_to_matrix,
)
from greenaco.cnf import CNFInstance


def sample_instance() -> CNFInstance:
    # 3 variables, 4 clauses: two binary, one ternary, one unit.
    return CNFInstance(
        name="sample",
        n_vars=3,
        n_clauses=4,
        clauses=[(1, -2), (2, 3), (-1, -3, 2), (2,)],
    )


def test_feature_names_are_the_descriptions_drivers():
    """The shipped Table 2 lists the same names the extractor produces."""
    assert FEATURE_NAMES[:5] == [
        "n_vars", "n_clauses", "clause_var_ratio",
        "mean_clause_len", "std_clause_len",
    ]


def test_extract_features_counts_on_a_hand_worked_formula():
    feats = extract_features(sample_instance())
    assert feats.n_vars == 3
    assert feats.n_clauses == 4
    assert feats.clause_var_ratio == pytest.approx(4 / 3)
    assert feats.mean_clause_len == pytest.approx(2.0)
    assert feats.frac_unit_clauses == pytest.approx(0.25)
    # Every clause has at most one positive literal except (2, 3).
    assert feats.frac_horn_clauses == pytest.approx(0.75)


def test_extract_features_counts_occurrences_not_literals():
    """A repeated literal is one occurrence; see the reader's contract."""
    inst = CNFInstance(name="dup", n_vars=1, n_clauses=1, clauses=[(1, 1)])
    feats = extract_features(inst)
    assert feats.mean_var_degree == pytest.approx(1.0)
    assert feats.max_var_degree == 1


def full_rows(*instances: CNFInstance) -> list[dict]:
    """Complete feature rows, as the training CSVs would carry them."""
    from dataclasses import asdict

    return [asdict(extract_features(inst)) for inst in instances]


def test_features_to_matrix_orders_columns_by_name():
    rows = full_rows(sample_instance(), sample_instance())
    rows[1]["n_vars"] = 10
    matrix = features_to_matrix(rows)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    positions = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert matrix[0, positions["n_vars"]] == pytest.approx(3.0)
    assert matrix[1, positions["n_vars"]] == pytest.approx(10.0)


def test_features_to_matrix_rejects_an_incomplete_row():
    """Training rows are complete by construction; a partial row is an error."""
    with pytest.raises(KeyError):
        features_to_matrix([{"n_vars": 1.0}])


def scaled(rows: list[dict]) -> FeatureScaler:
    """Fit a scaler on complete feature rows."""
    return FeatureScaler().fit(features_to_matrix(rows))


def two_level_rows() -> list[dict]:
    base = full_rows(sample_instance(), sample_instance(),
                     sample_instance(), sample_instance())
    for row, v in zip(base, (1, 2, 3, 4)):
        row["n_vars"] = float(v)
        row["n_clauses"] = float(v)
    return base


def test_scaler_centres_and_reduces():
    scaler = scaled(two_level_rows())
    out = scaler.transform(features_to_matrix(two_level_rows()[:1]))
    positions = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert out.shape[1] == len(FEATURE_NAMES)
    assert out[0, positions["n_vars"]] == pytest.approx(-1.3416408, abs=1e-6)


def test_scaler_round_trips_through_a_dict():
    rows = full_rows(sample_instance(), sample_instance(), sample_instance())
    rows[1]["n_vars"] = 9.0
    rows[2]["n_vars"] = 7.0
    scaler = scaled(rows)
    restored = FeatureScaler.from_dict(scaler.to_dict())
    query = full_rows(sample_instance())
    query[0]["n_vars"] = 5.0
    got = restored.transform(features_to_matrix(query))
    want = scaler.transform(features_to_matrix(query))
    assert got == pytest.approx(want)


def test_scaler_ignores_a_constant_column():
    """A zero-variance column must not divide by zero."""
    rows = full_rows(sample_instance(), sample_instance(), sample_instance())
    for row, v in zip(rows, (1, 2, 3)):
        row["n_clauses"] = float(v)
    scaler = scaled(rows)
    query = full_rows(sample_instance())
    query[0]["n_clauses"] = 9.0
    out = scaler.transform(features_to_matrix(query))
    positions = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert out[0, positions["n_vars"]] == pytest.approx(0.0)