"""Statistical tests, ranking and context standardisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greenaco.metrics import kruskal_wallis, operator_groups
from greenaco.metrics import pairwise_mannwhitney, rank_methods
from greenaco.metrics import run_operator_tests, wilcoxon_paired
from greenaco.metrics import zscore_by_context
from greenaco.metrics import tests_to_frame as _flatten_tests


def _profiles():
    return pd.DataFrame({
        "operator": ["walksat"] * 10 + ["focused_vns"] * 10
                    + ["clause_restart_greedy"] * 10,
        "mean_df": list(range(10)) + list(range(5, 15)) + list(range(50, 60)),
        "regime": ["easy"] * 30,
    })


def test_kruskal_detects_separated_groups():
    groups = {"a": [1, 2, 3, 4, 5], "b": [50, 51, 52, 53, 54],
              "c": [100, 101, 102, 103, 104]}
    assert kruskal_wallis(groups)["significant"]


def test_kruskal_finds_nothing_in_overlapping_groups():
    groups = {"a": [1, 2, 3, 4, 5, 6, 7, 8],
              "b": [2, 3, 4, 5, 6, 7, 8, 9]}
    assert not kruskal_wallis(groups)["significant"]


def test_kruskal_tolerates_a_single_group():
    assert kruskal_wallis({"only": [1, 2, 3]})["significant"] is False


def test_pairwise_tests_cover_every_pair():
    groups = {"a": [1, 2, 3], "b": [10, 11, 12], "c": [20, 21, 22]}
    assert len(pairwise_mannwhitney(groups)) == 3  # 3 choose 2


def test_pairwise_detects_a_large_separation():
    """With enough samples the difference is significant at alpha = 0.05."""
    groups = {"a": list(range(20)), "b": list(range(100, 120))}
    assert pairwise_mannwhitney(groups)[0]["significant"]


def test_wilcoxon_requires_equal_lengths():
    assert wilcoxon_paired([1, 2], [1, 2, 3])["significant"] is False


def test_wilcoxon_on_paired_samples():
    assert wilcoxon_paired([1, 2, 3, 4], [2, 3, 4, 5])["p_value"] >= 0


def test_operator_groups_respects_regime():
    df = pd.DataFrame({
        "operator": ["a"] * 4,
        "mean_df": [1, 2, 3, 4],
        "regime": ["easy", "easy", "hard", "hard"],
    })
    assert len(operator_groups(df, regime="easy")["a"]) == 2


def test_tests_frame_has_expected_columns():
    frame = _flatten_tests(run_operator_tests(_profiles()))
    assert list(frame.columns) == ["op1", "op2", "U", "p_value", "significant"]


def test_ranking_orders_by_quality_per_joule():
    df = pd.DataFrame({
        "method": ["slow-good", "fast-poor"],
        "qualite_pct": [99.0, 90.0],
        "score_per_joule": [0.01, 0.20],
        "energie_joules": [9900.0, 450.0],
        "co2_micrograms": [1e6, 6e4],
    })
    ranked = rank_methods(df)
    assert ranked.iloc[0]["method"] == "fast-poor"
    assert ranked.iloc[0]["rang_global"] == 1


def test_zscore_centres_within_context():
    df = pd.DataFrame({
        "context": ["a"] * 3 + ["b"] * 3,
        "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    out = zscore_by_context(df, "metric", ["context"])
    for context in ("a", "b"):
        block = out[out.context == context]["z"]
        assert block.mean() == pytest.approx(0.0, abs=1e-9)
        assert block.std() == pytest.approx(1.0, abs=1e-9)


def test_zscore_can_flip_sign():
    df = pd.DataFrame({"context": ["a"] * 2, "metric": [1.0, 3.0]})
    lower_better = zscore_by_context(df, "metric", ["context"],
                                     higher_is_better=False)
    assert lower_better["z"].iloc[0] > 0  # the small value is now positive


def test_zscore_handles_a_constant_context():
    df = pd.DataFrame({"context": ["a"] * 3, "metric": [2.0, 2.0, 2.0]})
    out = zscore_by_context(df, "metric", ["context"])
    assert np.isfinite(out["z"]).all()