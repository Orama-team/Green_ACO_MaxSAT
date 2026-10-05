"""Pairing and selection behaviour of tests, kept separate from the pure
statistical tests in ``test_metrics.py``.

Two design decisions in ``metrics.py`` are subtle enough to need their own
proof: the paired test must not depend on row order, and the group tests must
not run over transferred profiles, which would weight a few measured instances
by however many neighbours they happen to have.
"""

from __future__ import annotations

import pytest

from greenaco.metrics import (
    reference_profiles_only,
    run_operator_tests,
    wilcoxon_by_context,
    wilcoxon_table,
)

N_OPS = 3


def paired_frame(n_contexts: int = 6, shuffle: bool = False) -> "pd.DataFrame":
    """Two operators measured on the same (benchmark, regime) pairs.

    ``op_a`` beats ``op_b`` by exactly 2 units everywhere, so any sane paired
    test must see it, whatever the row order.
    """
    import pandas as pd

    rows = []
    for i in range(n_contexts):
        rows.append({"operator": "op_a", "benchmark": f"bench_{i}",
                     "regime": "sim", "mean_df": 10.0 + i})
        rows.append({"operator": "op_b", "benchmark": f"bench_{i}",
                     "regime": "sim", "mean_df": 8.0 + i})
    frame = pd.DataFrame(rows)
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return frame


def test_pairing_does_not_depend_on_row_order():
    ordered = wilcoxon_by_context(paired_frame(shuffle=False), "op_a", "op_b")
    shuffled = wilcoxon_by_context(paired_frame(shuffle=True), "op_a", "op_b")
    assert ordered["n_pairs"] == shuffled["n_pairs"] == 6
    assert ordered["W"] == pytest.approx(shuffled["W"])
    assert ordered["p_value"] == pytest.approx(shuffled["p_value"])


def test_pairing_detects_a_consistent_shift():
    result = wilcoxon_by_context(paired_frame(), "op_a", "op_b")
    assert result["n_pairs"] == 6
    assert result["significant"] is True


def test_pairing_refuses_too_few_pairs():
    result = wilcoxon_by_context(paired_frame(n_contexts=4), "op_a", "op_b")
    assert result["significant"] is False
    assert result["n_pairs"] == 4
    assert "fewer than 5" in result["note"]


def test_table_lists_one_row_per_pair():
    import pandas as pd

    frame = pd.concat([paired_frame(),
                       paired_frame().assign(operator="op_c")])
    table = wilcoxon_table(frame)
    assert len(table) == N_OPS
    assert set(table["op1"]) | set(table["op2"]) == {"op_a", "op_b", "op_c"}


def test_reference_filter_drops_transferred_copies():
    """Transferred profiles have ``n_runs == 0``; only measured rows remain."""
    import pandas as pd

    frame = pd.DataFrame([
        {"operator": "a", "mean_df": 5.0, "n_runs": 10},
        {"operator": "a", "mean_df": 5.0, "n_runs": 0},
        {"operator": "b", "mean_df": 3.0, "n_runs": None},
    ])
    kept = reference_profiles_only(frame)
    assert len(kept) == 1
    assert kept.iloc[0]["n_runs"] == 10


def test_reference_filter_without_column_passes_through():
    import pandas as pd

    frame = pd.DataFrame([{"operator": "a", "mean_df": 5.0}])
    assert reference_profiles_only(frame) is frame or \
        reference_profiles_only(frame).equals(frame)


def test_full_battery_on_measured_rows_only():
    """``run_operator_tests`` over rows with ``n_runs > 0`` must not crash and
    must count the measured samples only."""
    import pandas as pd

    measured = paired_frame().assign(n_runs=10)
    copies = paired_frame().assign(n_runs=0)
    tests = run_operator_tests(reference_profiles_only(
        pd.concat([measured, copies])))
    assert tests["n_per_operator"] == {"op_a": 6, "op_b": 6}
    assert "H" in tests["kruskal_wallis"]