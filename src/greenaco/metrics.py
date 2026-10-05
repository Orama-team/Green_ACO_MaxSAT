"""
metrics.py
==========
Statistical comparison and ranking utilities.

The operator profiles give repeated measurements per operator, which supports
non-parametric group comparison: Kruskal-Wallis across all operators, pairwise
Mann-Whitney U between each pair, and Wilcoxon signed-rank for paired samples
(same instance, two configurations) such as the ablation study.

Rankings aggregate quality, energy and CO2 per method, ordered by quality per
Joule, the metric the green comparison is built on.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

import pandas as pd
from scipy import stats as sp_stats

ALPHA = 0.05


def kruskal_wallis(groups: Dict[str, Sequence[float]]) -> Dict:
    """Kruskal-Wallis H-test across operator groups."""
    values = [list(v) for v in groups.values()]
    if len(values) < 2 or any(len(v) == 0 for v in values):
        return {"H": float("nan"), "p_value": float("nan"),
                "significant": False, "n_groups": len(values)}
    h, p = sp_stats.kruskal(*values)
    return {"H": float(h), "p_value": float(p),
            "significant": bool(p < ALPHA), "n_groups": len(values)}


def pairwise_mannwhitney(groups: Dict[str, Sequence[float]]) -> List[Dict]:
    """Pairwise two-sided Mann-Whitney U between every pair of operators."""
    out = []
    for op1, op2 in combinations(groups.keys(), 2):
        a, b = list(groups[op1]), list(groups[op2])
        if not a or not b:
            continue
        u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        out.append({"op1": op1, "op2": op2, "U": float(u),
                    "p_value": float(p), "significant": bool(p < ALPHA)})
    return out


def wilcoxon_by_context(df: pd.DataFrame, op1: str, op2: str,
                        value_col: str = "mean_df") -> Dict:
    """Paired Wilcoxon signed-rank between two operators.

    Pairs are formed by ``(benchmark, regime)``: for a given instance and
    regime, each operator was measured under the same conditions, so their gains
    are directly comparable. Rows are sorted on those keys before pairing, so
    the pairing does not depend on row order.

    ``zero_method="pratt"`` keeps the zero differences in the ranking, which is
    the right choice here because operators tie exactly on instances where one
    clause pattern dominates.
    """
    left = (df[df["operator"] == op1].sort_values(["benchmark", "regime"])
            [value_col].dropna().values)
    right = (df[df["operator"] == op2].sort_values(["benchmark", "regime"])
             [value_col].dropna().values)
    n = min(len(left), len(right))
    if n < 5:
        return {"op1": op1, "op2": op2, "W": float("nan"),
                "p_value": float("nan"), "significant": False,
                "n_pairs": n, "note": "fewer than 5 pairs"}
    w, p = sp_stats.wilcoxon(left[:n], right[:n], zero_method="pratt")
    return {"op1": op1, "op2": op2, "W": float(w), "p_value": float(p),
            "significant": bool(p < ALPHA), "n_pairs": n}


def wilcoxon_table(df: pd.DataFrame, value_col: str = "mean_df") -> pd.DataFrame:
    """Wilcoxon signed-rank for every operator pair."""
    operators = list(dict.fromkeys(df["operator"]))
    rows = [wilcoxon_by_context(df, a, b, value_col)
            for a, b in combinations(operators, 2)]
    return pd.DataFrame(rows, columns=["op1", "op2", "W", "p_value",
                                       "significant", "n_pairs"])


def wilcoxon_paired(sample_a: Sequence[float],
                    sample_b: Sequence[float]) -> Dict:
    """Wilcoxon signed-rank test for paired samples."""
    a, b = list(sample_a), list(sample_b)
    if len(a) != len(b) or not a:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False}
    stat, p = sp_stats.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p),
            "significant": bool(p < ALPHA)}


def reference_profiles_only(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a profiling table to the directly measured instances.

    Profiles are transferred to unprofiled instances by nearest-neighbour
    matching, so a table covering the whole benchmark contains many identical
    copies of the same handful of measured rows. Running the statistical tests
    over all of them would weight those few instances by however many
    neighbours they happen to have, and shift every p-value. The tests are
    therefore computed on the directly profiled rows only (``n_runs > 0``).
    """
    if "n_runs" not in df.columns:
        return df
    return df[df["n_runs"].fillna(0) > 0]


def operator_groups(df: pd.DataFrame, value_col: str = "mean_df",
                    group_col: str = "operator",
                    regime: str | None = None) -> Dict[str, List[float]]:
    """Collect the per-operator samples used by the group tests."""
    if regime is not None and "regime" in df.columns:
        df = df[df["regime"] == regime]
    return {op: grp[value_col].dropna().tolist()
            for op, grp in df.groupby(group_col)}


def run_operator_tests(df: pd.DataFrame, value_col: str = "mean_df",
                       regime: str | None = None) -> Dict:
    """Full statistical battery on a profiling table."""
    groups = operator_groups(df, value_col=value_col, regime=regime)
    return {
        "kruskal_wallis": kruskal_wallis(groups),
        "mann_whitney_pairwise": pairwise_mannwhitney(groups),
        "n_per_operator": {k: len(v) for k, v in groups.items()},
        "value_col": value_col,
        "regime": regime,
    }


def tests_to_frame(tests: Dict) -> pd.DataFrame:
    """Flatten the pairwise Mann-Whitney results into a tidy frame."""
    rows = tests.get("mann_whitney_pairwise", [])
    return pd.DataFrame(rows, columns=["op1", "op2", "U", "p_value", "significant"])


def rank_methods(df: pd.DataFrame, group_col: str = "method",
                 rank_col: str = "score_per_joule") -> pd.DataFrame:
    """Aggregate the green metrics per method and rank by quality per Joule."""
    agg = df.groupby(group_col).agg(
        qualite_moy_pct=("qualite_pct", "mean"),
        score_per_joule=("score_per_joule", "mean"),
        energie_moy_j=("energie_joules", "mean"),
        co2_moy_ug=("co2_micrograms", "mean"),
        n_runs=(rank_col, "count"),
    ).reset_index()
    agg["rang_global"] = agg[rank_col].rank(ascending=False,
                                            method="min").astype(int)
    return agg.sort_values(["rang_global", group_col]).reset_index(drop=True)


def zscore_by_context(df: pd.DataFrame, metric: str, context_cols: Sequence[str],
                      higher_is_better: bool = True) -> pd.DataFrame:
    """Standardise a metric within each context (one instance and budget).

    Raw quality and CO2 are not comparable across instances of very different
    size, so the ablation comparison z-scores within a context and ranks the
    configurations around their own mean.
    """
    out = df.copy()
    grouped = out.groupby(list(context_cols))[metric]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    std = std.where(std.abs() > 1e-12, 1.0)
    out["z"] = (out[metric] - mean) / std
    if not higher_is_better:
        out["z"] = -out["z"]
    return out