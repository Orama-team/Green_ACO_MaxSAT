"""
metrics.py
==========
Statistical comparison utilities matching section 1.4 of the notebook
("Tests Statistiques") — Kruskal-Wallis across all operators, pairwise
Mann-Whitney U, and Wilcoxon signed-rank for paired comparisons — run
on the Δf values collected during operator profiling.

Also provides the aggregation helpers used to build the summary /
ranking table of section 5.1 ("Table de synthèse finale").
"""

from itertools import combinations
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats


def kruskal_wallis(groups: Dict[str, List[float]]):
    """Kruskal-Wallis H-test across all operator groups."""
    values = list(groups.values())
    h, p = sp_stats.kruskal(*values)
    return {"H": h, "p_value": p, "significant": p < 0.05}


def pairwise_mannwhitney(groups: Dict[str, List[float]]):
    """Pairwise Mann-Whitney U test between every pair of operators."""
    results = []
    for op1, op2 in combinations(groups.keys(), 2):
        u, p = sp_stats.mannwhitneyu(groups[op1], groups[op2], alternative="two-sided")
        results.append({"op1": op1, "op2": op2, "U": u, "p_value": p, "significant": p < 0.05})
    return results


def wilcoxon_paired(sample_a: List[float], sample_b: List[float]):
    """Wilcoxon signed-rank test for paired samples (e.g. same instance,
    two configurations)."""
    stat, p = sp_stats.wilcoxon(sample_a, sample_b)
    return {"statistic": stat, "p_value": p, "significant": p < 0.05}


def run_statistical_tests(df_profiles, operator_col="operator", value_col="mean_df",
                           operator_pool=None) -> Dict:
    """Full statistical battery on an operator_profiles-like DataFrame,
    mirroring the notebook's `run_statistical_tests`."""
    if operator_pool:
        df_profiles = df_profiles[df_profiles[operator_col].isin(operator_pool)]

    groups = {op: grp[value_col].values for op, grp in df_profiles.groupby(operator_col)}
    return {
        "kruskal_wallis": kruskal_wallis(groups),
        "mann_whitney_pairwise": pairwise_mannwhitney(groups),
        "n_per_operator": {k: len(v) for k, v in groups.items()},
    }


def aggregate_ranking(df_results, group_col="method", rank_col="score_per_joule",
                       ascending=False):
    """Build the final ranking table of section 5.1: mean metrics per
    method, ranked by score/Joule (descending, higher is better)."""
    agg = df_results.groupby(group_col).agg(
        qualite_moy=("qualite_pct", "mean"),
        score_per_joule=(rank_col, "mean"),
        energie_moy_j=("energie_joules", "mean"),
        co2_moy=("co2_micrograms", "mean"),
    ).reset_index()
    agg["rang_global"] = agg[rank_col].rank(ascending=ascending).astype(int)
    return agg.sort_values(rank_col, ascending=ascending)
