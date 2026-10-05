"""
preprocessing.py
=================
Feature scaling and target construction for the budget predictor.

Target definition: given `best_params_results.csv` / `ablation_results.csv`
(instance x budget_j -> score_per_joule, qualite_pct), for each instance
we define the "optimal budget" as the smallest budget_j (among the
BUDGETS_JOULES tested, e.g. [400, 1000, 2000]) that reaches at least
`quality_target_pct` of solution quality. The predictor is trained to
regress this optimal-budget value (in Joules) directly from
instance-structural features, so it can be estimated for *new*
instances without running the solver at every candidate budget.
"""

from typing import List

import numpy as np
import pandas as pd

from .feature_extractor import FEATURE_NAMES


class FeatureScaler:
    """Simple standardization (zero mean, unit variance), stored so it
    can be re-applied identically at inference time."""

    def __init__(self):
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d) -> "FeatureScaler":
        obj = cls()
        obj.mean_ = np.array(d["mean"])
        obj.std_ = np.array(d["std"])
        return obj


def build_optimal_budget_targets(df_results: pd.DataFrame,
                                  quality_target_pct: float = 99.0) -> pd.DataFrame:
    """From a best_params_results / ablation_results-like DataFrame
    (columns: instance, budget_j, qualite_pct), compute, for every
    instance, the smallest tested budget reaching `quality_target_pct`.
    Falls back to the max tested budget if the target is never reached."""
    rows = []
    for instance, grp in df_results.groupby("instance"):
        grp = grp.sort_values("budget_j")
        reached = grp[grp["qualite_pct"] >= quality_target_pct]
        optimal_budget = reached["budget_j"].min() if len(reached) else grp["budget_j"].max()
        rows.append({"instance": instance, "optimal_budget_j": optimal_budget})
    return pd.DataFrame(rows)


def features_to_matrix(feature_dicts: List[dict]) -> np.ndarray:
    df = pd.DataFrame(feature_dicts)
    df = df[FEATURE_NAMES]
    return df.values.astype(float)
