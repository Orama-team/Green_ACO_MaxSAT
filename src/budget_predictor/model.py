"""
model.py
========

Thin wrapper around an XGBoost regressor predicting the optimal energy
budget (Joules) for a MAX-SAT instance, from its structural features.

Requires `pip install xgboost`. The wrapper also supports a scikit-learn
GradientBoostingRegressor fallback so the rest of the package can be
exercised without the xgboost dependency installed.
"""

import os
import pickle
from typing import Optional

import numpy as np


DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 0,
}


class BudgetXGBModel:
    def __init__(self, params: Optional[dict] = None, use_xgboost: bool = True):
        self.params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self.use_xgboost = use_xgboost
        self.model = None

    def _build_model(self):
        if self.use_xgboost:
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(**self.params)
            except ImportError:
                print("WARNING: xgboost not installed, falling back to "
                      "sklearn.GradientBoostingRegressor")

        from sklearn.ensemble import GradientBoostingRegressor

        sk_params = {
            k: v for k, v in self.params.items()
            if k in ("n_estimators", "max_depth", "learning_rate",
                     "subsample", "random_state")
        }
        return GradientBoostingRegressor(**sk_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0:
            raise ValueError("Cannot fit the budget predictor on an empty dataset.")
        if len(X) < 2:
            from sklearn.dummy import DummyRegressor

            self.model = DummyRegressor(strategy="constant", constant=float(y[0]))
            self.model.fit(X, y)
            return self

        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted / loaded yet.")
        return self.model.predict(X)

    def feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "params": self.params,
                "use_xgboost": self.use_xgboost,
                "model": self.model,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BudgetXGBModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(params=state["params"], use_xgboost=state["use_xgboost"])
        obj.model = state["model"]
        return obj
