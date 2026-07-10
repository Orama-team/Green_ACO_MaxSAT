"""
predictor.py
============
High-level API: given a path to a .cnf file, recommend the energy
budget (Joules) to allocate to Green ACO so it reaches the target
solution quality, without needing to run the solver at multiple
candidate budgets first (a "warm start" for `config.BUDGETS_JOULES`
/ `GreenACOConfig.budget_j`).
"""

import json
import os
from typing import Optional

import numpy as np

from .cnf_parser import CNFInstance, parse_dimacs_cnf
from .feature_extractor import extract_features
from .model import BudgetXGBModel
from .preprocessing import FeatureScaler


class BudgetPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model: Optional[BudgetXGBModel] = None
        self.scaler: Optional[FeatureScaler] = None
        self.metadata: dict = {}

    def load(self) -> "BudgetPredictor":
        model_path = os.path.join(self.model_dir, "budget_xgb_model.pkl")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.json")
        meta_path = os.path.join(self.model_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run `python -m budget_predictor.trainer` first."
            )

        self.model = BudgetXGBModel.load(model_path)
        with open(scaler_path) as f:
            self.scaler = FeatureScaler.from_dict(json.load(f))
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
        return self

    def extract_feature_vector(self, cnf_path: str) -> dict:
        """Return the exact named feature values used for model inference."""
        if self.model is None or self.scaler is None:
            self.load()

        instance = parse_dimacs_cnf(cnf_path)
        return self.extract_feature_vector_from_instance(instance)

    def extract_feature_vector_from_instance(self, instance: CNFInstance) -> dict:
        if self.model is None or self.scaler is None:
            self.load()

        feats = extract_features(instance).to_dict()
        feature_names = self.metadata.get("feature_names", list(feats.keys()))
        missing = [name for name in feature_names if name not in feats]
        if missing:
            raise ValueError(f"Trained model expects unavailable features: {missing}")
        return {name: feats[name] for name in feature_names}

    def predict_budget(self, cnf_path: str) -> float:
        if self.model is None or self.scaler is None:
            self.load()

        feature_vector = self.extract_feature_vector(cnf_path)
        feature_names = list(feature_vector.keys())
        X = np.array([[feature_vector[name] for name in feature_names]], dtype=float)
        X_scaled = self.scaler.transform(X)

        predicted = float(self.model.predict(X_scaled)[0])
        return max(50.0, predicted)  # floor to avoid degenerate near-zero budgets

    def predict_nearest_tested_budget(self, cnf_path: str,
                                       candidate_budgets=(400, 1000, 2000)) -> float:
        """Snap the raw prediction to the nearest budget actually
        benchmarked in the notebook (config.BUDGETS_JOULES), useful when
        feeding the result straight into `GreenACOConfig.budget_j` for
        reproducibility with the ablation/profiling results."""
        predicted = self.predict_budget(cnf_path)
        return min(candidate_budgets, key=lambda b: abs(b - predicted))
