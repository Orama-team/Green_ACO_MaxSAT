"""
trainer.py
==========

The training step consumes the tabular CSV exported from the benchmark
pickle, so dataset construction and model fitting are separate phases.
Only meaningful pre-solve structural features are used as model inputs.
"""

import argparse
import json
import os
from dataclasses import dataclass

from .dataset import BudgetDataset, build_dataset_from_csv
from .feature_extractor import FEATURE_DESCRIPTIONS
from .model import BudgetXGBModel
from .preprocessing import FeatureScaler

try:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-learn is required for training: pip install scikit-learn") from e


@dataclass
class TrainingReport:
    mae: float
    r2: float
    n_train: int
    n_test: int


def _train_and_evaluate(dataset: BudgetDataset,
                        model_dir: str,
                        test_size: float,
                        random_state: int,
                        metadata: dict,
                        source_label: str) -> TrainingReport:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if len(dataset.y) < 4:
        print("WARNING: Very small dataset (n=%d). With few instances, prefer "
              "cross-validation or collect more profiled instances before "
              "deploying this predictor in production." % len(dataset.y))

    if len(dataset.y) >= 4:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=random_state)
    else:
        X_train_raw, y_train = dataset.X, dataset.y
        X_test_raw, y_test = dataset.X, dataset.y

    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = BudgetXGBModel()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")

    importances = model.feature_importances()
    feature_importance = None
    if importances is not None:
        feature_importance = {
            name: float(value)
            for name, value in zip(dataset.feature_names, importances)
        }

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "budget_xgb_model.pkl"))
    with open(os.path.join(model_dir, "feature_scaler.json"), "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump({
            "feature_names": dataset.feature_names,
            "feature_descriptions": {
                name: FEATURE_DESCRIPTIONS.get(name, "")
                for name in dataset.feature_names
            },
            "feature_importance": feature_importance,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "train_fraction": len(y_train) / max(len(dataset.y), 1),
            "test_fraction": len(y_test) / max(len(dataset.y), 1),
            "mae": mae,
            "r2": r2,
            **metadata,
        }, f, indent=2)

    print(f"Model trained from {source_label}: MAE={mae:.1f} J, R^2={r2:.3f} "
          f"(n_train={len(y_train)}, n_test={len(y_test)})")
    print(f"Saved to {model_dir}/budget_xgb_model.pkl")

    return TrainingReport(mae=mae, r2=r2, n_train=len(y_train), n_test=len(y_test))


def train_budget_predictor(cnf_dir: str,
                           results_csv: str,
                           model_dir: str = "models",
                           test_size: float = 0.25,
                           quality_target_pct: float = 99.0,
                           random_state: int = 0) -> TrainingReport:
    raise NotImplementedError(
        "The trainer now consumes the exported tabular CSV. Build it with "
        "`python examples/build_budget_tabular_dataset.py ...`, then call "
        "`train_budget_predictor_from_csv(...)` or `python -m budget_predictor.trainer "
        "--dataset-csv models/budget_tabular_dataset.csv`."
    )


def train_budget_predictor_from_csv(dataset_csv: str,
                                    model_dir: str = "models",
                                    test_size: float = 0.2,
                                    random_state: int = 0) -> TrainingReport:
    dataset: BudgetDataset = build_dataset_from_csv(dataset_csv)
    return _train_and_evaluate(
        dataset=dataset,
        model_dir=model_dir,
        test_size=test_size,
        random_state=random_state,
        metadata={
            "instance_names": dataset.instance_names,
            "source_csv": dataset_csv,
        },
        source_label=os.path.basename(dataset_csv),
    )


def main():
    ap = argparse.ArgumentParser(description="Train the Green ACO budget predictor")
    ap.add_argument("--dataset-csv", required=True,
                    help="Path to the exported tabular CSV dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Fraction reserved for test; default 0.2 means train on 80%.")
    args = ap.parse_args()

    train_budget_predictor_from_csv(
        args.dataset_csv,
        model_dir=args.model_dir,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
