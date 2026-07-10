"""
train_and_predict_budget.py
============================
End-to-end demo of the `budget_predictor` package:

    1. Train an XGBoost model from the exported tabular CSV dataset.
    2. Use the trained model to recommend a budget for a new instance.

Usage:
        # phase 1: export the tabular dataset
        python examples/build_budget_tabular_dataset.py \
                --pck-path Bechmarks/final_merged_100_per_benchmark_3cat.pck \
                --out-csv models/budget_tabular_dataset.csv

        # phase 2: train + predict
    python examples/train_and_predict_budget.py \
                models/budget_tabular_dataset.csv \
        path/to/new_instance.cnf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from budget_predictor.predictor import BudgetPredictor
from budget_predictor.trainer import train_budget_predictor_from_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_csv", help="Path to the exported tabular CSV dataset")
    ap.add_argument("new_instance", help="Path to a new .cnf instance to predict a budget for")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Fraction of the exported tabular data reserved for test")
    ap.add_argument("--snap-to-tested", action="store_true",
                    help="Also show the nearest tested budget from [400, 1000, 2000] J")
    args = ap.parse_args()

    train_budget_predictor_from_csv(
        args.dataset_csv,
        model_dir=args.model_dir,
        test_size=args.test_size,
    )

    predictor = BudgetPredictor(model_dir=args.model_dir).load()
    budget = predictor.predict_budget(args.new_instance)

    print(f"\nPredicted budget for {os.path.basename(args.new_instance)}: "
          f"{budget:.1f} J")
    if args.snap_to_tested:
        snapped = predictor.predict_nearest_tested_budget(args.new_instance)
        print(f"Nearest tested budget: {snapped:.0f} J")


if __name__ == "__main__":
    main()
