"""
inference.py
============
Command-line entrypoint: predict the recommended Green ACO energy
budget for one or more CNF instances.

Usage:
    python -m budget_predictor.inference --cnf path/to/instance.cnf
    python -m budget_predictor.inference --cnf-dir path/to/instances/
"""

import argparse
import glob
import json
import os

from .predictor import BudgetPredictor


def main():
    ap = argparse.ArgumentParser(description="Predict Green ACO energy budget for CNF instances")
    ap.add_argument("--cnf", help="Path to a single .cnf file")
    ap.add_argument("--cnf-dir", help="Directory of .cnf files to batch-predict")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--snap-to-tested", action="store_true",
                    help="Also show the nearest tested budget from [400, 1000, 2000] J")
    ap.add_argument("--show-features", action="store_true",
                    help="Print the structural feature vector used for each prediction")
    args = ap.parse_args()

    predictor = BudgetPredictor(model_dir=args.model_dir).load()

    paths = []
    if args.cnf:
        paths.append(args.cnf)
    if args.cnf_dir:
        paths.extend(sorted(glob.glob(os.path.join(args.cnf_dir, "*.cnf"))))

    if not paths:
        ap.error("Provide --cnf or --cnf-dir")

    for path in paths:
        if args.show_features:
            features = predictor.extract_feature_vector(path)
            print(f"{os.path.basename(path):50s} features:")
            print(json.dumps(features, indent=2))
        budget = predictor.predict_budget(path)
        print(f"{os.path.basename(path):50s} -> predicted budget: {budget:.1f} J")
        if args.snap_to_tested:
            snapped = predictor.predict_nearest_tested_budget(path)
            print(f"{os.path.basename(path):50s} -> nearest tested budget: {snapped:.0f} J")


if __name__ == "__main__":
    main()
