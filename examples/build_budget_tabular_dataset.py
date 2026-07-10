"""
build_budget_tabular_dataset.py
===============================
Phase 1 of the budget-predictor workflow.

This script converts the benchmark pickle into a tabular CSV dataset
without training any model. The CSV contains the 12 structural features
plus the derived target energy column `energie_joules`.

Usage:
    python examples/build_budget_tabular_dataset.py \
        --pck-path Bechmarks/final_merged_100_per_benchmark_3cat.pck \
        --out-csv models/budget_tabular_dataset.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from budget_predictor.dataset import export_tabular_dataset_from_pck


def main():
    ap = argparse.ArgumentParser(description="Build the tabular budget dataset from the benchmark pickle")
    ap.add_argument("--pck-path", required=True,
                    help="Path to final_merged_100_per_benchmark_3cat.pck")
    ap.add_argument("--out-csv", required=True,
                    help="Where to write the exported tabular CSV dataset")
    ap.add_argument("--quality-target", type=float, default=95.0,
                    help="Quality threshold used to assign the budget label")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on the number of benchmark instances to process")
    ap.add_argument("--limit-strategy", choices=("stratified", "contiguous"), default="stratified",
                    help="How to choose records when --limit is set; stratified samples across categories/benchmarks")
    ap.add_argument("--flush-every", type=int, default=100,
                    help="Write completed rows to CSV after this many instances")
    ap.add_argument("--no-resume", action="store_true",
                    help="Overwrite the output CSV instead of appending missing rows")
    args = ap.parse_args()

    df = export_tabular_dataset_from_pck(
        args.pck_path,
        args.out_csv,
        quality_target_pct=args.quality_target,
        limit=args.limit,
        limit_strategy=args.limit_strategy,
        flush_every=args.flush_every,
        resume=not args.no_resume,
    )

    print(f"wrote {args.out_csv} ({len(df)} rows)")
    print(f"  columns: {', '.join(df.columns)}")
    if len(df) and {"category", "benchmark", "n_vars"}.issubset(df.columns):
        print(f"  categories: {df['category'].nunique()}")
        print(f"  benchmarks: {df['benchmark'].nunique()}")
        print(f"  n_vars range: {int(df['n_vars'].min())}..{int(df['n_vars'].max())}")


if __name__ == "__main__":
    main()
