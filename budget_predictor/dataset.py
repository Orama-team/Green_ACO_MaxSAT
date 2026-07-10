"""
dataset.py
==========
Utilities for building and loading the tabular dataset used to train
the budget predictor.

Two data paths are supported:
  - the benchmark pickle used in this repository
    (`Bechmarks/final_merged_100_per_benchmark_3cat.pck`)

The pickle is first converted into a CSV file
containing the 12 structural features plus the derived energy budget
label. Training then consumes that CSV independently of the export
step.
"""

import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .cnf_parser import parse_dimacs_cnf
from .feature_extractor import FEATURE_NAMES, extract_features
from .preprocessing import build_optimal_budget_targets, features_to_matrix


@dataclass
class BudgetDataset:
    X: np.ndarray
    y: np.ndarray
    instance_names: List[str]
    feature_names: List[str]


TABULAR_METADATA_COLUMNS = ["instance", "category", "benchmark", "file"]
TARGET_COLUMN = "energie_joules"
ROW_ID_COLUMNS = ["category", "benchmark", "file"]

def build_tabular_dataset_from_pck(pck_path: str,
                                   quality_target_pct: float = 95.0,
                                   candidate_budgets: Sequence[float] = (400, 1000, 2000),
                                   limit: int | None = None,
                                   limit_strategy: str = "stratified") -> pd.DataFrame:
    """Build the CSV-ready tabular dataset directly from the benchmark pickle."""
    from method.config import DEFAULT_BEST_PARAMS, GreenACOConfig
    from method.solver import GreenACOSolver

    with open(pck_path, "rb") as f:
        records = pickle.load(f)

    records = _select_records(records, limit, limit_strategy)

    rows: List[Dict] = []
    for record in records:
        rows.append(_record_to_tabular_row(
            record,
            quality_target_pct=quality_target_pct,
            candidate_budgets=candidate_budgets,
            default_best_params=DEFAULT_BEST_PARAMS,
            config_cls=GreenACOConfig,
            solver_cls=GreenACOSolver,
        ))

    columns = TABULAR_METADATA_COLUMNS + FEATURE_NAMES + [TARGET_COLUMN]
    return pd.DataFrame(rows, columns=columns)


def export_tabular_dataset_from_pck(pck_path: str,
                                    out_csv: str,
                                    quality_target_pct: float = 99.0,
                                    candidate_budgets: Sequence[float] = (400, 1000, 2000),
                                    limit: int | None = None,
                                    limit_strategy: str = "stratified",
                                    flush_every: int = 100,
                                    resume: bool = True) -> pd.DataFrame:
    """Export the tabular dataset as CSV and return the DataFrame."""
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    columns = TABULAR_METADATA_COLUMNS + FEATURE_NAMES + [TARGET_COLUMN]
    existing_keys = _existing_row_keys(out_csv, columns) if resume else set()
    if not existing_keys:
        _write_header(out_csv, columns)

    from method.config import DEFAULT_BEST_PARAMS, GreenACOConfig
    from method.solver import GreenACOSolver

    with open(pck_path, "rb") as f:
        records = pickle.load(f)

    records = _select_records(records, limit, limit_strategy)
    if existing_keys:
        records = [record for record in records if _record_key(record) not in existing_keys]
        print(f"  resuming {out_csv}; skipped {len(existing_keys)} existing rows", flush=True)

    batch: List[Dict] = []
    written = len(existing_keys)
    skipped = 0
    total = written + len(records)

    try:
        for record in records:
            try:
                batch.append(_record_to_tabular_row(
                    record,
                    quality_target_pct=quality_target_pct,
                    candidate_budgets=candidate_budgets,
                    default_best_params=DEFAULT_BEST_PARAMS,
                    config_cls=GreenACOConfig,
                    solver_cls=GreenACOSolver,
                ))
            except ValueError as exc:
                skipped += 1
                print(f"  skipped {record.get('file', '<unknown>')}: {exc}", flush=True)
                continue
            if len(batch) >= flush_every:
                written += _append_rows(out_csv, batch, columns)
                print(f"  flushed {written}/{total} rows -> {out_csv} ({skipped} skipped)", flush=True)
                batch = []
    except KeyboardInterrupt:
        if batch:
            written += _append_rows(out_csv, batch, columns)
        print(f"\nInterrupted. Kept {written} completed rows in {out_csv} ({skipped} skipped).", flush=True)
        return pd.read_csv(out_csv)

    if batch:
        written += _append_rows(out_csv, batch, columns)
        print(f"  flushed {written}/{total} rows -> {out_csv} ({skipped} skipped)", flush=True)

    return pd.read_csv(out_csv)


def build_dataset_from_csv(dataset_csv: str) -> BudgetDataset:
    """Load a tabular CSV dataset and convert it to model arrays."""
    df = pd.read_csv(dataset_csv)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {dataset_csv}")

    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {dataset_csv}: {missing}")

    feature_rows = df[FEATURE_NAMES].to_dict(orient="records")
    X = features_to_matrix(feature_rows) if feature_rows else np.empty((0, len(FEATURE_NAMES)))
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    instance_names = df["instance"].astype(str).tolist() if "instance" in df.columns else [str(i) for i in range(len(df))]
    return BudgetDataset(X=X, y=y, instance_names=instance_names, feature_names=FEATURE_NAMES)


def _find_cnf_file(cnf_dir: str, instance_name: str):
    for ext in (".cnf", ".wcnf", ""):
        candidate = os.path.join(cnf_dir, instance_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _select_records(records: Sequence[Dict], limit: int | None, limit_strategy: str) -> List[Dict]:
    if limit is None:
        return list(records)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit_strategy == "contiguous":
        return list(records[:limit])
    if limit_strategy != "stratified":
        raise ValueError(f"Unknown limit_strategy: {limit_strategy}")
    return _stratified_records(records, limit)


def _stratified_records(records: Sequence[Dict], limit: int) -> List[Dict]:
    """Round-robin records across categories and benchmarks for capped exports.

    The benchmark pickle is ordered in large category blocks. A plain
    records[:limit] slice can therefore hide later categories and collapse the
    observed variable/ratio distribution. This keeps the exported subset broad
    while preserving original order inside each benchmark.
    """
    by_category: "OrderedDict[str, OrderedDict[str, List[Dict]]]" = OrderedDict()
    for record in records:
        category = str(record.get("category", ""))
        benchmark = str(record.get("benchmark", ""))
        by_benchmark = by_category.setdefault(category, OrderedDict())
        by_benchmark.setdefault(benchmark, []).append(record)

    category_streams = [
        _round_robin_benchmarks(by_benchmark)
        for by_benchmark in by_category.values()
    ]

    selected: List[Dict] = []
    while len(selected) < limit and category_streams:
        next_streams = []
        for stream in category_streams:
            try:
                selected.append(next(stream))
            except StopIteration:
                continue
            if len(selected) >= limit:
                next_streams.append(stream)
                break
            next_streams.append(stream)
        category_streams = next_streams
    return selected


def _round_robin_benchmarks(by_benchmark: "OrderedDict[str, List[Dict]]"):
    positions = {benchmark: 0 for benchmark in by_benchmark}
    active = list(by_benchmark.keys())
    while active:
        next_active = []
        for benchmark in active:
            position = positions[benchmark]
            records = by_benchmark[benchmark]
            if position < len(records):
                yield records[position]
                positions[benchmark] = position + 1
            if positions[benchmark] < len(records):
                next_active.append(benchmark)
        active = next_active


def _record_to_tabular_row(record: Dict,
                           quality_target_pct: float,
                           candidate_budgets: Sequence[float],
                           default_best_params: Dict,
                           config_cls,
                           solver_cls) -> Dict:
    instance = _record_to_instance(record)
    feats = extract_features(instance).to_dict()

    selected_budget = float(max(candidate_budgets))
    selected_energy = None
    selected_quality = None
    target_reached = False
    for budget in candidate_budgets:
        cfg = config_cls.from_dict({**default_best_params, "budget_j": float(budget)})
        result = solver_cls(cfg).solve(instance)
        if result.qualite_pct >= quality_target_pct:
            selected_budget = float(budget)
            selected_energy = float(result.energie_joules)
            selected_quality = float(result.qualite_pct)
            target_reached = True
            break

    if selected_energy is None:
        cfg = config_cls.from_dict({**default_best_params, "budget_j": float(selected_budget)})
        result = solver_cls(cfg).solve(instance)
        selected_energy = float(result.energie_joules)
        selected_quality = float(result.qualite_pct)

    return {
        "instance": instance.name,
        "category": record.get("category", ""),
        "benchmark": record.get("benchmark", ""),
        "file": record.get("file", ""),
        **feats,
        "selected_budget_j": selected_budget,
        "qualite_pct": selected_quality,
        "target_reached": target_reached,
        TARGET_COLUMN: selected_energy,
    }


def _write_header(out_csv: str, columns: Sequence[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(out_csv, index=False)


def _existing_row_keys(out_csv: str, columns: Sequence[str]) -> set[tuple[str, str, str]]:
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return set()

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot resume {out_csv}; missing columns: {missing}")
    return {
        (
            str(row["category"]),
            str(row["benchmark"]),
            str(row["file"]),
        )
        for _, row in df.iterrows()
    }


def _record_key(record: Dict) -> tuple[str, str, str]:
    return tuple(str(record.get(column, "")) for column in ROW_ID_COLUMNS)


def _append_rows(out_csv: str, rows: List[Dict], columns: Sequence[str]) -> int:
    pd.DataFrame(rows, columns=columns).to_csv(
        out_csv,
        mode="a",
        header=False,
        index=False,
    )
    return len(rows)


def _record_to_instance(record):
    from method.parser import CNFInstance

    clauses = [tuple(int(lit) for lit in clause) for clause in record["clauses_list"]]
    instance_name = os.path.splitext(record["file"])[0]
    return CNFInstance(
        name=instance_name,
        n_vars=int(record["vars"]),
        n_clauses=int(record["clauses"]),
        clauses=clauses,
    )
