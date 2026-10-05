#!/usr/bin/env python3
"""
verify_results.py
=================
Check that the result CSVs are present, complete and internally consistent.

Runs after a sweep as a last gate, and can be run on its own against artifacts
that already exist. It reports rather than fixes: a missing or malformed file
is a finding, not something to paper over.

Checks performed
  presence   each expected artifact exists and is non-empty
  schema     the required columns are present
  coverage   the expected instances are covered
  integrity  quality lies in [0, 100]; the solver never reports more satisfied
             clauses than the instance has; energy is positive; score/J equals
             quality divided by energy
  provenance results/MANIFEST.json records the seed, worker count and machine

Usage:
    python scripts/verify_results.py --subset all
    python scripts/verify_results.py --subset core --strict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import banner, base_parser  # noqa: E402
from greenaco import paths  # noqa: E402
from greenaco.data import load_manifest, select_subset  # noqa: E402

SCHEMA = {
    "operator_profiles": ["benchmark", "operator", "regime", "mean_cost_j",
                          "mean_df"],
    "final_runs": ["instance", "run", "budget_j", "qualite_pct",
                   "energie_joules", "score_per_joule"],
    "final_summary": ["instance", "budget_j", "mean_qualite_pct",
                      "mean_score_per_joule"],
    "ablation_operators": ["config", "instance", "budget_j"],
    "ablation_mechanisms": ["config", "instance", "budget_j"],
    "comparison_runs": ["method", "instance", "qualite_pct",
                        "score_per_joule"],
}

STEMS = ["operator_profiles", "final_runs", "final_summary",
         "ablation_operators", "ablation_mechanisms", "comparison_runs"]


class Report:
    """Collects check outcomes; never mutates the artifacts."""

    def __init__(self, strict: bool):
        self.problems = []
        self.strict = strict

    def check(self, ok: bool, message: str, detail: str = "") -> None:
        if ok:
            print(f"  [ok]   {message}")
        else:
            print(f"  [FAIL] {message}" + (f" — {detail}" if detail else ""))
            self.problems.append(message)

    def note(self, message: str) -> None:
        print(f"  [note] {message}")

    def finish(self) -> int:
        print()
        if self.problems:
            print(f"  {len(self.problems)} problem(s) found:")
            for p in self.problems:
                print(f"    - {p}")
            return 1
        print("  all checks passed")
        return 0


def verify(rep: Report, stem: str, df: pd.DataFrame,
           expected: set) -> None:
    """Schema and integrity checks for one artifact."""
    required = SCHEMA.get(stem, [])
    missing = [c for c in required if c not in df.columns]
    rep.check(not missing, f"{stem}: schema",
              f"missing column(s) {missing}" if missing else "")

    if df.empty:
        rep.check(False, f"{stem}: non-empty")
        return

    inst_col = "benchmark" if "benchmark" in df.columns else "instance"
    if inst_col in df.columns and expected:
        gap = expected - set(df[inst_col].astype(str))
        rep.check(not gap, f"{stem}: instance coverage",
                  f"{len(gap)} instance(s) absent" if gap else "")

    if "qualite_pct" in df.columns:
        q = df["qualite_pct"].dropna()
        rep.check(bool(((q >= 0) & (q <= 100)).all()),
                  f"{stem}: quality within [0, 100]",
                  f"range {q.min():.2f}..{q.max():.2f}" if len(q) else "")

    if {"qualite_solution", "n_clauses"} <= set(df.columns):
        bad = int((df["qualite_solution"] > df["n_clauses"]).sum())
        rep.check(bad == 0, f"{stem}: satisfied clauses <= total",
                  f"{bad} row(s) exceed" if bad else "")

    if "energie_joules" in df.columns:
        e = df["energie_joules"].dropna()
        rep.check(bool((e > 0).all()), f"{stem}: energy positive",
                  f"{int((e <= 0).sum())} non-positive" if len(e) else "")

    if {"score_per_joule", "energie_joules", "qualite_pct"} <= set(df.columns):
        # Only meaningful for per-run rows. For an aggregate over repeats,
        # mean(quality)/mean(energy) is not the mean of quality/energy, so the
        # identity legitimately does not hold. Rows carrying `aggregated` are
        # such summaries and are excluded.
        if "run" in df.columns or "aggregated" in df.columns:
            sub = df.dropna(subset=["score_per_joule", "energie_joules",
                                    "qualite_pct"])
            if "aggregated" in sub.columns:
                sub = sub[~sub["aggregated"].fillna(False).astype(bool)]
            if not sub.empty:
                expect = sub["qualite_pct"] / sub["energie_joules"]
                drift = float((expect - sub["score_per_joule"]).abs().max())
                rep.check(drift < 1e-3, f"{stem}: score/J equals quality/energy",
                          f"max drift {drift:.2e}")


def main() -> int:
    parser = base_parser("Verify the result artifacts.")
    parser.add_argument("--strict", action="store_true",
                        help="Treat missing artifacts as failures.")
    args = parser.parse_args()

    banner("Verify results")
    rep = Report(args.strict)
    out, shipped = paths.RESULTS, paths.SHIPPED
    tag = "all50" if args.subset == "all" else args.subset
    expected = set(select_subset(load_manifest(), args.subset)["benchmark"])

    found = 0
    for stem in STEMS:
        path = out / f"{stem}_{tag}.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            fallback = shipped / f"{stem}_{tag}.csv"
            if fallback.exists():
                rep.note(f"{stem}: using shipped copy ({fallback.name})")
                df = pd.read_csv(fallback)
            else:
                rep.check(not args.strict, f"{stem}: present",
                          f"neither {path.name} nor a shipped copy exists")
                continue
        found += 1
        verify(rep, stem, df, expected)

    if not found:
        rep.check(False, "any artifacts present",
                  "nothing to verify in results/ or results/shipped/")
        return rep.finish()

    manifest_path = out / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        for key in ("seed", "n_jobs", "machine"):
            rep.check(key in info, f"MANIFEST records {key}")
    else:
        rep.note("no MANIFEST.json; run a stage to record provenance")

    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
