#!/usr/bin/env python3
"""
prepare_data.py
===============
Verify the benchmark: manifest integrity and instance-file presence.

This is the first stage. It checks that

  * the manifest exists, has the expected columns and no duplicate rows,
  * every ``best_known`` is populated,
  * every listed instance file is present and parses,
  * the parsed variable and clause counts match what the manifest declares.

The last check is the important one: it guarantees the pipeline cannot
silently run on a corpus different from the one the results were produced on.
A WCNF file whose parsed size disagrees with the manifest raises rather than
warning, because a mismatch means the corpus has drifted.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --subset core
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _pipeline import banner, base_parser  # noqa: E402
from greenaco.data import load_benchmark, load_manifest, select_subset  # noqa: E402
from greenaco.wcnf import parse_wcnf  # noqa: E402


def main() -> int:
    parser = base_parser("Verify the benchmark manifest and instance files.")
    args = parser.parse_args()

    banner("Prepare data")
    manifest = load_manifest()
    print(f"  manifest     : {len(manifest)} instances")

    duplicates = manifest["benchmark"].duplicated().sum()
    if duplicates:
        print(f"  [FAIL] {duplicates} duplicate benchmark row(s)")
        return 1
    missing_bk = manifest["best_known"].isna().sum()
    if missing_bk:
        print(f"  [FAIL] {missing_bk} row(s) missing best_known")
        return 1
    print("  [ok]   no duplicate rows; every best_known populated")

    rows = select_subset(manifest, args.subset)
    print(f"  subset       : {args.subset} ({len(rows)} instances)")

    ok = 0
    for row in rows.to_dict("records"):
        name = row["benchmark"]
        try:
            inst = parse_wcnf(Path("data/mse24") / str(row["file"]), name=name)
            inst.strip_empty_clauses()
        except FileNotFoundError as exc:
            print(f"  [FAIL] {name}: {exc}")
            return 1

        if inst.n_vars != int(row["vars"]) or inst.n_clauses != int(row["clauses"]):
            print(f"  [FAIL] {name}: parsed {inst.n_vars} vars / "
                  f"{inst.n_clauses} clauses but manifest declares "
                  f"{row['vars']} / {row['clauses']}")
            return 1
        ok += 1
        print(f"  [ok]   {name[:58]:58s} "
              f"{inst.n_vars:>7d} vars {inst.n_clauses:>8d} clauses")

    print(f"\n  {ok}/{len(rows)} instances verified")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())