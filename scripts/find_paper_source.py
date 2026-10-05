"""
find_paper_source.py
====================
Locate which stored result file backs a set of published figures.

Energy is a CodeCarbon *estimate* proportional to wall-clock time, so no two
runs give identical numbers even with the same seed. A published figure can
therefore only be matched by locating the run that produced it, not by
recomputing it. This scans candidate CSVs, groups them by shape (one row per
instance x budget, as the method comparison uses), and reports the mean energy
and mean quality-per-Joule of each group against the target values.

Usage:
    python scripts/find_paper_source.py --energy 986 --spj 0.135
    python scripts/find_paper_source.py --dir "C:/path/to/files" --energy 986
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Published values from the article (Section 4.9).
PAPER_ENERGY_J = 986.0
PAPER_SPJ = 0.135


def summarise(path: Path) -> dict | None:
    """Mean energy and SPJ of a candidate file, or None if it is not a run file."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "energie_joules" not in df.columns or "score_per_joule" not in df.columns:
        return None
    if df["energie_joules"].dropna().empty:
        return None

    inst = next((c for c in ("instance", "benchmark") if c in df.columns), None)
    return {
        "file": path.name,
        "rows": len(df),
        "instances": df[inst].nunique() if inst else None,
        "energy_j": float(df["energie_joules"].mean()),
        "spj": float(df["score_per_joule"].mean()),
        "quality": float(df["qualite_pct"].mean())
        if "qualite_pct" in df.columns else None,
        "carbon": df["ci_gco2_per_kwh"].iloc[0]
        if "ci_gco2_per_kwh" in df.columns else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse
                                     .RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=None,
                        help="Directory to scan (default: check this/ next to "
                             "the repository).")
    parser.add_argument("--energy", type=float, default=PAPER_ENERGY_J,
                        help="Published mean energy in joules.")
    parser.add_argument("--spj", type=float, default=PAPER_SPJ,
                        help="Published mean quality per Joule.")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Relative tolerance for a match (default 2%%).")
    args = parser.parse_args()

    here = Path(args.dir) if args.dir else ROOT.parent / "check this"
    if not here.exists():
        print(f"directory not found: {here}")
        return 1

    print(f"scanning {here}\n")
    print(f"{'file':44s} {'rows':>5s} {'inst':>5s} {'energy J':>10s} "
          f"{'SPJ':>8s} {'CI':>5s}")
    print("-" * 84)

    rows = []
    for path in sorted(here.glob("*.csv")):
        info = summarise(path)
        if info:
            rows.append(info)
            ci = f"{info['carbon']:.0f}" if info["carbon"] else "-"
            print(f"{info['file'][:44]:44s} {info['rows']:5d} "
                  f"{str(info['instances'] or '-'):>5s} "
                  f"{info['energy_j']:10.1f} {info['spj']:8.4f} {ci:>5s}")

    if not rows:
        print("\nno candidate run files found")
        return 1

    print(f"\ntarget: energy {args.energy} J, SPJ {args.spj} "
          f"(tolerance {args.tolerance:.1%})\n")
    hits = []
    for info in rows:
        d_e = abs(info["energy_j"] - args.energy) / args.energy
        d_s = abs(info["spj"] - args.spj) / args.spj
        if d_e <= args.tolerance and d_s <= args.tolerance:
            hits.append((info, d_e, d_s))

    if hits:
        print("MATCH:")
        for info, d_e, d_s in hits:
            print(f"  {info['file']}  energy off {d_e:.2%}, SPJ off {d_s:.2%}")
    else:
        print("no file matches both figures. Closest by energy:")
        for info in sorted(rows, key=lambda r: abs(r["energy_j"] - args.energy))[:5]:
            d_e = (info["energy_j"] - args.energy) / args.energy * 100
            d_s = (info["spj"] - args.spj) / args.spj * 100
            print(f"  {info['file'][:44]:44s} energy {info['energy_j']:8.1f} J "
                  f"({d_e:+6.1f}%)   SPJ {info['spj']:.4f} ({d_s:+6.1f}%)")

        carbons = {r["carbon"] for r in rows if r["carbon"]}
        if len(carbons) > 1:
            print(f"\nnote: carbon intensities present: {sorted(carbons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())