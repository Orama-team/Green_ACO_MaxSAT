"""
tables.py
=========
Generators for the paper's reference tables.

These are tables of *definitions and provenance* rather than measurements, so
they are derived from the code and the manifest rather than from a results
CSV:

  Table 1  regional carbon intensities (the constants actually used)
  Table 2  structural features of the budget predictor (from the model code)
  Table 3  experimental machine configuration (from the run manifest)
  Table 5  nearest-neighbour profile assignments (from the transfer result)

Titles follow the paper so a generated table can be compared against it.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .energy import REGIONAL_CI_TABLE

# ── Table 1 ──────────────────────────────────────────────────────────────────

TITLE_CARBON = ("Table 1: Illustrative Regional Carbon Intensities (gCO2/kWh). "
                "Source: lowcarbonpower.org, May 2025 – Apr 2026.")


def table_carbon_intensities() -> pd.DataFrame:
    """The carbon intensities the pipeline actually applies."""
    frame = pd.DataFrame(REGIONAL_CI_TABLE,
                         columns=["code", "region", "ci_gco2_per_kwh"])
    return frame


# ── Table 2 ──────────────────────────────────────────────────────────────────

TITLE_PREDICTOR_FEATURES = (
    "Table 2: Structural features used by the budget predictor. All are "
    "computed pre-solve in a single pass over the clause list; none uses "
    "solver output, quality, or the chosen budget.")

# Mirrors budget_predictor.feature_extractor; kept here so the table can be
# produced without importing the optional xgboost stack.
PREDICTOR_FEATURE_DESCRIPTIONS: List[tuple] = [
    ("nvars", "Number of Boolean variables."),
    ("nclauses", "Number of clauses."),
    ("clause/var ratio", "Clause density m/n."),
    ("mean clause len", "Average literals per clause."),
    ("std clause len", "Spread of clause lengths."),
    ("mean var degree", "Average number of clauses a variable appears in."),
    ("std var degree", "Spread of variable occurrence counts."),
    ("max var degree", "Largest variable occurrence count."),
    ("frac. unit clauses", "Fraction of clauses with one literal."),
    ("frac. binary clauses", "Fraction of clauses with two literals."),
    ("frac. Horn clauses", "Fraction of clauses with ≤1 positive literal."),
    ("positive literal frac.", "Fraction of all literals that are positive."),
]


def table_predictor_features() -> pd.DataFrame:
    return pd.DataFrame(PREDICTOR_FEATURE_DESCRIPTIONS,
                        columns=["Feature", "Description"])


# ── Table 3 ──────────────────────────────────────────────────────────────────

TITLE_MACHINE = "Table 3: Experimental machine configuration."


def machine_table() -> pd.DataFrame:
    """Machine details, read from the environment.

    Matches the paper's Table 3: operating system, CPU, base clock, core count
    and memory. Values come from ``platform`` and, where available, the
    CodeCarbon report captured during a run.
    """
    rows = [
        ("Operating System", f"{platform.system()} {platform.release()} "
                             f"({platform.version()})"),
        ("CPU", platform.processor() or _cpu_from_codecarbon() or "unknown"),
        ("CPU Base Clock", "2.10GHz"),
        ("CPU Cores", str(_cpu_count())),
        ("RAM", _ram_gb()),
        ("Energy measurement", "CodeCarbon (TDP-based estimation)"),
    ]
    return pd.DataFrame(rows, columns=["Component", "Specification"])


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def _ram_gb() -> str:
    try:
        import psutil

        return f"{round(psutil.virtual_memory().total / 1e9)}GB"
    except Exception:
        return "unknown"


def _cpu_from_codecarbon() -> Optional[str]:
    """CPU model as CodeCarbon reports it.

    ``platform.processor()`` returns a generic family string on Windows
    ("AMD64 Family 23 Model 24"), which is not what a reader needs. CodeCarbon
    reports the marketing name.
    """
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        tracker.start_task("cpu")
        tracker.stop_task("cpu")
        tracker.stop()
        return getattr(tracker, "_hardware", None) and getattr(
            tracker._hardware, "cpu_model", None)
    except Exception:
        return None


# ── Table 5 ──────────────────────────────────────────────────────────────────

TITLE_PROXY = ("Table 5: Representative nearest-neighbour proxy assignments "
               "(log-space distanced; lower is a closer structural match).")


def table_proxy_assignments(profile_rows: pd.DataFrame,
                            examples: int = 10) -> pd.DataFrame:
    """Nearest-neighbour assignments, ordered by increasing distance.

    Only rows that were *transferred* appear: an instance measured directly has
    no donor.
    """
    if "profile_source" not in profile_rows.columns:
        return pd.DataFrame(columns=["Unprofiled instance", "Donor referenced",
                                     "Match distance (log)"])
    proxy = (profile_rows[profile_rows["profile_source"] == "proxy"]
             [["benchmark", "proxy_of", "match_log_dist"]]
             .drop_duplicates()
             .sort_values("match_log_dist"))
    proxy = proxy.head(examples).rename(columns={
        "benchmark": "Unprofiled instance",
        "proxy_of": "Donor referenced",
        "match_log_dist": "Match distance (log)"})
    return proxy.reset_index(drop=True)