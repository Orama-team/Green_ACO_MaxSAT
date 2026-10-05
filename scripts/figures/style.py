"""
figures/style.py
================
Shared plotting conventions, so every figure in the report looks like it came
from the same study.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402

# Operator colours, matching the palette used in the report.
OPERATOR_COLORS = {
    "walksat": "#378ADD",
    "focused_vns": "#D85A30",
    "clause_restart_greedy": "#888888",
}

# Method families in the comparison table.
FAMILY_COLORS = {
    "GA": "#4C9F70",
    "ACO": "#378ADD",
    "ACO + EI/J": "#D85A30",
}

METRIC_LABELS = {
    "qualite_pct": "Quality (%)",
    "energie_joules": "Energy (J)",
    "score_per_joule": "Quality per Joule (%/J)",
    "co2_micrograms": "CO2 footprint (ug CO2eq)",
    "mean_qualite_pct": "Mean quality (%)",
    "mean_energie_joules": "Mean energy (J)",
    "mean_score_per_joule": "Mean quality per Joule (%/J)",
    "mean_co2_micrograms": "Mean CO2 footprint (ug CO2eq)",
}

DPI = 150


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "legend.frameon": False,
    })


def figure_dir() -> Path:
    from greenaco import paths

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    return paths.FIGURES


def save(fig, name: str) -> Path:
    """Write a figure to ``figures/`` and close it."""
    target = figure_dir() / name
    fig.savefig(target)
    plt.close(fig)
    print(f"  -> wrote figures/{name}")
    return target


def short_instance(name: str) -> str:
    """Compact instance label for figure axes."""
    if name.startswith("decision-tree-"):
        name = name[len("decision-tree-"):]
    for suffix in ("-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree",
                   "-un-wcnf_incomplete_improved"):
        name = name.split(suffix)[0]
    if name.startswith("min-fill-MinFill_R0_"):
        name = "minfill-" + name[len("min-fill-MinFill_R0_"):]
    return name[:26]