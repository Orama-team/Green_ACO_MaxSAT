#!/usr/bin/env python3
"""
make_figures.py
===============
Build every figure and table in the report from the CSVs in ``results/``.

This script **never runs an experiment**. It only reads ``results/*.csv`` (or
``results/shipped/*.csv`` with ``--use-shipped``) and writes to ``figures/``.
That separation is deliberate: a figure can be regenerated at any time without
re-running hours of search, and re-running an experiment can never silently
alter a figure.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --use-shipped --subset all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from figures.style import (FAMILY_COLORS, OPERATOR_COLORS,  # noqa: E402
                           apply_style, save, short_instance)
from greenaco import paths  # noqa: E402
from greenaco.metrics import (rank_methods, reference_profiles_only,  # noqa: E402
                              run_operator_tests, tests_to_frame,
                              zscore_by_context)
from greenaco.tables import (TITLE_CARBON, TITLE_MACHINE,  # noqa: E402
                             TITLE_PREDICTOR_FEATURES, TITLE_PROXY,
                             machine_table, table_carbon_intensities,
                             table_predictor_features,
                             table_proxy_assignments)


def read(name: str, results: Path, shipped: Path):
    """Read an artifact, preferring the freshly computed copy.

    Falls back to the shipped CSV so figures can always be produced, even when a
    stage has not been re-run.
    """
    for candidate in (results / name, shipped / name):
        if candidate.exists():
            return pd.read_csv(candidate)
    print(f"  [skip] {name} not found in results/ or results/shipped/")
    return None


def fig_operator_profiles(df: pd.DataFrame) -> None:
    """Energy share, cost by regime, and gain-per-Joule."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("Operator profiling — cost and quality gain per call",
                 fontsize=13, fontweight="bold")

    ops = [o for o in OPERATOR_COLORS if o in set(df["operator"])]
    regimes = sorted(df["regime"].unique())

    share = df.groupby("operator")["mean_cost_j"].mean().sort_values(
        ascending=False)
    axes[0].pie(share.values,
                labels=[o.replace("_", "\n") for o in share.index],
                colors=[OPERATOR_COLORS.get(o, "#aaa") for o in share.index],
                autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
                startangle=140, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("Share of measured energy cost")

    width = 0.38
    x = np.arange(len(ops))
    for i, regime in enumerate(regimes):
        vals = [df[(df.operator == o) & (df.regime == regime)]
                ["mean_cost_j"].mean() for o in ops]
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=f"{regime} regime",
                    color=["#378ADD", "#D85A30"][i % 2])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([o.replace("_", "\n") for o in ops], fontsize=9)
    axes[1].set_ylabel("Mean energy cost (J)")
    axes[1].set_title("Energy cost per operator and regime")
    axes[1].legend()

    for regime, marker in zip(regimes, ["o", "s"]):
        sub = (df[df.regime == regime].groupby("operator")
               .agg(cost=("mean_cost_j", "mean"), gain=("mean_df", "mean"))
               .reset_index())
        for _, r in sub.iterrows():
            axes[2].scatter(r["cost"], r["gain"],
                            color=OPERATOR_COLORS.get(r["operator"], "#888"),
                            marker=marker, s=90)
            axes[2].annotate(r["operator"].replace("_", " "),
                             (r["cost"], r["gain"]), fontsize=7,
                             xytext=(4, 4), textcoords="offset points")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Mean energy cost (J, log scale)")
    axes[2].set_ylabel("Mean quality gain (clauses)")
    axes[2].set_title("Efficiency: gain per Joule")
    save(fig, "fig_operator_profiles.png")


def fig_operator_tests(df: pd.DataFrame, out: Path) -> None:
    """Non-parametric comparison of the operators on measured gains.

    Run on the directly measured reference instances only: the rest of the
    table is nearest-neighbour copies of those same rows, which would distort
    the p-values. See ``metrics.reference_profiles_only``.
    """
    apply_style()
    measured = reference_profiles_only(df)
    tests = run_operator_tests(measured, value_col="mean_df")
    kw = tests["kruskal_wallis"]
    table = tests_to_frame(tests)
    if not table.empty:
        table.to_csv(out / "table_operator_tests.csv", index=False)
        print("  -> wrote figures/table_operator_tests.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [g for _, g in df.groupby("operator")["mean_df"]]
    labels = [o.replace("_", "\n") for o in sorted(df["operator"].unique())]
    ax.boxplot(data, tick_labels=labels, patch_artist=True,
               boxprops={"facecolor": "#378ADD", "alpha": 0.6})
    ax.set_ylabel("Mean quality gain per call (clauses)")
    ax.set_title("Operator quality gain\n"
                 f"Kruskal-Wallis H={kw['H']:.2f}, p={kw['p_value']:.2e}"
                 f"{' (significant)' if kw['significant'] else ''}")
    save(fig, "fig_operator_tests.png")


def fig_ablation(df: pd.DataFrame) -> None:
    """Ablation, z-scored within each (instance, budget) context.

    Instances differ so much in size and baseline quality that raw values are
    not comparable across them; standardising within a context puts the
    configurations on a common scale.
    """
    apply_style()
    metrics = [("score_per_joule", "Quality per Joule (%/J)", True),
               ("co2_micrograms", "CO2 footprint (ug CO2eq)", False)]
    work = df.rename(columns={c: c.replace("mean_", "") for c in df.columns
                              if c.startswith("mean_")})
    work["context"] = (work["instance"].map(short_instance) + "_"
                       + work["budget_j"].astype(str) + "J")
    contexts = sorted(work["context"].unique(),
                      key=lambda s: (s.rsplit("_", 1)[0],
                                     int(s.rsplit("_", 1)[1].rstrip("J"))))
    configs = list(dict.fromkeys(work["config"]))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5.5 * len(metrics)),
                             squeeze=False)
    for ax, (metric, label, higher) in zip(axes[:, 0], metrics):
        zs = zscore_by_context(work, metric, ["context"],
                               higher_is_better=higher)
        matrix = np.full((len(configs), len(contexts)), np.nan)
        for i, cfg in enumerate(configs):
            for j, ctx in enumerate(contexts):
                hit = zs[(zs.config == cfg) & (zs.context == ctx)]
                if not hit.empty:
                    matrix[i, j] = hit["z"].iloc[0]
        vmax = max(np.nanpercentile(np.abs(matrix), 95), 0.1)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels(contexts, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{label} — z-score within each instance and budget "
                     f"(green = above the mean)")
        for sep in range(3, len(contexts), 3):
            ax.axvline(sep - 0.5, color="white", linewidth=2)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z-score")
    fig.suptitle("Ablation study", fontsize=13, fontweight="bold")
    save(fig, "fig_ablation.png")


def fig_convergence(summary: pd.DataFrame) -> None:
    """Search depth and quality against the energy budget."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test results across the benchmark, by energy budget "
                 "(mean over instances)", fontsize=13, fontweight="bold")
    groups = summary.groupby(summary["instance"].map(short_instance))

    for name, grp in groups:
        if "mean_nb_iterations" in summary.columns:
            axes[0].plot(grp["budget_j"], grp["mean_nb_iterations"],
                         marker="o", label=name)
        axes[1].plot(grp["budget_j"], grp["mean_qualite_pct"],
                     marker="o", label=name)

    axes[0].set_xlabel("Energy budget (J)")
    axes[0].set_ylabel("Iterations performed")
    axes[0].set_title("Search depth against budget")
    axes[1].set_xlabel("Energy budget (J)")
    axes[1].set_ylabel("Quality (%)")
    axes[1].set_title("Quality against budget")
    axes[1].legend(fontsize=8)
    save(fig, "fig_convergence.png")


def fig_method_comparison(runs: pd.DataFrame, variants=None) -> None:
    """Green ACO against the comparison methods, on four green metrics."""
    apply_style()
    metrics = [("qualite_pct", "Quality (%)"),
               ("score_per_joule", "Quality per Joule (%/J)"),
               ("energie_joules", "Energy (J)"),
               ("co2_micrograms", "CO2 footprint (ug CO2eq)")]

    frames = [runs]
    if variants is not None and not variants.empty:
        keep = {"method", "family", "qualite_pct", "energie_joules",
                "score_per_joule", "co2_micrograms"}
        frames.append(variants[[c for c in keep if c in variants.columns]])
    data = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    fig.suptitle("Comparison of solution quality and energy efficiency "
                 "across Green ACO, ACO, RSS and GA variants",
                 fontsize=13, fontweight="bold")

    for ax, (metric, label) in zip(axes, metrics):
        agg = (data[data[metric].notna()]
               .groupby(["method", "family"])[metric].mean().reset_index()
               .sort_values(metric, ascending=False))
        colors = [FAMILY_COLORS.get(f, "#888") for f in agg["family"]]
        bars = ax.barh(agg["method"], agg[metric], color=colors, alpha=0.9)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(label.split("(")[0].strip(), fontsize=10)
        ax.invert_yaxis()
        for bar, val in zip(bars, agg[metric]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:,.4g}", va="center", fontsize=7)
    save(fig, "fig_method_comparison.png")


def table_summary(runs: pd.DataFrame, out: Path, variants=None) -> None:
    """The ranking table, written as CSV and Markdown.

    ``variants`` optionally carries additional single-method variants (RSS, DE)
    measured on the same instances, so they appear in the same comparison.
    """
    frames = [runs]
    if variants is not None and not variants.empty:
        keep = {"method", "family", "instance", "n_clauses", "qualite_pct",
                "energie_joules", "score_per_joule", "co2_micrograms"}
        frames.append(variants[[c for c in keep if c in variants.columns]])
    combined = pd.concat(frames, ignore_index=True)

    valid = combined[combined["qualite_pct"].notna()]
    if valid.empty:
        return
    ranking = rank_methods(valid, group_col="method")
    ranking.to_csv(out / "table_summary_ranking.csv", index=False)
    print("  -> wrote figures/table_summary_ranking.csv")

    md = ranking[["rang_global", "method", "qualite_moy_pct",
                  "score_per_joule", "energie_moy_j", "co2_moy_ug"]].copy()
    md.columns = ["Rank", "Method", "Quality (%)", "Quality/J (%/J)",
                  "Energy (J)", "CO2 (ug)"]
    for col in ("Quality (%)", "Quality/J (%/J)"):
        md[col] = md[col].map(lambda v: f"{v:.4f}")
    for col in ("Energy (J)", "CO2 (ug)"):
        md[col] = md[col].map(lambda v: f"{v:,.1f}")

    # Rendered by hand rather than via pandas.to_markdown, which would pull in
    # the optional `tabulate` dependency for a six-column table.
    header = "| " + " | ".join(md.columns) + " |"
    divider = "| " + " | ".join("---" for _ in md.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in md.values]
    (out / "table_summary_ranking.md").write_text(
        "\n".join([header, divider, *body]) + "\n", encoding="utf-8")
    print("  -> wrote figures/table_summary_ranking.md")


def reference_tables(profiles=None) -> None:
    """Write the paper's reference tables (1, 2, 3, 5)."""
    out = paths.FIGURES

    def _write(frame: pd.DataFrame, stem: str, title: str) -> None:
        path = out / f"{stem}.csv"
        frame.to_csv(path, index=False)
        header = "| " + " | ".join(frame.columns) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        body = ["| " + " | ".join(str(v) for v in row)
                for row in frame.itertuples(index=False)]
        (out / f"{stem}.md").write_text(
            f"{title}\n\n" + "\n".join([header, divider, *body]) + "\n",
            encoding="utf-8")
        print(f"  -> wrote {path.name} and {stem}.md")

    _write(table_carbon_intensities(), "table1_carbon_intensities",
           TITLE_CARBON)
    _write(table_predictor_features(), "table2_predictor_features",
           TITLE_PREDICTOR_FEATURES)
    _write(machine_table(), "table3_machine_configuration", TITLE_MACHINE)
    if profiles is not None:
        proxy = table_proxy_assignments(profiles)
        if not proxy.empty:
            _write(proxy, "table5_profile_transfer", TITLE_PROXY)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all figures.")
    parser.add_argument("--subset", default="all", choices=["all", "core"])
    parser.add_argument("--use-shipped", action="store_true",
                        help="Read the shipped CSVs instead of results/.")
    args = parser.parse_args()

    paths.ensure_dirs()
    apply_style()
    tag = "all54" if args.subset == "all" else args.subset
    results = paths.SHIPPED if args.use_shipped else paths.RESULTS
    out = paths.FIGURES
    print(f"  reading from {results}")

    profiles = read(f"operator_profiles_{tag}.csv", results, paths.SHIPPED)
    reference_tables(profiles)
    if profiles is not None:
        fig_operator_profiles(profiles)
        fig_operator_tests(profiles, out)

    for axis in ("operators", "mechanisms"):
        abl = read(f"ablation_{axis}_{tag}.csv", results, paths.SHIPPED)
        if abl is not None:
            fig_ablation(abl)

    summary = read(f"final_summary_{tag}.csv", results, paths.SHIPPED)
    if summary is not None:
        fig_convergence(summary)

    runs = read(f"comparison_runs_{tag}.csv", results, paths.SHIPPED)

    # Green ACO's own comparison row comes from the 4-instance sweep used for
    # the method comparison (results/shipped/green_aco_variants.csv), not from
    # the full benchmark sweep. Using the latter would compare a mean over 54
    # instances against methods measured on 4, and at a single budget against
    # means over all three.
    own = read("green_aco_variants.csv", results, paths.SHIPPED)
    if own is not None and runs is not None:
        # Replace any existing Green ACO rows rather than appending: the file
        # carries a row derived from the full benchmark sweep, which would
        # otherwise be averaged together with the comparison row.
        runs = runs[runs["method"] != "Green ACO"]
        own = own.rename(columns={"variant": "method"})
        own["method"] = "Green ACO"
        own["family"] = "ACO + EI/J"
        shared = {"method", "family", "qualite_pct", "energie_joules",
                  "score_per_joule", "co2_micrograms"}
        runs = pd.concat([runs, own[[c for c in shared if c in own.columns]]],
                         ignore_index=True)

    # Variants are averaged over all budgets, matching every other row: the
    # comparison is a mean across the three energy budgets, not a single one.
    variants = None
    for name in ("green_rss_results.csv", "green_de_results.csv"):
        frame = read(name, results, paths.SHIPPED)
        if frame is None:
            continue
        frame = frame.rename(columns={"variant": "method"})
        frame["family"] = "Green ACO variant"
        variants = frame if variants is None else pd.concat(
            [variants, frame], ignore_index=True)

    if runs is not None:
        fig_method_comparison(runs, variants)
        table_summary(runs, out, variants)

    print("\n  figures complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
