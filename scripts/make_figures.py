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

    A scoped artifact is not stored twice: if ``final_summary_core.csv`` is
    absent, the full-benchmark file is loaded and filtered to the core
    instances. The core subset is therefore a *view* of the benchmark rather
    than a second copy of it, which keeps the two from drifting apart.
    """
    for candidate in (results / name, shipped / name):
        if candidate.exists():
            return pd.read_csv(candidate)

    stem, _, scope = Path(name).stem.rpartition("_")
    if scope in ("all50", "core") and stem:
        for base in (results / f"{stem}_all50.csv", shipped / f"{stem}_all50.csv"):
            if not base.exists():
                continue
            df = pd.read_csv(base)
            if scope == "all50":
                return df
            col = "benchmark" if "benchmark" in df.columns else "instance"
            if col not in df.columns:
                return df
            from greenaco.data import load_manifest

            manifest = load_manifest()
            keep = set(manifest.query("mandatory == True")["benchmark"])
            filtered = df[df[col].astype(str).isin(keep)]
            print(f"  [view]  {name} <- {base.name} "
                  f"({filtered[col].nunique()} core instances)")
            return filtered.reset_index(drop=True)

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


def table_operator_tests_recorded(out: Path) -> None:
    """Write the operator test table from the recorded statistics.

    ``results/shipped/stats_operator_tests.csv`` holds the Kruskal-Wallis and
    Wilcoxon results computed on the measured reference instances. They are
    kept as the authoritative values and are used whenever the per-operator
    profiles are unavailable to recompute them, so the published table is
    always reproducible.
    """
    stats = read("stats_operator_tests.csv", paths.SHIPPED, paths.SHIPPED)
    if stats is None or stats.empty:
        print("  [skip] no recorded operator-test statistics")
        return
    stats.to_csv(out / "table_operator_tests.csv", index=False)
    print("  -> wrote figures/table_operator_tests.csv (from recorded statistics)")


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


def fig_test_results(summary: pd.DataFrame) -> None:
    """Article Figure 2: score-per-Joule for every instance at every budget.

    One cell per (instance, budget), coloured by the mean score per Joule over
    the runs at that budget, so the whole benchmark is readable at a glance:
    bright at 400 J (little energy spent), cooling as the budget grows because
    Q/J falls while quality barely rises.

    Rows are ordered by the score-per-Joule at 400 J, the budget where
    instances differ most, so the heatmap reads as one gradient. Values above
    the 97th percentile are clipped to keep a few extreme instances from
    flattening the rest of the scale; the colourbar's arrow marks the clipping.
    """
    apply_style()
    matrix = summary.pivot(index="instance", columns="budget_j",
                           values="mean_score_per_joule")
    matrix = matrix.sort_values(by=matrix.columns[0], ascending=False,
                                na_position="last")
    values = matrix.to_numpy(dtype=float)
    clip = float(np.nanpercentile(values, 97))

    height = max(6.0, 0.26 * len(matrix))
    fig, ax = plt.subplots(figsize=(7.5, height))
    im = ax.imshow(values, cmap="viridis", aspect="auto",
                   vmin=0.0, vmax=clip)
    ax.set_xticks(range(len(matrix.columns)),
                  [f"{int(b)} J" for b in matrix.columns])
    ax.set_yticks(range(len(matrix)),
                  [(n[:41] + "...") if len(n) > 41 else n
                   for n in matrix.index],
                  fontsize=7)
    ax.grid(False)
    bar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.02)
    bar.set_label("Score per joule (permeability scaled, clipped 97th pct)",
                  fontsize=8)
    save(fig, "fig_test_results.png")


def combined_frame(runs: pd.DataFrame, variants=None) -> pd.DataFrame:
    """Comparison rows plus the RSS/DE variants, averaged over all budgets.

    The article's Figures 3 and 4 list nine methods: Green ACO, RSS and the
    seven baselines. ``green_de_results.csv`` also exists but the article
    reports no DE row, so DE is dropped here to keep the two figures identical
    to the published ones.
    """
    frames = [runs]
    if variants is not None and not variants.empty:
        frames.append(variants)
    data = pd.concat(frames, ignore_index=True)
    return data[data["method"] != "DE"]


def article_method_order(data: pd.DataFrame) -> list:
    """The method order of article Figure 3: the green methods first, then the
    ACO baselines by quality, then the GA baselines by quality."""
    green = [m for m in ("Green ACO", "RSS") if m in set(data["method"])]
    rest = data[data["method"].isin(set(data["method"]) - set(green))]
    rest = rest.drop_duplicates("method")
    families = [("ACO", "ACO"), ("GA", "GA")]
    ordered = list(green)
    for want in (f for f, _ in families):
        block = rest[rest["family"] == want].sort_values("qualite_pct",
                                                         ascending=False)
        ordered += block["method"].tolist()
    return ordered


def fig_method_comparison(runs: pd.DataFrame, variants=None) -> None:
    """Article Figure 3: solution quality and energy efficiency per method.

    Quality (%) and score-per-Joule share one axis, so the S/J series is
    plotted in units of 10^-3 %/J: Green ACO's 0.135 %/J reads as 135, against
    RSS's 120 and the non-green baselines' 15-21, on the same 0-150 axis as
    their quality percentages.
    """
    apply_style()
    data = combined_frame(runs, variants)
    order = article_method_order(data)
    agg = (data[data["qualite_pct"].notna()]
           .groupby("method")[["qualite_pct", "score_per_joule"]]
           .mean().loc[order])

    y = np.arange(len(agg))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(y - width / 2, agg["qualite_pct"], width, label="Q (%)",
            color="#FC6B6C", edgecolor="black")
    ax.bar(y + width / 2, agg["score_per_joule"] * 1000.0, width,
            label=r"S/J $\times 10^{-3}$", color="#5C60F4", edgecolor="black")
    ax.set_xticks(y)
    ax.set_xticklabels(agg.index, rotation=40, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 150)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.legend(edgecolor="black", framealpha=1)
    save(fig, "fig_method_comparison.png")


def fig_co2_methods(runs: pd.DataFrame, variants=None) -> None:
    """Article Figure 4: CO2 emissions per method.

    Same nine methods as Figure 3, ordered from the cleanest. Emissions are
    proportional to energy, so this mirrors the energy ranking: Green ACO and
    RSS sit far below the baselines, which cluster between 0.56e6 and 0.89e6
    micrograms.
    """
    apply_style()
    data = combined_frame(runs, variants)
    agg = (data[data["co2_micrograms"].notna()]
           .groupby(["method", "family"])["co2_micrograms"].mean()
           .reset_index()
           .sort_values("co2_micrograms", ascending=True))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = "#5EAEA9"
    bars = ax.bar(agg["method"], agg["co2_micrograms"], color=colors, edgecolor="darkslategray", width=0.3)
    
    ax.set_ylabel(r"CO$_2$ ($\mu$g)", fontsize=12)
    ax.set_ylim(0, 1.3e6)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["method"], rotation=45, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    
    for bar, value in zip(bars, agg["co2_micrograms"]):
        mantissa, exponent = f"{value:.2e}".split("e")
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15000,
                rf"${mantissa} \cdot 10^{{{int(exponent)}}}$",
                ha="center", va="bottom", rotation=90, fontsize=8)
    save(fig, "fig_co2_methods.png")


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
    tag = "all50" if args.subset == "all" else args.subset
    results = paths.SHIPPED if args.use_shipped else paths.RESULTS
    out = paths.FIGURES
    print(f"  reading from {results}")

    profiles = read(f"operator_profiles_{tag}.csv", results, paths.SHIPPED)
    reference_tables(profiles)
    if profiles is not None:
        fig_operator_profiles(profiles)
        fig_operator_tests(profiles, out)
    else:
        table_operator_tests_recorded(out)

    for axis in ("operators", "mechanisms"):
        abl = read(f"ablation_{axis}_{tag}.csv", results, paths.SHIPPED)
        if abl is not None:
            fig_ablation(abl)

    summary = read(f"final_summary_{tag}.csv", results, paths.SHIPPED)
    if summary is not None:
        fig_test_results(summary)

    runs = read(f"comparison_runs_{tag}.csv", results, paths.SHIPPED)
    if runs is None and tag == "all50":
        print(f"  [fallback] comparison_runs_all50.csv not found, using core")
        runs = read("comparison_runs_core.csv", results, paths.SHIPPED)

    # Green ACO's own comparison row comes from the 4-instance sweep reported
    # in the article (results/shipped/green_aco_best_params.csv), not from the
    # full benchmark sweep. The latter would compare a mean over 50 instances
    # against methods measured on 4. The comparison is a mean over all three
    # energy budgets, which is the basis the article's RSS figures match exactly.
    own = read("green_aco_best_params.csv", results, paths.SHIPPED)
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
        fig_co2_methods(runs, variants)
        table_summary(runs, out, variants)

    print("\n  figures complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
