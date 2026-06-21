#!/usr/bin/env python3
"""
Aggregate Results — cross-method comparative analysis for the dissertation.

Reads all experiment runs from the runs_registry (or discovers run_metrics.json
files), builds comparison tables, exports LaTeX, generates plots, and runs
statistical significance tests.

Usage:
    python scripts/aggregate_results.py [--results-dir results/] [--output-dir results/analysis/]

Filters:
    --method cosmet --method-type replication --dataset albergate --llm-model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.results_registry import load_registry

NUMERIC_COLS = [
    "mae", "mdae", "mmre", "nmae", "bre", "mibre",
    "pred25", "pred30", "r2", "sa", "exact_match",
    "mean_signed_error", "spearman_rho", "pearson_r",
    "wall_clock_s", "total_tokens", "total_cost_usd", "throughput_fp_min",
    "n_fps", "n_errors",
]


# ---------------------------------------------------------------------------
# Step 1 — Discovery
# ---------------------------------------------------------------------------
def discover(results_dir: Path, filters: dict) -> pd.DataFrame:
    """Load the registry and apply optional filters."""
    df = load_registry(results_dir)
    if df.empty:
        df = _fallback_scan(results_dir)
    if df.empty:
        return df

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for key, val in filters.items():
        if val and key in df.columns:
            df = df[df[key] == val]

    return df.reset_index(drop=True)


def _fallback_scan(results_dir: Path) -> pd.DataFrame:
    """Walk results/ for run_metrics.json when no registry exists."""
    rows = []
    for metrics_file in results_dir.rglob("run_metrics.json"):
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        acc = data.get("accuracy", {})
        perf = data.get("performance", {})
        cost = data.get("cost", {})
        tokens = data.get("tokens", {})
        rows.append({
            "run_id": meta.get("run_id", metrics_file.parent.name),
            "method": meta.get("method", metrics_file.parent.parent.name),
            "method_type": meta.get("method_type", ""),
            "dataset": meta.get("dataset", ""),
            "llm_model": meta.get("llm_model", ""),
            "timestamp": meta.get("timestamp", ""),
            "n_fps": meta.get("n_fps", 0),
            "n_errors": meta.get("n_errors", 0),
            "total_cfp_pred": acc.get("total_cfp_pred", ""),
            "mae": acc.get("mae", ""),
            "mmre": acc.get("mmre", ""),
            "pred25": acc.get("pred25", ""),
            "pred30": acc.get("pred30", ""),
            "r2": acc.get("r2", ""),
            "wall_clock_s": perf.get("wall_clock_s", ""),
            "total_tokens": tokens.get("total", ""),
            "total_cost_usd": cost.get("total_usd", ""),
            "throughput_fp_min": perf.get("fp_per_min", ""),
            "status": "ok" if meta.get("n_errors", 0) == 0 else "partial",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison table with method x dataset x llm_model axes."""
    if df.empty:
        return df

    agg_cols = [c for c in NUMERIC_COLS if c in df.columns and df[c].notna().any()]
    group_cols = [c for c in ["method", "method_type", "dataset", "llm_model"] if c in df.columns]

    if not group_cols:
        return df

    grouped = df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
    return grouped.sort_values(group_cols)


# ---------------------------------------------------------------------------
# Step 3 — LaTeX export
# ---------------------------------------------------------------------------
def export_latex(df: pd.DataFrame, output_path: Path) -> None:
    """Export the comparison table as a LaTeX tabular environment."""
    if df.empty:
        return

    display_cols = [c for c in df.columns if c != "method_type"]
    lines = []
    col_spec = "l" * len(display_cols)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_map = {
        "method": "Method", "dataset": "Dataset", "llm_model": "LLM",
        "mae": "MAE", "mdae": "MdAE", "mmre": "MMRE",
        "nmae": "NMAE", "bre": "BRE", "mibre": "MIBRE",
        "pred25": "PRED(25)", "pred30": "PRED(30)", "r2": "$R^2$",
        "sa": "SA", "exact_match": "Exact",
        "mean_signed_error": "Bias", "spearman_rho": "$\\rho$", "pearson_r": "$r$",
        "wall_clock_s": "Time (s)", "total_tokens": "Tokens",
        "total_cost_usd": "Cost (USD)", "throughput_fp_min": "FP/min",
        "n_fps": "n", "n_errors": "Errors",
    }
    headers = [header_map.get(c, c) for c in display_cols]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        is_own = str(row.get("method_type", "")).strip() == "own"
        cells = []
        for c in display_cols:
            val = row[c]
            if isinstance(val, float) and not np.isnan(val):
                if c in ("mae", "mdae", "mmre", "pred25", "pred30", "r2"):
                    formatted = f"{val:.3f}"
                elif c == "total_cost_usd":
                    formatted = f"{val:.4f}"
                elif c in ("wall_clock_s", "throughput_fp_min"):
                    formatted = f"{val:.1f}"
                else:
                    formatted = f"{val:.0f}"
            else:
                formatted = str(val) if pd.notna(val) else "--"
            cells.append(formatted)

        line = " & ".join(cells) + " \\\\"
        if is_own:
            line = f"\\textbf{{{line[:-3]}}}" + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table: {output_path}")


# ---------------------------------------------------------------------------
# Step 4 — Plots
# ---------------------------------------------------------------------------
def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparative visualisation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "mae" not in df.columns or df["mae"].isna().all():
        print("  (no accuracy data to plot)")
        return

    palette = {"replication": "#5b9bd5", "own": "#e86c3a"}

    _plot_mae_by_method(df, output_dir, palette)
    _plot_cost_vs_accuracy(df, output_dir, palette)
    _plot_timing(df, output_dir, palette)
    _plot_token_distribution(df, output_dir)
    _plot_own_vs_baselines(df, output_dir, palette)


def _safe_palette(df: pd.DataFrame, col: str, palette: dict) -> dict:
    """Return only palette entries that exist in the data."""
    vals = df[col].dropna().unique()
    return {k: v for k, v in palette.items() if k in vals}


def _plot_mae_by_method(df: pd.DataFrame, out: Path, palette: dict) -> None:
    plot_df = df.dropna(subset=["mae"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    safe = _safe_palette(plot_df, "method_type", palette)
    sns.barplot(data=plot_df, x="dataset", y="mae", hue="method",
                palette=None, ax=ax)
    ax.set_title("MAE by Method and Dataset")
    ax.set_ylabel("MAE (lower is better)")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "mae_by_method.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: mae_by_method.png")


def _plot_cost_vs_accuracy(df: pd.DataFrame, out: Path, palette: dict) -> None:
    plot_df = df.dropna(subset=["mae", "total_cost_usd"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    safe = _safe_palette(plot_df, "method_type", palette)
    for mt, grp in plot_df.groupby("method_type"):
        color = palette.get(mt, "#999")
        marker = "D" if mt == "own" else "o"
        sizes = grp["n_fps"].fillna(10).astype(float) * 5
        ax.scatter(grp["total_cost_usd"], grp["mae"], s=sizes,
                   c=color, marker=marker, alpha=0.8, label=mt, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Cost vs Accuracy")
    ax.legend(title="Type")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: cost_vs_accuracy.png")


def _plot_timing(df: pd.DataFrame, out: Path, palette: dict) -> None:
    plot_df = df.dropna(subset=["wall_clock_s"])
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=plot_df, x="method", y="wall_clock_s", hue="llm_model", ax=axes[0])
    axes[0].set_title("Wall Clock Time by Method")
    axes[0].set_ylabel("Seconds")

    tp_df = plot_df.dropna(subset=["throughput_fp_min"])
    if not tp_df.empty:
        sns.barplot(data=tp_df, x="method", y="throughput_fp_min", hue="llm_model", ax=axes[1])
    axes[1].set_title("Throughput by Method")
    axes[1].set_ylabel("FP / min")

    fig.tight_layout()
    fig.savefig(out / "timing_by_model.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: timing_by_model.png")


def _plot_token_distribution(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.dropna(subset=["total_tokens"])
    if plot_df.empty or len(plot_df) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="method", y="total_tokens", ax=ax)
    ax.set_title("Token Distribution by Method")
    ax.set_ylabel("Total Tokens")
    fig.tight_layout()
    fig.savefig(out / "token_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: token_distribution.png")


def _plot_own_vs_baselines(df: pd.DataFrame, out: Path, palette: dict) -> None:
    """Grouped bar: own model vs each baseline on the same dataset."""
    own = df[df["method_type"] == "own"]
    rep = df[df["method_type"] == "replication"]
    if own.empty or rep.empty:
        return

    metric_cols = ["mae", "mmre", "pred25"]
    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return

    combined = pd.concat([own, rep])
    datasets = combined["dataset"].dropna().unique()

    for ds in datasets:
        sub = combined[combined["dataset"] == ds].dropna(subset=available)
        if sub.empty or len(sub) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        melted = sub.melt(
            id_vars=["method", "method_type"],
            value_vars=available,
            var_name="metric",
            value_name="value",
        )
        colors = [palette.get(mt, "#999") for mt in melted["method_type"]]
        sns.barplot(data=melted, x="metric", y="value", hue="method", ax=ax)
        ax.set_title(f"Own Model vs Baselines — {ds}")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        safe_ds = ds.replace("/", "_")
        fig.savefig(out / f"own_vs_baselines_{safe_ds}.png", dpi=150)
        plt.close(fig)
        print(f"  Plot: own_vs_baselines_{safe_ds}.png")


# ---------------------------------------------------------------------------
# Step 5 — Statistical tests
# ---------------------------------------------------------------------------
def run_statistical_tests(df: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    """Wilcoxon signed-rank test + Cliff's delta between own model and each replication."""
    own = df[df["method_type"] == "own"]
    rep = df[df["method_type"] == "replication"]
    if own.empty or rep.empty:
        print("  (skip statistical tests — need both own and replication runs)")
        return

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  (scipy not installed — skipping statistical tests)")
        return

    from shared.metrics import cliffs_delta

    test_results = []
    datasets = df["dataset"].dropna().unique()
    rep_methods = rep["method"].unique()

    for ds in datasets:
        own_ds = own[own["dataset"] == ds]
        if own_ds.empty:
            continue

        for rm in rep_methods:
            rep_ds = rep[(rep["dataset"] == ds) & (rep["method"] == rm)]
            if rep_ds.empty:
                continue

            for metric in ["mae", "mmre", "pred25"]:
                own_vals = _load_per_fp_metric(results_dir, own_ds, metric)
                rep_vals = _load_per_fp_metric(results_dir, rep_ds, metric)

                if own_vals is None or rep_vals is None or len(own_vals) != len(rep_vals):
                    continue
                if len(own_vals) < 5:
                    continue

                try:
                    stat, p_value = wilcoxon(own_vals, rep_vals)
                    n = len(own_vals)
                    effect_r = abs(stat) / np.sqrt(n) if n > 0 else 0
                    cd = cliffs_delta(own_vals, rep_vals)
                    cd_mag = (
                        "negligible" if abs(cd) < 0.147
                        else "small" if abs(cd) < 0.33
                        else "medium" if abs(cd) < 0.474
                        else "large"
                    )
                    test_results.append({
                        "dataset": ds,
                        "own_method": own_ds["method"].iloc[0],
                        "baseline_method": rm,
                        "metric": metric,
                        "n_pairs": n,
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "significant_005": p_value < 0.05,
                        "effect_size_r": round(effect_r, 4),
                        "cliffs_delta": round(cd, 4),
                        "cliffs_magnitude": cd_mag,
                    })
                except Exception:
                    pass

    if test_results:
        out_path = output_dir / "statistical_tests.csv"
        pd.DataFrame(test_results).to_csv(out_path, index=False)
        print(f"  Statistical tests: {out_path}")
    else:
        print("  (no paired comparisons available for statistical tests)")


def _load_per_fp_metric(
    results_dir: Path, runs_df: pd.DataFrame, metric: str
) -> np.ndarray | None:
    """Load per-FP comparison_results.csv and extract paired values."""
    for _, run in runs_df.iterrows():
        run_id = run.get("run_id", "")
        method = run.get("method", "")
        comp_file = results_dir / method / run_id / "comparison_results.csv"
        if not comp_file.exists():
            comp_file = results_dir / method / run_id / "measurements" / "comparison_results.csv"
        if not comp_file.exists():
            continue
        try:
            comp_df = pd.read_csv(comp_file)
            if "y_true" in comp_df.columns and "y_pred" in comp_df.columns:
                matched = comp_df.dropna(subset=["y_true", "y_pred"])
                errors = np.abs(matched["y_true"].values - matched["y_pred"].values)
                return errors
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Step 6 — Report
# ---------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Write analysis_report.json with structured summary."""
    if df.empty:
        return

    own = df[df["method_type"] == "own"]
    rep = df[df["method_type"] == "replication"]

    report = {
        "total_runs": len(df),
        "methods": df["method"].unique().tolist(),
        "datasets": df["dataset"].unique().tolist(),
        "llm_models": df["llm_model"].unique().tolist() if "llm_model" in df.columns else [],
        "own_model_runs": len(own),
        "replication_runs": len(rep),
    }

    if not own.empty and not rep.empty:
        for metric in ["mae", "mmre", "pred25"]:
            if metric in df.columns:
                own_mean = own[metric].mean()
                rep_mean = rep[metric].mean()
                delta = own_mean - rep_mean
                report[f"own_vs_rep_{metric}"] = {
                    "own_mean": round(float(own_mean), 4) if pd.notna(own_mean) else None,
                    "rep_mean": round(float(rep_mean), 4) if pd.notna(rep_mean) else None,
                    "delta": round(float(delta), 4) if pd.notna(delta) else None,
                }

    ranking_metric = "mae"
    if ranking_metric in df.columns and df[ranking_metric].notna().any():
        ranked = df.groupby("method")[ranking_metric].mean().sort_values()
        report["ranking_by_mae"] = {
            m: round(float(v), 4) for m, v in ranked.items()
        }

    path = output_dir / "analysis_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Report: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate cross-method analysis")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root results directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir for analysis (default: results/analysis/)")
    parser.add_argument("--method", default=None, help="Filter by method name")
    parser.add_argument("--method-type", default=None, help="Filter: replication | own")
    parser.add_argument("--dataset", default=None, help="Filter by dataset")
    parser.add_argument("--llm-model", default=None, help="Filter by LLM model")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    filters = {
        "method": args.method,
        "method_type": args.method_type,
        "dataset": args.dataset,
        "llm_model": args.llm_model,
    }

    print(f"=== Aggregate Analysis ===")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Output dir:  {output_dir}")
    filters_str = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
    if filters_str:
        print(f"  Filters:     {filters_str}")

    df = discover(args.results_dir, filters)

    if df.empty:
        print("\n  No runs found. Nothing to analyse.")
        return

    print(f"\n  Found {len(df)} runs")
    print(f"  Methods: {df['method'].unique().tolist()}")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")

    table = build_comparison_table(df)
    table_csv = output_dir / "comparison_table.csv"
    table.to_csv(table_csv, index=False)
    print(f"\n  Comparison table: {table_csv}")

    export_latex(table, output_dir / "comparison_table.tex")
    generate_plots(table, plots_dir)
    run_statistical_tests(df, args.results_dir, output_dir)
    generate_report(df, output_dir)

    print(f"\nAnalysis complete: {output_dir}")


if __name__ == "__main__":
    main()
