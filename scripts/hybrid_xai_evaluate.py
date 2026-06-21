#!/usr/bin/env python3
"""Hybrid XAI Evaluation — compares all experiment runs and generates
the final comparison table + Wilcoxon statistical tests.

Usage:
    python scripts/hybrid_xai_evaluate.py [--results-dir results/hybrid-xai]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.metrics import (
    mae, mdae, mmre, pred_x, standardized_accuracy, accuracy,
    spearman_corr, pearson_corr,
)


def load_run(run_dir: Path) -> dict:
    """Load comparison_results.csv and metrics.json from a run."""
    metrics_path = run_dir / "metrics.json"
    csv_path = run_dir / "comparison_results.csv"

    data = {"run_name": run_dir.name, "run_dir": str(run_dir)}

    if metrics_path.exists():
        data["metrics"] = json.loads(metrics_path.read_text())

    if csv_path.exists():
        y_true, y_pred = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("y_true") and row["y_true"] != "":
                    y_true.append(int(row["y_true"]))
                    y_pred.append(int(row["y_pred"]))
        data["y_true"] = y_true
        data["y_pred"] = y_pred
    else:
        data["y_true"] = []
        data["y_pred"] = []

    return data


def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test between two error vectors."""
    from scipy.stats import wilcoxon

    diffs = errors_a - errors_b
    diffs = diffs[diffs != 0]
    if len(diffs) < 5:
        return {"statistic": None, "p_value": None, "n_pairs": len(diffs)}

    stat, p = wilcoxon(diffs)
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": len(diffs)}


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / n)


def main():
    parser = argparse.ArgumentParser(description="Hybrid XAI Evaluation")
    parser.add_argument(
        "--results-dir", type=Path,
        default=BASE / "results" / "hybrid-xai",
    )
    parser.add_argument(
        "--baseline-csv", type=Path, default=None,
        help="Path to COSMET comparison_results.csv for cross-method comparison",
    )
    args = parser.parse_args()

    runs_dir = args.results_dir
    if not runs_dir.exists():
        print(f"No results directory found at {runs_dir}")
        return

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        print("No experiment runs found.")
        return

    runs = [load_run(d) for d in run_dirs]

    # --- Summary table ---
    print("=" * 90)
    print("Hybrid XAI — Experiment Comparison")
    print("=" * 90)
    print(f"{'Run':<45} {'N':>4} {'MAE':>6} {'MdAE':>6} {'SA':>6} {'Acc':>6} {'PRED25':>7} {'t/FP':>6}")
    print("-" * 90)

    for run in runs:
        m = run.get("metrics", {})
        name = run["run_name"][:44]
        n = m.get("n_with_gt", m.get("n_fps", 0))
        print(
            f"{name:<45} "
            f"{n:>4} "
            f"{m.get('mae', '-'):>6} "
            f"{m.get('mdae', '-'):>6} "
            f"{m.get('sa', '-'):>6} "
            f"{m.get('accuracy', '-'):>6} "
            f"{m.get('pred_25', '-'):>7} "
            f"{m.get('avg_time_s', '-'):>6}"
        )

    # --- Paired Wilcoxon tests between runs ---
    print(f"\n{'=' * 90}")
    print("Paired Wilcoxon Signed-Rank Tests")
    print("=" * 90)

    valid_runs = [r for r in runs if len(r["y_true"]) > 5]

    if len(valid_runs) >= 2:
        print(f"{'Run A':<30} {'Run B':<30} {'p-value':>10} {'delta':>8}")
        print("-" * 90)

        for i, run_a in enumerate(valid_runs):
            for run_b in valid_runs[i + 1:]:
                yt_a, yp_a = np.array(run_a["y_true"]), np.array(run_a["y_pred"])
                yt_b, yp_b = np.array(run_b["y_true"]), np.array(run_b["y_pred"])

                min_len = min(len(yt_a), len(yt_b))
                if min_len < 5:
                    continue

                err_a = np.abs(yt_a[:min_len] - yp_a[:min_len])
                err_b = np.abs(yt_b[:min_len] - yp_b[:min_len])

                result = wilcoxon_test(err_a, err_b)
                delta = cliffs_delta(err_a, err_b)

                p_str = f"{result['p_value']:.4f}" if result["p_value"] is not None else "N/A"
                name_a = run_a["run_name"][:29]
                name_b = run_b["run_name"][:29]
                print(f"{name_a:<30} {name_b:<30} {p_str:>10} {delta:>+8.3f}")

    # --- Cross-method comparison with COSMET ---
    if args.baseline_csv and args.baseline_csv.exists():
        print(f"\n{'=' * 90}")
        print("Cross-Method Comparison (vs COSMET baseline)")
        print("=" * 90)

        cosmet_gt, cosmet_pred = [], []
        with open(args.baseline_csv) as f:
            for row in csv.DictReader(f):
                cosmet_gt.append(int(row["y_true"]))
                cosmet_pred.append(int(row["y_pred"]))

        cosmet_mae = mae(cosmet_gt, cosmet_pred)
        cosmet_sa = standardized_accuracy(cosmet_gt, cosmet_pred)

        print(f"COSMET baseline: MAE={cosmet_mae:.3f}, SA={cosmet_sa:.3f}")

        for run in valid_runs:
            m = run.get("metrics", {})
            run_mae = m.get("mae", "?")
            run_sa = m.get("sa", "?")
            name = run["run_name"][:40]
            print(f"  {name}: MAE={run_mae}, SA={run_sa}")

    print()


if __name__ == "__main__":
    main()
