#!/usr/bin/env python3
"""Final comparison of Hybrid-XAI experiments vs COSMET/R2/R3.

Loads all experiment results, computes comprehensive metrics,
runs Wilcoxon signed-rank tests, and outputs summary tables.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from shared.metrics import (
    mae, mdae, mmre, pred_x, standardized_accuracy as sa,
    accuracy, r_squared, spearman_corr, mean_signed_error,
    error_distribution, cliffs_delta, nmae,
)
from shared.seeds import set_global_seeds

RESULTS_DIR = BASE / "results" / "hybrid-xai"
COSMET_DIR = BASE / "results" / "cosmet"


def load_comparison_csv(path: Path) -> dict[str, dict]:
    """Load comparison_results.csv → {fp_id: {y_true, y_pred}}."""
    data: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            fp = row.get("fp", "")
            y_true = row.get("y_true", "")
            y_pred = row.get("y_pred", "")
            if y_true == "" or y_pred == "":
                continue
            data[fp] = {
                "y_true": int(float(y_true)),
                "y_pred": int(float(y_pred)),
            }
    return data


def load_cosmet_devito() -> dict[str, dict]:
    d = COSMET_DIR / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv"
    return load_comparison_csv(d) if d.exists() else {}


def load_hybrid_xai_results() -> dict[str, dict[str, dict]]:
    """Load all Hybrid-XAI experiment results → {exp_name: {fp_id: {y_true, y_pred}}}."""
    results = {}
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        csv_path = exp_dir / "comparison_results.csv"
        if not csv_path.exists():
            continue
        data = load_comparison_csv(csv_path)
        if data:
            results[exp_dir.name] = data
    return results


def compute_all_metrics(data: dict[str, dict]) -> dict:
    fps_with_gt = {fp: d for fp, d in data.items() if d.get("y_true") is not None}
    if not fps_with_gt:
        return {}
    y_true = np.array([d["y_true"] for d in fps_with_gt.values()])
    y_pred = np.array([d["y_pred"] for d in fps_with_gt.values()])

    sp = spearman_corr(y_true, y_pred)
    err_dist = error_distribution(y_true, y_pred)

    return {
        "n": len(y_true),
        "mae": round(mae(y_true, y_pred), 3),
        "mdae": round(mdae(y_true, y_pred), 3),
        "mmre": round(mmre(y_true, y_pred), 3),
        "nmae": round(nmae(y_true, y_pred), 3),
        "pred_25": round(pred_x(y_true, y_pred, 0.25), 3),
        "sa": round(sa(y_true, y_pred), 3),
        "accuracy": round(accuracy(y_true, y_pred), 3),
        "r2": round(r_squared(y_true, y_pred), 3),
        "spearman_rho": round(sp["rho"], 3),
        "mean_signed_err": round(mean_signed_error(y_true, y_pred), 3),
        "err_p50": err_dist["p50"],
        "err_p95": err_dist["p95"],
    }


def wilcoxon_test(data_a: dict[str, dict], data_b: dict[str, dict], name_a: str, name_b: str):
    """Paired Wilcoxon signed-rank test on absolute errors."""
    common_fps = set(data_a.keys()) & set(data_b.keys())
    common_fps = [fp for fp in common_fps if data_a[fp].get("y_true") is not None and data_b[fp].get("y_true") is not None]

    if len(common_fps) < 10:
        return {"n_common": len(common_fps), "note": "too few common FPs for Wilcoxon"}

    errors_a = np.array([abs(data_a[fp]["y_pred"] - data_a[fp]["y_true"]) for fp in common_fps])
    errors_b = np.array([abs(data_b[fp]["y_pred"] - data_b[fp]["y_true"]) for fp in common_fps])

    diff = errors_a - errors_b
    if np.all(diff == 0):
        return {"n_common": len(common_fps), "statistic": 0, "p_value": 1.0, "note": "identical errors"}

    stat, p_value = sp_stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
    delta = cliffs_delta(errors_a, errors_b)

    effect = "negligible"
    if abs(delta) >= 0.474:
        effect = "large"
    elif abs(delta) >= 0.33:
        effect = "medium"
    elif abs(delta) >= 0.147:
        effect = "small"

    return {
        "n_common": len(common_fps),
        "mae_a": round(float(np.mean(errors_a)), 3),
        "mae_b": round(float(np.mean(errors_b)), 3),
        "wilcoxon_stat": round(float(stat), 3),
        "p_value": round(float(p_value), 6),
        "cliffs_delta": round(delta, 3),
        "effect_size": effect,
        "better": name_a if np.mean(errors_a) < np.mean(errors_b) else name_b,
    }


def main():
    set_global_seeds()
    print("=" * 80)
    print("FINAL COMPARISON: Hybrid-XAI vs COSMET (De Vito)")
    print("=" * 80)

    cosmet = load_cosmet_devito()
    hxai = load_hybrid_xai_results()

    if cosmet:
        print(f"\n## COSMET (gpt-4-turbo) baseline")
        m = compute_all_metrics(cosmet)
        for k, v in m.items():
            print(f"  {k:>20}: {v}")

    print(f"\n## Hybrid-XAI experiments ({len(hxai)} runs)")
    all_metrics = {}
    for exp_name, data in sorted(hxai.items()):
        m = compute_all_metrics(data)
        if not m:
            continue
        all_metrics[exp_name] = m
        print(f"\n### {exp_name}")
        for k, v in m.items():
            print(f"  {k:>20}: {v}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"{'Experiment':60} | {'N':>4} | {'MAE':>6} | {'MdAE':>5} | {'Acc':>5} | {'SA':>6} | {'PRED25':>6} | {'Rho':>5}"
    print(header)
    print("-" * len(header))

    if cosmet:
        mc = compute_all_metrics(cosmet)
        print(f"{'COSMET gpt-4-turbo (baseline)':60} | {mc['n']:>4} | {mc['mae']:>6} | {mc['mdae']:>5} | {mc['accuracy']:>5} | {mc['sa']:>6} | {mc['pred_25']:>6} | {mc['spearman_rho']:>5}")

    for exp_name, m in sorted(all_metrics.items()):
        print(f"{exp_name:60} | {m['n']:>4} | {m['mae']:>6} | {m['mdae']:>5} | {m['accuracy']:>5} | {m['sa']:>6} | {m['pred_25']:>6} | {m['spearman_rho']:>5}")

    # Wilcoxon tests (Hybrid-XAI vs COSMET)
    if cosmet:
        print(f"\n{'='*80}")
        print("WILCOXON SIGNED-RANK TESTS vs COSMET")
        print(f"{'='*80}")
        for exp_name, data in sorted(hxai.items()):
            w = wilcoxon_test(data, cosmet, exp_name, "COSMET")
            print(f"\n### {exp_name} vs COSMET")
            for k, v in w.items():
                print(f"  {k:>20}: {v}")

    # Save combined results
    output_path = RESULTS_DIR / "final_comparison.json"
    combined = {
        "cosmet_devito": compute_all_metrics(cosmet) if cosmet else {},
        "hybrid_xai": all_metrics,
    }
    output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
