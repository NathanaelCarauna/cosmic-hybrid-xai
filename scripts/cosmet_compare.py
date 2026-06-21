#!/usr/bin/env python3
"""
COSMET Comparison — compares batch runner output against ground truth.

Supports three usage patterns:

1. **original**: Compare against COSMET repo XLSX ground truth.
2. **zenodo**: Compare against Zenodo CSV ground truth.
3. Both modes accept an optional --run-id to locate predictions inside
   results/<method>/<run_id>/measurements/ and update the runs registry.

Usage:
    # Legacy (flat pred-dir)
    python scripts/cosmet_compare.py original \\
        --gt-dir replications/cosmet-devito/datasets/MIS/ALBERGATE/ground\\ truth \\
        --pred-dir output/cosmet-albergate/

    # Structured (via run-id)
    python scripts/cosmet_compare.py original \\
        --gt-dir replications/cosmet-devito/datasets/MIS/ALBERGATE/ground\\ truth \\
        --run-id albergate-gpt4turbo-20260215T143022 \\
        --method cosmet --results-dir results/

    # Zenodo
    python scripts/cosmet_compare.py zenodo \\
        --gt-csv datasets/prepared/zenodo-cosmet/sentence/ground_truth.csv \\
        --pred-dir output/cosmet-zenodo-sentence/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.metrics import (
    accuracy, mae, mdae, mmre, pred_x, r_squared, standardized_accuracy,
    movement_f1, nmae, bre, mibre, mean_signed_error, spearman_corr,
    pearson_corr, error_distribution,
)
from shared.io_utils import extract_cfp_from_csv, extract_cfp_from_xlsx, extract_movements_from_csv
from shared.results_registry import update_run
from shared.pricing import estimate_cost


# ---------------------------------------------------------------------------
# Efficiency metrics from batch_summary.csv
# ---------------------------------------------------------------------------
def _load_efficiency_metrics(pred_dir: Path) -> dict:
    """Read batch_summary.csv and compute time/cost/token efficiency stats.

    Looks in pred_dir itself and in its parent (for structured runs where
    pred_dir = .../measurements/ but batch_summary.csv sits one level up).
    """
    candidates = [
        pred_dir / "batch_summary.csv",
        pred_dir.parent / "batch_summary.csv",
    ]
    summary_path = None
    for c in candidates:
        if c.exists():
            summary_path = c
            break
    if summary_path is None:
        return {}

    df = pd.read_csv(summary_path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return {}

    for col in ["duration_s", "split_duration_s", "analyze_duration_s",
                 "input_tokens", "output_tokens", "estimated_cost_usd",
                 "split_input_tokens", "split_output_tokens",
                 "analyze_input_tokens", "analyze_output_tokens"]:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    n = len(ok)
    total_dur = ok["duration_s"].sum()
    total_cost = ok["estimated_cost_usd"].sum()
    total_input = ok["input_tokens"].sum()
    total_output = ok["output_tokens"].sum()

    result = {
        "n_processed": n,
        "total_time_s": round(total_dur, 2),
        "mean_time_per_fp_s": round(total_dur / n, 2),
        "median_time_per_fp_s": round(ok["duration_s"].median(), 2),
        "throughput_fp_per_min": round(n / (total_dur / 60), 2) if total_dur > 0 else 0,
        "total_cost_usd": round(total_cost, 4),
        "mean_cost_per_fp_usd": round(total_cost / n, 5),
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "mean_tokens_per_fp": round((total_input + total_output) / n, 0),
    }

    if "split_duration_s" in ok.columns and ok["split_duration_s"].notna().any():
        result["mean_split_time_s"] = round(ok["split_duration_s"].mean(), 2)
        result["mean_analyze_time_s"] = round(ok["analyze_duration_s"].mean(), 2)
        result["split_pct"] = round(
            ok["split_duration_s"].sum() / total_dur * 100, 1
        ) if total_dur > 0 else 0

    if "split_input_tokens" in ok.columns and ok["split_input_tokens"].notna().any():
        split_tok = ok["split_input_tokens"].sum() + ok["split_output_tokens"].sum()
        analyze_tok = ok["analyze_input_tokens"].sum() + ok["analyze_output_tokens"].sum()
        result["split_tokens_total"] = int(split_tok)
        result["analyze_tokens_total"] = int(analyze_tok)
        result["split_tokens_pct"] = round(
            split_tok / (split_tok + analyze_tok) * 100, 1
        ) if (split_tok + analyze_tok) > 0 else 0

    return result


def _print_efficiency(eff: dict) -> None:
    """Print the efficiency metrics block."""
    if not eff:
        print("\n  --- Efficiency ---")
        print("  (batch_summary.csv not found — timing/cost data unavailable)")
        return

    print(f"\n  --- Efficiency ({eff['n_processed']} FPs) ---")
    print(f"  Total time:        {eff['total_time_s']:.1f}s")
    print(f"  Mean time/FP:      {eff['mean_time_per_fp_s']:.1f}s")
    print(f"  Median time/FP:    {eff['median_time_per_fp_s']:.1f}s")
    print(f"  Throughput:        {eff['throughput_fp_per_min']:.1f} FP/min")

    if "mean_split_time_s" in eff:
        print(f"  Mean split time:   {eff['mean_split_time_s']:.1f}s ({eff['split_pct']:.0f}%)")
        print(f"  Mean analyze time: {eff['mean_analyze_time_s']:.1f}s ({100 - eff['split_pct']:.0f}%)")

    print(f"  Total cost:        US$ {eff['total_cost_usd']:.4f}")
    print(f"  Mean cost/FP:      US$ {eff['mean_cost_per_fp_usd']:.5f}")
    print(f"  Total tokens:      {eff['total_input_tokens'] + eff['total_output_tokens']:,} "
          f"(in={eff['total_input_tokens']:,} out={eff['total_output_tokens']:,})")
    print(f"  Mean tokens/FP:    {eff['mean_tokens_per_fp']:.0f}")

    if "split_tokens_pct" in eff:
        print(f"  Split tokens:      {eff['split_tokens_total']:,} ({eff['split_tokens_pct']:.0f}%)")
        print(f"  Analyze tokens:    {eff['analyze_tokens_total']:,} ({100 - eff['split_tokens_pct']:.0f}%)")


# ---------------------------------------------------------------------------
# Original mode (COSMET datasets with XLSX ground truth)
# ---------------------------------------------------------------------------
def _extract_movements_from_xlsx(path: Path) -> dict | None:
    """Extract per-type movement counts from a COSMET-format XLSX ground truth."""
    try:
        df = pd.read_excel(path, engine="openpyxl")
        dm_col = [c for c in df.columns if "DM" in c.upper()]
        if not dm_col:
            return None
        movements = df[dm_col[0]].dropna().astype(str).tolist()
        entry = sum(1 for m in movements if "entry" in m.lower())
        read = sum(1 for m in movements if "read" in m.lower())
        write = sum(1 for m in movements if "write" in m.lower())
        exit_ = sum(
            1 for m in movements
            if "exit" in m.lower() and "must be counted" not in m.lower()
        )
        exit_once = sum(1 for m in movements if "must be counted only once" in m.lower())
        exit_total = min(1, exit_once) + exit_ if exit_once else exit_
        return {"entry": entry, "read": read, "write": write, "exit": exit_total}
    except Exception:
        return None


def compare_original(gt_dir: Path, pred_dir: Path) -> dict:
    """Compare and return metrics dict (also prints to stdout)."""
    print(f"\n=== COSMET Original Ground Truth Comparison ===")
    print(f"  GT:   {gt_dir}")
    print(f"  Pred: {pred_dir}")

    gt_files = sorted(
        gt_dir.glob("*.xlsx"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if not gt_files:
        print("  No ground truth XLSX files found.")
        return {}

    results = []
    for gt_file in gt_files:
        fp_num = re.search(r"(\d+)", gt_file.stem).group(1)
        y_true = extract_cfp_from_xlsx(gt_file)
        gt_movements = _extract_movements_from_xlsx(gt_file)

        pred_xlsx = pred_dir / f"{fp_num}_measurement.xlsx"
        pred_csv = pred_dir / f"{fp_num}_measurement.csv"
        y_pred = None
        pred_movements = None
        if pred_csv.exists():
            y_pred = extract_cfp_from_csv(pred_csv)
            pred_movements = extract_movements_from_csv(pred_csv)
        elif pred_xlsx.exists():
            y_pred = extract_cfp_from_xlsx(pred_xlsx)

        row = {"fp": gt_file.stem, "y_true": y_true, "y_pred": y_pred}
        for mov in ["entry", "read", "write", "exit"]:
            row[f"{mov}_gt"] = gt_movements[mov] if gt_movements else None
            row[f"{mov}_pred"] = pred_movements[mov] if pred_movements else None
        results.append(row)

    df = pd.DataFrame(results)
    matched = df.dropna(subset=["y_true", "y_pred"])
    missing = df[df["y_pred"].isna()]

    print(f"\n  Ground truth FPs: {len(df)}")
    print(f"  Matched predictions: {len(matched)}")
    if not missing.empty:
        print(f"  Missing predictions: {len(missing)} ({', '.join(missing['fp'].tolist()[:10])})")

    if matched.empty:
        print("  No matched predictions to compare.")
        return {}

    yt = matched["y_true"].values.astype(float)
    yp = matched["y_pred"].values.astype(float)

    sp = spearman_corr(yt, yp)
    pr = pearson_corr(yt, yp)
    err_dist = error_distribution(yt, yp)

    computed = {
        "mae": mae(yt, yp),
        "mdae": mdae(yt, yp),
        "mmre": mmre(yt, yp),
        "nmae": nmae(yt, yp),
        "bre": bre(yt, yp),
        "mibre": mibre(yt, yp),
        "pred25": pred_x(yt, yp, 0.25),
        "pred30": pred_x(yt, yp, 0.30),
        "r2": r_squared(yt, yp),
        "sa": standardized_accuracy(yt, yp),
        "exact_match": accuracy(yt, yp),
        "mean_signed_error": mean_signed_error(yt, yp),
        "spearman_rho": sp["rho"],
        "pearson_r": pr["r"],
        "error_p5": err_dist["p5"],
        "error_p25": err_dist["p25"],
        "error_p50": err_dist["p50"],
        "error_p75": err_dist["p75"],
        "error_p95": err_dist["p95"],
        "error_max": err_dist["max"],
    }

    print(f"\n  --- CFP Accuracy (n={len(matched)}) ---")
    print(f"  MAE:          {computed['mae']:.3f}")
    print(f"  MdAE:         {computed['mdae']:.3f}")
    print(f"  MMRE:         {computed['mmre']:.4f}")
    print(f"  NMAE:         {computed['nmae']:.4f}")
    print(f"  BRE:          {computed['bre']:.4f}")
    print(f"  MIBRE:        {computed['mibre']:.4f}")
    print(f"  PRED(25):     {computed['pred25']:.3f}")
    print(f"  PRED(30):     {computed['pred30']:.3f}")
    print(f"  R²:           {computed['r2']:.3f}")
    print(f"  SA:           {computed['sa']:.3f}")
    print(f"  Exact Match:  {computed['exact_match']:.3f}")

    print(f"\n  --- Bias & Correlation ---")
    bias_dir = "overestimation" if computed["mean_signed_error"] > 0 else "underestimation"
    print(f"  Mean Signed Error: {computed['mean_signed_error']:+.3f} ({bias_dir})")
    print(f"  Spearman ρ:   {computed['spearman_rho']:.3f}")
    print(f"  Pearson r:    {computed['pearson_r']:.3f}")

    print(f"\n  --- Error Distribution ---")
    print(f"  P5={err_dist['p5']:.1f}  P25={err_dist['p25']:.1f}  "
          f"P50={err_dist['p50']:.1f}  P75={err_dist['p75']:.1f}  "
          f"P95={err_dist['p95']:.1f}  Max={err_dist['max']:.1f}")

    m_df = matched.dropna(subset=["entry_pred", "read_pred", "write_pred", "exit_pred"])
    if not m_df.empty:
        m_df_gt = m_df.dropna(subset=["entry_gt", "read_gt", "write_gt", "exit_gt"])
        if not m_df_gt.empty:
            print(f"\n  --- Per-Movement Metrics (n={len(m_df_gt)}) ---")
            for mov in ["entry", "read", "write", "exit"]:
                gt_col = m_df_gt[f"{mov}_gt"].values.astype(float)
                pred_col = m_df_gt[f"{mov}_pred"].values.astype(float)
                acc = accuracy(gt_col, pred_col)
                m = mae(gt_col, pred_col)
                print(f"  {mov:>6s}:  accuracy={acc:.3f}  MAE={m:.3f}")
                computed[f"{mov}_accuracy"] = acc
                computed[f"{mov}_mae"] = m

    eff = _load_efficiency_metrics(pred_dir)
    _print_efficiency(eff)
    for k, v in eff.items():
        computed[f"eff_{k}"] = v

    out_csv = pred_dir / "comparison_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Detailed results: {out_csv}")

    return computed


# ---------------------------------------------------------------------------
# Zenodo mode (CSV ground truth)
# ---------------------------------------------------------------------------
def compare_zenodo(gt_csv: Path, pred_dir: Path) -> dict:
    """Compare and return metrics dict (also prints to stdout)."""
    print(f"\n=== Zenodo Ground Truth Comparison ===")
    print(f"  GT:   {gt_csv}")
    print(f"  Pred: {pred_dir}")

    gt = pd.read_csv(gt_csv)
    results = []

    for _, row in gt.iterrows():
        fp_num = row["fp_num"]
        cfp_gt = row["cfp_gt"]

        pred_csv = pred_dir / f"{fp_num}_measurement.csv"
        pred_xlsx = pred_dir / f"{fp_num}_measurement.xlsx"

        cfp_pred = None
        movements_pred = None
        if pred_csv.exists():
            cfp_pred = extract_cfp_from_csv(pred_csv)
            movements_pred = extract_movements_from_csv(pred_csv)
        elif pred_xlsx.exists():
            cfp_pred = extract_cfp_from_xlsx(pred_xlsx)

        entry_pred = movements_pred["entry"] if movements_pred else None
        read_pred = movements_pred["read"] if movements_pred else None
        write_pred = movements_pred["write"] if movements_pred else None
        exit_pred = movements_pred["exit"] if movements_pred else None

        results.append({
            "fp_num": fp_num,
            "cfp_gt": cfp_gt,
            "cfp_pred": cfp_pred,
            "entry_gt": row.get("entry_gt"),
            "read_gt": row.get("read_gt"),
            "write_gt": row.get("write_gt"),
            "exit_gt": row.get("exit_gt"),
            "entry_pred": entry_pred,
            "read_pred": read_pred,
            "write_pred": write_pred,
            "exit_pred": exit_pred,
        })

    df = pd.DataFrame(results)
    matched = df.dropna(subset=["cfp_gt", "cfp_pred"])

    print(f"\n  Ground truth items: {len(df)}")
    print(f"  Matched predictions: {len(matched)}")
    n_missing = len(df) - len(matched)
    if n_missing:
        print(f"  Missing predictions: {n_missing}")

    if matched.empty:
        print("  No matched predictions to compare.")
        return {}

    yt = matched["cfp_gt"].values.astype(float)
    yp = matched["cfp_pred"].values.astype(float)

    sp = spearman_corr(yt, yp)
    pr = pearson_corr(yt, yp)
    err_dist = error_distribution(yt, yp)

    computed = {
        "mae": mae(yt, yp),
        "mdae": mdae(yt, yp),
        "mmre": mmre(yt, yp),
        "nmae": nmae(yt, yp),
        "bre": bre(yt, yp),
        "mibre": mibre(yt, yp),
        "pred25": pred_x(yt, yp, 0.25),
        "pred30": pred_x(yt, yp, 0.30),
        "r2": r_squared(yt, yp),
        "sa": standardized_accuracy(yt, yp),
        "exact_match": accuracy(yt, yp),
        "mean_signed_error": mean_signed_error(yt, yp),
        "spearman_rho": sp["rho"],
        "pearson_r": pr["r"],
        "error_p5": err_dist["p5"],
        "error_p25": err_dist["p25"],
        "error_p50": err_dist["p50"],
        "error_p75": err_dist["p75"],
        "error_p95": err_dist["p95"],
        "error_max": err_dist["max"],
    }

    print(f"\n  --- CFP Accuracy (n={len(matched)}) ---")
    print(f"  MAE:          {computed['mae']:.3f}")
    print(f"  MdAE:         {computed['mdae']:.3f}")
    print(f"  MMRE:         {computed['mmre']:.4f}")
    print(f"  NMAE:         {computed['nmae']:.4f}")
    print(f"  BRE:          {computed['bre']:.4f}")
    print(f"  MIBRE:        {computed['mibre']:.4f}")
    print(f"  PRED(25):     {computed['pred25']:.3f}")
    print(f"  PRED(30):     {computed['pred30']:.3f}")
    print(f"  R²:           {computed['r2']:.3f}")
    print(f"  SA:           {computed['sa']:.3f}")
    print(f"  Exact Match:  {computed['exact_match']:.3f}")

    print(f"\n  --- Bias & Correlation ---")
    bias_dir = "overestimation" if computed["mean_signed_error"] > 0 else "underestimation"
    print(f"  Mean Signed Error: {computed['mean_signed_error']:+.3f} ({bias_dir})")
    print(f"  Spearman ρ:   {computed['spearman_rho']:.3f}")
    print(f"  Pearson r:    {computed['pearson_r']:.3f}")

    print(f"\n  --- Error Distribution ---")
    print(f"  P5={err_dist['p5']:.1f}  P25={err_dist['p25']:.1f}  "
          f"P50={err_dist['p50']:.1f}  P75={err_dist['p75']:.1f}  "
          f"P95={err_dist['p95']:.1f}  Max={err_dist['max']:.1f}")

    has_movements = (
        all(c in matched.columns for c in ["entry_gt", "entry_pred"])
        and matched["entry_pred"].notna().any()
    )
    if has_movements:
        m_df = matched.dropna(subset=["entry_pred", "read_pred", "write_pred", "exit_pred"])
        if not m_df.empty:
            print(f"\n  --- Per-Movement Accuracy (n={len(m_df)}) ---")
            for mov in ["entry", "read", "write", "exit"]:
                gt_col = m_df[f"{mov}_gt"].values.astype(float)
                pred_col = m_df[f"{mov}_pred"].values.astype(float)
                acc = accuracy(gt_col, pred_col)
                m = mae(gt_col, pred_col)
                print(f"  {mov:>6s}:  accuracy={acc:.3f}  MAE={m:.3f}")

    eff = _load_efficiency_metrics(pred_dir)
    _print_efficiency(eff)
    for k, v in eff.items():
        computed[f"eff_{k}"] = v

    out_csv = pred_dir / "comparison_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Detailed results: {out_csv}")

    return computed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare predictions against ground truth")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("original", help="Compare against COSMET XLSX ground truth")
    p1.add_argument("--gt-dir", type=Path, required=True, help="Dir with ground truth FPn.xlsx files")
    p1.add_argument("--pred-dir", type=Path, default=None, help="Dir with batch runner output")
    p1.add_argument("--run-id", default=None, help="Run ID to locate in results/<method>/<run_id>/measurements/")
    p1.add_argument("--method", default="cosmet", help="Method name for structured lookup")
    p1.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results dir")

    p2 = sub.add_parser("zenodo", help="Compare against Zenodo CSV ground truth")
    p2.add_argument("--gt-csv", type=Path, required=True, help="ground_truth.csv from zenodo adapter")
    p2.add_argument("--pred-dir", type=Path, default=None, help="Dir with batch runner output")
    p2.add_argument("--run-id", default=None, help="Run ID to locate in results/<method>/<run_id>/measurements/")
    p2.add_argument("--method", default="cosmet", help="Method name for structured lookup")
    p2.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results dir")

    args = parser.parse_args()

    pred_dir = args.pred_dir
    run_dir = None

    if args.run_id:
        run_dir = args.results_dir / args.method / args.run_id
        pred_dir = run_dir / "measurements"
        if not pred_dir.exists():
            sys.exit(f"ERROR: measurements dir not found: {pred_dir}")

    if pred_dir is None:
        parser.error("Either --pred-dir or --run-id is required")

    if args.mode == "original":
        computed = compare_original(args.gt_dir, pred_dir)
    elif args.mode == "zenodo":
        computed = compare_zenodo(args.gt_csv, pred_dir)
    else:
        computed = {}

    if args.run_id and computed:
        update_run(args.results_dir, args.run_id, {
            "mae": round(computed.get("mae", 0), 4),
            "mmre": round(computed.get("mmre", 0), 4),
            "pred25": round(computed.get("pred25", 0), 4),
        })
        print(f"\n  Registry updated for run_id={args.run_id}")
        print(f"  (Full metric set stored in run_metrics.json and comparison_results.csv)")

        if run_dir:
            import json
            metrics_path = run_dir / "run_metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics.setdefault("accuracy", {})
                metrics["accuracy"].update({
                    k: round(v, 4) for k, v in computed.items()
                })
                metrics_path.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"  run_metrics.json updated with accuracy block")


if __name__ == "__main__":
    main()
