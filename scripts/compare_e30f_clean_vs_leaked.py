#!/usr/bin/env python3
"""T6 — Compare E30f-clean-LOOCV vs E30f-leaked to quantify leakage inflation.

Reads metrics.json from two experiment dirs (leaked baseline and clean run)
and produces a side-by-side comparison table in CSV + Markdown.

Usage:
  python3 scripts/compare_e30f_clean_vs_leaked.py \
    --leaked  results/hybrid-xai/bert-improvements-turbo/E30f-correction-ensemble \
    --clean   results/hybrid-xai/bert-improvements-clean-loocv/E30f-correction-ensemble \
    --output  results/leakage-impact
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = [
    ("mae",         "MAE",        4, "lower is better"),
    ("mdae",        "MdAE",       4, "lower is better"),
    ("mmre",        "MMRE",       4, "lower is better"),
    ("pred_25",     "PRED(25)",   4, "higher is better"),
    ("sa",          "SA",         4, "higher is better"),
    ("r2",          "R²",         4, "higher is better"),
    ("exact_match", "ExactMatch", 4, "higher is better"),
    ("spearman",    "Spearman",   4, "higher is better"),
]


def load(path: Path) -> dict:
    p = Path(path)
    candidates = [p, p / "metrics.json"]
    for c in candidates:
        if c.exists() and c.name == "metrics.json":
            return json.loads(c.read_text())
        if c.is_dir():
            m = c / "metrics.json"
            if m.exists():
                return json.loads(m.read_text())
    raise FileNotFoundError(f"metrics.json not found at {path}")


def pct_change(new_val: float, old_val: float) -> float:
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / abs(old_val) * 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaked", required=True,
                        help="Path to leaked (original) experiment dir")
    parser.add_argument("--clean", required=True,
                        help="Path to clean (no-leakage) experiment dir")
    parser.add_argument("--output", default="results/leakage-impact",
                        help="Output directory")
    args = parser.parse_args()

    leaked = load(args.leaked)
    clean = load(args.clean)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, decimals, direction in METRICS:
        lv = leaked.get(key)
        cv = clean.get(key)
        if lv is None and cv is None:
            continue
        lv_s = f"{lv:.{decimals}f}" if lv is not None else "N/A"
        cv_s = f"{cv:.{decimals}f}" if cv is not None else "N/A"

        delta_abs = (cv - lv) if (cv is not None and lv is not None) else None
        delta_pct = pct_change(cv, lv) if (cv is not None and lv is not None) else None
        delta_abs_s = f"{delta_abs:+.{decimals}f}" if delta_abs is not None else "N/A"
        delta_pct_s = f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A"

        rows.append({
            "metric": label,
            "leaked": lv_s,
            "clean": cv_s,
            "delta_abs": delta_abs_s,
            "delta_pct": delta_pct_s,
            "direction": direction,
        })

    # CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "leaked", "clean", "delta_abs", "delta_pct", "direction"])
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md_lines = [
        "# E30f Leakage Impact — Leaked vs Clean",
        "",
        f"- **Leaked baseline:** `{args.leaked}`",
        f"- **Clean run:** `{args.clean}`",
        "",
        "| Metric | Leaked | Clean | Δ abs | Δ % | Direction |",
        "|--------|--------|-------|-------|-----|-----------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['metric']} | {r['leaked']} | {r['clean']} "
            f"| {r['delta_abs']} | {r['delta_pct']} | {r['direction']} |"
        )

    # Highlight MAE delta
    mae_row = next((r for r in rows if r["metric"] == "MAE"), None)
    if mae_row:
        md_lines += [
            "",
            f"> **Key finding:** ΔMAE (clean − leaked) = **{mae_row['delta_abs']}** "
            f"({mae_row['delta_pct']}). "
            f"Positive delta = clean is worse (leakage inflated results). "
            f"Negative delta = clean is better (unexpected).",
        ]

    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Comparison written to {out_dir}/")
    print(f"  {csv_path.name}")
    print(f"  {md_path.name}")
    print()
    for r in rows:
        print(f"  {r['metric']:12s}: leaked={r['leaked']:8s} clean={r['clean']:8s} Δ={r['delta_abs']:10s} ({r['delta_pct']})")


if __name__ == "__main__":
    main()
