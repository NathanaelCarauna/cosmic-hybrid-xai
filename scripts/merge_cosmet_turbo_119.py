#!/usr/bin/env python3
"""Merge 7 De Vito COSMET gpt-4-turbo per-project runs into combined 119 FP result."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
COSMET_DIR = BASE / "results" / "cosmet"

SYSTEMS = {
    "ALBERGATE": "ALBERGATE-gpt-4-turbo-20260305T023001",
    "K01719": "K01719-gpt-4-turbo-20260323T005355",
    "K01720": "K01720-gpt-4-turbo-20260323T005553",
    "K01726": "K01726-gpt-4-turbo-20260323T005703",
    "U-CURE": "U-CURE-gpt-4-turbo-20260323T010152",
    "AutomaticLineSwitching": "AutomaticLineSwitching-gpt-4-turbo-20260323T010407",
    "RiceCooker": "RiceCooker-gpt-4-turbo-20260323T010514",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))
    mdae = float(np.median(ae))
    re = ae / np.maximum(y_true, 1)
    mmre = float(np.mean(re))
    pred_25 = float(np.mean(re <= 0.25))
    mean_gt = float(np.mean(y_true))
    random_mae = float(np.mean(np.abs(y_true - mean_gt)))
    sa = 1 - mae / random_mae if random_mae > 0 else 0.0
    exact_match = float(np.mean(ae == 0))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_gt) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    signed_err = float(np.mean(y_pred - y_true))
    sp, _ = spearmanr(y_true, y_pred)
    return {
        "n_with_gt": int(len(y_true)),
        "mae": round(mae, 3),
        "mdae": round(mdae, 3),
        "mmre": round(mmre, 3),
        "pred_25": round(pred_25, 3),
        "sa": round(sa, 3),
        "r2": round(r2, 3),
        "exact_match": round(exact_match, 3),
        "mean_signed_error": round(signed_err, 3),
        "spearman": round(float(sp), 3) if not np.isnan(sp) else 0.0,
    }


def main():
    all_rows: list[dict] = []

    for system, run_dir_name in SYSTEMS.items():
        csv_path = COSMET_DIR / run_dir_name / "measurements" / "comparison_results.csv"
        if not csv_path.exists():
            print(f"WARN: missing {csv_path}")
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                all_rows.append({
                    "fp": f"{system}/{row['fp']}",
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                })

    print(f"Merged {len(all_rows)} FPs from {len(SYSTEMS)} systems")

    out_dir = COSMET_DIR / "devito-119-combined-gpt-4-turbo-merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "y_pred"])
        w.writeheader()
        w.writerows(all_rows)

    y_true = np.array([r["y_true"] for r in all_rows], dtype=float)
    y_pred = np.array([r["y_pred"] for r in all_rows], dtype=float)
    metrics = compute_metrics(y_true, y_pred)

    with open(out_dir / "comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCOSMET gpt-4-turbo (119 FPs combined):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
