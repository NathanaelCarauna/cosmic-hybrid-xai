#!/usr/bin/env python3
"""E16: Stacked calibration on E11 (correction model) predictions."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

pkg_base = BASE / "models" / "hybrid-xai"
for name, path in [
    ("models.hybrid_xai", pkg_base / "__init__.py"),
    ("models.hybrid_xai.schema", pkg_base / "schema.py"),
    ("models.hybrid_xai.normalizers", pkg_base / "normalizers" / "__init__.py"),
    ("models.hybrid_xai.normalizers.base", pkg_base / "normalizers" / "base.py"),
    ("models.hybrid_xai.normalizers.regression_normalizer", pkg_base / "normalizers" / "regression_normalizer.py"),
]:
    if not path.exists():
        continue
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

from models.hybrid_xai.schema import SourceFormat
from models.hybrid_xai.normalizers.regression_normalizer import extract_features


def load_gt():
    p = BASE / "results" / "cosmet" / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv"
    gt = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            gt[row["fp"]] = int(row["y_true"])
    return gt


def load_preds(dirname):
    p = BASE / "results" / "hybrid-xai" / dirname / "comparison_results.csv"
    preds = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            preds[row["fp"]] = int(row["y_pred"])
    return preds


def run_stacked(base_name, base_dir):
    print("=" * 60)
    print(f"E16: Stacked Calibration on {base_name}")
    print("=" * 60)

    fp_dir = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"
    gt = load_gt()
    base_preds = load_preds(base_dir)

    all_data = []
    for fp_file in sorted(fp_dir.glob("*.txt")):
        fp_id = fp_file.stem
        if fp_id not in gt or fp_id not in base_preds:
            continue
        text = fp_file.read_text(encoding="utf-8", errors="replace")
        text_features = extract_features(text, SourceFormat.USE_CASE)
        features = np.concatenate([[base_preds[fp_id]], text_features])
        all_data.append((fp_id, gt[fp_id], features))

    n = len(all_data)

    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        y_true_list, y_pred_list = [], []
        for i in range(n):
            X_train = np.array([d[2] for j, d in enumerate(all_data) if j != i])
            y_train = np.array([d[1] for j, d in enumerate(all_data) if j != i])
            X_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
            I = np.eye(X_bias.shape[1])
            I[-1, -1] = 0
            w = np.linalg.solve(X_bias.T @ X_bias + alpha * I, X_bias.T @ y_train)
            x_test = np.append(all_data[i][2], 1.0)
            pred = max(1, round(float(x_test @ w)))
            y_true_list.append(all_data[i][1])
            y_pred_list.append(pred)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        errors = np.abs(y_true - y_pred)
        rel_errors = errors / np.maximum(y_true, 1)
        random_mae = np.mean(np.abs(y_true - np.mean(y_true)))
        mae = float(np.mean(errors))
        sa = float(1.0 - mae / random_mae)
        pred25 = float(np.mean(rel_errors <= 0.25))
        mdae = float(np.median(errors))
        exact = float(np.mean(y_true == y_pred))

        print(f"  alpha={alpha:5.1f} | MAE={mae:.3f} | MdAE={mdae:.1f} | SA={sa:.3f} | PRED(25)={pred25:.3f} | Exact={exact:.3f}")

    best_alpha = 1.0
    y_true_list, y_pred_list, fp_ids = [], [], []
    for i in range(n):
        X_train = np.array([d[2] for j, d in enumerate(all_data) if j != i])
        y_train = np.array([d[1] for j, d in enumerate(all_data) if j != i])
        X_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        I = np.eye(X_bias.shape[1])
        I[-1, -1] = 0
        w = np.linalg.solve(X_bias.T @ X_bias + best_alpha * I, X_bias.T @ y_train)
        x_test = np.append(all_data[i][2], 1.0)
        pred = max(1, round(float(x_test @ w)))
        y_true_list.append(all_data[i][1])
        y_pred_list.append(pred)
        fp_ids.append(all_data[i][0])

        err = pred - all_data[i][1]
        status = "OK" if err == 0 else f"err={err:+d}"
        print(f"  [{i+1:4d}] {all_data[i][0]:>8s} | pred={pred:3d} | gt={all_data[i][1]:3d} | base={all_data[i][2][0]:.0f} | {status}")

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    errors = np.abs(y_true - y_pred)
    rel_errors = errors / np.maximum(y_true, 1)
    random_mae = np.mean(np.abs(y_true - np.mean(y_true)))

    metrics = {
        "n_fps": n,
        "mae": round(float(np.mean(errors)), 3),
        "mdae": round(float(np.median(errors)), 3),
        "mmre": round(float(np.mean(rel_errors)), 3),
        "pred_25": round(float(np.mean(rel_errors <= 0.25)), 3),
        "sa": round(float(1.0 - np.mean(errors) / random_mae), 3),
        "mean_signed_error": round(float(np.mean(y_pred - y_true)), 3),
        "exact_match": round(float(np.mean(y_true == y_pred)), 3),
    }

    out_dir = BASE / "results" / "hybrid-xai" / f"devito-E16-stacked-{base_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "y_pred", "error"])
        w.writeheader()
        for fid, yt, yp in zip(fp_ids, y_true_list, y_pred_list):
            w.writerow({"fp": fid, "y_true": int(yt), "y_pred": int(yp), "error": int(yp - yt)})

    print(f"\n  FINAL: MAE={metrics['mae']} | SA={metrics['sa']} | PRED(25)={metrics['pred_25']}")
    return metrics


if __name__ == "__main__":
    run_stacked("correction", "devito-E11-correction-7b-gpu-20260327T032310")
    print()
    run_stacked("slm-v2", "devito-E4-7b-v2-gpu-20260326T225122")
