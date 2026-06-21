#!/usr/bin/env python3
"""Run all new experiments (E7-E12) on De Vito dataset and compare.

Experiments:
  E7:  Stepwise SLM (already ran via runner)
  E8:  Ensemble voting (already ran via runner)
  E9:  Adaptive NLP+Stepwise (runs via runner)
  E10: Keyword counting (already ran via runner)
  E11: Correction model on 7B output (runs via runner)
  E12: Ridge regression with LOOCV (special — handled here)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

pkg_base = BASE / "models" / "hybrid-xai"
mod_list = [
    ("models.hybrid_xai", pkg_base / "__init__.py"),
    ("models.hybrid_xai.schema", pkg_base / "schema.py"),
    ("models.hybrid_xai.rule_engine", pkg_base / "rule_engine.py"),
    ("models.hybrid_xai.normalizers", pkg_base / "normalizers" / "__init__.py"),
    ("models.hybrid_xai.normalizers.base", pkg_base / "normalizers" / "base.py"),
    ("models.hybrid_xai.normalizers.nlp_normalizer", pkg_base / "normalizers" / "nlp_normalizer.py"),
    ("models.hybrid_xai.normalizers.slm_normalizer", pkg_base / "normalizers" / "slm_normalizer.py"),
    ("models.hybrid_xai.normalizers.hybrid_normalizer", pkg_base / "normalizers" / "hybrid_normalizer.py"),
    ("models.hybrid_xai.normalizers.stepwise_normalizer", pkg_base / "normalizers" / "stepwise_normalizer.py"),
    ("models.hybrid_xai.normalizers.keyword_normalizer", pkg_base / "normalizers" / "keyword_normalizer.py"),
    ("models.hybrid_xai.normalizers.ensemble_normalizer", pkg_base / "normalizers" / "ensemble_normalizer.py"),
    ("models.hybrid_xai.normalizers.regression_normalizer", pkg_base / "normalizers" / "regression_normalizer.py"),
]
for name, path in mod_list:
    if not path.exists():
        continue
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

pkg = sys.modules["models.hybrid_xai"]
pkg.schema = sys.modules["models.hybrid_xai.schema"]
pkg.rule_engine = sys.modules["models.hybrid_xai.rule_engine"]
pkg.normalizers = sys.modules["models.hybrid_xai.normalizers"]
norms = sys.modules["models.hybrid_xai.normalizers"]
norms.base = sys.modules["models.hybrid_xai.normalizers.base"]

for nm, _ in mod_list:
    if nm in sys.modules:
        sys.modules[nm].__spec__.loader.exec_module(sys.modules[nm])

from models.hybrid_xai.schema import SourceFormat
from models.hybrid_xai.normalizers.regression_normalizer import (
    RegressionNormalizer, extract_features,
)


def load_devito_gt() -> dict[str, int]:
    gt_path = (
        BASE / "results" / "cosmet"
        / "ALBERGATE-gpt-4-turbo-20260305T023001"
        / "measurements" / "comparison_results.csv"
    )
    gt = {}
    with open(gt_path) as f:
        for row in csv.DictReader(f):
            gt[row["fp"]] = int(row["y_true"])
    return gt


def iter_devito_fps(input_dir: Path):
    for fp_file in sorted(input_dir.glob("*.txt")):
        fp_id = fp_file.stem
        text = fp_file.read_text(encoding="utf-8", errors="replace")
        yield fp_id, text, SourceFormat.USE_CASE


def run_regression_loocv():
    """Run Ridge regression with leave-one-out cross-validation."""
    print("=" * 60)
    print("E12: Ridge Regression with LOOCV")
    print("=" * 60)

    fp_dir = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"
    gt = load_devito_gt()

    all_data = []
    for fp_id, text, sf in iter_devito_fps(fp_dir):
        if fp_id not in gt:
            continue
        features = extract_features(text, sf)
        all_data.append((fp_id, text, sf, gt[fp_id], features))

    n = len(all_data)
    print(f"  Samples: {n}")

    y_true_list = []
    y_pred_list = []
    results = []

    for i in range(n):
        train = [(text, sf, cfp) for j, (fid, text, sf, cfp, _) in enumerate(all_data) if j != i]

        model = RegressionNormalizer(training_data=train, alpha=1.0)

        test_id, test_text, test_sf, test_gt, _ = all_data[i]
        pred = model.predict_cfp(test_text, test_sf)

        y_true_list.append(test_gt)
        y_pred_list.append(pred)
        err = pred - test_gt
        results.append({"fp": test_id, "y_true": test_gt, "y_pred": pred, "error": err})
        status = "OK" if err == 0 else f"err={err:+d}"
        print(f"  [{i+1:4d}] {test_id:>8s} | pred={pred:3d} | gt={test_gt:3d} | {status}")

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

    print(f"\n  MAE:  {metrics['mae']}")
    print(f"  MdAE: {metrics['mdae']}")
    print(f"  MMRE: {metrics['mmre']}")
    print(f"  PRED(25): {metrics['pred_25']}")
    print(f"  SA:   {metrics['sa']}")
    print(f"  Exact match: {metrics['exact_match']}")

    out_dir = BASE / "results" / "hybrid-xai" / "devito-E12-regression-loocv"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "y_pred", "error"])
        w.writeheader()
        w.writerows(results)

    train_all = [(text, sf, cfp) for _, text, sf, cfp, _ in all_data]
    model_all = RegressionNormalizer(training_data=train_all, alpha=1.0)
    print(f"\n  Learned weights (full dataset):")
    for name, w in zip(RegressionNormalizer.FEATURE_NAMES, model_all.weights):
        print(f"    {name:>20s}: {w:+.4f}")
    print(f"    {'bias':>20s}: {model_all.bias:+.4f}")

    with open(out_dir / "learned_weights.json", "w") as f:
        json.dump({
            "weights": dict(zip(RegressionNormalizer.FEATURE_NAMES, model_all.weights.tolist())),
            "bias": model_all.bias,
        }, f, indent=2)

    return metrics


def run_regression_default():
    """Run regression with hand-tuned default weights (no training)."""
    print("\n" + "=" * 60)
    print("E12b: Regression with default hand-tuned weights")
    print("=" * 60)

    fp_dir = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"
    gt = load_devito_gt()

    model = RegressionNormalizer()

    y_true_list = []
    y_pred_list = []
    results = []
    i = 0
    for fp_id, text, sf in iter_devito_fps(fp_dir):
        if fp_id not in gt:
            continue
        pred = model.predict_cfp(text, sf)
        true = gt[fp_id]
        y_true_list.append(true)
        y_pred_list.append(pred)
        err = pred - true
        results.append({"fp": fp_id, "y_true": true, "y_pred": pred, "error": err})
        i += 1
        status = "OK" if err == 0 else f"err={err:+d}"
        print(f"  [{i:4d}] {fp_id:>8s} | pred={pred:3d} | gt={true:3d} | {status}")

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    errors = np.abs(y_true - y_pred)
    rel_errors = errors / np.maximum(y_true, 1)
    random_mae = np.mean(np.abs(y_true - np.mean(y_true)))

    metrics = {
        "n_fps": len(y_true),
        "mae": round(float(np.mean(errors)), 3),
        "mdae": round(float(np.median(errors)), 3),
        "mmre": round(float(np.mean(rel_errors)), 3),
        "pred_25": round(float(np.mean(rel_errors <= 0.25)), 3),
        "sa": round(float(1.0 - np.mean(errors) / random_mae), 3),
    }

    print(f"\n  MAE:  {metrics['mae']}")
    print(f"  MdAE: {metrics['mdae']}")
    print(f"  SA:   {metrics['sa']}")

    out_dir = BASE / "results" / "hybrid-xai" / "devito-E12b-regression-default"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "y_pred", "error"])
        w.writeheader()
        w.writerows(results)

    return metrics


if __name__ == "__main__":
    run_regression_loocv()
    run_regression_default()
    print("\n" + "=" * 60)
    print("All regression experiments complete.")
    print("=" * 60)
