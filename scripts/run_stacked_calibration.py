#!/usr/bin/env python3
"""E13: Stacked Calibration — correct SLM predictions with text features.

Uses the SLM 7B v2 predictions as a primary signal, combined with
text-derived features, to train a second-stage Ridge regression
that corrects systematic underestimation bias.

Also runs:
  E14: Simple average of SLM v2 + Ridge regression predictions.
  E15: Weighted median of Keyword + NLP + SLM + Regression.
"""
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
mod_list = [
    ("models.hybrid_xai", pkg_base / "__init__.py"),
    ("models.hybrid_xai.schema", pkg_base / "schema.py"),
    ("models.hybrid_xai.rule_engine", pkg_base / "rule_engine.py"),
    ("models.hybrid_xai.normalizers", pkg_base / "normalizers" / "__init__.py"),
    ("models.hybrid_xai.normalizers.base", pkg_base / "normalizers" / "base.py"),
    ("models.hybrid_xai.normalizers.nlp_normalizer", pkg_base / "normalizers" / "nlp_normalizer.py"),
    ("models.hybrid_xai.normalizers.regression_normalizer", pkg_base / "normalizers" / "regression_normalizer.py"),
]
for name, path in mod_list:
    if not path.exists():
        continue
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
for nm, _ in mod_list:
    if nm in sys.modules and sys.modules[nm].__spec__:
        sys.modules[nm].__spec__.loader.exec_module(sys.modules[nm])

from models.hybrid_xai.schema import SourceFormat
from models.hybrid_xai.normalizers.regression_normalizer import extract_features
from models.hybrid_xai.normalizers.base import BaseNormalizer


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


def load_slm_predictions(result_dir: str) -> dict[str, int]:
    preds = {}
    csv_path = BASE / "results" / "hybrid-xai" / result_dir / "comparison_results.csv"
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            preds[row["fp"]] = int(row["y_pred"])
    return preds


def load_all_model_predictions() -> dict[str, dict[str, int]]:
    """Load predictions from all available models."""
    models = {}

    nlp_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E1*"))
    if nlp_dirs:
        models["nlp"] = load_preds_from_dir(nlp_dirs[0])

    kw_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E10*"))
    if kw_dirs:
        models["keyword"] = load_preds_from_dir(kw_dirs[0])

    slm_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E4*7b*v2*"))
    if slm_dirs:
        models["slm_7b_v2"] = load_preds_from_dir(slm_dirs[0])

    reg_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E12-regression*"))
    if reg_dirs:
        models["regression"] = load_preds_from_dir(reg_dirs[0])

    sw_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E7*"))
    if sw_dirs:
        models["stepwise"] = load_preds_from_dir(sw_dirs[0])

    ens_dirs = list((BASE / "results" / "hybrid-xai").glob("devito-E8*"))
    if ens_dirs:
        models["ensemble"] = load_preds_from_dir(ens_dirs[0])

    return models


def load_preds_from_dir(d: Path) -> dict[str, int]:
    preds = {}
    csv_path = d / "comparison_results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                preds[row["fp"]] = int(row["y_pred"])
    return preds


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    rel_errors = errors / np.maximum(y_true, 1)
    random_mae = np.mean(np.abs(y_true - np.mean(y_true)))
    return {
        "n_fps": len(y_true),
        "mae": round(float(np.mean(errors)), 3),
        "mdae": round(float(np.median(errors)), 3),
        "mmre": round(float(np.mean(rel_errors)), 3),
        "pred_25": round(float(np.mean(rel_errors <= 0.25)), 3),
        "sa": round(float(1.0 - np.mean(errors) / random_mae), 3),
        "mean_signed_error": round(float(np.mean(y_pred - y_true)), 3),
        "exact_match": round(float(np.mean(y_true == y_pred)), 3),
    }


def save_results(name, y_true_list, y_pred_list, fp_ids, metrics):
    out_dir = BASE / "results" / "hybrid-xai" / f"devito-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "y_pred", "error"])
        w.writeheader()
        for fid, yt, yp in zip(fp_ids, y_true_list, y_pred_list):
            w.writerow({"fp": fid, "y_true": int(yt), "y_pred": int(yp), "error": int(yp - yt)})


def iter_devito_fps(input_dir: Path):
    for fp_file in sorted(input_dir.glob("*.txt")):
        yield fp_file.stem, fp_file.read_text(encoding="utf-8", errors="replace"), SourceFormat.USE_CASE


def run_stacked_calibration():
    """E13: SLM prediction + text features → calibrated Ridge regression (LOOCV)."""
    print("=" * 60)
    print("E13: Stacked Calibration (SLM 7B v2 + text features)")
    print("=" * 60)

    fp_dir = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"
    gt = load_devito_gt()
    slm_preds = load_slm_predictions("devito-E4-7b-v2-gpu-20260326T225122")

    all_data = []
    for fp_id, text, sf in iter_devito_fps(fp_dir):
        if fp_id not in gt or fp_id not in slm_preds:
            continue
        text_features = extract_features(text, sf)
        slm_pred = slm_preds[fp_id]
        features = np.concatenate([[slm_pred], text_features])
        all_data.append((fp_id, gt[fp_id], features))

    n = len(all_data)
    alpha = 1.0

    y_true_list, y_pred_list, fp_ids = [], [], []
    for i in range(n):
        X_train = np.array([d[2] for j, d in enumerate(all_data) if j != i])
        y_train = np.array([d[1] for j, d in enumerate(all_data) if j != i])

        X_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        I = np.eye(X_bias.shape[1])
        I[-1, -1] = 0
        w = np.linalg.solve(X_bias.T @ X_bias + alpha * I, X_bias.T @ y_train)

        test_id, test_gt, test_feat = all_data[i]
        x_test = np.append(test_feat, 1.0)
        pred = max(1, round(float(x_test @ w)))

        y_true_list.append(test_gt)
        y_pred_list.append(pred)
        fp_ids.append(test_id)
        err = pred - test_gt
        status = "OK" if err == 0 else f"err={err:+d}"
        print(f"  [{i+1:4d}] {test_id:>8s} | pred={pred:3d} | gt={test_gt:3d} | slm={all_data[i][2][0]:.0f} | {status}")

    metrics = compute_metrics(y_true_list, y_pred_list)
    save_results("E13-stacked-calibration", y_true_list, y_pred_list, fp_ids, metrics)

    print(f"\n  MAE:  {metrics['mae']}")
    print(f"  MdAE: {metrics['mdae']}")
    print(f"  PRED(25): {metrics['pred_25']}")
    print(f"  SA:   {metrics['sa']}")
    print(f"  Exact match: {metrics['exact_match']}")

    X_all = np.array([d[2] for d in all_data])
    y_all = np.array([d[1] for d in all_data])
    X_bias = np.hstack([X_all, np.ones((X_all.shape[0], 1))])
    I = np.eye(X_bias.shape[1])
    I[-1, -1] = 0
    w = np.linalg.solve(X_bias.T @ X_bias + alpha * I, X_bias.T @ y_all)

    feat_names = ["slm_pred"] + [
        "word_count", "sentence_count", "step_count",
        "entry_verbs", "read_verbs", "write_verbs", "exit_verbs",
        "unique_entities", "conjunctions", "error_mentions",
    ]
    print("\n  Learned weights:")
    for name, ww in zip(feat_names, w[:-1]):
        print(f"    {name:>20s}: {ww:+.4f}")
    print(f"    {'bias':>20s}: {w[-1]:+.4f}")

    return metrics


def run_model_average():
    """E14: Simple average of SLM 7B v2 + Ridge regression predictions."""
    print("\n" + "=" * 60)
    print("E14: Model Average (SLM 7B v2 + Ridge Regression)")
    print("=" * 60)

    gt = load_devito_gt()
    slm_preds = load_slm_predictions("devito-E4-7b-v2-gpu-20260326T225122")

    reg_dir = BASE / "results" / "hybrid-xai" / "devito-E12-regression-loocv"
    reg_preds = {}
    with open(reg_dir / "comparison_results.csv") as f:
        for row in csv.DictReader(f):
            reg_preds[row["fp"]] = int(row["y_pred"])

    y_true_list, y_pred_list, fp_ids = [], [], []
    for fp_id in sorted(gt.keys()):
        if fp_id not in slm_preds or fp_id not in reg_preds:
            continue
        avg = round((slm_preds[fp_id] + reg_preds[fp_id]) / 2.0)
        avg = max(1, avg)
        y_true_list.append(gt[fp_id])
        y_pred_list.append(avg)
        fp_ids.append(fp_id)
        err = avg - gt[fp_id]
        status = "OK" if err == 0 else f"err={err:+d}"
        print(f"  {fp_id:>8s} | slm={slm_preds[fp_id]:3d} | reg={reg_preds[fp_id]:3d} | avg={avg:3d} | gt={gt[fp_id]:3d} | {status}")

    metrics = compute_metrics(y_true_list, y_pred_list)
    save_results("E14-model-average", y_true_list, y_pred_list, fp_ids, metrics)

    print(f"\n  MAE:  {metrics['mae']}")
    print(f"  MdAE: {metrics['mdae']}")
    print(f"  PRED(25): {metrics['pred_25']}")
    print(f"  SA:   {metrics['sa']}")

    return metrics


def run_mega_ensemble():
    """E15: Weighted ensemble of ALL models (median of available predictions)."""
    print("\n" + "=" * 60)
    print("E15: Mega-Ensemble (median of all available models)")
    print("=" * 60)

    gt = load_devito_gt()
    all_models = load_all_model_predictions()

    print(f"  Models loaded: {list(all_models.keys())}")

    y_true_list, y_pred_list, fp_ids = [], [], []
    for fp_id in sorted(gt.keys()):
        preds = []
        for model_name, model_preds in all_models.items():
            if fp_id in model_preds:
                preds.append(model_preds[fp_id])

        if not preds:
            continue

        med = round(float(np.median(preds)))
        med = max(1, med)

        y_true_list.append(gt[fp_id])
        y_pred_list.append(med)
        fp_ids.append(fp_id)
        err = med - gt[fp_id]
        status = "OK" if err == 0 else f"err={err:+d}"
        pred_str = " | ".join(f"{n}={all_models[n].get(fp_id, '?'):>3}" for n in all_models)
        print(f"  {fp_id:>8s} | {pred_str} | med={med:3d} | gt={gt[fp_id]:3d} | {status}")

    metrics = compute_metrics(y_true_list, y_pred_list)
    save_results("E15-mega-ensemble", y_true_list, y_pred_list, fp_ids, metrics)

    print(f"\n  MAE:  {metrics['mae']}")
    print(f"  MdAE: {metrics['mdae']}")
    print(f"  PRED(25): {metrics['pred_25']}")
    print(f"  SA:   {metrics['sa']}")

    return metrics


if __name__ == "__main__":
    results = {}
    results["E13"] = run_stacked_calibration()
    results["E14"] = run_model_average()
    results["E15"] = run_mega_ensemble()

    print("\n" + "=" * 60)
    print("SUMMARY — All Stacked/Ensemble Experiments")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name}: MAE={m['mae']:.3f} | MdAE={m['mdae']:.1f} | SA={m['sa']:.3f} | PRED(25)={m['pred_25']:.3f}")
