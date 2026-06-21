#!/usr/bin/env python3
"""E31a–E31b: GradientBoosting variants of E30f/E30g.

Replaces Ridge with GradientBoostingRegressor to capture non-linear patterns.

  E31a  Correction+GBR ensemble   — E30f variant with GBR instead of Ridge
  E31b  PCA+GBR stacking          — E30g variant with GBR instead of Ridge

Usage:
  python scripts/bert_hybrid_gbr.py \
    --e28a-dir results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921 \
    --output-dir results/hybrid-xai/gbr-experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from bert_hybrid_improvements_v2 import (
    load_e28a_data,
    extract_bert_embeddings,
    compute_metrics,
    save_experiment,
    per_size_analysis,
    degradation_analysis,
    _apply_tiered_correction,
    _grid_search_correction,
)


def run_e31a_correction_gbr_ensemble(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """E30f variant: Correction + GBR ensemble instead of Ridge.

    Uses PCA-reduced BERT (768→20) + hand features to make GBR tractable.
    Inner CV uses pre-computed GBR predictions (one train per fold, not per weight)
    to avoid combinatorial explosion.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_raw = np.array([d["y_raw_slm"] for d in data])
    n_steps = np.array([d["n_steps"] for d in data])

    hand_feats = np.array([
        [d["n_steps"], d["n_words"], d["n_movements"], d["y_raw_slm"]]
        for d in data
    ], dtype=float)

    preds = np.zeros(n)
    weights_used = []

    for i in range(n):
        if i % 20 == 0:
            print(f"    LOOCV fold {i+1}/{n}...")
        mask = np.arange(n) != i

        params = _grid_search_correction(y_true, y_raw, n_steps, mask)
        corrected_i = _apply_tiered_correction(y_raw[i], n_steps[i], params)
        y_corrected_train = np.array([
            _apply_tiered_correction(y_raw[j], n_steps[j], params)
            for j in np.where(mask)[0]
        ])

        sc_bert = StandardScaler()
        bert_tr = sc_bert.fit_transform(bert_emb[mask])
        bert_te = sc_bert.transform(bert_emb[i : i + 1])
        pca = PCA(n_components=20)
        pca_tr = pca.fit_transform(bert_tr)
        pca_te = pca.transform(bert_te)

        X_tr = np.hstack([hand_feats[mask], pca_tr])
        X_te = np.hstack([hand_feats[i : i + 1], pca_te])
        sc_all = StandardScaler()
        X_tr = sc_all.fit_transform(X_tr)
        X_te = sc_all.transform(X_te)

        gbr = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        gbr.fit(X_tr, y_true[mask])
        gbr_pred_i = gbr.predict(X_te)[0]

        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        gbr_inner_preds = np.zeros(len(inner_idxs))
        for train_fold, val_fold in kf.split(inner_idxs):
            tr = inner_idxs[train_fold]
            va = inner_idxs[val_fold]
            sc2 = StandardScaler()
            b2 = sc2.fit_transform(bert_emb[tr])
            bv = sc2.transform(bert_emb[va])
            p2 = PCA(n_components=20)
            pb2 = p2.fit_transform(b2)
            pbv = p2.transform(bv)
            Xtr2 = np.hstack([hand_feats[tr], pb2])
            Xva2 = np.hstack([hand_feats[va], pbv])
            sc3 = StandardScaler()
            Xtr2 = sc3.fit_transform(Xtr2)
            Xva2 = sc3.transform(Xva2)
            g2 = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            g2.fit(Xtr2, y_true[tr])
            gbr_inner_preds[val_fold] = g2.predict(Xva2)

        best_w = 0.7
        best_inner_mae = float("inf")
        for w in np.arange(0.40, 0.90, 0.05):
            ens = w * y_corrected_train + (1 - w) * gbr_inner_preds
            mae_inner = np.mean(np.abs(y_true[inner_idxs] - np.round(np.maximum(1, ens))))
            if mae_inner < best_inner_mae:
                best_inner_mae = mae_inner
                best_w = w

        weights_used.append(best_w)
        preds[i] = max(1, round(best_w * corrected_i + (1 - best_w) * gbr_pred_i))

    print(f"    Weights: mean={np.mean(weights_used):.2f} "
          f"std={np.std(weights_used):.2f} "
          f"min={np.min(weights_used):.2f} max={np.max(weights_used):.2f}")

    return _build_results(data, preds)


def run_e31b_pca_gbr_stacking(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """E30g variant: PCA + GBR stacking instead of Ridge.

    Fixed PCA k=20 (same as E30g best), GBR n_estimators=50 to keep
    the LOOCV tractable. No nested hyperparameter search.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])

    hand_feats = np.array([
        [
            d["y_pred_slm"],
            d["y_raw_slm"],
            d["correction_count"],
            d["n_movements"],
            d["n_steps"],
            d["n_words"],
        ]
        for d in data
    ], dtype=float)

    preds = np.zeros(n)
    for i in range(n):
        if i % 20 == 0:
            print(f"    LOOCV fold {i+1}/{n}...")
        mask = np.arange(n) != i

        sc_bert = StandardScaler()
        bert_tr = sc_bert.fit_transform(bert_emb[mask])
        bert_te = sc_bert.transform(bert_emb[i : i + 1])

        pca = PCA(n_components=20)
        pca_tr = pca.fit_transform(bert_tr)
        pca_te = pca.transform(bert_te)

        X_tr = np.hstack([hand_feats[mask], pca_tr])
        X_te = np.hstack([hand_feats[i : i + 1], pca_te])

        sc_all = StandardScaler()
        X_tr = sc_all.fit_transform(X_tr)
        X_te = sc_all.transform(X_te)

        gbr = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        gbr.fit(X_tr, y_true[mask])
        preds[i] = max(1, round(gbr.predict(X_te)[0]))

    return _build_results(data, preds)


def _build_results(data: list[dict], preds: np.ndarray) -> list[dict]:
    return [
        {"fp": d["fp_id"], "y_pred": int(preds[i]), "y_true": d["y_true"]}
        for i, d in enumerate(data)
    ]


def main():
    parser = argparse.ArgumentParser(description="E31a–E31b: GBR variants of E30f/E30g")
    parser.add_argument("--e28a-dir", type=Path,
                        default=Path("results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/hybrid-xai/gbr-experiments"))
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading E28a data (119 FPs)...")
    data = load_e28a_data(args.e28a_dir)
    ref_results = [
        {"fp": d["fp_id"], "y_pred": d["y_pred_slm"], "y_true": d["y_true"]}
        for d in data
    ]
    ref_metrics = compute_metrics(ref_results)
    print(f"  E28a reference: MAE={ref_metrics['mae']:.3f}")

    texts = [d["raw_text"] for d in data]
    print("\nExtracting BERT [CLS] embeddings...")
    t0 = time.time()
    bert_emb = extract_bert_embeddings(texts)
    print(f"  Shape: {bert_emb.shape}, took {time.time()-t0:.1f}s")
    print("=" * 70)

    if "e31a" not in args.skip:
        print("\n[E31a] Correction + GBR ensemble (E30f variant)...")
        t0 = time.time()
        res = run_e31a_correction_gbr_ensemble(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E31a-correction-gbr-ensemble", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E31a")
        degradation_analysis(res, ref_results, "E31a")

    if "e31b" not in args.skip:
        print("\n[E31b] PCA + GBR stacking (E30g variant)...")
        t0 = time.time()
        res = run_e31b_pca_gbr_stacking(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E31b-pca-gbr-stacking", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E31b")
        degradation_analysis(res, ref_results, "E31b")

    print("\n" + "=" * 70)
    print("SUMMARY — E31 GBR experiments vs references")
    print("=" * 70)
    print(f"{'Experiment':<42} {'MAE':>6} {'PRED(25)':>9} {'SA':>7} {'Exact':>7}")
    print("-" * 72)
    print(f"{'E28a Qwen3-8B (reference)':<42} {ref_metrics['mae']:>6.3f} "
          f"{ref_metrics['pred_25']*100:>8.1f}% {ref_metrics['sa']*100:>6.1f}% "
          f"{ref_metrics['exact_match']*100:>6.1f}%")

    e30f = Path("results/hybrid-xai/bert-improvements/E30f-correction-ensemble/metrics.json")
    if e30f.exists():
        m30 = json.loads(e30f.read_text())
        print(f"{'E30f correction+Ridge (prev best)':<42} {m30['mae']:>6.3f} "
              f"{m30['pred_25']*100:>8.1f}% {m30['sa']*100:>6.1f}% "
              f"{m30['exact_match']*100:>6.1f}%")

    print("-" * 72)
    for exp_name in sorted(args.output_dir.iterdir()):
        mf = exp_name / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            print(f"{exp_name.name:<42} {m['mae']:>6.3f} "
                  f"{m['pred_25']*100:>8.1f}% {m['sa']*100:>6.1f}% "
                  f"{m['exact_match']*100:>6.1f}%")
    print("-" * 72)


if __name__ == "__main__":
    main()
