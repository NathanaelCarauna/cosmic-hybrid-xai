#!/usr/bin/env python3
"""BERT-based Hybrid-XAI improvement experiments v2 (E30f–E30i).

Addresses v1 failures by:
- Combining re-optimized TieredCorrection (E30a) with BERT ensemble (E29e)
- PCA dimensionality reduction for stacking
- Adaptive per-prediction weight blending

Experiments:
  E30f  Correction+Ensemble    — E30a correction → BERT ensemble (nested LOOCV)
  E30g  PCA stacking           — PCA-reduced BERT + SLM features → Ridge
  E30h  Adaptive weight blend  — Weight varies by predicted size (higher SLM for small)
  E30i  Best-of combo          — Combine top-2 approaches from E30f/g/h

Usage:
  python scripts/bert_hybrid_improvements_v2.py \
    --e28a-dir results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921 \
    --output-dir results/hybrid-xai/bert-improvements
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from shared.seeds import set_global_seeds


# ── Data loading ─────────────────────────────────────────────────────

def load_e28a_data(e28a_dir: Path) -> list[dict]:
    comp = e28a_dir / "comparison_results.csv"
    rows = list(csv.DictReader(comp.open()))

    data = []
    for row in rows:
        fp_id = row["fp"]
        safe = fp_id.replace("/", "_")
        ir_path = e28a_dir / f"{safe}_ir.json"
        ir = json.loads(ir_path.read_text()) if ir_path.exists() else None

        raw_count = 0
        correction_count = 0
        if ir:
            for fp_proc in ir["functional_processes"]:
                for dm in fp_proc["data_movements"]:
                    if dm["data_group"].startswith("correction_read"):
                        correction_count += 1
                    else:
                        raw_count += 1

        text = ir["raw_text"] if ir else ""
        feats = _extract_features(text)

        data.append({
            "fp_id": fp_id,
            "y_true": int(row["y_true"]),
            "y_pred_slm": int(row["y_pred"]),
            "y_raw_slm": raw_count if ir else int(row["y_pred"]),
            "correction_count": correction_count,
            "raw_text": text,
            "n_movements": int(row["n_movements"]),
            "n_processes": int(row["n_processes"]),
            "n_steps": feats["n_steps"],
            "n_words": feats["n_words"],
            "n_lines": feats["n_lines"],
            "text_len": feats["text_len"],
            "ir": ir,
        })
    return data


def _extract_features(text: str) -> dict:
    lines = text.strip().split("\n")
    words = text.split()
    step_like = [l for l in lines if re.match(r"^\s*\d+\s*\t", l)]
    return {
        "n_lines": len(lines),
        "n_words": len(words),
        "n_steps": len(step_like),
        "text_len": len(text),
    }


# ── BERT embedding extraction ───────────────────────────────────────

def extract_bert_embeddings(texts: list[str], max_length: int = 512) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "google-bert/bert-base-uncased"
    _BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=_BERT_REVISION)
    model = AutoModel.from_pretrained(model_name, revision=_BERT_REVISION)
    model.eval()

    embeddings = []
    batch_sz = 8
    for i in range(0, len(texts), batch_sz):
        batch = texts[i : i + batch_sz]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls)

    return np.vstack(embeddings)


# ── TieredCorrection helper ─────────────────────────────────────────

def _apply_tiered_correction(y_raw_val, n_steps_val, params):
    alpha, beta, gamma, alpha_l, beta_l, gamma_l, threshold = params
    if n_steps_val >= threshold:
        a, b, g = alpha_l, beta_l, gamma_l
    else:
        a, b, g = alpha, beta, gamma
    raw_corr = a * max(0, n_steps_val - b)
    correction = min(raw_corr, y_raw_val * g)
    extra = round(correction) if correction > 0.5 else 0
    return y_raw_val + extra


def _grid_search_correction(y_true, y_raw, n_steps, mask=None):
    """Find best TieredCorrection params on given data subset."""
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    idxs = np.where(mask)[0]
    best_params = (0.10, 6, 0.20, 0.60, 2, 0.60, 12)
    best_mae = float("inf")

    for threshold in [10, 12, 14]:
        for alpha in np.arange(0.05, 0.35, 0.05):
            for beta in [4, 6, 8]:
                for gamma in np.arange(0.1, 0.5, 0.1):
                    for alpha_l in np.arange(0.3, 0.8, 0.1):
                        for beta_l in [2, 3, 4, 5]:
                            for gamma_l in np.arange(0.4, 0.8, 0.1):
                                params = (alpha, beta, gamma, alpha_l, beta_l, gamma_l, threshold)
                                preds = np.array([
                                    _apply_tiered_correction(y_raw[j], n_steps[j], params)
                                    for j in idxs
                                ])
                                mae = np.mean(np.abs(y_true[idxs] - preds))
                                if mae < best_mae:
                                    best_mae = mae
                                    best_params = params
    return best_params


# ── E30f: Correction + Ensemble (nested LOOCV) ──────────────────────

def run_e30f_correction_ensemble(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """Combine re-optimized TieredCorrection with BERT Ridge ensemble.

    For each LOOCV fold:
    1. Grid-search correction params on training set → apply to test raw SLM
    2. Train Ridge on BERT embeddings → get BERT prediction for test
    3. Nested 10-fold CV to find optimal blend weight
    4. Apply: w * corrected_SLM + (1-w) * BERT_pred
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_raw = np.array([d["y_raw_slm"] for d in data])
    n_steps = np.array([d["n_steps"] for d in data])

    text_feats = np.array([
        [d["n_steps"], d["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    weights_used = []

    for i in range(n):
        mask = np.arange(n) != i

        # Step 1: Grid search correction on training set
        params = _grid_search_correction(y_true, y_raw, n_steps, mask)
        corrected_i = _apply_tiered_correction(y_raw[i], n_steps[i], params)
        y_corrected_train = np.array([
            _apply_tiered_correction(y_raw[j], n_steps[j], params)
            for j in np.where(mask)[0]
        ])

        # Step 2: BERT prediction
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred_i = reg.predict(X_test)[0]

        # Get BERT predictions for training set (inner LOOCV-like via 10-fold)
        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        best_w = 0.7
        best_inner_mae = float("inf")

        for w in np.arange(0.40, 0.90, 0.05):
            inner_errors = []
            for train_fold, val_fold in kf.split(inner_idxs):
                tr = inner_idxs[train_fold]
                va = inner_idxs[val_fold]
                sc2 = StandardScaler()
                X2_tr = sc2.fit_transform(X[tr])
                X2_va = sc2.transform(X[va])
                r2 = Ridge(alpha=10.0)
                r2.fit(X2_tr, y_true[tr])
                bp = r2.predict(X2_va)

                y_corr_va = y_corrected_train[np.searchsorted(inner_idxs, va)]
                ens = w * y_corr_va + (1 - w) * bp
                inner_errors.extend(np.abs(y_true[va] - np.round(np.maximum(1, ens))))

            mae_inner = np.mean(inner_errors)
            if mae_inner < best_inner_mae:
                best_inner_mae = mae_inner
                best_w = w

        weights_used.append(best_w)
        preds[i] = max(1, round(best_w * corrected_i + (1 - best_w) * bert_pred_i))

    print(f"    Weights: mean={np.mean(weights_used):.2f} "
          f"std={np.std(weights_used):.2f} "
          f"min={np.min(weights_used):.2f} max={np.max(weights_used):.2f}")

    results = _build_results(data, preds)
    for i, r in enumerate(results):
        r["blend_weight"] = round(float(weights_used[i]), 4)
    return results


# ── E30g: PCA stacking ──────────────────────────────────────────────

def run_e30g_pca_stacking(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """PCA-reduced BERT embeddings + hand-crafted features → Ridge.

    Reduces 768-dim BERT to k components (k selected via nested CV)
    to avoid overfitting with n=119.
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
        mask = np.arange(n) != i
        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        best_k = 20
        best_alpha = 10.0
        best_inner_mae = float("inf")

        for k in [5, 10, 15, 20, 30, 50]:
            for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
                inner_errors = []
                for train_fold, val_fold in kf.split(inner_idxs):
                    tr = inner_idxs[train_fold]
                    va = inner_idxs[val_fold]

                    sc_bert = StandardScaler()
                    bert_tr = sc_bert.fit_transform(bert_emb[tr])
                    bert_va = sc_bert.transform(bert_emb[va])

                    pca = PCA(n_components=min(k, bert_tr.shape[0], bert_tr.shape[1]))
                    pca_tr = pca.fit_transform(bert_tr)
                    pca_va = pca.transform(bert_va)

                    X_tr = np.hstack([hand_feats[tr], pca_tr])
                    X_va = np.hstack([hand_feats[va], pca_va])

                    sc_all = StandardScaler()
                    X_tr = sc_all.fit_transform(X_tr)
                    X_va = sc_all.transform(X_va)

                    reg = Ridge(alpha=alpha)
                    reg.fit(X_tr, y_true[tr])
                    bp = reg.predict(X_va)
                    inner_errors.extend(np.abs(y_true[va] - np.round(np.maximum(1, bp))))

                mae_inner = np.mean(inner_errors)
                if mae_inner < best_inner_mae:
                    best_inner_mae = mae_inner
                    best_k = k
                    best_alpha = alpha

        # Apply with best hyperparams
        sc_bert = StandardScaler()
        bert_tr = sc_bert.fit_transform(bert_emb[mask])
        bert_te = sc_bert.transform(bert_emb[i : i + 1])

        pca = PCA(n_components=min(best_k, bert_tr.shape[0], bert_tr.shape[1]))
        pca_tr = pca.fit_transform(bert_tr)
        pca_te = pca.transform(bert_te)

        X_tr = np.hstack([hand_feats[mask], pca_tr])
        X_te = np.hstack([hand_feats[i : i + 1], pca_te])

        sc_all = StandardScaler()
        X_tr = sc_all.fit_transform(X_tr)
        X_te = sc_all.transform(X_te)

        reg = Ridge(alpha=best_alpha)
        reg.fit(X_tr, y_true[mask])
        preds[i] = max(1, round(reg.predict(X_te)[0]))

    return _build_results(data, preds)


# ── E30h: Adaptive weight blend ──────────────────────────────────────

def run_e30h_adaptive_weight(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """Blend SLM and BERT with weight that varies by predicted size.

    Small predictions (SLM ≤ threshold): higher SLM weight (less BERT noise)
    Large predictions: lower SLM weight (more BERT correction needed)
    Threshold and weights selected via nested LOOCV.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    text_feats = np.array([
        [d["n_steps"], d["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i

        # BERT prediction for test point
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred_i = reg.predict(X_test)[0]

        # Nested CV to find best (threshold, w_small, w_large)
        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        best_cfg = (4, 0.90, 0.70)
        best_inner_mae = float("inf")

        for thr in [3, 4, 5]:
            for w_small in np.arange(0.85, 1.01, 0.05):
                for w_large in np.arange(0.50, 0.85, 0.05):
                    inner_errors = []
                    for train_fold, val_fold in kf.split(inner_idxs):
                        tr = inner_idxs[train_fold]
                        va = inner_idxs[val_fold]
                        sc2 = StandardScaler()
                        X2_tr = sc2.fit_transform(X[tr])
                        X2_va = sc2.transform(X[va])
                        r2 = Ridge(alpha=10.0)
                        r2.fit(X2_tr, y_true[tr])
                        bp = r2.predict(X2_va)

                        for k, vi in enumerate(va):
                            w = w_small if y_slm[vi] <= thr else w_large
                            ens = w * y_slm[vi] + (1 - w) * bp[k]
                            inner_errors.append(abs(y_true[vi] - round(max(1, ens))))

                    mae_inner = np.mean(inner_errors)
                    if mae_inner < best_inner_mae:
                        best_inner_mae = mae_inner
                        best_cfg = (thr, w_small, w_large)

        thr, w_small, w_large = best_cfg
        w = w_small if y_slm[i] <= thr else w_large
        preds[i] = max(1, round(w * y_slm[i] + (1 - w) * bert_pred_i))

    return _build_results(data, preds)


# ── E30i: Correction + Adaptive ensemble ─────────────────────────────

def run_e30i_correction_adaptive(
    data: list[dict], bert_emb: np.ndarray,
) -> list[dict]:
    """Best of E30f + E30h: re-optimized correction + adaptive-weight BERT blend.

    1. Apply per-fold grid-searched correction to raw SLM
    2. Blend corrected SLM with BERT prediction using size-adaptive weight
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_raw = np.array([d["y_raw_slm"] for d in data])
    n_steps = np.array([d["n_steps"] for d in data])

    text_feats = np.array([
        [d["n_steps"], d["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i

        # Grid search correction on training set
        params = _grid_search_correction(y_true, y_raw, n_steps, mask)
        corrected_i = _apply_tiered_correction(y_raw[i], n_steps[i], params)
        y_corrected_train = np.array([
            _apply_tiered_correction(y_raw[j], n_steps[j], params)
            for j in np.where(mask)[0]
        ])

        # BERT prediction
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred_i = reg.predict(X_test)[0]

        # Adaptive weight selection via inner CV
        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        best_cfg = (4, 0.85, 0.65)
        best_inner_mae = float("inf")

        for thr in [3, 4, 5]:
            for w_small in np.arange(0.80, 1.01, 0.05):
                for w_large in np.arange(0.45, 0.80, 0.05):
                    inner_errors = []
                    for train_fold, val_fold in kf.split(inner_idxs):
                        tr = inner_idxs[train_fold]
                        va = inner_idxs[val_fold]
                        sc2 = StandardScaler()
                        X2_tr = sc2.fit_transform(X[tr])
                        X2_va = sc2.transform(X[va])
                        r2 = Ridge(alpha=10.0)
                        r2.fit(X2_tr, y_true[tr])
                        bp = r2.predict(X2_va)

                        for k, vi in enumerate(va):
                            corr_val = y_corrected_train[np.searchsorted(inner_idxs, vi)]
                            w = w_small if corr_val <= thr else w_large
                            ens = w * corr_val + (1 - w) * bp[k]
                            inner_errors.append(abs(y_true[vi] - round(max(1, ens))))

                    mae_inner = np.mean(inner_errors)
                    if mae_inner < best_inner_mae:
                        best_inner_mae = mae_inner
                        best_cfg = (thr, w_small, w_large)

        thr, w_small, w_large = best_cfg
        w = w_small if corrected_i <= thr else w_large
        preds[i] = max(1, round(w * corrected_i + (1 - w) * bert_pred_i))

    return _build_results(data, preds)


# ── Metrics + output ─────────────────────────────────────────────────

def _build_results(data: list[dict], preds: np.ndarray) -> list[dict]:
    return [
        {"fp": d["fp_id"], "y_pred": int(preds[i]), "y_true": d["y_true"]}
        for i, d in enumerate(data)
    ]


def compute_metrics(results: list[dict]) -> dict:
    y_true = np.array([r["y_true"] for r in results], dtype=float)
    y_pred = np.array([r["y_pred"] for r in results], dtype=float)
    n = len(y_true)

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
        "n_with_gt": n,
        "mae": mae,
        "mdae": mdae,
        "mmre": mmre,
        "pred_25": pred_25,
        "sa": sa,
        "r2": r2,
        "exact_match": exact_match,
        "mean_signed_error": signed_err,
        "spearman": float(sp) if not np.isnan(sp) else 0.0,
    }


def _augment_metrics_with_optional_fields(metrics: dict, results: list[dict]) -> dict:
    """Add per-experiment optional fields to metrics if present in results."""
    if results and "blend_weight" in results[0]:
        weights = [r["blend_weight"] for r in results]
        metrics["mean_blend_weight"] = round(float(np.mean(weights)), 4)
        metrics["std_blend_weight"] = round(float(np.std(weights)), 4)
    return metrics


def save_experiment(
    name: str, results: list[dict], metrics: dict,
    elapsed: float, output_dir: Path,
):
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "comparison_results.csv", "w", newline="") as f:
        fieldnames = list(results[0].keys()) if results else ["fp", "y_pred", "y_true"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    metrics["total_time_s"] = elapsed
    metrics["avg_time_s"] = elapsed / max(len(results), 1)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✅ {name}: MAE={metrics['mae']:.3f} PRED(25)={metrics['pred_25']*100:.1f}% "
          f"SA={metrics['sa']*100:.1f}% ExactMatch={metrics['exact_match']*100:.1f}% "
          f"({elapsed:.1f}s)")


def per_size_analysis(results: list[dict], label: str):
    y_true = np.array([r["y_true"] for r in results], dtype=float)
    y_pred = np.array([r["y_pred"] for r in results], dtype=float)
    print(f"    {label} per-tier:")
    for tag, lo, hi in [("Small ≤4", 0, 4), ("Med 5-8", 5, 8), ("Large ≥9", 9, 999)]:
        m = (y_true >= lo) & (y_true <= hi)
        if m.sum() == 0:
            continue
        mae = np.mean(np.abs(y_true[m] - y_pred[m]))
        improved = np.sum(np.abs(y_true[m] - y_pred[m]) == 0)
        print(f"      {tag}: n={m.sum():3d} MAE={mae:.3f} exact={improved}")


def degradation_analysis(results: list[dict], ref_data: list[dict], label: str):
    ref_map = {r["fp"]: r["y_pred"] for r in ref_data}
    improved = degraded = same = 0
    for r in results:
        ae_new = abs(r["y_true"] - r["y_pred"])
        ae_ref = abs(r["y_true"] - ref_map[r["fp"]])
        if ae_new < ae_ref:
            improved += 1
        elif ae_new > ae_ref:
            degraded += 1
        else:
            same += 1
    print(f"    {label} vs E28a: improve={improved} degrade={degraded} same={same}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    set_global_seeds()
    parser = argparse.ArgumentParser(description="BERT Hybrid-XAI improvements v2 E30f–E30i")
    parser.add_argument("--e28a-dir", type=Path,
                        default=Path("results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/hybrid-xai/bert-improvements"))
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading E28a data (119 FPs)...")
    data = load_e28a_data(args.e28a_dir)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])
    ref_results = [{"fp": d["fp_id"], "y_pred": d["y_pred_slm"], "y_true": d["y_true"]} for d in data]

    ref_metrics = compute_metrics(ref_results)
    print(f"  E28a reference: MAE={ref_metrics['mae']:.3f}")

    texts = [d["raw_text"] for d in data]

    print("\nExtracting BERT [CLS] embeddings...")
    t0 = time.time()
    bert_emb = extract_bert_embeddings(texts)
    print(f"  Shape: {bert_emb.shape}, took {time.time()-t0:.1f}s")
    print("=" * 70)

    # ── E30f: Correction + Ensemble ────────────────────────────────
    if "e30f" not in args.skip:
        print("\n[E30f] Re-optimized correction + BERT ensemble...")
        t0 = time.time()
        res = run_e30f_correction_ensemble(data, bert_emb)
        elapsed = time.time() - t0
        met = _augment_metrics_with_optional_fields(compute_metrics(res), res)
        save_experiment("E30f-correction-ensemble", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30f")
        degradation_analysis(res, ref_results, "E30f")

    # ── E30g: PCA stacking ─────────────────────────────────────────
    if "e30g" not in args.skip:
        print("\n[E30g] PCA-reduced BERT + features → Ridge stacking...")
        t0 = time.time()
        res = run_e30g_pca_stacking(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30g-pca-stacking", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30g")
        degradation_analysis(res, ref_results, "E30g")

    # ── E30h: Adaptive weight blend ────────────────────────────────
    if "e30h" not in args.skip:
        print("\n[E30h] Adaptive weight blend (size-dependent SLM/BERT mix)...")
        t0 = time.time()
        res = run_e30h_adaptive_weight(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30h-adaptive-weight", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30h")
        degradation_analysis(res, ref_results, "E30h")

    # ── E30i: Correction + Adaptive ────────────────────────────────
    if "e30i" not in args.skip:
        print("\n[E30i] Re-optimized correction + adaptive-weight BERT blend...")
        t0 = time.time()
        res = run_e30i_correction_adaptive(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30i-correction-adaptive", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30i")
        degradation_analysis(res, ref_results, "E30i")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — E30 v2 improvements vs references")
    print("=" * 70)
    print(f"{'Experiment':<42} {'MAE':>6} {'PRED(25)':>9} {'SA':>7} {'Exact':>7}")
    print("-" * 72)
    print(f"{'E28a Qwen3-8B (reference)':<42} {ref_metrics['mae']:>6.3f} "
          f"{ref_metrics['pred_25']*100:>8.1f}% {ref_metrics['sa']*100:>6.1f}% "
          f"{ref_metrics['exact_match']*100:>6.1f}%")

    e29ev2 = Path("results/hybrid-xai/bert-experiments/E29e-v2-nested-ensemble/metrics.json")
    if e29ev2.exists():
        m29 = json.loads(e29ev2.read_text())
        print(f"{'E29e-v2 nested ensemble (prev best)':<42} {m29['mae']:>6.3f} "
              f"{m29['pred_25']*100:>8.1f}% {m29['sa']*100:>6.1f}% "
              f"{m29['exact_match']*100:>6.1f}%")

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
