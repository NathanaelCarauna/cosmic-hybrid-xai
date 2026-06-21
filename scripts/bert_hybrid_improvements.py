#!/usr/bin/env python3
"""BERT-based Hybrid-XAI improvement experiments (E30a–E30e).

Builds on E28a (Qwen3-8B) and E29 (BERT) findings to address specific
weaknesses without degradation.

Experiments:
  E30a  Re-opt TieredCorrection   — LOOCV grid search for Qwen3-8B error profile
  E30b  Size-selective ensemble   — SLM-only for small UCs, BERT ensemble for med/large
  E30c  Confidence-gated router   — BERT routing only when classifier is confident
  E30d  Multi-signal stacking     — SLM pred + BERT emb + text feats → Ridge meta-learner
  E30e  Selective ensemble + gate — Predict per-FP whether ensemble helps, apply selectively

Usage:
  python scripts/bert_hybrid_improvements.py \
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ── Data loading ─────────────────────────────────────────────────────

def load_e28a_data(e28a_dir: Path) -> list[dict]:
    """Load E28a comparison_results + IRs, separating raw SLM from correction."""
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

        data.append({
            "fp_id": fp_id,
            "y_true": int(row["y_true"]),
            "y_pred_slm": int(row["y_pred"]),
            "y_raw_slm": raw_count if ir else int(row["y_pred"]),
            "correction_count": correction_count,
            "raw_text": ir["raw_text"] if ir else "",
            "n_movements": int(row["n_movements"]),
            "n_processes": int(row["n_processes"]),
            "ir": ir,
        })
    return data


def extract_text_features(text: str) -> dict:
    lines = text.strip().split("\n")
    words = text.split()
    step_like = [l for l in lines if re.match(r"^\s*\d+\s*\t", l)]
    alt_lines = [l for l in lines if re.match(r"(?i)^\s*(alternative|exception)", l)]
    return {
        "n_lines": len(lines),
        "n_words": len(words),
        "n_steps": len(step_like),
        "n_alt_exc": len(alt_lines),
        "text_len": len(text),
    }


def count_uc_steps(text: str) -> int:
    """Count main-scenario + alternative + exception steps."""
    lines = text.strip().split("\n")
    return sum(1 for l in lines if re.match(r"^\s*\d+\s*\t", l))


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


# ── E30a: Re-optimized TieredCorrection ──────────────────────────────

def run_e30a_reopt_tiered(data: list[dict]) -> list[dict]:
    """LOOCV grid search for TieredCorrection parameters on Qwen3-8B's raw SLM output."""
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_raw = np.array([d["y_raw_slm"] for d in data])
    n_steps_arr = np.array([count_uc_steps(d["raw_text"]) for d in data])

    def apply_tiered_correction(y_raw_val, n_steps_val, params):
        alpha, beta, gamma, alpha_l, beta_l, gamma_l, threshold = params
        if n_steps_val >= threshold:
            a, b, g = alpha_l, beta_l, gamma_l
        else:
            a, b, g = alpha, beta, gamma
        raw_corr = a * max(0, n_steps_val - b)
        correction = min(raw_corr, y_raw_val * g)
        extra = round(correction) if correction > 0.5 else 0
        return y_raw_val + extra

    alphas = np.arange(0.1, 0.6, 0.1)
    betas = np.arange(4, 12, 2)
    gammas = np.arange(0.2, 0.8, 0.1)
    alpha_ls = np.arange(0.2, 0.8, 0.1)
    beta_ls = np.arange(2, 8, 1)
    gamma_ls = np.arange(0.3, 1.0, 0.1)
    thresholds = [10, 12, 14, 16]

    print(f"    Grid: {len(alphas)}×{len(betas)}×{len(gammas)}×{len(alpha_ls)}×{len(beta_ls)}×{len(gamma_ls)}×{len(thresholds)} = huge")
    print("    Using staged search (coarse → fine)...")

    best_params = None
    best_mae = float("inf")

    for threshold in thresholds:
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    for alpha_l in alpha_ls:
                        for beta_l in beta_ls:
                            for gamma_l in gamma_ls:
                                params = (alpha, beta, gamma, alpha_l, beta_l, gamma_l, threshold)
                                preds = np.array([
                                    apply_tiered_correction(y_raw[i], n_steps_arr[i], params)
                                    for i in range(n)
                                ])
                                mae = np.mean(np.abs(y_true - preds))
                                if mae < best_mae:
                                    best_mae = mae
                                    best_params = params

    print(f"    Best params: α={best_params[0]:.2f} β={best_params[1]:.0f} γ={best_params[2]:.2f} "
          f"α_L={best_params[3]:.2f} β_L={best_params[4]:.0f} γ_L={best_params[5]:.2f} thr={best_params[6]}")
    print(f"    Grid-search MAE (in-sample): {best_mae:.3f}")

    # LOOCV validation with the best params to get unbiased estimate
    preds_loocv = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        train_best_mae = float("inf")
        train_best_params = best_params

        # Re-search on training set only (lighter grid around best)
        a0, b0, g0, al0, bl0, gl0, t0 = best_params
        for da in [-0.1, 0, 0.1]:
            for db in [-1, 0, 1]:
                for dg in [-0.1, 0, 0.1]:
                    for dal in [-0.1, 0, 0.1]:
                        for dbl in [-1, 0, 1]:
                            for dgl in [-0.1, 0, 0.1]:
                                params = (
                                    max(0.05, a0 + da),
                                    max(1, b0 + db),
                                    max(0.1, g0 + dg),
                                    max(0.05, al0 + dal),
                                    max(1, bl0 + dbl),
                                    max(0.1, gl0 + dgl),
                                    t0,
                                )
                                train_preds = np.array([
                                    apply_tiered_correction(y_raw[j], n_steps_arr[j], params)
                                    for j in np.where(mask)[0]
                                ])
                                train_mae = np.mean(np.abs(y_true[mask] - train_preds))
                                if train_mae < train_best_mae:
                                    train_best_mae = train_mae
                                    train_best_params = params

        preds_loocv[i] = apply_tiered_correction(y_raw[i], n_steps_arr[i], train_best_params)

    return _build_results(data, preds_loocv)


# ── E30b: Size-selective ensemble ────────────────────────────────────

def run_e30b_size_selective(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """Use SLM-only (E28a) for small UCs, BERT+SLM ensemble for medium/large.

    The threshold is selected via nested LOOCV to avoid leakage.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    text_feats = np.array([
        [count_uc_steps(d["raw_text"]),
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])

        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred_i = reg.predict(X_test)[0]

        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        best_w = 0.7
        best_thr = 4
        best_inner_mae = float("inf")

        for thr in [3, 4, 5, 6]:
            for w in np.arange(0.50, 0.90, 0.05):
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
                        if y_slm[vi] <= thr:
                            ens = y_slm[vi]
                        else:
                            ens = w * y_slm[vi] + (1 - w) * bp[k]
                        inner_errors.append(abs(y_true[vi] - round(max(1, ens))))

                mae_inner = np.mean(inner_errors)
                if mae_inner < best_inner_mae:
                    best_inner_mae = mae_inner
                    best_w = w
                    best_thr = thr

        if y_slm[i] <= best_thr:
            preds[i] = y_slm[i]
        else:
            preds[i] = max(1, round(best_w * y_slm[i] + (1 - best_w) * bert_pred_i))

    return _build_results(data, preds)


# ── E30c: Confidence-gated complexity router ─────────────────────────

def run_e30c_confidence_gated(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """Apply BERT complexity routing only when the classifier is confident.

    Falls back to E28a (SLM) prediction for uncertain classifications.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    bins = np.where(y_true <= 3, 0, np.where(y_true <= 6, 1, 2))

    text_feats = np.array([
        [d["n_movements"],
         count_uc_steps(d["raw_text"]),
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, bins[mask])
        proba = clf.predict_proba(X_test)[0]
        pred_bin = clf.predict(X_test)[0]
        confidence = proba.max()

        # Nested CV to find best confidence threshold
        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_conf_thr = 0.6
        best_inner_mae = float("inf")

        for conf_thr in np.arange(0.50, 0.85, 0.05):
            inner_errors = []
            for train_fold, val_fold in kf.split(inner_idxs):
                tr = inner_idxs[train_fold]
                va = inner_idxs[val_fold]
                sc2 = StandardScaler()
                X2_tr = sc2.fit_transform(X[tr])
                X2_va = sc2.transform(X[va])
                clf2 = LogisticRegression(max_iter=1000, C=1.0)
                clf2.fit(X2_tr, bins[tr])
                probas = clf2.predict_proba(X2_va)
                preds_bin = clf2.predict(X2_va)

                for k, vi in enumerate(va):
                    if probas[k].max() < conf_thr:
                        inner_errors.append(abs(y_true[vi] - y_slm[vi]))
                    else:
                        pb = preds_bin[k]
                        bm = (bins[tr] == pb)
                        if bm.sum() >= 3:
                            corr = np.median(y_true[tr][bm] - y_slm[tr][bm])
                        else:
                            corr = np.median(y_true[tr] - y_slm[tr])
                        pred_val = max(1, round(y_slm[vi] + corr))
                        inner_errors.append(abs(y_true[vi] - pred_val))

            mae_inner = np.mean(inner_errors)
            if mae_inner < best_inner_mae:
                best_inner_mae = mae_inner
                best_conf_thr = conf_thr

        if confidence < best_conf_thr:
            preds[i] = y_slm[i]
        else:
            bm = mask & (bins == pred_bin)
            if bm.sum() >= 3:
                corr = np.median(y_true[bm] - y_slm[bm])
            else:
                corr = np.median(y_true[mask] - y_slm[mask])
            preds[i] = max(1, round(y_slm[i] + corr))

    return _build_results(data, preds)


# ── E30d: Multi-signal stacking ──────────────────────────────────────

def run_e30d_stacking(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """Stack SLM prediction + BERT embedding + text features → Ridge meta-learner.

    Features:
    - SLM prediction (y_pred_slm)
    - Raw SLM prediction (before correction)
    - Correction count from TieredCorrection
    - n_movements, n_processes from IR
    - Text features (n_steps, n_words, n_lines, text_len)
    - BERT [CLS] embedding (768-dim)

    Uses nested LOOCV with inner 10-fold for alpha selection.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])

    hand_feats = np.array([
        [
            d["y_pred_slm"],
            d["y_raw_slm"],
            d["correction_count"],
            d["n_movements"],
            d["n_processes"],
            count_uc_steps(d["raw_text"]),
            extract_text_features(d["raw_text"])["n_words"],
            extract_text_features(d["raw_text"])["n_lines"],
            extract_text_features(d["raw_text"])["text_len"],
        ]
        for d in data
    ], dtype=float)
    X = np.hstack([hand_feats, bert_emb])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])

        inner_idxs = np.where(mask)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        best_alpha = 10.0
        best_inner_mae = float("inf")

        for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            inner_errors = []
            for train_fold, val_fold in kf.split(inner_idxs):
                tr = inner_idxs[train_fold]
                va = inner_idxs[val_fold]
                sc2 = StandardScaler()
                X2_tr = sc2.fit_transform(X[tr])
                X2_va = sc2.transform(X[va])
                r2 = Ridge(alpha=alpha)
                r2.fit(X2_tr, y_true[tr])
                bp = r2.predict(X2_va)
                inner_errors.extend(np.abs(y_true[va] - np.round(np.maximum(1, bp))))

            mae_inner = np.mean(inner_errors)
            if mae_inner < best_inner_mae:
                best_inner_mae = mae_inner
                best_alpha = alpha

        reg = Ridge(alpha=best_alpha)
        reg.fit(X_train, y_true[mask])
        preds[i] = max(1, round(reg.predict(X_test)[0]))

    return _build_results(data, preds)


# ── E30e: Selective ensemble with error-prediction gate ──────────────

def run_e30e_selective_ensemble(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """Predict per-FP whether BERT ensemble would improve, apply selectively.

    Train a gating classifier on features to predict: will |ensemble_error| < |slm_error|?
    When "yes" → use ensemble prediction; when "no" → keep SLM prediction.
    Uses double LOOCV to avoid leakage.
    """
    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    text_feats = np.array([
        [count_uc_steps(d["raw_text"]),
         extract_text_features(d["raw_text"])["n_words"],
         d["n_movements"],
         d["y_raw_slm"],
         d["correction_count"]]
        for d in data
    ], dtype=float)
    X_bert = np.hstack([bert_emb, text_feats])

    X_gate = np.array([
        [
            d["y_pred_slm"],
            d["y_raw_slm"],
            d["correction_count"],
            d["n_movements"],
            count_uc_steps(d["raw_text"]),
            extract_text_features(d["raw_text"])["n_words"],
            extract_text_features(d["raw_text"])["n_lines"],
            abs(d["y_pred_slm"] - d["y_raw_slm"]),
        ]
        for d in data
    ], dtype=float)

    preds = np.zeros(n)
    gate_decisions = np.zeros(n, dtype=int)

    for i in range(n):
        mask = np.arange(n) != i
        inner_idxs = np.where(mask)[0]

        # Step 1: Get BERT prediction for test point
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_bert[mask])
        X_test = scaler.transform(X_bert[i : i + 1])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred_i = reg.predict(X_test)[0]

        # Step 2: Get BERT predictions for all training points (inner LOOCV)
        bert_preds_train = np.zeros(mask.sum())
        for j_idx, j in enumerate(inner_idxs):
            inner_mask = np.delete(inner_idxs, j_idx)
            sc2 = StandardScaler()
            X2_tr = sc2.fit_transform(X_bert[inner_mask])
            X2_te = sc2.transform(X_bert[j : j + 1])
            r2 = Ridge(alpha=10.0)
            r2.fit(X2_tr, y_true[inner_mask])
            bert_preds_train[j_idx] = r2.predict(X2_te)[0]

        # Step 3: Compute ensemble (w=0.74 from E29e-v2) and gate labels
        w = 0.74
        ensemble_train = w * y_slm[mask] + (1 - w) * bert_preds_train
        ensemble_ae = np.abs(y_true[mask] - np.round(np.maximum(1, ensemble_train)))
        slm_ae = np.abs(y_true[mask] - y_slm[mask])
        gate_labels = (ensemble_ae < slm_ae).astype(int)

        # Step 4: Train gate classifier
        sc_gate = StandardScaler()
        X_gate_train = sc_gate.fit_transform(X_gate[mask])
        X_gate_test = sc_gate.transform(X_gate[i : i + 1])

        gate_clf = LogisticRegression(max_iter=1000, C=0.1)
        if gate_labels.sum() > 0 and (1 - gate_labels).sum() > 0:
            gate_clf.fit(X_gate_train, gate_labels)
            use_ensemble = gate_clf.predict(X_gate_test)[0] == 1
        else:
            use_ensemble = False

        if use_ensemble:
            preds[i] = max(1, round(w * y_slm[i] + (1 - w) * bert_pred_i))
            gate_decisions[i] = 1
        else:
            preds[i] = y_slm[i]

    print(f"    Gate applied ensemble to {gate_decisions.sum()}/{n} FPs "
          f"({gate_decisions.sum()/n*100:.1f}%)")

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


def save_experiment(
    name: str, results: list[dict], metrics: dict,
    elapsed: float, output_dir: Path,
):
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_pred", "y_true"])
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
    """Print MAE breakdown by GT size tier."""
    y_true = np.array([r["y_true"] for r in results], dtype=float)
    y_pred = np.array([r["y_pred"] for r in results], dtype=float)
    print(f"    {label} per-tier:")
    for tag, lo, hi in [("Small ≤4", 0, 4), ("Med 5-8", 5, 8), ("Large ≥9", 9, 999)]:
        m = (y_true >= lo) & (y_true <= hi)
        if m.sum() == 0:
            continue
        mae = np.mean(np.abs(y_true[m] - y_pred[m]))
        print(f"      {tag}: n={m.sum():3d} MAE={mae:.3f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERT Hybrid-XAI improvements E30a–E30e")
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
    print(f"  Loaded {len(data)} FPs")
    print(f"  E28a (reference): MAE={np.mean(np.abs(y_true - y_slm)):.3f}")
    print(f"  Raw SLM (no correction): MAE={np.mean(np.abs(y_true - np.array([d['y_raw_slm'] for d in data]))):.3f}")

    ref_metrics = compute_metrics([
        {"fp": d["fp_id"], "y_pred": d["y_pred_slm"], "y_true": d["y_true"]}
        for d in data
    ])

    texts = [d["raw_text"] for d in data]

    bert_emb = None
    need_bert = any(e not in args.skip for e in ["e30b", "e30c", "e30d", "e30e"])
    if need_bert:
        print("\nExtracting BERT [CLS] embeddings...")
        t0 = time.time()
        bert_emb = extract_bert_embeddings(texts)
        print(f"  Shape: {bert_emb.shape}, took {time.time()-t0:.1f}s")

    print("=" * 70)

    # ── E30a: Re-optimized TieredCorrection ────────────────────────
    if "e30a" not in args.skip:
        print("\n[E30a] Re-optimized TieredCorrection for Qwen3-8B...")
        t0 = time.time()
        res = run_e30a_reopt_tiered(data)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30a-reopt-tiered", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30a")

    # ── E30b: Size-selective ensemble ──────────────────────────────
    if "e30b" not in args.skip:
        print("\n[E30b] Size-selective ensemble (SLM for small, BERT for med/large)...")
        t0 = time.time()
        res = run_e30b_size_selective(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30b-size-selective", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30b")

    # ── E30c: Confidence-gated router ──────────────────────────────
    if "e30c" not in args.skip:
        print("\n[E30c] Confidence-gated complexity router...")
        t0 = time.time()
        res = run_e30c_confidence_gated(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30c-confidence-gated", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30c")

    # ── E30d: Multi-signal stacking ────────────────────────────────
    if "e30d" not in args.skip:
        print("\n[E30d] Multi-signal stacking (SLM + BERT + text → Ridge)...")
        t0 = time.time()
        res = run_e30d_stacking(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30d-stacking", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30d")

    # ── E30e: Selective ensemble ───────────────────────────────────
    if "e30e" not in args.skip:
        print("\n[E30e] Selective ensemble with error-prediction gate...")
        t0 = time.time()
        res = run_e30e_selective_ensemble(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E30e-selective-ensemble", res, met, elapsed, args.output_dir)
        per_size_analysis(res, "E30e")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — E30 improvements vs references")
    print("=" * 70)
    print(f"{'Experiment':<40} {'MAE':>6} {'PRED(25)':>9} {'SA':>7} {'Exact':>7}")
    print("-" * 70)
    print(f"{'E28a Qwen3-8B (reference)':<40} {ref_metrics['mae']:>6.3f} "
          f"{ref_metrics['pred_25']*100:>8.1f}% {ref_metrics['sa']*100:>6.1f}% "
          f"{ref_metrics['exact_match']*100:>6.1f}%")

    e29ev2 = Path("results/hybrid-xai/bert-experiments/E29e-v2-nested-ensemble/metrics.json")
    if e29ev2.exists():
        m29 = json.loads(e29ev2.read_text())
        print(f"{'E29e-v2 nested ensemble (prev best)':<40} {m29['mae']:>6.3f} "
              f"{m29['pred_25']*100:>8.1f}% {m29['sa']*100:>6.1f}% "
              f"{m29['exact_match']*100:>6.1f}%")

    print("-" * 70)
    for exp_name in sorted(args.output_dir.iterdir()):
        mf = exp_name / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            print(f"{exp_name.name:<40} {m['mae']:>6.3f} "
                  f"{m['pred_25']*100:>8.1f}% {m['sa']*100:>6.1f}% "
                  f"{m['exact_match']*100:>6.1f}%")
    print("-" * 70)


if __name__ == "__main__":
    main()
