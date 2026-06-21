#!/usr/bin/env python3
"""D1 — Ablation study: E30f-clean pipeline variants on De Vito holdout (30 FPs).

Compares 4 ablation variants of the E30f pipeline against the full E30f-clean
reference on the 30 De Vito holdout functional processes.

Ablation variants:
  slm-raw          — raw SLM output only (no correction, no BERT)
  correction-only  — TieredCorrection applied, no BERT
  bert-only        — BERT Ridge only, no SLM correction
  ensemble-fixed-w — full pipeline with frozen blend weight=0.7264

Data paths (relative to repo root):
  Train  (89 FPs): results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245/
  Holdout(30 FPs): results/hybrid-xai/119fp-E28a-qwen3-8b-holdout-devito-20260525T133453/
  Hparams:         results/hybrid-xai/bert-improvements-clean-train/hparams_final.json
  Ref metrics:     results/hybrid-xai/holdout-final-devito/E30f-correction-ensemble/metrics.json

Usage (DO NOT run on CPU — needs GPU, ~30 min):
  python scripts/ablation_e30f_holdout.py

Output:
  results/ablations/E30f-holdout-summary.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── Repo root & shared imports ────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.seeds import set_global_seeds

# Reuse all data-loading and correction helpers from bert_hybrid_improvements_v2
from bert_hybrid_improvements_v2 import (
    load_e28a_data,
    extract_bert_embeddings,
    _grid_search_correction,
    _apply_tiered_correction,
    compute_metrics,
)

# ── Hardcoded paths ───────────────────────────────────────────────────
TRAIN_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245"
HOLDOUT_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-holdout-devito-20260525T133453"
HPARAMS_PATH = BASE / "results/hybrid-xai/bert-improvements-clean-train/hparams_final.json"
REF_METRICS_PATH = (
    BASE
    / "results/hybrid-xai/holdout-final-devito/E30f-correction-ensemble/metrics.json"
)
OUTPUT_DIR = BASE / "results/ablations"
OUTPUT_CSV = OUTPUT_DIR / "E30f-holdout-summary.csv"

# Fixed blend weight from hparams_final.json → mean_blend_weight
FIXED_W = 0.7264


# ── Metrics ───────────────────────────────────────────────────────────

def _compute_ablation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, PRED25, SA, R² for the ablation."""
    n = len(y_true)
    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))

    re = ae / np.maximum(y_true, 1)
    pred25 = float(np.mean(re <= 0.25))

    mean_gt = float(np.mean(y_true))
    random_mae = float(np.mean(np.abs(y_true - mean_gt)))
    sa = 1.0 - mae / random_mae if random_mae > 0 else 0.0

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_gt) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"mae": mae, "pred25": pred25, "sa": sa, "r2": r2, "n": n}


# ── Ablation runners ──────────────────────────────────────────────────

def ablation_slm_raw(
    holdout_data: list[dict],
) -> dict:
    """Variant 1: raw SLM count, no correction, no BERT."""
    y_true = np.array([d["y_true"] for d in holdout_data], dtype=float)
    y_pred = np.array([d["y_raw_slm"] for d in holdout_data], dtype=float)
    y_pred = np.maximum(1, y_pred)
    return _compute_ablation_metrics(y_true, y_pred)


def ablation_correction_only(
    train_data: list[dict],
    holdout_data: list[dict],
) -> dict:
    """Variant 2: TieredCorrection on full train set → apply to holdout raw SLM."""
    y_true_tr = np.array([d["y_true"] for d in train_data])
    y_raw_tr = np.array([d["y_raw_slm"] for d in train_data])
    n_steps_tr = np.array([d["n_steps"] for d in train_data])

    print("  Grid-searching TieredCorrection params on full train set...")
    best_params = _grid_search_correction(y_true_tr, y_raw_tr, n_steps_tr)

    y_true_ho = np.array([d["y_true"] for d in holdout_data], dtype=float)
    y_pred = np.array([
        max(1, _apply_tiered_correction(d["y_raw_slm"], d["n_steps"], best_params))
        for d in holdout_data
    ], dtype=float)
    return _compute_ablation_metrics(y_true_ho, y_pred)


def ablation_bert_only(
    train_data: list[dict],
    holdout_data: list[dict],
    bert_emb_train: np.ndarray,
    bert_emb_holdout: np.ndarray,
) -> dict:
    """Variant 3: BERT Ridge only, no SLM correction component."""
    y_true_tr = np.array([d["y_true"] for d in train_data])

    # Build feature matrix: BERT + hand-crafted text features
    text_feats_tr = np.array(
        [[d["n_steps"], d["n_words"]] for d in train_data], dtype=float
    )
    text_feats_ho = np.array(
        [[d["n_steps"], d["n_words"]] for d in holdout_data], dtype=float
    )

    X_train = np.hstack([bert_emb_train, text_feats_tr])
    X_holdout = np.hstack([bert_emb_holdout, text_feats_ho])

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_holdout_sc = scaler.transform(X_holdout)

    reg = Ridge(alpha=10.0)
    reg.fit(X_train_sc, y_true_tr)

    y_pred_raw = reg.predict(X_holdout_sc)
    y_pred = np.maximum(1, np.round(y_pred_raw))

    y_true_ho = np.array([d["y_true"] for d in holdout_data], dtype=float)
    return _compute_ablation_metrics(y_true_ho, y_pred)


def ablation_ensemble_fixed_w(
    train_data: list[dict],
    holdout_data: list[dict],
    bert_emb_train: np.ndarray,
    bert_emb_holdout: np.ndarray,
    fixed_w: float,
) -> dict:
    """Variant 4: full pipeline but with frozen blend weight (no per-sample CV)."""
    y_true_tr = np.array([d["y_true"] for d in train_data])
    y_raw_tr = np.array([d["y_raw_slm"] for d in train_data])
    n_steps_tr = np.array([d["n_steps"] for d in train_data])

    print("  Grid-searching TieredCorrection params on full train set...")
    best_params = _grid_search_correction(y_true_tr, y_raw_tr, n_steps_tr)

    # Corrected SLM predictions for holdout
    y_corrected_ho = np.array([
        _apply_tiered_correction(d["y_raw_slm"], d["n_steps"], best_params)
        for d in holdout_data
    ], dtype=float)

    # BERT Ridge fit on full train, predict holdout
    text_feats_tr = np.array(
        [[d["n_steps"], d["n_words"]] for d in train_data], dtype=float
    )
    text_feats_ho = np.array(
        [[d["n_steps"], d["n_words"]] for d in holdout_data], dtype=float
    )
    X_train = np.hstack([bert_emb_train, text_feats_tr])
    X_holdout = np.hstack([bert_emb_holdout, text_feats_ho])

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_holdout_sc = scaler.transform(X_holdout)

    reg = Ridge(alpha=10.0)
    reg.fit(X_train_sc, y_true_tr)
    y_bert_ho = reg.predict(X_holdout_sc)

    # Ensemble with fixed weight
    y_pred = np.maximum(1, np.round(fixed_w * y_corrected_ho + (1 - fixed_w) * y_bert_ho))

    y_true_ho = np.array([d["y_true"] for d in holdout_data], dtype=float)
    return _compute_ablation_metrics(y_true_ho, y_pred)


# ── Output ────────────────────────────────────────────────────────────

def _load_ref_metrics(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    print(f"  [WARN] Reference metrics not found: {path}")
    return None


def _save_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "mae", "pred25", "sa", "r2", "n_fps", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary saved to: {path}")


def _print_comparison_table(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("ABLATION STUDY — E30f pipeline variants on De Vito holdout (30 FPs)")
    print("=" * 72)
    header = f"{'Variant':<22} {'MAE':>6}  {'PRED25':>7}  {'SA':>7}  {'R²':>7}  {'n':>4}"
    print(header)
    print("-" * 72)

    for row in rows:
        marker = " ← reference" if row["variant"] == "E30f-clean" else ""
        print(
            f"{row['variant']:<22} {row['mae']:>6.3f}  "
            f"{row['pred25']*100:>6.1f}%  "
            f"{row['sa']*100:>6.1f}%  "
            f"{row['r2']:>7.3f}  "
            f"{int(row['n_fps']):>4}{marker}"
        )
    print("-" * 72)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    set_global_seeds()

    print("=" * 72)
    print("D1 — Ablation: E30f-clean variants on De Vito holdout")
    print("=" * 72)

    # Load fixed blend weight from hparams
    if HPARAMS_PATH.exists():
        hparams = json.loads(HPARAMS_PATH.read_text())
        fixed_w = float(hparams.get("mean_blend_weight", FIXED_W))
        print(f"Fixed blend weight from hparams: {fixed_w}")
    else:
        fixed_w = FIXED_W
        print(f"[WARN] hparams_final.json not found — using default w={fixed_w}")

    # 1. Load train data (89 FPs)
    print(f"\nLoading train data from: {TRAIN_DIR.name}")
    train_data = load_e28a_data(TRAIN_DIR)
    print(f"  Loaded {len(train_data)} train FPs")

    # 2. Load holdout data (30 FPs)
    print(f"Loading holdout data from: {HOLDOUT_DIR.name}")
    holdout_data = load_e28a_data(HOLDOUT_DIR)
    print(f"  Loaded {len(holdout_data)} holdout FPs")

    # 3. Extract BERT embeddings
    print("\nExtracting BERT [CLS] embeddings (train)...")
    train_texts = [d["raw_text"] for d in train_data]
    bert_emb_train = extract_bert_embeddings(train_texts)
    print(f"  Train BERT shape: {bert_emb_train.shape}")

    print("Extracting BERT [CLS] embeddings (holdout)...")
    holdout_texts = [d["raw_text"] for d in holdout_data]
    bert_emb_holdout = extract_bert_embeddings(holdout_texts)
    print(f"  Holdout BERT shape: {bert_emb_holdout.shape}")

    # 4. Run ablation variants
    rows: list[dict] = []

    print("\n[1/4] Variant: slm-raw (no correction, no BERT)...")
    m = ablation_slm_raw(holdout_data)
    rows.append({
        "variant": "slm-raw",
        "mae": round(m["mae"], 4),
        "pred25": round(m["pred25"], 4),
        "sa": round(m["sa"], 4),
        "r2": round(m["r2"], 4),
        "n_fps": m["n"],
        "note": "Raw SLM y_raw_slm only",
    })
    print(f"  MAE={m['mae']:.3f} PRED25={m['pred25']*100:.1f}% SA={m['sa']*100:.1f}% R²={m['r2']:.3f}")

    print("\n[2/4] Variant: correction-only (TieredCorrection, no BERT)...")
    m = ablation_correction_only(train_data, holdout_data)
    rows.append({
        "variant": "correction-only",
        "mae": round(m["mae"], 4),
        "pred25": round(m["pred25"], 4),
        "sa": round(m["sa"], 4),
        "r2": round(m["r2"], 4),
        "n_fps": m["n"],
        "note": "TieredCorrection grid-searched on 89 train FPs",
    })
    print(f"  MAE={m['mae']:.3f} PRED25={m['pred25']*100:.1f}% SA={m['sa']*100:.1f}% R²={m['r2']:.3f}")

    print("\n[3/4] Variant: bert-only (Ridge on BERT, no SLM correction)...")
    m = ablation_bert_only(train_data, holdout_data, bert_emb_train, bert_emb_holdout)
    rows.append({
        "variant": "bert-only",
        "mae": round(m["mae"], 4),
        "pred25": round(m["pred25"], 4),
        "sa": round(m["sa"], 4),
        "r2": round(m["r2"], 4),
        "n_fps": m["n"],
        "note": "BERT [CLS] + text feats → Ridge(alpha=10), no SLM",
    })
    print(f"  MAE={m['mae']:.3f} PRED25={m['pred25']*100:.1f}% SA={m['sa']*100:.1f}% R²={m['r2']:.3f}")

    print(f"\n[4/4] Variant: ensemble-fixed-w (w={fixed_w})...")
    m = ablation_ensemble_fixed_w(
        train_data, holdout_data, bert_emb_train, bert_emb_holdout, fixed_w
    )
    rows.append({
        "variant": "ensemble-fixed-w",
        "mae": round(m["mae"], 4),
        "pred25": round(m["pred25"], 4),
        "sa": round(m["sa"], 4),
        "r2": round(m["r2"], 4),
        "n_fps": m["n"],
        "note": f"Full E30f pipeline, frozen w={fixed_w} (no per-sample CV)",
    })
    print(f"  MAE={m['mae']:.3f} PRED25={m['pred25']*100:.1f}% SA={m['sa']*100:.1f}% R²={m['r2']:.3f}")

    # 5. Load reference E30f-clean metrics
    ref = _load_ref_metrics(REF_METRICS_PATH)
    if ref is not None:
        rows.append({
            "variant": "E30f-clean",
            "mae": round(ref["mae"], 4),
            "pred25": round(ref["pred_25"], 4),
            "sa": round(ref["sa"], 4),
            "r2": round(ref["r2"], 4),
            "n_fps": int(ref.get("n_with_gt", 30)),
            "note": "Full E30f-clean with per-sample nested CV blend weight",
        })

    # 6. Save and print
    _save_summary(rows, OUTPUT_CSV)
    _print_comparison_table(rows)


if __name__ == "__main__":
    main()
