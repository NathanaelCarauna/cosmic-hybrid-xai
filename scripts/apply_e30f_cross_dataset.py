#!/usr/bin/env python3
"""A5/A7 — Apply E30f-clean pipeline (frozen on De Vito train-89) to cross-dataset.

Calibrates TieredCorrection + BERT Ridge on the 89 De Vito train FPs, then
applies with frozen parameters to Zenodo-200 or Ochodek-200 E28a results.

Requires that E28a has already been run on the target dataset.

Usage:
    python scripts/apply_e30f_cross_dataset.py zenodo \
        --e28a-dir results/hybrid-xai/zenodo-E28a-clean-zenodo200-20260525T183254 \
        --gt-csv datasets/cross-sample/zenodo-200-r1.csv

    python scripts/apply_e30f_cross_dataset.py ochodek \
        --e28a-dir results/hybrid-xai/ochodek-E28a-clean-ochodek200-... \
        --gt-csv datasets/cross-sample/ochodek-200-r1.csv

Output:
    results/hybrid-xai/e30f-cross-<dataset>-<timestamp>/metrics.json
    results/hybrid-xai/e30f-cross-<dataset>-<timestamp>/comparison_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.seeds import set_global_seeds
from bert_hybrid_improvements_v2 import (
    load_e28a_data,
    extract_bert_embeddings,
    _grid_search_correction,
    _apply_tiered_correction,
)

TRAIN_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245"
HPARAMS_PATH = BASE / "results/hybrid-xai/bert-improvements-clean-train/hparams_final.json"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))
    re = ae / np.maximum(y_true, 1)
    pred25 = float(np.mean(re <= 0.25))
    mmre = float(np.mean(re))
    mgt = float(np.mean(y_true))
    random_mae = float(np.mean(np.abs(y_true - mgt)))
    sa = (1.0 - mae / random_mae) if random_mae > 0 else 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mgt) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mdae = float(np.median(ae))
    return {
        "n_with_gt": n,
        "mae": round(mae, 4),
        "mdae": round(mdae, 4),
        "mmre": round(mmre, 4),
        "pred_25": round(pred25, 4),
        "sa": round(sa, 4),
        "r2": round(r2, 4),
    }


def load_gt_csv(gt_csv: Path, dataset: str) -> dict[str, int]:
    """Load ground truth CFP from sample CSV. Returns {fp_id: cfp}.

    The fp_id key format must match what load_e28a_data returns:
    - zenodo: "FP001" (input dir has FP001.txt files → iter_zenodo_fps uses stem)
    - ochodek: "O001" (ochodek-200-r1.csv has id="O001" → iter_ochodek_fps uses id col)
    """
    gt = {}
    if dataset == "user-stories":
        # ground_truth.csv has columns fp_id (FP001) + cfp_gt; FP*.txt stems
        with gt_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                fp_id = r.get("fp_id", "")
                cfp_val = r.get("cfp_gt", "")
                if fp_id and cfp_val:
                    gt[fp_id] = int(cfp_val)
        return gt
    with gt_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            item_id = r.get("id", "")
            cfp_key = "CFP" if "CFP" in r else "cfp"
            cfp_val = r.get(cfp_key, "")
            if item_id and cfp_val:
                if dataset == "zenodo":
                    # Z001 → FP001 (iter_zenodo_fps uses FP*.txt stems)
                    fp_id = f"FP{int(item_id.lstrip('Z')):03d}"
                else:
                    # O001 stays as O001 (iter_ochodek_fps uses "id" column)
                    fp_id = item_id
                gt[fp_id] = int(cfp_val)
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply E30f-clean (frozen hparams) to cross-dataset E28a results"
    )
    parser.add_argument("dataset", choices=["zenodo", "ochodek", "user-stories"],
                        help="Target dataset label")
    parser.add_argument("--e28a-dir", type=Path, required=True,
                        help="Directory with E28a IR JSON files for the target dataset")
    parser.add_argument("--gt-csv", type=Path, required=True,
                        help="Ground-truth CSV (from datasets/cross-sample/)")
    parser.add_argument("--blend-weight", type=float, default=None,
                        help="Override blend weight (default: from hparams_final.json)")
    args = parser.parse_args()

    set_global_seeds()

    if not args.e28a_dir.exists():
        print(f"ERROR: E28a dir not found: {args.e28a_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.gt_csv.exists():
        print(f"ERROR: GT CSV not found: {args.gt_csv}", file=sys.stderr)
        sys.exit(1)

    # Load frozen blend weight
    blend_w = args.blend_weight
    if blend_w is None and HPARAMS_PATH.exists():
        hparams = json.loads(HPARAMS_PATH.read_text())
        blend_w = float(hparams.get("mean_blend_weight", 0.7264))
    if blend_w is None:
        blend_w = 0.7264
    print(f"Blend weight: {blend_w:.4f}")

    # Load De Vito train set (89 FPs) for calibration
    print(f"\nLoading De Vito train data from: {TRAIN_DIR.name}")
    train_data = load_e28a_data(TRAIN_DIR)
    print(f"  Loaded {len(train_data)} train FPs")

    y_true_tr = np.array([d["y_true"] for d in train_data])
    y_raw_tr = np.array([d["y_raw_slm"] for d in train_data])
    n_steps_tr = np.array([d["n_steps"] for d in train_data])

    # Grid-search TieredCorrection params on De Vito train
    print("Grid-searching TieredCorrection params on De Vito train...")
    best_params = _grid_search_correction(y_true_tr, y_raw_tr, n_steps_tr)
    print(f"  Best params: {best_params}")

    # BERT Ridge on De Vito train
    print("Extracting BERT embeddings (train)...")
    train_texts = [d["raw_text"] for d in train_data]
    bert_emb_train = extract_bert_embeddings(train_texts)
    text_feats_tr = np.array([[d["n_steps"], d["n_words"]] for d in train_data], dtype=float)
    X_train = np.hstack([bert_emb_train, text_feats_tr])
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    reg = Ridge(alpha=10.0)
    reg.fit(X_train_sc, y_true_tr)
    print(f"  BERT Ridge trained: train shape={X_train_sc.shape}")

    # Load cross-dataset E28a results
    print(f"\nLoading cross-dataset E28a from: {args.e28a_dir.name}")
    eval_data = load_e28a_data(args.e28a_dir)
    print(f"  Loaded {len(eval_data)} items")

    # Load ground truth
    gt = load_gt_csv(args.gt_csv, args.dataset)
    print(f"  Ground truth: {len(gt)} items")

    # Filter to items with ground truth
    eval_with_gt = [d for d in eval_data if d.get("fp_id") in gt]
    if not eval_with_gt:
        # Try matching by index
        eval_with_gt = eval_data
        print(f"  [WARN] fp_id matching failed; using all {len(eval_data)} items")

    print(f"  Items with GT: {len(eval_with_gt)}")

    # Extract BERT embeddings for eval set
    print("Extracting BERT embeddings (eval)...")
    eval_texts = [d["raw_text"] for d in eval_with_gt]
    bert_emb_eval = extract_bert_embeddings(eval_texts)
    text_feats_ev = np.array([[d["n_steps"], d["n_words"]] for d in eval_with_gt], dtype=float)
    X_eval = np.hstack([bert_emb_eval, text_feats_ev])
    X_eval_sc = scaler.transform(X_eval)

    # Apply pipeline to each eval item
    results = []
    for i, d in enumerate(eval_with_gt):
        fp_id = d.get("fp_id", f"FP{i+1:03d}")
        y_true = gt.get(fp_id, d.get("y_true", 0))

        # TieredCorrection
        y_corrected = _apply_tiered_correction(d["y_raw_slm"], d["n_steps"], best_params)

        # BERT Ridge prediction
        y_bert = float(reg.predict(X_eval_sc[i:i+1])[0])

        # Ensemble
        y_pred = blend_w * y_corrected + (1 - blend_w) * y_bert
        y_pred_rounded = max(0, round(y_pred))

        results.append({
            "fp_id": fp_id,
            "y_true": y_true,
            "y_raw_slm": d["y_raw_slm"],
            "y_corrected": round(y_corrected, 3),
            "y_bert": round(y_bert, 3),
            "y_pred": y_pred_rounded,
            "blend_w": blend_w,
            "ae": abs(y_true - y_pred_rounded),
        })

    # Compute metrics
    y_true_arr = np.array([r["y_true"] for r in results], dtype=float)
    y_pred_arr = np.array([r["y_pred"] for r in results], dtype=float)
    m = compute_metrics(y_true_arr, y_pred_arr)

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = BASE / "results/hybrid-xai" / f"e30f-cross-{args.dataset}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison_results.csv
    with (out_dir / "comparison_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp_id", "y_true", "y_raw_slm", "y_corrected", "y_bert", "y_pred", "blend_w", "ae"])
        w.writeheader()
        w.writerows(results)

    # Save metrics.json
    metrics_out = {
        **m,
        "dataset": args.dataset,
        "e28a_dir": str(args.e28a_dir),
        "gt_csv": str(args.gt_csv),
        "blend_weight": blend_w,
        "best_correction_params": best_params,
        "timestamp": ts,
        "mean_signed_error": round(float(np.mean(y_pred_arr - y_true_arr)), 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    print(f"\n{'='*60}")
    print(f"E30f-clean (frozen) → {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  n={m['n_with_gt']}, MAE={m['mae']:.3f}, PRED25={m['pred_25']*100:.1f}%, "
          f"SA={m['sa']:.3f}, R²={m['r2']:.3f}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
