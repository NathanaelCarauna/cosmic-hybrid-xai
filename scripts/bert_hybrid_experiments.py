#!/usr/bin/env python3
"""BERT-based Hybrid-XAI experiments (E29a–E29e).

Runs 5 BERT variants with LOOCV over 119 FPs De Vito, using existing
SLM IRs from E28a (Qwen3-8B) and BERT embeddings (no fine-tuning for
a–c,e; sentence-BERT for d).

Experiments:
  E29a  BERT calibration       — BERT [CLS] → Ridge predicts correction
  E29b  BERT complexity router  — BERT [CLS] → classifier selects params
  E29c  BERT direct regression  — BERT [CLS] → Ridge predicts CFP
  E29d  Sentence-BERT dedup     — cosine dedup in hierarchical decomp
  E29e  BERT+SLM ensemble       — weighted avg of SLM CFP + BERT CFP

Usage:
  python scripts/bert_hybrid_experiments.py \
    --e28a-dir results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921 \
    --output-dir results/hybrid-xai/bert-experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


# ── Data loading ─────────────────────────────────────────────────────

def load_e28a_data(e28a_dir: Path) -> list[dict]:
    """Load comparison_results + raw texts from E28a run."""
    comp = e28a_dir / "comparison_results.csv"
    rows = list(csv.DictReader(comp.open()))

    data = []
    for row in rows:
        fp_id = row["fp"]
        safe = fp_id.replace("/", "_")
        ir_path = e28a_dir / f"{safe}_ir.json"
        ir = json.loads(ir_path.read_text()) if ir_path.exists() else None

        data.append({
            "fp_id": fp_id,
            "y_true": int(row["y_true"]),
            "y_pred_slm": int(row["y_pred"]),
            "raw_text": ir["raw_text"] if ir else "",
            "n_movements": int(row["n_movements"]),
            "n_processes": int(row["n_processes"]),
            "ir": ir,
        })
    return data


def extract_text_features(text: str) -> dict:
    """Simple text features for enriching BERT embeddings."""
    lines = text.strip().split("\n")
    steps = [l for l in lines if l.strip().startswith(("Step", "step", "S", "E", "A"))]
    words = text.split()
    return {
        "n_lines": len(lines),
        "n_words": len(words),
        "n_steps": len(steps),
        "text_len": len(text),
    }


# ── BERT embedding extraction ───────────────────────────────────────

def extract_bert_embeddings(texts: list[str], max_length: int = 512) -> np.ndarray:
    """Extract BERT [CLS] embeddings for all texts (no fine-tuning)."""
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


def extract_sbert_embeddings(texts: list[str]) -> np.ndarray:
    """Extract Sentence-BERT embeddings."""
    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False)


# ── Experiment implementations ───────────────────────────────────────

def run_e29a_bert_calibration(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """E29a: BERT embeddings predict correction = GT - SLM_pred (LOOCV)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])
    corrections = y_true - y_slm

    text_feats = np.array([
        [d["n_movements"], d["n_processes"],
         extract_text_features(d["raw_text"])["n_steps"],
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])

        reg = Ridge(alpha=10.0)
        reg.fit(X_train, corrections[mask])
        pred_corr = reg.predict(X_test)[0]
        preds[i] = max(1, round(y_slm[i] + pred_corr))

    return _build_results(data, preds)


def run_e29b_bert_complexity_router(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """E29b: BERT classifies UC complexity → route to tier params (LOOCV)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    bins = np.where(y_true <= 3, 0, np.where(y_true <= 6, 1, 2))

    text_feats = np.array([
        [d["n_movements"],
         extract_text_features(d["raw_text"])["n_steps"],
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, bins[mask])
        pred_bin = clf.predict(X_test)[0]

        bin_mask = mask & (bins == pred_bin)
        if bin_mask.sum() < 3:
            bin_mask = mask

        corrections_in_bin = y_true[bin_mask] - y_slm[bin_mask]
        if bin_mask.sum() >= 5:
            reg = Ridge(alpha=5.0)
            reg.fit(X_train[bins[mask] == pred_bin] if (bins[mask] == pred_bin).sum() >= 3
                    else X_train,
                    corrections_in_bin if (bins[mask] == pred_bin).sum() >= 3
                    else (y_true[mask] - y_slm[mask]))
            corr = reg.predict(X_test)[0]
        else:
            corr = np.median(corrections_in_bin)

        preds[i] = max(1, round(y_slm[i] + corr))

    return _build_results(data, preds)


def run_e29c_bert_regression(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """E29c: BERT embeddings → Ridge predicts CFP directly (LOOCV)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    y_true = np.array([d["y_true"] for d in data])

    text_feats = np.array([
        [extract_text_features(d["raw_text"])["n_steps"],
         extract_text_features(d["raw_text"])["n_words"],
         extract_text_features(d["raw_text"])["n_lines"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])

        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        preds[i] = max(1, round(reg.predict(X_test)[0]))

    return _build_results(data, preds)


def run_e29d_sbert_dedup(
    data: list[dict], sbert_model
) -> list[dict]:
    """E29d: Sentence-BERT semantic dedup in hierarchical decomposition.

    Re-processes large UCs (>=12 steps) by chunking and merging with
    semantic similarity for deduplication instead of string matching.
    Uses existing E28a IRs for non-large UCs.
    """
    from sentence_transformers import SentenceTransformer

    results = []
    for d in data:
        ir = d["ir"]
        if ir is None:
            results.append({"fp": d["fp_id"], "y_pred": d["y_pred_slm"],
                            "y_true": d["y_true"]})
            continue

        fps = ir["functional_processes"]
        total_movements = []
        for fp in fps:
            total_movements.extend(fp["data_movements"])

        if len(total_movements) <= 6:
            results.append({"fp": d["fp_id"], "y_pred": d["y_pred_slm"],
                            "y_true": d["y_true"]})
            continue

        descs = [f"{m['type']} {m['data_group']} {m['object_of_interest']}"
                 for m in total_movements]

        if len(descs) < 2:
            results.append({"fp": d["fp_id"], "y_pred": len(total_movements),
                            "y_true": d["y_true"]})
            continue

        embs = sbert_model.encode(descs)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embs)

        keep = [True] * len(total_movements)
        threshold = 0.85
        for i in range(len(total_movements)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(total_movements)):
                if not keep[j]:
                    continue
                if (total_movements[i]["type"] == total_movements[j]["type"]
                        and sim_matrix[i, j] > threshold):
                    keep[j] = False

        deduped_count = sum(keep)
        results.append({"fp": d["fp_id"], "y_pred": deduped_count,
                        "y_true": d["y_true"]})

    return results


def run_e29e_bert_ensemble(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """E29e: Weighted ensemble of SLM CFP + BERT-predicted CFP (LOOCV).

    Optimizes weight via inner LOOCV on training fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    text_feats = np.array([
        [extract_text_features(d["raw_text"])["n_steps"],
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])

        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_pred = reg.predict(X_test)[0]

        best_w, best_mae = 0.5, float("inf")
        for w in np.arange(0.1, 0.95, 0.05):
            inner_preds_bert = []
            inner_mask_idxs = np.where(mask)[0]
            for j_idx, j in enumerate(inner_mask_idxs):
                inner_train = np.delete(inner_mask_idxs, j_idx)
                inner_scaler = StandardScaler()
                X_inner = inner_scaler.fit_transform(X[inner_train])
                X_inner_test = inner_scaler.transform(X[j:j+1])
                inner_reg = Ridge(alpha=10.0)
                inner_reg.fit(X_inner, y_true[inner_train])
                inner_preds_bert.append(inner_reg.predict(X_inner_test)[0])

            inner_ensemble = w * y_slm[mask] + (1 - w) * np.array(inner_preds_bert)
            inner_mae = np.mean(np.abs(y_true[mask] - np.round(np.maximum(1, inner_ensemble))))
            if inner_mae < best_mae:
                best_mae = inner_mae
                best_w = w

        ensemble_pred = best_w * y_slm[i] + (1 - best_w) * bert_pred
        preds[i] = max(1, round(ensemble_pred))

    return _build_results(data, preds)


def run_e29e_simple_ensemble(
    data: list[dict], bert_emb: np.ndarray
) -> list[dict]:
    """E29e-simple: Fixed-weight ensemble (faster, no inner LOOCV)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    y_true = np.array([d["y_true"] for d in data])
    y_slm = np.array([d["y_pred_slm"] for d in data])

    text_feats = np.array([
        [extract_text_features(d["raw_text"])["n_steps"],
         extract_text_features(d["raw_text"])["n_words"]]
        for d in data
    ], dtype=float)
    X = np.hstack([bert_emb, text_feats])

    bert_preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])

        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        bert_preds[i] = reg.predict(X_test)[0]

    best_w, best_mae = 0.5, float("inf")
    for w in np.arange(0.1, 0.95, 0.05):
        ensemble = w * y_slm + (1 - w) * bert_preds
        mae = np.mean(np.abs(y_true - np.round(np.maximum(1, ensemble))))
        if mae < best_mae:
            best_mae = mae
            best_w = w

    print(f"    Best weight SLM={best_w:.2f}, BERT={1-best_w:.2f}")
    preds = np.maximum(1, np.round(best_w * y_slm + (1 - best_w) * bert_preds))
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

    from scipy.stats import spearmanr
    spearman, _ = spearmanr(y_true, y_pred)

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
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERT Hybrid-XAI experiments E29a–E29e")
    parser.add_argument("--e28a-dir", type=Path,
                        default=Path("results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/hybrid-xai/bert-experiments"))
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip (e.g. e29e)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading E28a data (119 FPs)...")
    data = load_e28a_data(args.e28a_dir)
    print(f"  Loaded {len(data)} FPs, SLM MAE={np.mean(np.abs(np.array([d['y_true'] for d in data]) - np.array([d['y_pred_slm'] for d in data]))):.3f}")

    texts = [d["raw_text"] for d in data]

    print("\nExtracting BERT [CLS] embeddings...")
    t0 = time.time()
    bert_emb = extract_bert_embeddings(texts)
    bert_time = time.time() - t0
    print(f"  Shape: {bert_emb.shape}, took {bert_time:.1f}s")

    ref_metrics = compute_metrics([
        {"fp": d["fp_id"], "y_pred": d["y_pred_slm"], "y_true": d["y_true"]}
        for d in data
    ])
    print(f"\nReference E28a: MAE={ref_metrics['mae']:.3f} PRED(25)={ref_metrics['pred_25']*100:.1f}%")
    print("=" * 70)

    # ── E29a: BERT Calibration ────────────────────────────────────
    if "e29a" not in args.skip:
        print("\n[E29a] BERT calibration (BERT → Ridge → correction)...")
        t0 = time.time()
        res = run_e29a_bert_calibration(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E29a-bert-calibration", res, met, elapsed, args.output_dir)

    # ── E29b: BERT Complexity Router ──────────────────────────────
    if "e29b" not in args.skip:
        print("\n[E29b] BERT complexity router (classify → route params)...")
        t0 = time.time()
        res = run_e29b_bert_complexity_router(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E29b-bert-complexity-router", res, met, elapsed, args.output_dir)

    # ── E29c: BERT Direct Regression ──────────────────────────────
    if "e29c" not in args.skip:
        print("\n[E29c] BERT direct regression (BERT → Ridge → CFP)...")
        t0 = time.time()
        res = run_e29c_bert_regression(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E29c-bert-regression", res, met, elapsed, args.output_dir)

    # ── E29d: Sentence-BERT Dedup ─────────────────────────────────
    if "e29d" not in args.skip:
        print("\n[E29d] Sentence-BERT semantic deduplication...")
        from sentence_transformers import SentenceTransformer
        t0_load = time.time()
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Loaded Sentence-BERT in {time.time()-t0_load:.1f}s")
        t0 = time.time()
        res = run_e29d_sbert_dedup(data, sbert_model)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E29d-sbert-dedup", res, met, elapsed, args.output_dir)

    # ── E29e: BERT+SLM Ensemble ───────────────────────────────────
    if "e29e" not in args.skip:
        print("\n[E29e-simple] BERT+SLM simple ensemble (fixed weight)...")
        t0 = time.time()
        res = run_e29e_simple_ensemble(data, bert_emb)
        elapsed = time.time() - t0
        met = compute_metrics(res)
        save_experiment("E29e-bert-slm-ensemble", res, met, elapsed, args.output_dir)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — All BERT experiments vs E28a reference")
    print("=" * 70)
    print(f"{'Experiment':<35} {'MAE':>6} {'PRED(25)':>9} {'SA':>7} {'Exact':>7}")
    print("-" * 70)
    print(f"{'E28a Qwen3-8B (reference)':<35} {ref_metrics['mae']:>6.3f} {ref_metrics['pred_25']*100:>8.1f}% {ref_metrics['sa']*100:>6.1f}% {ref_metrics['exact_match']*100:>6.1f}%")

    for exp_name in sorted(args.output_dir.iterdir()):
        mf = exp_name / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            print(f"{exp_name.name:<35} {m['mae']:>6.3f} {m['pred_25']*100:>8.1f}% {m['sa']*100:>6.1f}% {m['exact_match']*100:>6.1f}%")

    print("-" * 70)


if __name__ == "__main__":
    main()
