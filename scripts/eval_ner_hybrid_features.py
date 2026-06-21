#!/usr/bin/env python3
"""E33e — Hybrid NER+Ontology features as BERT replacement/complement.

Extract interpretable NER+Ontology features and use them in the SLM+ensemble
pipeline, replacing or complementing the 768-d BERT embeddings.

Features extracted per FP:
  - n_actors: count of unique functional users detected
  - n_data_objs: count of unique data objects
  - n_persist_stores: count of persistence markers
  - n_ui_elements: count of UI elements
  - n_conditions: count of conditions
  - ner_cfp: CFP predicted by ontology pipeline (strong interpretable feature)
  - n_entries_ont: Entry count from ontology
  - n_reads_ont: Read count from ontology
  - n_writes_ont: Write count from ontology
  - n_exits_ont: Exit count from ontology
  - verb_ambiguity: ratio of ambiguous verbs to total verbs

Usage:
  python3 scripts/eval_ner_hybrid_features.py \
    --e28a-dir results/hybrid-xai/119fp-E28a-qwen3-8b-20260330T215921
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        lines = text.strip().split("\n")
        words = text.split()
        step_like = [l for l in lines if re.match(r"^\s*\d+\s*\t", l)]

        data.append({
            "fp_id": fp_id,
            "y_true": int(row["y_true"]),
            "y_pred_slm": int(row["y_pred"]),
            "y_raw_slm": raw_count if ir else int(row["y_pred"]),
            "correction_count": correction_count,
            "raw_text": text,
            "n_steps": len(step_like),
            "n_words": len(words),
            "ir": ir,
        })
    return data


def extract_ner_ontology_features(data: list[dict]) -> np.ndarray:
    """Extract interpretable NER+Ontology features for each FP."""
    pipe_mod = load_mod(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
    )
    pipeline = pipe_mod.NerOntologyPipeline()

    features = []
    for d in data:
        text = d["raw_text"]
        result = pipeline.measure(text, fp_id=d["fp_id"])

        ner_entities = []
        if pipeline.ner_nlp:
            doc = pipeline.ner_nlp(text)
            ner_entities = [(ent.label_, ent.text) for ent in doc.ents]

        n_actors = len(set(t for l, t in ner_entities if l == "FUNC_USER"))
        n_data_objs = len(set(t for l, t in ner_entities if l == "DATA_OBJ"))
        n_persist = len(set(t for l, t in ner_entities if l == "PERSIST_STORE"))
        n_ui = len(set(t for l, t in ner_entities if l == "UI_ELEMENT"))

        active_movements = [m for m in result.movements if m.movement_type is not None]
        n_rules = len(set(m.rule_id for m in result.movements))

        feats = [
            result.cfp,           # ontology CFP prediction (strongest feature)
            result.n_entries,
            result.n_reads,
            result.n_writes,
            result.n_exits,
            n_actors,
            n_data_objs,
            n_persist,
            n_ui,
            n_rules,
            d["n_steps"],
            d["n_words"],
        ]
        features.append(feats)

    return np.array(features, dtype=float)


def extract_bert_embeddings(texts: list[str], max_length: int = 512) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "google-bert/bert-base-uncased"
    _BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
    print("  Loading BERT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=_BERT_REVISION)
    model = AutoModel.from_pretrained(model_name, revision=_BERT_REVISION)
    model.eval()

    embeddings = []
    batch_sz = 8
    for i in range(0, len(texts), batch_sz):
        batch = texts[i : i + batch_sz]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls)
    return np.vstack(embeddings)


def grid_search_correction(y_true, y_raw, n_steps, mask):
    best_mae = float("inf")
    best_params = {"alpha": 0.35, "beta": 8, "gamma": 0.6,
                    "alpha_lg": 0.50, "beta_lg": 6, "gamma_lg": 0.8, "thr": 13}
    for alpha in np.arange(0.2, 0.6, 0.05):
        for beta in range(5, 12):
            for gamma in np.arange(0.4, 0.9, 0.1):
                for alpha_lg in np.arange(0.3, 0.7, 0.05):
                    corrected = np.array([
                        _apply_correction(y_raw[j], n_steps[j],
                                          alpha, beta, gamma, alpha_lg, 6, 0.8, 13)
                        for j in np.where(mask)[0]
                    ])
                    mae = np.mean(np.abs(y_true[mask] - corrected))
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {"alpha": alpha, "beta": beta, "gamma": gamma,
                                       "alpha_lg": alpha_lg, "beta_lg": 6, "gamma_lg": 0.8, "thr": 13}
    return best_params


def _apply_correction(y, n_steps, alpha, beta, gamma, alpha_lg, beta_lg, gamma_lg, thr):
    if n_steps >= thr:
        a, b, g = alpha_lg, beta_lg, gamma_lg
    else:
        a, b, g = alpha, beta, gamma
    correction = a * max(0, n_steps - b)
    correction = min(correction, y * g)
    return max(1, round(y + correction))


def run_experiment(name, X, data, y_true, y_raw, n_steps):
    """Run ensemble experiment with given features."""
    n = len(data)
    preds = np.zeros(n)

    for i in range(n):
        mask = np.arange(n) != i

        params = grid_search_correction(y_true, y_raw, n_steps, mask)
        corrected_i = _apply_correction(
            y_raw[i], n_steps[i],
            params["alpha"], params["beta"], params["gamma"],
            params["alpha_lg"], params["beta_lg"], params["gamma_lg"], params["thr"])
        y_corrected_train = np.array([
            _apply_correction(
                y_raw[j], n_steps[j],
                params["alpha"], params["beta"], params["gamma"],
                params["alpha_lg"], params["beta_lg"], params["gamma_lg"], params["thr"])
            for j in np.where(mask)[0]
        ])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[i:i+1])
        reg = Ridge(alpha=10.0)
        reg.fit(X_train, y_true[mask])
        feat_pred_i = reg.predict(X_test)[0]

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

        preds[i] = max(1, round(best_w * corrected_i + (1 - best_w) * feat_pred_i))

    mae = np.mean(np.abs(y_true - preds))
    pred25 = np.mean(np.abs(preds - y_true) / np.maximum(y_true, 1) <= 0.25)
    mean_mag = np.mean(y_true)
    sa = 1 - mae / mean_mag
    exact = np.sum(preds == y_true)

    print(f"\n  {name}:")
    print(f"    MAE={mae:.3f}, PRED(25)={pred25*100:.1f}%, SA={sa*100:.1f}%")
    print(f"    Exact matches: {exact}/{n} ({exact/n*100:.1f}%)")
    return {"name": name, "mae": round(float(mae), 3),
            "pred25": round(float(pred25), 3), "sa": round(float(sa), 3),
            "exact": int(exact), "preds": preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e28a-dir", type=str, required=True)
    args = parser.parse_args()

    e28a_dir = Path(args.e28a_dir)
    print("Loading E28a data...")
    data = load_e28a_data(e28a_dir)
    n = len(data)

    y_true = np.array([d["y_true"] for d in data])
    y_raw = np.array([d["y_raw_slm"] for d in data])
    n_steps_arr = np.array([d["n_steps"] for d in data])
    texts = [d["raw_text"] for d in data]

    print(f"  Loaded {n} FPs")

    print("\nExtracting NER+Ontology features...")
    t0 = time.time()
    ner_features = extract_ner_ontology_features(data)
    t_ner = time.time() - t0
    print(f"  NER features shape: {ner_features.shape}, time: {t_ner:.1f}s")

    print("\nExtracting BERT embeddings...")
    t0 = time.time()
    bert_emb = extract_bert_embeddings(texts)
    t_bert = time.time() - t0
    print(f"  BERT embeddings shape: {bert_emb.shape}, time: {t_bert:.1f}s")

    text_feats = np.array([[d["n_steps"], d["n_words"]] for d in data], dtype=float)

    # E30f baseline: BERT + text features
    X_bert = np.hstack([bert_emb, text_feats])

    # E33e-a: NER+Ontology features only (12 features, fully interpretable)
    X_ner = ner_features

    # E33e-b: NER+Ontology + text features
    X_ner_text = np.hstack([ner_features, text_feats])

    # E33e-c: BERT + NER+Ontology (combined)
    X_combined = np.hstack([bert_emb, ner_features])

    print("\n" + "=" * 60)
    print("  E33e — Hybrid NER+Ontology Feature Experiments")
    print("=" * 60)

    results = []

    print("\n  Running E30f baseline (BERT + text features)...")
    r = run_experiment("E30f_baseline_BERT", X_bert, data, y_true, y_raw, n_steps_arr)
    results.append(r)

    print("\n  Running E33e-a (NER+Ontology features only)...")
    r = run_experiment("E33e_a_NER_only", X_ner, data, y_true, y_raw, n_steps_arr)
    results.append(r)

    print("\n  Running E33e-b (NER+Ontology + text features)...")
    r = run_experiment("E33e_b_NER_text", X_ner_text, data, y_true, y_raw, n_steps_arr)
    results.append(r)

    print("\n  Running E33e-c (BERT + NER+Ontology combined)...")
    r = run_experiment("E33e_c_BERT_NER", X_combined, data, y_true, y_raw, n_steps_arr)
    results.append(r)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:30s}: MAE={r['mae']:.3f}, PRED(25)={r['pred25']*100:.1f}%, SA={r['sa']*100:.1f}%")

    output_dir = BASE / "results" / "ner-ontology"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "e33e_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "preds"} for r in results], f, indent=2)
    print(f"\n  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
