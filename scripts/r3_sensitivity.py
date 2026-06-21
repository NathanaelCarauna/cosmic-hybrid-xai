#!/usr/bin/env python3
"""T4 (Sprint 3) — Sensibilidade do R3-HKO a re-tuning per-dataset.

Grid: min_samples_leaf x top_k (C4.5, n_classes conforme melhor variante por
dataset), stratified CV identico ao runner canonico. Objetivo: mostrar que
mesmo otimizando hiperparametros POR dataset, o teto do R3 (SA ~0,1-0,3) nao
muda de ordem de grandeza — reforca que o "SA>0 nas 4 celulas" do R3 e um
teto estrutural do classificador, nao transferencia.

Output: results/r3-sensitivity/per-dataset-tuning.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from shared.seeds import set_global_seeds
from r3_hko_runner import (
    _load_dataset, annotate_classes,
)
from shared.r3_features import (
    build_keyword_lists, extract_syntactic_matrix,
    keyword_features_from_cache, assemble_features, extract_pos_tokens_batch,
)

GRID_LEAF = [2, 4, 6, 8, 12]
GRID_TOPK = [5, 10, 20, 40]
# melhor n_classes por dataset (variantes canonicas da matriz)
N_CLASSES = {"devito": 2, "zenodo": 4, "ochodek": 3, "user-stories": 3}


def cv_eval(texts, cfp_values, labels, class_data, *, leaf, top_k,
            syn_cache, pos_cache):
    min_class = min(Counter(labels).values())
    folds = min(10, min_class)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cfp_t, cfp_p = [], []
    for tr, va in skf.split(texts, labels):
        kw = build_keyword_lists([texts[i] for i in tr],
                                 labels[tr].tolist(), top_k=top_k)
        Xtr = assemble_features(syn_cache[tr],
                                keyword_features_from_cache(
                                    [pos_cache[i] for i in tr], kw))
        Xva = assemble_features(syn_cache[va],
                                keyword_features_from_cache(
                                    [pos_cache[i] for i in va], kw))
        clf = DecisionTreeClassifier(criterion="entropy",
                                     min_samples_leaf=leaf, random_state=42)
        clf.fit(Xtr, labels[tr])
        preds = clf.predict(Xva)
        cfp_p += [class_data.get(p, {}).get("median", 0) for p in preds]
        cfp_t += list(cfp_values[va])
    cfp_t, cfp_p = np.array(cfp_t, float), np.array(cfp_p, float)
    mae = float(np.mean(np.abs(cfp_t - cfp_p)))
    mad = float(np.mean(np.abs(cfp_t - cfp_t.mean())))
    return mae, 1 - mae / mad


def main() -> None:
    set_global_seeds()
    out = BASE / "results" / "r3-sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ["devito", "zenodo", "ochodek", "user-stories"]:
        df, label = _load_dataset(ds, BASE / "results")
        texts = df["text"].tolist()
        cfp = df["cfp"].to_numpy()
        nc = N_CLASSES[ds]
        labels, class_data = annotate_classes(cfp, n_classes=nc)
        labels = np.asarray(labels)
        print(f"\n=== {ds} (n={len(texts)}, {nc}-cls) — cache spaCy...")
        t0 = time.perf_counter()
        syn_cache = extract_syntactic_matrix(texts)
        pos_cache = extract_pos_tokens_batch(texts)
        print(f"  cache em {time.perf_counter()-t0:.1f}s")
        best = None
        for leaf in GRID_LEAF:
            for tk in GRID_TOPK:
                mae, sa = cv_eval(texts, cfp, labels, class_data,
                                  leaf=leaf, top_k=tk,
                                  syn_cache=syn_cache, pos_cache=pos_cache)
                rows.append({"dataset": ds, "n_classes": nc,
                             "min_samples_leaf": leaf, "top_k": tk,
                             "cfp_mae": round(mae, 4), "cfp_sa": round(sa, 4)})
                if best is None or mae < best[0]:
                    best = (mae, sa, leaf, tk)
        canon = next((r for r in rows if r["dataset"] == ds
                      and r["min_samples_leaf"] == 6 and r["top_k"] == 20))
        print(f"  canonico (leaf=6,k=20): MAE={canon['cfp_mae']} SA={canon['cfp_sa']}")
        print(f"  melhor do grid:         MAE={best[0]:.4f} SA={best[1]:.4f} "
              f"(leaf={best[2]}, k={best[3]})")
    with (out / "per-dataset-tuning.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} linhas -> {out/'per-dataset-tuning.csv'}")


if __name__ == "__main__":
    main()
