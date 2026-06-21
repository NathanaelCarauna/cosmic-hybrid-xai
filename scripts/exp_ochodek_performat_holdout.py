#!/usr/bin/env python3
"""Tarefa A (reenquadramento multi-formato): número HONESTO per-formato para
nomes de UC (Ochodek), com estimador calibrado SÓ no dev e validado held-out —
mesma disciplina dev/teste do E34a.

Problema: o handler atual (_try_short_input_estimate, tabela palavra-chave->CFP)
foi tunado à mão olhando o GT do Ochodek e medido no mesmo Ochodek => in-sample
(SA +0.045 otimista). Aqui calibramos estimadores reproduzíveis e medimos por
LEAVE-ONE-PROJECT-OUT CV (cada item predito fora-de-fold; sem vazamento entre
UCs do mesmo projeto).

Estimadores (todos calibrados no train de cada fold):
  baseline_mean : prediz a média de CFP do train (SA=0 por construção).
  verb_mean     : média de CFP do train para o verbo-cabeça (1o token);
                  verbo não visto -> média global do train.
  linreg        : regressão linear sobre [n_tokens, has_crud, n_objetos]
                  (features interpretáveis), fit no train.

Compara com: heurístico in-sample (4.515/SA+0.045), pipeline puro (8.263/-0.749),
e competidores no Ochodek (R2 home 3.591; R1 6.7; E30f 7.4).

Saída: results/ner-ontology/ochodek_performat_holdout.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OCHODEK = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"

_CRUD_MARKERS = {"crud"}
_OP_MARKERS = {"crud", "edit", "delete", "add", "create", "update", "remove",
               "insert", "view", "list", "manage", "modify", "search"}


def load():
    items = []
    for r in csv.DictReader(open(OCHODEK)):
        toks = r["TitleTokens"].strip().lower().split()
        items.append({
            "title": r["TitleTokens"].strip(),
            "cfp": int(r["Cfp"]),
            "project": r["ProjectID"],
            "toks": toks,
            "verb": toks[0] if toks else "",
            "n_tokens": len(toks),
            "has_crud": 1 if (_CRUD_MARKERS & set(toks)) else 0,
            # objetos ~ tokens que não são marcadores de operação
            "n_obj": max(1, sum(1 for t in toks[1:] if t not in _OP_MARKERS)),
        })
    return items


def feats(it):
    return [it["n_tokens"], it["has_crud"], it["n_obj"]]


def fit_predict(train, test):
    """Retorna dict estimador -> lista de predições para 'test'."""
    ytr = np.array([t["cfp"] for t in train], float)
    gmean = ytr.mean()

    # verb_mean (target encoding no train)
    vsum, vcnt = defaultdict(float), defaultdict(int)
    for t in train:
        vsum[t["verb"]] += t["cfp"]; vcnt[t["verb"]] += 1
    vmean = {v: vsum[v] / vcnt[v] for v in vsum}

    # linreg (normal equations, sem dependência externa)
    X = np.array([[1.0] + feats(t) for t in train], float)
    coef, *_ = np.linalg.lstsq(X, ytr, rcond=None)

    preds = {"baseline_mean": [], "verb_mean": [], "linreg": []}
    for t in test:
        preds["baseline_mean"].append(gmean)
        preds["verb_mean"].append(vmean.get(t["verb"], gmean))
        preds["linreg"].append(float(np.array([1.0] + feats(t)) @ coef))
    return preds


def metrics(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.round(np.clip(np.array(y_pred, float), 1, None))
    err = np.abs(y_pred - y_true)
    mae = err.mean()
    base = np.abs(y_true - y_true.mean()).mean()
    sa = 1 - mae / base if base > 0 else float("nan")
    # bootstrap IC95 da SA
    rng = np.random.default_rng(42)
    idx = np.arange(len(y_true)); sas = []
    for _ in range(10000):
        s = rng.choice(idx, len(idx), replace=True)
        b = np.abs(y_true[s] - y_true[s].mean()).mean()
        sas.append(1 - np.abs(y_pred[s] - y_true[s]).mean() / b if b > 0 else np.nan)
    sas = np.array(sas); sas = sas[~np.isnan(sas)]
    return {"mae": round(float(mae), 3), "sa": round(float(sa), 3),
            "sa_ci95": [round(float(np.percentile(sas, 2.5)), 3),
                        round(float(np.percentile(sas, 97.5)), 3)],
            "exact": int((err == 0).sum())}


def main():
    items = load()
    projects = sorted({it["project"] for it in items})

    # Leave-one-project-out CV: predição fora-de-fold para cada item
    oof = {"baseline_mean": {}, "verb_mean": {}, "linreg": {}}
    for held in projects:
        train = [it for it in items if it["project"] != held]
        test = [it for it in items if it["project"] == held]
        preds = fit_predict(train, test)
        for est in oof:
            for it, p in zip(test, preds[est]):
                oof[est][id(it)] = p

    y_true = [it["cfp"] for it in items]
    out = {
        "n": len(items), "n_projetos": len(projects),
        "cv": "leave-one-project-out",
        "estimadores_holdout": {
            est: metrics(y_true, [oof[est][id(it)] for it in items]) for est in oof
        },
        "referencias": {
            "heuristico_in_sample": {"mae": 4.515, "sa": 0.045, "nota": "tunado+testado no mesmo Ochodek"},
            "pipeline_puro_sem_heuristico": {"mae": 8.263, "sa": -0.749},
            "R2_DEEP-COSMIC_home": {"mae": 3.591, "nota": "treinado no Ochodek (home)"},
            "R1_COSMET": {"mae": 6.7, "sa": -0.336},
            "E30f_neural": {"mae": 7.4, "sa": -0.475},
        },
    }
    OUT = BASE / "results" / "ner-ontology" / "ochodek_performat_holdout.json"
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
