#!/usr/bin/env python3
"""Validação held-out do Read implícito na exibição (Zenodo / formato de
sentença única). Princípio COSMIC: exibir um grupo de dados ARMAZENADO exige um
Read antes do Exit; numa sentença isolada (processo funcional próprio, sem
Entry/Read anterior) esse Read precisa ser contado. Gateado por sentence_mode
(não dispara em UC multi-passo, onde o dado pode já ter sido lido antes — razão
de inject_implicit_reads ter PIORADO o De Vito).

Roda o pipeline E33d com a regra OFF e ON sobre os 2040, faz split estratificado
70/30 por faixa de CFP (seed 42) e reporta MAE/SA em train, test (held-out) e
full — para mostrar que o ganho NÃO é overfitting de regex (regra é principiada,
sem ajuste numérico).

Uso: python3.10 scripts/exp_zenodo_display_read.py
"""
from __future__ import annotations

import csv
import importlib.util
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ZENODO_DIR = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sentence"


def load_pipeline():
    path = BASE / "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py"
    spec = importlib.util.spec_from_file_location("ner_ontology_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod.NerOntologyPipeline(use_ner=False)


def main():
    gt = {}
    with open(ZENODO_DIR / "ground_truth.csv") as f:
        for r in csv.DictReader(f):
            gt[r["fp_file"].replace(".txt", "")] = int(r["cfp_gt"])

    pipe = load_pipeline()
    prompt_dir = ZENODO_DIR / "FP_prompts"

    rows = []
    t0 = time.time()
    for fp_file in sorted(prompt_dir.glob("FP*.txt")):
        fid = fp_file.stem
        if fid not in gt:
            continue
        text = fp_file.read_text(encoding="utf-8")
        off = pipe.measure(text, fp_id=fid, sentence_mode=False).cfp
        on = pipe.measure(text, fp_id=fid, sentence_mode=True).cfp
        rows.append({"fp": fid, "y_true": gt[fid], "off": off, "on": on})
    print(f"medido {len(rows)} sentenças em {time.time()-t0:.1f}s")

    # split estratificado 70/30 por faixa de CFP (cap em 3), seed 42
    random.seed(42)
    buckets: dict[int, list] = {}
    for r in rows:
        buckets.setdefault(min(r["y_true"], 3), []).append(r)
    train, test = [], []
    for b in sorted(buckets):
        rs = buckets[b][:]
        random.shuffle(rs)
        k = int(len(rs) * 0.7)
        train += rs[:k]
        test += rs[k:]

    def report(name, sub):
        n = len(sub)
        mean_gt = sum(r["y_true"] for r in sub) / n
        base = sum(abs(r["y_true"] - mean_gt) for r in sub) / n
        for col in ("off", "on"):
            mae = sum(abs(r[col] - r["y_true"]) for r in sub) / n
            sa = 1 - mae / base if base else float("nan")
            pred25 = sum(1 for r in sub
                         if abs(r[col] - r["y_true"]) / max(r["y_true"], 1) <= 0.25) / n
            print(f"  {name:14s} {col:3s}  n={n:4d}  MAE={mae:.4f}  "
                  f"SA={sa:+.3f}  PRED25={pred25*100:.1f}%")

    print("\n=== Read implícito na exibição: OFF (baseline) vs ON (regra) ===")
    for name, sub in [("TRAIN", train), ("TEST(held-out)", test), ("FULL", rows)]:
        report(name, sub)

    out = BASE / "results/ner-ontology/zenodo_display_read.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "y_true", "off", "on"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
