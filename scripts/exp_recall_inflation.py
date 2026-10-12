#!/usr/bin/env python3
"""Família de "inflação por recall": todas as alternativas da VIA SIMBÓLICA que
aumentam o recall de movimentos sobre uma extração não calibrada.

O achado metodológico da Seção de alternativas descartadas ("aumentar o recall
sobre uma extração não calibrada superestima o tamanho") é sustentado por vários
experimentos avulsos, feitos em momentos diferentes e nunca reunidos em um
artefato regerável. Este script reexecuta a família inteira sobre o MESMO
conjunto (De Vito, 119 processos) e no MESMO commit, para que os números citados
na Discussão tenham um gerador.

Todas as variantes são opções já existentes no construtor do pipeline, desligadas
por padrão. Nenhuma toca o resultado canônico.

Saída: results/ablations/recall_inflation.{csv,md}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "models" / "hybrid-xai"))
import eval_ner_ontology_pipeline as ev  # noqa: E402
from ner_srl.ner_ontology_pipeline import NerOntologyPipeline  # noqa: E402

# (rótulo, kwargs do construtor, kwargs de measure_item)
VARIANTS = [
    ("canônico (E33d implantado)", {}, {}),
    ("+ leitura implícita na exibição (global)", {}, {"sentence_mode": True}),
    ("+ dedup por verbo+objeto", {"dedup_by_object": True}, {}),
    ("+ injeção de leituras implícitas", {"inject_implicit_reads": True}, {}),
    ("+ dedup melhorada (semântica)", {"improved_dedup": True}, {}),
]


def run_devito(pipeline, measure_kw):
    res = []
    for sys_name, cfg in ev.DEVITO_SYSTEMS.items():
        gt = ev.load_devito_gt(cfg["gt_csv"])
        for fp_file in sorted(cfg["prompts"].glob("FP*.txt"),
                              key=lambda p: ev.fp_sort_key(p.stem)):
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue
            rec, _ = ev.measure_item(pipeline, fp_file.read_text(encoding="utf-8"),
                                     fp_id, sys_name, **measure_kw)
            rec["y_true"] = gt[fp_id]["y_true"]
            rec["error"] = rec["y_pred"] - rec["y_true"]
            rec["abs_error"] = abs(rec["error"])
            res.append(rec)
    return res


def summ(res):
    m = ev.compute_metrics(res)
    bias = float(np.mean([r["error"] for r in res]))
    return {"n": m["n"], "mae": round(m["mae"], 4), "sa": round(m["sa"], 4),
            "exact": m["exact"], "bias": round(bias, 4)}


def main() -> None:
    rows = []
    for label, ctor_kw, meas_kw in VARIANTS:
        print(f"== {label}")
        pipe = NerOntologyPipeline(use_ner=False, **ctor_kw)
        s = summ(run_devito(pipe, meas_kw))
        s["variante"] = label
        rows.append(s)
        print(f"   MAE={s['mae']} SA={s['sa']:+.4f} exatos={s['exact']} viés={s['bias']:+.4f}")

    base = rows[0]
    for r in rows:
        r["delta_mae"] = round(r["mae"] - base["mae"], 4)

    out = BASE / "results" / "ablations"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["variante", "n", "mae", "delta_mae", "sa", "exact", "bias"]
    with (out / "recall_inflation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])

    md = ["# Inflação por recall — via simbólica, De Vito (119 processos)", "",
          "Todas as variantes são opções do construtor do pipeline, desligadas por",
          "padrão. Reexecutadas no mesmo commit e sobre o mesmo conjunto.", "",
          "| Variante | MAE | Δ MAE | SA | exatos | viés/proc. |",
          "|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['variante']} | {r['mae']:.4f} | {r['delta_mae']:+.4f} | "
                  f"{r['sa']:+.4f} | {r['exact']} | {r['bias']:+.4f} |")
    md += ["", "Leitura: toda alternativa que aumenta o recall de movimentos sobre a",
           "extração não calibrada piora o MAE e desloca o viés para cima. O",
           "gargalo é a calibração da extração, não a cobertura de regras."]
    (out / "recall_inflation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}/recall_inflation.{{csv,md}}")


if __name__ == "__main__":
    main()
