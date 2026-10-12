#!/usr/bin/env python3
"""Ablacao do NER (tarefa #32) — duas perguntas:

1) Quanto o NER treinado contribui? Roda E33d com e SEM o modelo de NER
   (ner_model_path=None) em De Vito (home) e Zenodo (transferencia limpa).
   Se o MAE quase nao muda, a ontologia (listas de verbos + spaCy) faz o
   trabalho e "NER + ontologia" superdimensiona o NER.

2) Celula-casa LIMPA do De Vito: o NER treinou em ALBERGATE/K01726/K01719
   (94/119). Reporta o E33d (com NER, canonico) APENAS nos 13 FPs dos sistemas
   held-out do NER (RiceCooker + U-CURE) como estimativa nao-in-sample.

NAO sobrescreve resultados canonicos — salva em e33d_ner_ablation_summary.json.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import eval_ner_ontology_pipeline as ev  # noqa: E402

NER_TRAIN = {"ALBERGATE", "K01726", "K01719"}
NER_VAL = {"AutomaticLineSwitching", "K01720"}
NER_TEST = {"RiceCooker", "U-CURE"}  # held-out do NER


def build_pipeline(use_ner: bool):
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    ner = BASE / "models" / "cosmic-ner" / "best"
    path = str(ner) if (use_ner and ner.exists()) else None
    # use_ner precisa ser passado explicitamente: desde a correção de 2026-06-16
    # o default do construtor é False, e passar só o caminho do modelo não liga o
    # NER — os dois braços da ablação rodavam SEM NER (o assert abaixo pega isso).
    return mod.NerOntologyPipeline(ner_model_path=path, use_ner=use_ner)


def sa_ci(results, n=10000, seed=42):
    errs = np.array([r["abs_error"] for r in results], float)
    yt = np.array([r["y_true"] for r in results], float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(results))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        base = np.abs(yt[s] - yt[s].mean()).mean()
        out.append(1 - errs[s].mean() / base if base > 0 else np.nan)
    out = np.array(out)
    out = out[~np.isnan(out)]
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def run_devito(pipeline):
    res = []
    for sys_name, cfg in ev.DEVITO_SYSTEMS.items():
        gt = ev.load_devito_gt(cfg["gt_csv"])
        for fp_file in sorted(cfg["prompts"].glob("FP*.txt"), key=lambda p: ev.fp_sort_key(p.stem)):
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue
            rec, _ = ev.measure_item(pipeline, fp_file.read_text(encoding="utf-8"), fp_id, sys_name)
            rec["y_true"] = gt[fp_id]["y_true"]
            rec["error"] = rec["y_pred"] - rec["y_true"]
            rec["abs_error"] = abs(rec["error"])
            rec["system"] = sys_name
            res.append(rec)
    return res


def run_zenodo(pipeline):
    gt = ev.load_zenodo_gt()
    res = []
    for fp_file in sorted((ev.ZENODO_DIR / "FP_prompts").glob("FP*.txt"), key=lambda p: ev.fp_sort_key(p.stem)):
        fp_id = fp_file.stem
        if fp_id not in gt:
            continue
        # sentence_mode=True é a configuração canônica do formato de sentenças
        # (habilita R_DISPLAY_IMPLICIT_READ). Sem ele a ablação rodava sobre uma
        # configuração que não é a avaliada na dissertação, e o MAE não batia com
        # o de results/ner-ontology/e33d_zenodo_summary.json.
        rec, _ = ev.measure_item(pipeline, fp_file.read_text(encoding="utf-8"), fp_id,
                                 "ZENODO", sentence_mode=True)
        rec["y_true"] = gt[fp_id]["y_true"]
        rec["error"] = rec["y_pred"] - rec["y_true"]
        rec["abs_error"] = abs(rec["error"])
        res.append(rec)
    return res


def summ(results):
    m = ev.compute_metrics(results)
    return {"n": m["n"], "mae": round(m["mae"], 4), "sa": round(m["sa"], 4),
            "sa_ci95": sa_ci(results), "exact": m["exact"]}


def main():
    out = {}

    # 1) NER ablation
    print("Building pipelines...")
    pipe_ner = build_pipeline(use_ner=True)
    pipe_non = build_pipeline(use_ner=False)
    # IMPORTANTE: __init__ faz `ner_path = ner_model_path or default`, entao passar
    # None NAO desliga o NER (cai no default). Forca-se ner_nlp=None aqui para que
    # _run_ner retorne [] de fato. (Bug latente: a flag --no-ner do eval tem o
    # mesmo problema e nunca desligou o NER.)
    pipe_non.ner_nlp = None
    assert pipe_ner.ner_nlp is not None, "pipeline COM NER deveria ter carregado o NER"
    print(f"  pipe_ner.ner_nlp loaded={pipe_ner.ner_nlp is not None}; "
          f"pipe_non.ner_nlp loaded={pipe_non.ner_nlp is not None}")

    print("De Vito with NER...");    dv_ner = run_devito(pipe_ner)
    print("De Vito without NER..."); dv_non = run_devito(pipe_non)
    print("Zenodo with NER...");     zn_ner = run_zenodo(pipe_ner)
    print("Zenodo without NER...");  zn_non = run_zenodo(pipe_non)

    out["ner_ablation"] = {
        "devito_with_ner": summ(dv_ner),
        "devito_no_ner": summ(dv_non),
        "zenodo_with_ner": summ(zn_ner),
        "zenodo_no_ner": summ(zn_non),
    }

    # 2) De Vito home cell por particao do NER (usando a run COM NER)
    by = {"train": [], "val": [], "test_heldout": []}
    for r in dv_ner:
        s = r["system"]
        if s in NER_TRAIN: by["train"].append(r)
        elif s in NER_VAL: by["val"].append(r)
        elif s in NER_TEST: by["test_heldout"].append(r)
    out["devito_by_ner_partition"] = {k: summ(v) for k, v in by.items() if v}
    out["note"] = ("test_heldout = RiceCooker+U-CURE (sistemas que o NER nunca "
                   "viu) = estimativa nao-in-sample da celula-casa.")

    OUT = BASE / "results" / "ner-ontology" / "e33d_ner_ablation_summary.json"
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
