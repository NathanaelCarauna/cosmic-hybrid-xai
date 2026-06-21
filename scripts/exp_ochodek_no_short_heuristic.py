#!/usr/bin/env python3
"""Experimento: E33d no Ochodek-437 SEM o estimador de input curto.

Pergunta: a celula Ochodek (SA +0.045, "indistinguivel de zero") depende do
heuristico _try_short_input_estimate (tabela palavra-chave->CFP, parcialmente
calibrada a escala do dataset) ou o pipeline NER+ontologia puro a sustenta?

Roda o MESMO pipeline e os MESMOS loaders/metricas do eval canonico, apenas
desligando _try_short_input_estimate (monkeypatch -> None). Compara com o
baseline (results/ner-ontology/e33d_ochodek_summary.json: MAE 4.515, SA 0.045).

Saida: results/ner-ontology/e33d_ochodek_NO_SHORT_summary.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import eval_ner_ontology_pipeline as ev  # noqa: E402


def load_pipeline_with_ner():
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    ner = BASE / "models" / "cosmic-ner" / "best"
    return mod.NerOntologyPipeline(ner_model_path=str(ner) if ner.exists() else None)


def bootstrap_sa_ci(results, n=10000, seed=42):
    errs = np.array([r["abs_error"] for r in results], float)
    yt = np.array([r["y_true"] for r in results], float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(results))
    sas = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        e, y = errs[s], yt[s]
        base = np.abs(y - y.mean()).mean()
        sas.append(1 - e.mean() / base if base > 0 else np.nan)
    sas = np.array(sas)
    sas = sas[~np.isnan(sas)]
    return float(np.percentile(sas, 2.5)), float(np.percentile(sas, 97.5))


def main():
    pipe = load_pipeline_with_ner()
    # DESLIGA o estimador de input curto: forca o pipeline NER+ontologia a rodar
    pipe._try_short_input_estimate = lambda *a, **k: None

    items = ev.load_ochodek_items()
    results = []
    n_fallback_min = 0
    for i, item in enumerate(items):
        fp_id = f"UC{i+1}"
        text = ev.wrap_as_uc(item["title"], fp_id)
        rec, _ = ev.measure_item(pipe, text, fp_id, item["project"])
        rec["y_true"] = item["cfp"]
        rec["error"] = rec["y_pred"] - item["cfp"]
        rec["abs_error"] = abs(rec["error"])
        results.append(rec)
        if rec["y_pred"] <= 1:
            n_fallback_min += 1

    m = ev.compute_metrics(results)
    lo, hi = bootstrap_sa_ci(results)

    out = {
        "experiment": "E33D_OCHODEK_NO_SHORT_HEURISTIC",
        "n_fps": m["n"],
        "mae": round(m["mae"], 4),
        "pred25": round(m["pred25"], 4),
        "sa": round(m["sa"], 4),
        "sa_ci95": [round(lo, 4), round(hi, 4)],
        "exact_matches": m["exact"],
        "mean_signed_error": round(m["mean_signed_error"], 4),
        "n_pred_le_1": n_fallback_min,
        "baseline_with_heuristic": {"mae": 4.515, "sa": 0.045},
    }
    OUT = BASE / "results" / "ner-ontology" / "e33d_ochodek_NO_SHORT_summary.json"
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
