#!/usr/bin/env python3
"""Mede o efeito das correções de implementação (auditoria 2026-06-16) no E33d,
isolando cada uma. NAO sobrescreve resultados canonicos.

Correções avaliadas (flags do pipeline/ontologia):
  - inject_implicit_reads: liga _inject_implicit_reads (era codigo morto).
  - improved_dedup: normalizacao de grupo de dados menos rasa.
  - use_short_input_estimate=False: desliga o heuristico do Ochodek (pipeline real).

Configs por dataset:
  canon  = (impl=F, dedup=F, short=T)  -> deve reproduzir o canonico (sanity).
  +impl  = (impl=T, dedup=F, short=T)
  +dedup = (impl=F, dedup=T, short=T)
  both   = (impl=T, dedup=T, short=T)
  both_noheur = (impl=T, dedup=T, short=F)  -> Ochodek/US honestos.

Saida: results/ner-ontology/e33d_corrections_summary.json
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

CANON = {"devito": {"mae": 0.933, "sa": 0.555}, "zenodo": {"mae": 0.379, "sa": 0.077},
         "ochodek": {"mae": 4.515, "sa": 0.045}, "user_stories_e34a": {"mae": 0.459, "sa": 0.364}}


def build():
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    ner = BASE / "models" / "cosmic-ner" / "best"
    return mod.NerOntologyPipeline(ner_model_path=str(ner) if ner.exists() else None)


def set_flags(p, impl, dedup, short):
    p.ontology.inject_implicit_reads = impl
    p.ontology.improved_dedup = dedup
    p.use_short_input_estimate = short


def sa_ci(results, n=10000, seed=42):
    e = np.array([r["abs_error"] for r in results], float)
    y = np.array([r["y_true"] for r in results], float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(results))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        base = np.abs(y[s] - y[s].mean()).mean()
        out.append(1 - e[s].mean() / base if base > 0 else np.nan)
    out = np.array(out); out = out[~np.isnan(out)]
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def summ(results):
    m = ev.compute_metrics(results)
    return {"n": m["n"], "mae": round(m["mae"], 3), "sa": round(m["sa"], 3),
            "sa_ci95": sa_ci(results), "exact": m["exact"]}


def run_devito(p):
    res = []
    for sysn, cfg in ev.DEVITO_SYSTEMS.items():
        gt = ev.load_devito_gt(cfg["gt_csv"])
        for fp in sorted(cfg["prompts"].glob("FP*.txt"), key=lambda q: ev.fp_sort_key(q.stem)):
            if fp.stem not in gt:
                continue
            rec, _ = ev.measure_item(p, fp.read_text(encoding="utf-8"), fp.stem, sysn)
            rec["y_true"] = gt[fp.stem]["y_true"]; rec["abs_error"] = abs(rec["y_pred"] - rec["y_true"])
            rec["error"] = rec["y_pred"] - rec["y_true"]; res.append(rec)
    return res


def run_zenodo(p):
    gt = ev.load_zenodo_gt(); res = []
    for fp in sorted((ev.ZENODO_DIR / "FP_prompts").glob("FP*.txt"), key=lambda q: ev.fp_sort_key(q.stem)):
        if fp.stem not in gt:
            continue
        rec, _ = ev.measure_item(p, fp.read_text(encoding="utf-8"), fp.stem, "ZENODO")
        rec["y_true"] = gt[fp.stem]["y_true"]; rec["abs_error"] = abs(rec["y_pred"] - rec["y_true"])
        rec["error"] = rec["y_pred"] - rec["y_true"]; res.append(rec)
    return res


def run_ochodek(p):
    res = []
    for i, item in enumerate(ev.load_ochodek_items()):
        rec, _ = ev.measure_item(p, ev.wrap_as_uc(item["title"], f"UC{i+1}"), f"UC{i+1}", item["project"])
        rec["y_true"] = item["cfp"]; rec["abs_error"] = abs(rec["y_pred"] - rec["y_true"])
        rec["error"] = rec["y_pred"] - rec["y_true"]; res.append(rec)
    return res


def run_us_e34a(p):
    from adapters.connextra_adapter import connextra_to_ucs
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = {}
    import csv as _csv
    with (fp_dir / "ground_truth.csv").open(newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            gt[r["fp_id"]] = int(r["cfp_gt"])
    res = []
    for fp in sorted(fp_dir.glob("FP*.txt"), key=lambda q: ev.fp_sort_key(q.stem)):
        if fp.stem not in gt:
            continue
        raw = fp.read_text(encoding="utf-8").strip()
        ucs = connextra_to_ucs(raw, fp.stem) or [ev.wrap_as_uc(raw, fp.stem)]
        rec = None
        for uc in ucs:
            sub, _ = ev.measure_item(p, uc, fp.stem, "US")
            if rec is None:
                rec = sub
            else:
                for k in ("y_pred", "E_pred", "R_pred", "W_pred", "X_pred"):
                    rec[k] += sub[k]
        rec["y_true"] = gt[fp.stem]; rec["abs_error"] = abs(rec["y_pred"] - rec["y_true"])
        rec["error"] = rec["y_pred"] - rec["y_true"]; res.append(rec)
    return res


def main():
    p = build()
    runners = {"devito": run_devito, "zenodo": run_zenodo,
               "ochodek": run_ochodek, "user_stories_e34a": run_us_e34a}
    # configs por dataset
    cfgs_full = [("canon", False, False, True), ("+impl", True, False, True),
                 ("+dedup", False, True, True), ("both", True, True, True)]
    cfgs_short = [("canon(heur)", False, False, True), ("both_noheur", True, True, False)]

    out = {}
    for ds, runner in runners.items():
        cfgs = cfgs_short if ds in ("ochodek", "user_stories_e34a") else cfgs_full
        out[ds] = {"canonical_ref": CANON[ds], "configs": {}}
        for name, impl, dedup, short in cfgs:
            set_flags(p, impl, dedup, short)
            out[ds]["configs"][name] = summ(runner(p))
            print(f"{ds:18s} {name:14s} -> {out[ds]['configs'][name]}")

    OUT = BASE / "results" / "ner-ontology" / "e33d_corrections_summary.json"
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
