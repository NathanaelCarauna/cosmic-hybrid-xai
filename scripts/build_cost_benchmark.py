#!/usr/bin/env python3
"""T5 (Sprint 3) — Benchmark de custo computacional cross-method.

Consolida custo (US$) e tempo (s/item) por método × dataset a partir dos
artefatos persistidos (run_metrics.json, summaries, runs_registry.csv).

Output: results/cost-benchmark/comparison.csv + comparison.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "cost-benchmark"
OUT.mkdir(parents=True, exist_ok=True)

rows: list[dict] = []


def add(method, dataset, n, s_per_item, usd_total, hardware, source):
    usd_item = usd_total / n if n else 0.0
    rows.append({
        "method": method, "dataset": dataset, "n_items": n,
        "s_per_item": round(s_per_item, 3),
        "usd_total": round(usd_total, 4),
        "usd_per_item": round(usd_item, 5),
        "usd_per_100_items": round(usd_item * 100, 3),
        "hardware": hardware, "source": source,
    })


# ---- R1 COSMET (gpt-4-turbo, API) ----
for run, ds, n in [
    ("zenodo-200-r1-gpt-4-turbo-20260526T042716", "zenodo-200", 200),
    ("ochodek-200-r1-gpt-4-turbo-20260526T044427", "ochodek-200", 200),
    ("user-stories-98-r1-gpt-4-turbo-20260611T020933", "user-stories-98", 98),
]:
    p = BASE / "results" / "cosmet" / run / "run_metrics.json"
    if p.exists():
        d = json.loads(p.read_text())
        add("R1-COSMET (gpt-4-turbo)", ds, n,
            d["performance"]["fp_mean_s"], d["cost"]["total_usd"],
            "OpenAI API", run)
# De Vito: soma das 7 runs por projeto (custo documentado $9,10; wall do registry)
reg = list(csv.DictReader((BASE / "results" / "runs_registry.csv").open()))
devito_runs = [r for r in reg if r.get("method") == "cosmet"
               and r.get("llm_model") == "gpt-4-turbo"
               and r.get("dataset") in {"ALBERGATE", "K01719", "K01720", "K01726",
                                        "U-CURE", "AutomaticLineSwitching", "RiceCooker"}
               and r.get("wall_clock_s")]
if devito_runs:
    tot_s = sum(float(r["wall_clock_s"]) for r in devito_runs)
    tot_c = sum(float(r["total_cost_usd"]) for r in devito_runs if r.get("total_cost_usd"))
    n = sum(int(r["n_fps"]) for r in devito_runs)
    add("R1-COSMET (gpt-4-turbo)", "devito-119", n, tot_s / n,
        tot_c if tot_c else 9.10, "OpenAI API", "registry (7 projetos)")

# ---- E28a (Qwen3-8B local) = etapa SLM do E30f ----
for run, ds in [
    ("119fp-E28a-qwen3-8b-clean-train-20260525T130245", "devito-89-train"),
    ("zenodo-E28a-clean-zenodo200-20260525T183254", "zenodo-200"),
    ("ochodek-E28a-clean-ochodek200-20260525T185618", "ochodek-200"),
    ("zenodo-E28a-clean-us98-20260604T163714", "user-stories-98"),
]:
    p = BASE / "results" / "hybrid-xai" / run / "metrics.json"
    if p.exists():
        d = json.loads(p.read_text())
        add("E30f/E28a (Qwen3-8B local)", ds, d["n_fps"],
            d["total_time_s"] / d["n_fps"], 0.0,
            "GPU AMD RX 6700 XT (Vulkan)", run)

# ---- E33d / E34a (NER+Ontologia, simbólico) ----
for fn, ds, method in [
    ("e33d_summary.json", "devito-119", "E33d (NER+Ontologia)"),
    ("e33d_zenodo_summary.json", "zenodo-2040", "E33d (NER+Ontologia)"),
    ("e33d_ochodek_summary.json", "ochodek-437", "E33d (NER+Ontologia)"),
    ("e34a_user_stories_summary.json", "user-stories-98", "E34a (E33d+adaptador)"),
]:
    p = BASE / "results" / "ner-ontology" / fn
    if p.exists():
        d = json.loads(p.read_text())
        add(method, ds, d["n_fps"], d["avg_time_per_fp_s"], 0.0,
            "CPU (spaCy, sem GPU)", fn)

# ---- R2 (treino 10x10 CV no Ochodek) ----
for run_id, model in [("ochodek-uc-437-cnn-20260323T092843", "CNN"),
                      ("ochodek-uc-437-rf-20260323T090813", "RF")]:
    r = next((x for x in reg if x["run_id"] == run_id), None)
    if r:
        add(f"R2-DEEP-COSMIC ({model})", "ochodek-437 (treino CV)",
            int(r["n_fps"]), float(r["wall_clock_s"]) / int(r["n_fps"]), 0.0,
            "CPU/GPU local", run_id)

# ---- R3 HKO (CV completo) ----
for run_id, ds in [("hko-devito-119-c45-4cls-20260323T204644", "devito-119"),
                   ("hko-user-stories-98-c45-3cls-20260611T015226", "user-stories-98")]:
    r = next((x for x in reg if x["run_id"] == run_id), None)
    if r:
        add("R3-HKO (C4.5, CV completo)", ds, int(r["n_fps"]),
            float(r["wall_clock_s"]) / int(r["n_fps"]), 0.0,
            "CPU (spaCy features)", run_id)

# ---- escreve ----
fields = list(rows[0].keys())
with (OUT / "comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

lines = ["# Benchmark de Custo Cross-Method (Sprint 3, T5)\n",
         "| Método | Dataset | n | s/item | US$/item | US$/100 itens | Hardware |",
         "|--------|---------|--:|-------:|---------:|--------------:|----------|"]
for r in rows:
    lines.append(f"| {r['method']} | {r['dataset']} | {r['n_items']} | "
                 f"{r['s_per_item']} | {r['usd_per_item']} | "
                 f"{r['usd_per_100_items']} | {r['hardware']} |")
lines += [
    "",
    "Notas:",
    "- E30f = E28a (SLM, custo dominante) + TieredCorrection + BERT Ridge "
    "(batch ~1–2 min por dataset, marginal).",
    "- E33d/E34a: pipeline simbólico puro — 2 a 3 ordens de magnitude mais "
    "rápido que LLM e custo zero.",
    "- R1 De Vito: agregado das 7 runs por projeto.",
    "- R2/R3: tempo de CV completo (treino+avaliação) dividido por item; "
    "inferência pura é menor.",
]
(OUT / "comparison.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} linhas -> {OUT/'comparison.csv'} + comparison.md")
for r in rows:
    print(f"  {r['method'][:34]:34s} {r['dataset']:22s} {r['s_per_item']:>8.3f}s/item  ${r['usd_per_item']:.5f}/item")
