#!/usr/bin/env python3
"""Experimento (b) — SLM extrai AÇÕES (temp 0, determinístico) -> a ontologia
classifica e CITA as regras. Combina a extração melhor do SLM (recall de
implícitos) com a explicabilidade regra-a-trecho da via simbólica.

O SLM NÃO classifica E/X/R/W — só lista ações atômicas, aflorando movimentos
implícitos (read-antes-de-exibir, cada objeto persistido como ação separada).
As ActionTuples vão para ontology.classify_actions (mesmas 26 regras), que
produz o CFP e o trace com rule_id. Few-shots SINTÉTICOS (sem dado de teste).

Uso: python3.10 scripts/exp_slm_extract.py [--n N] [--systems S1,S2] [--out FILE]
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

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

_SYS = ("You extract atomic data-movement actions from a use case for COSMIC sizing.\n"
        "Output ONLY JSON: {\"actions\":[{\"actor\":\"user|system\",\"verb\":\"...\",\"object\":\"...\"}]}.\n"
        "CRITICAL: the fields/data the USER enters or fills are ONE single user action "
        "(verb \"enter\", object = the whole form/record), NOT separate reads and NOT one per "
        "field.\n"
        "Add a system \"read\" ONLY for data the system must FETCH FROM STORAGE that the user "
        "did NOT provide (e.g. to display an existing record, or to validate against a stored "
        "list). Do not invent reads for entered fields.\n"
        "When the system records/saves/stores/updates/deletes, list EACH distinct persisted "
        "object as a SEPARATE system action.\n"
        "Ignore pure UI/control (click, press, navigate, select a menu/button). Do NOT output "
        "Entry/Exit/Read/Write. /no_think")
_EX_U = ("Use case: Save profile. 1 User: the user edits the fields and clicks Save. "
         "2 System: the system validates the data and records the new profile.")
_EX_A = json.dumps({"actions": [
    {"actor": "user", "verb": "edit", "object": "profile fields"},
    {"actor": "system", "verb": "read", "object": "existing profile"},
    {"actor": "system", "verb": "store", "object": "profile"},
    {"actor": "system", "verb": "show", "object": "confirmation message"}]})


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


DEVITO = {
    "ALBERGATE": BASE/"replications/cosmet-devito/datasets/MIS/ALBERGATE",
    "K01719": BASE/"replications/cosmet-devito/datasets/IoT_Telemedicine/K01719",
    "K01720": BASE/"replications/cosmet-devito/datasets/IoT_Telemedicine/K01720",
    "K01726": BASE/"replications/cosmet-devito/datasets/IoT_Telemedicine/K01726",
    "U-CURE": BASE/"replications/cosmet-devito/datasets/ML/U-CURE",
    "AutomaticLineSwitching": BASE/"replications/cosmet-devito/datasets/Real-time/AutomaticLineSwitching",
    "RiceCooker": BASE/"replications/cosmet-devito/datasets/Real-time/RiceCooker",
}


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("actions", [])
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limite de FPs (0 = todos)")
    ap.add_argument("--systems", type=str, default="")
    ap.add_argument("--out", type=str, default="results/ner-ontology/slm_extract_results.csv")
    args = ap.parse_args()

    from llama_cpp import Llama
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # cosmic_ontology.py agora importa language_profile via import relativo
    # (from .language_profile import ...), então não pode ser carregado como
    # módulo raiz avulso (load_mod). ner_ontology_pipeline.py já resolve isso
    # com um shim de pacote (_load_ner_srl_submodule); carregá-lo aqui reusa
    # essa resolução em vez de duplicar o mecanismo.
    pipeline_mod = load_mod(
        "ner_ontology_pipeline",
        str(BASE/"models/hybrid-xai/ner_srl/ner_ontology_pipeline.py"),
    )
    CosmicOntology = pipeline_mod.CosmicOntology
    ActionTuple = pipeline_mod.ActionTuple
    ontology = CosmicOntology(improved_dedup=True)  # dedup por grupo de dados

    llm = Llama(model_path=str(BASE/"models/gguf/Qwen3-8B-Q4_K_M.gguf"),
                n_ctx=8192, n_gpu_layers=35, verbose=False)

    def extract(uc_text):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": _EX_U},
                      {"role": "assistant", "content": _EX_A},
                      {"role": "user", "content": "Use case:\n" + uc_text[:6000]}],
            temperature=0.0, max_tokens=1500)
        return parse_json(r["choices"][0]["message"]["content"])

    def measure(uc_text):
        actions = extract(uc_text)
        tuples = []
        for i, a in enumerate(actions):
            verb = str(a.get("verb", "")).strip().lower()
            if not verb:
                continue
            lemma = nlp(verb)[0].lemma_.lower() if verb else ""
            obj = str(a.get("object", "")).strip()
            actor = str(a.get("actor", "user")).strip().lower()
            tuples.append(ActionTuple(
                step_id=str(i+1), agent="system" if actor == "system" else actor,
                verb=verb, verb_lemma=lemma, theme=obj,
                source_text=f"{actor} {verb} {obj}", entities=[],
                section="main", is_negated=False))
        movs = ontology.classify_actions(tuples, primary_actor="User", n_steps=len(tuples))
        cfp = sum(1 for m in movs if m.movement_type is not None)
        return cfp, movs

    # GT
    def gt_map(sysname):
        gtcsv = BASE/"results/cosmet"/f"{sysname}-gpt-4-turbo-*"
        import glob
        d = glob.glob(str(BASE/"results/cosmet"/f"{sysname}-gpt-4-turbo-*"/"measurements"/"comparison_results.csv"))
        gt = {}
        if d:
            for r in csv.DictReader(open(d[0])):
                gt[r["fp"]] = int(r["y_true"])
        return gt

    systems = args.systems.split(",") if args.systems else list(DEVITO)
    rows = []
    t0 = time.time()
    for sysname in systems:
        root = DEVITO[sysname]; gt = gt_map(sysname)
        for fp in sorted(root.glob("FP_prompts/FP*.txt"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
            if fp.stem not in gt:
                continue
            if args.n and len(rows) >= args.n:
                break
            cfp, movs = measure(fp.read_text(encoding="utf-8"))
            rows.append({"system": sysname, "fp": fp.stem, "y_pred": cfp, "y_true": gt[fp.stem],
                         "abs_error": abs(cfp - gt[fp.stem])})
            print(f"  {sysname} {fp.stem}: pred={cfp} gt={gt[fp.stem]} ({time.time()-t0:.0f}s)", flush=True)
        if args.n and len(rows) >= args.n:
            break

    import numpy as np
    e = np.array([r["abs_error"] for r in rows], float)
    y = np.array([r["y_true"] for r in rows], float)
    sa = 1 - e.mean()/np.abs(y - y.mean()).mean()
    print(f"\nSLM-extract -> ontologia: n={len(rows)} MAE={e.mean():.3f} SA={sa:+.3f}")
    out = BASE/args.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "fp", "y_pred", "y_true", "abs_error"])
        w.writeheader(); w.writerows(rows)
    print("saved:", out)


if __name__ == "__main__":
    main()
