#!/usr/bin/env python3
"""D14 — Robustez a perturbação linguística meaning-preserving.

Mede a estabilidade do CFP do pipeline simbólico (E33d em casos de uso De Vito;
E34a em user stories) sob reescritas que NÃO mudam o tamanho funcional COSMIC,
portanto o ground truth é fixo por construção (não precisa re-anotar):

  (1) reordenação dos passos do Main Scenario — CFP é invariante à ORDEM (o
      tamanho é o conjunto de movimentos, não a sequência). Testa se o parser
      depende de ordem/posição do gatilho.
  (2) paráfrase lexical COSMIC-neutra — sinônimos DENTRO da mesma classe de
      movimento (show↔display, enter↔input, save↔store, clicks↔presses, the
      actor↔the user, an interface↔a screen). Por construção não altera E/X/R/W.
  (3) combinada — (1)+(2).

Reporta, por dataset: MAE limpo vs MAE sob perturbação (drift), e estabilidade do
CFP (fração de variantes idênticas ao limpo; desvio-padrão médio por item; taxa
de "flip"). Determinístico (seeds 1..K). Não toca no .tex.

Uso: python3.10 scripts/exp_robustness_perturbation.py [--k 5]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import re
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

EV = importlib.util.spec_from_file_location(
    "evalpipe", str(BASE / "scripts" / "eval_ner_ontology_pipeline.py"))
evalpipe = importlib.util.module_from_spec(EV)
sys.modules["evalpipe"] = evalpipe
EV.loader.exec_module(evalpipe)

# ---- motor de perturbação (determinístico) -------------------------------

# Sinônimos DENTRO da mesma classe de movimento COSMIC (não mudam E/X/R/W).
_SYN = [
    (r"clicks on", ["selects", "presses"]),
    (r"fills out", ["completes", "fills in"]),
    (r"the actor", ["the user"]),
    (r"an interface", ["a screen", "a form"]),
    (r"shows", ["displays", "presents"]),
    (r"show", ["display", "present"]),
    (r"displays", ["shows", "presents"]),
    (r"display", ["show", "present"]),
    (r"lists", ["displays", "shows"]),
    (r"enters", ["inputs", "supplies"]),
    (r"enter", ["input", "supply"]),
    (r"provides", ["enters", "supplies"]),
    (r"saves", ["stores", "records"]),
    (r"save", ["store", "record"]),
    (r"stores", ["saves", "records"]),
    (r"records", ["saves", "stores"]),
    (r"updates", ["modifies", "edits"]),
    (r"clicks", ["presses", "taps"]),
    (r"selects", ["chooses", "picks"]),
]
_SYN_RES = [(re.compile(rf"\b{p}\b", re.IGNORECASE), s) for p, s in _SYN]


def _match_case(repl: str, original: str) -> str:
    if original[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def paraphrase(text: str, rng: random.Random, prob: float = 0.7) -> str:
    """Substitui sinônimos COSMIC-neutros com probabilidade prob (seeded)."""
    out = text
    for rex, syns in _SYN_RES:
        def _sub(m):
            if rng.random() > prob:
                return m.group(0)
            return _match_case(rng.choice(syns), m.group(0))
        out = rex.sub(_sub, out)
    return out


_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s+(.*)$")
_SECTION_RE = re.compile(
    r"^\s*(alternative\s+flows?|exceptions?|reference|pre-?conditions?|"
    r"post-?conditions?|main\s+scenario)\b", re.IGNORECASE)


def reorder_steps(text: str, rng: random.Random) -> str:
    """Embaralha os passos do Main Scenario (CFP invariante à ordem)."""
    lines = text.split("\n")
    # localizar bloco Main Scenario
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*main\s+scenario\s*$", ln, re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return text
    steps = []
    end = start
    for j in range(start, len(lines)):
        if _SECTION_RE.match(lines[j]) and not lines[j].strip()[0:1].isdigit():
            break
        m = _STEP_RE.match(lines[j])
        if m:
            steps.append(m.group(1))
            end = j + 1
        elif lines[j].strip() == "":
            end = j + 1
        else:
            break
    if len(steps) < 2:
        return text
    order = steps[:]
    # garantir uma permutação diferente da identidade quando possível
    for _ in range(8):
        rng.shuffle(order)
        if order != steps:
            break
    renum = [f"{k+1}.\t{s}" for k, s in enumerate(order)]
    return "\n".join(lines[:start] + renum + lines[end:])


def perturb(text, kind, rng):
    if kind == "reorder":
        return reorder_steps(text, rng)
    if kind == "paraphrase":
        return paraphrase(text, rng)
    if kind == "combined":
        return paraphrase(reorder_steps(text, rng), rng)
    return text


# ---- carga de dados -------------------------------------------------------

def load_devito():
    items = []
    for sysname, cfg in evalpipe.DEVITO_SYSTEMS.items():
        gt = evalpipe.load_devito_gt(cfg["gt_csv"])
        for fp in sorted(cfg["prompts"].glob("FP*.txt"),
                         key=lambda p: evalpipe.fp_sort_key(p.stem)):
            if fp.stem in gt:
                items.append({"fp": fp.stem, "text": fp.read_text(encoding="utf-8"),
                              "gt": gt[fp.stem]["y_true"], "system": sysname})
    return items


def load_user_stories():
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = {}
    with (fp_dir / "ground_truth.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["fp_id"]] = int(r["cfp_gt"])
    items = []
    for fp in sorted(fp_dir.glob("FP*.txt"), key=lambda p: evalpipe.fp_sort_key(p.stem)):
        if fp.stem in gt:
            items.append({"fp": fp.stem, "text": fp.read_text(encoding="utf-8").strip(),
                          "gt": gt[fp.stem], "system": "USER-STORIES"})
    return items


# ---- medição --------------------------------------------------------------

def measure_devito(pipe, text):
    return pipe.measure(text, fp_id="").cfp


def measure_us(pipe, text, fp_id):
    from adapters.connextra_adapter import connextra_to_ucs
    ucs = connextra_to_ucs(text, fp_id)
    if ucs is None:
        ucs = [evalpipe.wrap_as_uc(text, fp_id)]
    return sum(pipe.measure(uc, fp_id="").cfp for uc in ucs)


def run(name, items, measure_fn, kinds, k):
    # chave ÚNICA por item: o stem FP repete entre sistemas De Vito (FP1 existe
    # em ALBERGATE, K01726, ...). Usar só o stem colidiria as chaves.
    for it in items:
        it["uid"] = f"{it['system']}/{it['fp']}"
    clean = {it["uid"]: measure_fn(it["text"], it["fp"]) for it in items}
    n = len(items)
    clean_mae = sum(abs(clean[it["uid"]] - it["gt"]) for it in items) / n
    print(f"\n{'='*64}\n  {name}  (n={n})\n{'='*64}")
    print(f"  MAE limpo: {clean_mae:.4f}")

    rows = []
    for kind in kinds:
        per_variant_mae = [0.0] * k
        flips = 0
        total_variants = 0
        per_item_std = []
        for it in items:
            cfps = []
            for si, seed in enumerate(range(1, k + 1)):
                rng = random.Random(hash((it["uid"], kind, seed)) & 0xFFFFFFFF)
                pt = perturb(it["text"], kind, rng)
                cfp = measure_fn(pt, it["fp"]) if pt != it["text"] else clean[it["uid"]]
                cfps.append(cfp)
                total_variants += 1
                per_variant_mae[si] += abs(cfp - it["gt"])
                if cfp != clean[it["uid"]]:
                    flips += 1
            per_item_std.append(statistics.pstdev(cfps) if len(cfps) > 1 else 0.0)
            rows.append((it["uid"], kind, it["gt"], clean[it["uid"]], cfps))
        per_variant_mae = [t / n for t in per_variant_mae]
        mean_mae = statistics.mean(per_variant_mae)
        exact_stab = 1 - flips / total_variants
        print(f"\n  [{kind}]  k={k}")
        print(f"    MAE sob perturbação: {mean_mae:.4f} "
              f"(min {min(per_variant_mae):.4f} / max {max(per_variant_mae):.4f})"
              f"  | drift vs limpo: {mean_mae-clean_mae:+.4f}")
        print(f"    estabilidade exata do CFP: {exact_stab*100:.1f}% "
              f"(variantes idênticas ao limpo)")
        print(f"    desvio-padrão médio do CFP por item: {statistics.mean(per_item_std):.3f}")
    return clean_mae, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    pipe = evalpipe  # só p/ namespace; carrega pipeline real abaixo
    nop_path = BASE / "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py"
    spec = importlib.util.spec_from_file_location("ner_ontology_pipeline", nop_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    P = mod.NerOntologyPipeline(use_ner=False)

    devito = load_devito()
    us = load_user_stories()

    run("E33d — De Vito (casos de uso, 119)", devito,
        lambda t, f: measure_devito(P, t), ["reorder", "paraphrase", "combined"], args.k)
    run("E34a — User Stories (98)", us,
        lambda t, f: measure_us(P, t, f), ["paraphrase"], args.k)


if __name__ == "__main__":
    main()
