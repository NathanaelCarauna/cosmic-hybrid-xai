#!/usr/bin/env python3
"""T6 (Sprint 3) — Gera materiais REAIS para o survey M3 (condições A/B/C).

Para 2 UCs reais do De Vito (FP1 simples CFP=4, FP40 complexo CFP=8):
  A — Hybrid-XAI/E33d: trace real (movimento → grupo de dados → regra → trecho)
  B — COSMET cru: apenas a predição (do run gpt-4-turbo real)
  C — LIME real sobre BERT-Ridge (treinado no train-89, mesmo braço do E30f)

Substitui os exemplos fabricados do apêndice do instrumento
(m3-survey-instrument.md) por saídas reais dos sistemas.

Output: results/m3-materials/condicoes-{uc1,uc2}.md
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
OUT = BASE / "results" / "m3-materials"
OUT.mkdir(parents=True, exist_ok=True)

UCS = [
    ("uc1", "COSMET-ALBERGATE-FP1", "ALBERGATE/FP1", "simples"),
    ("uc2", "COSMET-ALBERGATE-FP40", "ALBERGATE/FP40", "complexo"),
]


from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("xai_symbolic",
    str(BASE / "models" / "hybrid-xai" / "xai_symbolic.py"))
_xai = module_from_spec(_spec); _spec.loader.exec_module(_xai)
to_mermaid = _xai.diagram_mermaid


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    ner = BASE / "models" / "cosmic-ner" / "best"
    return mod.NerOntologyPipeline(ner_model_path=str(ner) if ner.exists() else None)


FP_DIR = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"


def main() -> None:
    manifest = {r["id"]: r for r in csv.DictReader(
        (BASE / "datasets" / "manifest.csv").open(newline="", encoding="utf-8"))}
    cosmet = {r["fp"]: int(float(r["y_pred"])) for r in csv.DictReader(
        (BASE / "results" / "cosmet" / "devito-119-combined-gpt-4-turbo-merged"
         / "comparison_results.csv").open(newline="", encoding="utf-8"))}

    pipeline = load_pipeline()

    # --- BERT-Ridge p/ LIME: treina no train-89 (mesmo braço do E30f) ---
    from bert_hybrid_improvements_v2 import load_e28a_data, extract_bert_embeddings
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from lime.lime_text import LimeTextExplainer

    train = load_e28a_data(
        BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245")
    tr_texts = [d["raw_text"] for d in train]
    y_tr = np.array([d["y_true"] for d in train], float)
    emb_tr = extract_bert_embeddings(tr_texts)
    scaler = StandardScaler().fit(emb_tr)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(emb_tr), y_tr)

    def predict_fn(texts: list[str]) -> np.ndarray:
        e = extract_bert_embeddings(list(texts))
        return ridge.predict(scaler.transform(e)).reshape(-1, 1)

    explainer = LimeTextExplainer(random_state=42)

    for tag, man_id, cosmet_id, kind in UCS:
        row = manifest[man_id]
        gt = int(row["cfp"])
        # texto do arquivo FP original (formatação de UC preservada — é o que
        # o parser do E33d e o COSMET receberam nas runs reais)
        fp_name = cosmet_id.split("/")[1]  # ALBERGATE/FP1 -> FP1
        text = (FP_DIR / f"{fp_name}.txt").read_text(encoding="utf-8")
        print(f"\n=== {tag} ({man_id}, GT={gt}, {kind}) ===")

        # A — trace E33d real
        res = pipeline.measure(text, fp_id=man_id)
        a_lines = [f"Este caso de uso recebeu **CFP = {res.cfp}** porque o "
                   "pipeline identificou os seguintes movimentos de dados:", ""]
        a_lines.append("| # | Movimento | Grupo de dados | Regra COSMIC | Trecho do requisito |")
        a_lines.append("|---|-----------|----------------|--------------|---------------------|")
        n = 0
        for m in res.movements:
            mt = str(m.movement_type).split(".")[-1]
            if mt == "None":
                continue
            n += 1
            src = (getattr(m, "source_text", "") or "").replace("|", "/")[:70]
            a_lines.append(f"| {n} | {mt} | {m.data_group} | {m.rule_id} | \"{src}\" |")
        a_lines += ["", f"Total: E={res.n_entries}, R={res.n_reads}, "
                        f"W={res.n_writes}, X={res.n_exits} → **CFP {res.cfp}**"]
        uc_label = row["text"].split("\n")[1].strip() if "\n" in row["text"] else man_id
        actor = "Usuário"
        for ln in text.splitlines():
            if ln.lower().startswith("primary actor"):
                actor = ln.split("\t", 1)[-1].strip() or actor
                break
        mermaid = to_mermaid(res, uc_label[:50], actor=actor)
        (OUT / f"diagrama-{tag}.mmd").write_text(mermaid + "\n", encoding="utf-8")
        a_lines += ["", "**Diagrama de rastreabilidade (renderizar e embutir no formulário"
                        " — ex.: mermaid.live → exportar PNG):**", "",
                    "```mermaid", mermaid, "```"]
        cond_a = "\n".join(a_lines)
        print(f"  A (E33d): CFP={res.cfp}, {n} movimentos + diagrama")

        # B — COSMET cru real
        cfp_b = cosmet.get(cosmet_id, "?")
        cond_b = f"**CFP = {cfp_b}**"
        print(f"  B (COSMET gpt-4-turbo): CFP={cfp_b}")

        # C — LIME real
        # regressão via truque do label único: predict_fn retorna (n,1) e
        # explicamos o "label" 0 (a própria saída contínua)
        exp = explainer.explain_instance(text, predict_fn, labels=(0,),
                                         num_features=8, num_samples=400)
        pred_c = float(predict_fn([text])[0][0])
        c_lines = [f"Estimativa do modelo (BERT+Ridge): **CFP ≈ {pred_c:.1f}**",
                   "", "Palavras que mais contribuíram (LIME):", "",
                   "| Palavra/trecho | Contribuição |", "|---|---|"]
        for tok, w in exp.as_list(label=0):
            c_lines.append(f"| \"{tok}\" | {w:+.3f} |")
        cond_c = "\n".join(c_lines)
        print(f"  C (LIME): pred={pred_c:.1f}, top tokens ok")

        md = [f"# Materiais M3 — {tag.upper()} ({kind})",
              "", f"**Fonte:** {man_id} (De Vito/ALBERGATE) · GT (não mostrar ao participante): CFP={gt}",
              "", "## Texto do caso de uso", "", "```", text.strip()[:1500], "```",
              "", "## Condição A — Hybrid-XAI (E33d, trace real)", "", cond_a,
              "", "## Condição B — COSMET cru (gpt-4-turbo, run real)", "", cond_b,
              "", "## Condição C — LIME post-hoc (BERT-Ridge real, 400 amostras)", "", cond_c,
              "", "---", "*Gerado por scripts/m3_generate_materials.py — saídas reais dos sistemas, sem edição manual.*"]
        (OUT / f"condicoes-{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  -> {OUT / f'condicoes-{tag}.md'}")


if __name__ == "__main__":
    main()
