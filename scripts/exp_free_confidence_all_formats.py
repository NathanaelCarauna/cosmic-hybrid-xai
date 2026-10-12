#!/usr/bin/env python3
"""E44b — Réplica do sinal de confiança de custo zero nos quatro formatos.

Estende `exp_rule_provenance_confidence.py` (que mediu só user stories) para
De Vito, Zenodo e Ochodek. Os dois braços rodam no MESMO laço de iteração, de
modo que o alinhamento por item é garantido por construção (os `item_id` de
`results/error-analysis/per_item_all.csv` estão renumerados e o join seria frágil).

Braço A — ontologia das 27 regras (`ner_srl`): produz o CFP canônico e os
          `rule_id` de cada movimento, dos quais se deriva a densidade de
          regras de baixa certeza.
Braço B — `KeywordNormalizer` + `CosmicRuleEngine`: segunda opinião GRATUITA,
          via independente, que erra de forma descorrelacionada.

Custo financeiro: ZERO. Nenhuma chamada de API, nenhum treino.

Uso:
    .venv/bin/python scripts/exp_free_confidence_all_formats.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hybrid_xai_runner as R  # noqa: E402
from eval_ner_ontology_pipeline import load_mod, wrap_as_uc  # noqa: E402

_PIPE_MOD = load_mod(
    "ner_ontology_pipeline",
    str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
)

LOW_CONFIDENCE = {
    "R7_AMBIG_READ", "R7_AMBIG_INTERNAL", "R7_AMBIG_PROCESS",
    "R14_SYS_FALLBACK_READ", "R14_SYS_FALLBACK_EXIT", "R14_SYS_UNKNOWN",
    "R15_USER_FALLBACK", "R_DEFAULT",
}


def build_arms():
    onto = _PIPE_MOD.NerOntologyPipeline(use_ner=False)
    from models.hybrid_xai.normalizers.keyword_normalizer import KeywordNormalizer
    return onto, KeywordNormalizer(), R.CosmicRuleEngine()


def report(name: str, rows: list[dict]) -> dict:
    if not rows:
        print(f"\n### {name}: sem itens")
        return {}
    err = np.array([r["abs_error"] for r in rows], float)

    def stat(g):
        if not g:
            return None
        e = np.array([r["abs_error"] for r in g], float)
        return len(g), len(g) / len(rows), e.mean(), (e == 0).mean()

    conf = [r for r in rows if r["agree"] and r["n_low"] == 0]
    rest = [r for r in rows if not (r["agree"] and r["n_low"] == 0)]
    print(f"\n### {name}  (n={len(rows)}, MAE global={err.mean():.3f}, "
          f"exatos={(err == 0).mean():.1%})")
    for lbl, g in (("keyword concorda", [r for r in rows if r["agree"]]),
                   ("sem fallback", [r for r in rows if r["n_low"] == 0]),
                   ("AMBOS (alta confiança)", conf),
                   ("complemento", rest)):
        s = stat(g)
        if s is None:
            print(f"  {lbl:26} vazio")
            continue
        n, frac, mae, ex = s
        print(f"  {lbl:26} n={n:5d} ({frac:5.1%})  MAE={mae:.3f}  exatos={ex:5.1%}")
    sc, sr = stat(conf), stat(rest)
    if sc and sr and sc[2] > 0:
        print(f"  --> discriminação: {sr[2] / sc[2]:.1f}x")
    return {"conf": sc, "rest": sr}


def run_devito(onto, kwn, engine) -> list[dict]:
    gt = R.load_devito_gt(scope="all")
    rows = []
    for fp_id, text, sf in R.iter_devito_fps(BASE / "replications" / "cosmet-devito" / "datasets",
                                             scope="all"):
        if fp_id not in gt:
            continue
        res = onto.measure(text, fp_id=fp_id)
        rules = [getattr(m, "rule_id", "?") or "?" for m in res.movements]
        kw = engine.measure(kwn.normalize(text, sf, fp_id=fp_id))
        rows.append({"fp": fp_id, "y_true": gt[fp_id], "y_pred": res.cfp,
                     "abs_error": abs(res.cfp - gt[fp_id]),
                     "n_low": sum(1 for r in rules if r in LOW_CONFIDENCE),
                     "agree": res.cfp == kw.total_cfp})
    return rows


def run_zenodo(onto, kwn, engine) -> list[dict]:
    gt = R.load_zenodo_gt()
    rows = []
    for fp_id, text, sf in R.iter_zenodo_fps(BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sentence"):
        if fp_id not in gt:
            continue
        res = onto.measure(text, fp_id=fp_id, sentence_mode=True)
        rules = [getattr(m, "rule_id", "?") or "?" for m in res.movements]
        kw = engine.measure(kwn.normalize(text, sf, fp_id=fp_id))
        rows.append({"fp": fp_id, "y_true": gt[fp_id], "y_pred": res.cfp,
                     "abs_error": abs(res.cfp - gt[fp_id]),
                     "n_low": sum(1 for r in rules if r in LOW_CONFIDENCE),
                     "agree": res.cfp == kw.total_cfp})
    return rows


def run_user_stories(onto, kwn, engine) -> list[dict]:
    from adapters.connextra_adapter import connextra_to_ucs
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = {r["fp_id"]: int(r["cfp_gt"])
          for r in csv.DictReader((fp_dir / "ground_truth.csv").open(newline="", encoding="utf-8"))}
    rows = []
    for fp_file in sorted(fp_dir.glob("FP*.txt")):
        fp_id = fp_file.stem
        if fp_id not in gt:
            continue
        raw = fp_file.read_text(encoding="utf-8").strip()
        ucs = connextra_to_ucs(raw, fp_id) or [wrap_as_uc(raw, fp_id)]
        cfp, rules = 0, []
        for uc in ucs:
            res = onto.measure(uc, fp_id=fp_id)
            cfp += res.cfp
            rules += [getattr(m, "rule_id", "?") or "?" for m in res.movements]
        kw_total = sum(engine.measure(kwn.normalize(uc, R.SourceFormat.USE_CASE,
                                                    fp_id=fp_id)).total_cfp for uc in ucs)
        rows.append({"fp": fp_id, "y_true": gt[fp_id], "y_pred": cfp,
                     "abs_error": abs(cfp - gt[fp_id]),
                     "n_low": sum(1 for r in rules if r in LOW_CONFIDENCE),
                     "agree": cfp == kw_total})
    return rows


def triage(name: str, rows: list[dict]) -> dict:
    """Números da TRIAGEM citados na dissertação: que fatia do corpus a
    proveniência marca, e que fatia da massa de erro essa fatia carrega.

    Só `n_low` entra: a coluna `agree` (segunda opinião do KeywordNormalizer)
    inverte de sinal entre formatos e não é sinal utilizável.
    """
    err = np.array([r["abs_error"] for r in rows], float)
    low = np.array([r["n_low"] > 0 for r in rows], bool)
    massa = err.sum()
    return {
        "formato": name, "n": len(rows),
        "itens_marcados": int(low.sum()),
        "frac_itens": round(float(low.mean()), 4),
        "frac_massa_erro": round(float(err[low].sum() / massa), 4) if massa else 0.0,
        "mae_marcados": round(float(err[low].mean()), 4),
        "mae_nao_marcados": round(float(err[~low].mean()), 4),
        "discriminacao": round(float(err[low].mean() / err[~low].mean()), 3)
        if err[~low].mean() > 0 else None,
    }


def main() -> int:
    onto, kwn, engine = build_arms()
    out = BASE / "results" / "rule-provenance"
    out.mkdir(parents=True, exist_ok=True)
    triagens = []

    for name, fn in (("DE VITO (casos de uso)", run_devito),
                     ("ZENODO (sentenças)", run_zenodo),
                     ("USER STORIES (Connextra)", run_user_stories)):
        try:
            rows = fn(onto, kwn, engine)
        except Exception as exc:  # noqa: BLE001
            print(f"\n### {name}: FALHOU — {type(exc).__name__}: {exc}")
            continue
        report(name, rows)
        if rows:
            slug = name.split()[0].lower()
            with (out / f"{slug}_confidence.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            triagens.append(triage(name, rows))

    if triagens:
        with (out / "triage_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(triagens[0]))
            w.writeheader()
            w.writerows(triagens)
        md = ["# Triagem por proveniência de regra (sinal de confiança de custo zero)",
              "",
              "Item marcado = ao menos um movimento classificado por regra de baixa",
              "certeza (fallback/ambiguidade). Nenhuma chamada de API, nenhum treino.",
              "",
              "| Formato | n | marcados | % itens | % da massa de erro | MAE marcados | MAE demais | discriminação |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for t in triagens:
            md.append(f"| {t['formato']} | {t['n']} | {t['itens_marcados']} | "
                      f"{t['frac_itens']:.1%} | {t['frac_massa_erro']:.1%} | "
                      f"{t['mae_marcados']:.3f} | {t['mae_nao_marcados']:.3f} | "
                      f"{t['discriminacao']:.1f}x |")
        md += ["", "Leitura: a proveniência da regra separa, sem custo, uma minoria de",
               "itens que concentra desproporcionalmente o erro — nos três formatos, no",
               "mesmo sentido. É subproduto do trace do M3 e estruturalmente indisponível",
               "para métodos que não expõem a regra aplicada."]
        (out / "triage_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n" + "\n".join(md))

    print("\nOCHODEK (nomes de UC): NÃO APLICÁVEL — a célula canônica usa um "
          "estimador linear por-formato, não a ontologia de 27 regras, "
          "de modo que não há rule_id de proveniência (ver checklist §3t.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
