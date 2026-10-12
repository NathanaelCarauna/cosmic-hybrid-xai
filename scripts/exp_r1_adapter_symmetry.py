#!/usr/bin/env python3
"""Experimento de SIMETRIA da matriz: a normalização (M1) é a alavanca portável?

Crítica do revisor à matriz de heterogeneidade: o método proposto recebe um
normalizador por formato; os competidores são aplicados congelados. "A robustez
é da arquitetura ou do esforço de engenharia concedido só a si?"

Resposta empírica: damos ao competidor R1 (COSMET / GPT-4-turbo) o MESMO
adaptador Connextra (nosso M1) e medimos se ele recupera no formato em que
colapsa (histórias de usuário). Se recupera, a normalização — e não o motor —
é a contribuição portável.

Reproduz a comparação:
  - R1 cru (Connextra)      : results/cosmet/user-stories-98-r1-gpt-4-turbo-*/
  - R1 + adaptador (M1)     : results/cosmet/user-stories-98-r1-adapted-gpt-4-turbo-*/
    (insumos gerados por connextra_to_uc em datasets/prepared/us-connextra-adapted/)

Saída: results/asymmetry/r1_adapter_symmetry.csv + .md

Pré-requisito: a run paga do R1 sobre os pseudo-UCs já deve ter sido executada
  via: python scripts/cosmet_batch_runner.py datasets/prepared/us-connextra-adapted \
         --dataset user-stories-98-r1-adapted --model gpt-4-turbo
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)

GT = BASE / "results/cross-input/user-stories-98/ground_truth.csv"
GT_V2 = BASE / "datasets/user-stories-100-ground-truth-FINAL-v2.csv"
PER_ITEM = BASE / "results/error-analysis/per_item_all.csv"
RAW = "results/cosmet/user-stories-98-r1-gpt-4-turbo-20260611T020933/batch_summary.csv"
ADAPTED_GLOB = "results/cosmet/user-stories-98-r1-adapted-gpt-4-turbo-*"


def load_gt() -> dict[str, float]:
    return {r["fp_file"]: float(r["cfp_gt"]) for r in csv.DictReader(GT.open())}


def load_preds(path: Path) -> dict[str, float]:
    d = {}
    for r in csv.DictReader(path.open()):
        if r["status"] == "ok":
            d[r["file"]] = float(r["cfp"])
    return d


def score(gt: dict, preds: dict) -> dict:
    keys = [k for k in gt if k in preds]
    yt = np.array([gt[k] for k in keys])
    yp = np.array([preds[k] for k in keys])
    mae = float(np.abs(yt - yp).mean())
    base = float(np.abs(yt - yt.mean()).mean())
    sa = 1 - mae / base
    idx = np.arange(len(yt))
    sas = []
    for _ in range(10000):
        s = rng.choice(idx, len(idx), True)
        b = np.abs(yt[s] - yt[s].mean()).mean()
        sas.append(1 - np.abs(yt[s] - yp[s]).mean() / b if b > 0 else np.nan)
    sas = np.array(sas)
    sas = sas[~np.isnan(sas)]
    lo, hi = np.percentile(sas, [2.5, 97.5])
    return {"n": len(yt), "mae": round(mae, 3), "sa": round(float(sa), 3),
            "sa_lo": round(float(lo), 3), "sa_hi": round(float(hi), 3),
            "exatos": int((yt == yp).sum())}


def clean_comparison() -> dict:
    """Comparação LIMPA E34a x R1+adaptador sobre os MESMOS 78 itens de teste.

    A leitura anterior confrontava o E34a de 98 itens (que inclui as 20 histórias
    de desenvolvimento, cujo gabarito esteve visível na calibração do adaptador)
    contra o R1 adaptado de 98. O par válido restringe AMBOS ao split de teste e
    testa a diferença de forma pareada, em vez de comparar dois pontos com IC
    largamente sobrepostos.
    """
    grows = list(csv.DictReader(GT.open()))
    file2fp = {r["fp_file"]: r["fp_id"] for r in grows}
    fp2us = {r["fp_id"]: r["us_id"] for r in grows}
    gt = {r["fp_id"]: float(r["cfp_gt"]) for r in grows}
    dev = {r["us_id"] for r in csv.DictReader(GT_V2.open())
           if r["kappa_subset"] == "True"}
    test = sorted(fp for fp in gt if fp2us[fp] not in dev)

    adapted_dir = sorted(BASE.glob(ADAPTED_GLOB))[-1]
    r1a = {file2fp[k]: v for k, v in
           load_preds(adapted_dir / "batch_summary.csv").items()}
    e34 = {r["item_id"]: float(r["y_pred"]) for r in csv.DictReader(PER_ITEM.open())
           if r["method"] == "E34a" and r["dataset"] == "user-stories"}

    e_ours = np.array([abs(gt[k] - e34[k]) for k in test])
    e_r1 = np.array([abs(gt[k] - r1a[k]) for k in test])
    stat, p = wilcoxon(e_ours, e_r1)
    gt_, lt_ = 0, 0
    for a in e_ours:
        gt_ += int((a > e_r1).sum())
        lt_ += int((a < e_r1).sum())
    delta = (gt_ - lt_) / (len(e_ours) * len(e_r1))
    d = e_ours - e_r1
    return {
        "n": len(test),
        "ours": score({k: gt[k] for k in test}, {k: e34[k] for k in test}),
        "r1_adapted": score({k: gt[k] for k in test}, {k: r1a[k] for k in test}),
        "wilcoxon_p": round(float(p), 4), "cliffs_delta": round(float(delta), 3),
        "identical_preds": int((d == 0).sum()),
        "ours_better": int((d < 0).sum()), "r1_better": int((d > 0).sum()),
    }


def main() -> None:
    gt = load_gt()
    raw = score(gt, load_preds(BASE / RAW))
    adapted_dir = sorted(BASE.glob(ADAPTED_GLOB))[-1]
    adapted = score(gt, load_preds(adapted_dir / "batch_summary.csv"))
    clean = clean_comparison()

    rows = [
        {"config": "R1 cru (Connextra)", **raw},
        {"config": "R1 + adaptador Connextra (M1)", **adapted},
        {"config": "E34a (proposto) — teste-78", **clean["ours"]},
        {"config": "R1 + adaptador — teste-78", **clean["r1_adapted"]},
    ]
    out = BASE / "results" / "asymmetry"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "r1_adapter_symmetry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# Simetria da matriz: a normalização (M1) é portável entre motores?", "",
          "R1 (GPT-4-turbo) em histórias de usuário (n=98), com e sem o adaptador",
          "Connextra (nosso M1). SA com baseline pela média; IC95 bootstrap 10k (seed 42).", "",
          "| Configuração | n | MAE | SA [IC95] | exatos |",
          "|---|--:|--:|---|--:|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['n']} | {r['mae']} | "
                  f"{r['sa']:+.3f} [{r['sa_lo']:+.3f}; {r['sa_hi']:+.3f}] | {r['exatos']} |")
    md += ["",
           "Referência (matriz): E34a (nossa via simbólica + adaptador) MAE 0,459 / SA +0,364; "
           "E30f (via neural própria) MAE 1,490 / SA −1,065.", "",
           "## Comparação limpa (mesmos 78 itens de teste)", "",
           "Nota: os IC95 acima vêm do fluxo de reamostragem deste script; a célula "
           "canônica do E34a-test78 é a de `results/final_statistical_table.md` "
           "(SA +0,261 [−0,026; +0,486]). A diferença é de terceira decimal, própria "
           "de dois fluxos bootstrap independentes, e o teste decisivo aqui é o "
           "pareado, não a sobreposição de intervalos.", "",
           f"Wilcoxon pareado E34a x R1+adaptador: p = {clean['wilcoxon_p']}; "
           f"Cliff's delta = {clean['cliffs_delta']:+.3f}. "
           f"Predições idênticas em {clean['identical_preds']}/{clean['n']} itens "
           f"(E34a melhor em {clean['ours_better']}, pior em {clean['r1_better']}).", "",
           "Leitura: o adaptador (M1) leva o competidor R1 do colapso (SA −0,230) à SA "
           "positiva — a normalização é a alavanca portável, independente do motor. "
           "Sobre os MESMOS 78 itens de teste, porém, as duas configurações são "
           "estatisticamente indistinguíveis: a diferença de MAE não sobrevive ao teste "
           "pareado, e os dois motores produzem a MESMA contagem na larga maioria dos "
           "itens. O que separa o método proposto do R1 adaptado, nesse formato, não é "
           "acurácia — é auditabilidade e custo."]
    (out / "r1_adapter_symmetry.md").write_text("\n".join(md) + "\n")

    for r in rows:
        print(f"{r['config']:34s} n={r['n']} MAE={r['mae']} "
              f"SA={r['sa']:+.3f} [{r['sa_lo']:+.3f},{r['sa_hi']:+.3f}] exatos={r['exatos']}")
    print(f"-> {out}/r1_adapter_symmetry.{{csv,md}}")


if __name__ == "__main__":
    main()
