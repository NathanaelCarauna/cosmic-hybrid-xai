#!/usr/bin/env python3
"""Validação externa out-of-sample nos case studies canônicos do COSMIC Consortium.

Bases (formato caso de uso, GT canônico do Measurement Practices Committee):
  - C-REG: 19 FPs / 102 CFP    - ACME Car Hire: 8 / 33
  - RestoSys: 26 / 108 (requisitos de §1.C, NL)
  - Web Advice Module: 9 / 40  (reportado À PARTE: teste pouco justo, texto esparso)

Compara as vias nativas de medição de texto: E33d (simbólico, custo nulo) e
R1-COSMET (GPT-4-turbo). Aplica o MESMO aparato estatístico do resto da tese:
SA com IC95 bootstrap (10k, seed 42) por estudo e no agregado, para AMBOS os
métodos; teste pareado Wilcoxon (|erro| por item) + Cliff's δ no agregado justo.

Saída: results/cosmic-casestudies/external-validation.csv + .md
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
rng = np.random.default_rng(42)

DATASETS = [  # (slug, nome, glob_R1, justo)
    ("c-reg", "C-REG", "c-reg-19-r1-gpt-4-turbo-*", True),
    ("acme", "ACME Car Hire", "acme-r1-gpt-4-turbo-*", True),
    ("restosys", "RestoSys", "restosys-r1-gpt-4-turbo-*", True),
    ("webadvice", "Web Advice Module", "webadvice-r1-gpt-4-turbo-*", False),
]


def load_pipe():
    spec = importlib.util.spec_from_file_location(
        "nop", str(BASE / "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["nop"] = m
    spec.loader.exec_module(m)
    return m.NerOntologyPipeline(use_ner=False)


def sa_point(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mae = np.abs(yt - yp).mean()
    base = np.abs(yt - yt.mean()).mean()
    return (1 - mae / base) if base > 0 else float("nan")


def sa_ci(yt, yp, n=10000):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    idx = rng.integers(0, len(yt), size=(n, len(yt)))
    sas = []
    for i in idx:
        b = np.abs(yt[i] - yt[i].mean()).mean()
        sas.append(1 - np.abs(yt[i] - yp[i]).mean() / b if b > 0 else np.nan)
    sas = np.array(sas)
    sas = sas[~np.isnan(sas)]
    lo, hi = np.percentile(sas, [2.5, 97.5])
    return round(float(sa_point(yt, yp)), 3), round(float(lo), 3), round(float(hi), 3)


def mae_point(yt, yp):
    return float(np.abs(np.asarray(yt, float) - np.asarray(yp, float)).mean())


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def load_r1(glob):
    if glob is None:
        return {}
    dirs = sorted((BASE / "results/cosmet").glob(glob))
    if not dirs:
        return {}
    return {r["file"].replace(".txt", ""): float(r["cfp"])
            for r in csv.DictReader((dirs[-1] / "batch_summary.csv").open())
            if r["status"] == "ok"}


def main() -> None:
    pipe = load_pipe()
    rows_out = []
    fair = {"gt": [], "e33d": [], "r1": []}  # itens pareados (E33d e R1) das bases justas
    per_item_all = []

    for slug, name, glob, is_fair in DATASETS:
        rows = list(csv.DictReader((BASE / f"datasets/cosmic-casestudies/{slug}.csv").open()))
        r1 = load_r1(glob)
        gt = [int(r["cfp_gt"]) for r in rows]
        e33d = [pipe.measure(r["text"]).cfp for r in rows]
        es, elo, ehi = sa_ci(gt, e33d)
        row = {"base": name, "n": len(rows), "justo": "sim" if is_fair else "não",
               "e33d_mae": round(mae_point(gt, e33d), 3),
               "e33d_sa": es, "e33d_ci": f"[{elo:+.3f}, {ehi:+.3f}]"}
        keys = [i for i, r in enumerate(rows) if r["fp_id"] in r1]
        if keys:
            gtk = [gt[i] for i in keys]
            r1k = [r1[rows[i]["fp_id"]] for i in keys]
            rs, rlo, rhi = sa_ci(gtk, r1k)
            row.update({"r1_mae": round(mae_point(gtk, r1k), 3),
                        "r1_sa": rs, "r1_ci": f"[{rlo:+.3f}, {rhi:+.3f}]"})
            if is_fair:
                # alinhar E33d e R1 nos mesmos itens
                for i in keys:
                    fair["gt"].append(gt[i])
                    fair["e33d"].append(e33d[i])
                    fair["r1"].append(r1[rows[i]["fp_id"]])
        else:
            row.update({"r1_mae": "", "r1_sa": "", "r1_ci": "N/A"})
        rows_out.append(row)
        for r, g, e in zip(rows, gt, e33d):
            per_item_all.append({"base": name, "fp_id": r["fp_id"], "cfp_gt": g,
                                 "e33d": e, "r1": int(r1[r["fp_id"]]) if r["fp_id"] in r1 else ""})

    # agregado justo
    gt = np.array(fair["gt"], float)
    e = np.array(fair["e33d"], float)
    r = np.array(fair["r1"], float)
    es, elo, ehi = sa_ci(gt, e)
    rs, rlo, rhi = sa_ci(gt, r)
    simple = gt <= 4
    comp = gt >= 5
    agg = {
        "n": len(gt),
        "e33d_mae": round(mae_point(gt, e), 3), "e33d_sa": es, "e33d_ci": f"[{elo:+.3f}, {ehi:+.3f}]",
        "r1_mae": round(mae_point(gt, r), 3), "r1_sa": rs, "r1_ci": f"[{rlo:+.3f}, {rhi:+.3f}]",
        "e33d_mae_simple": round(float(np.abs(gt[simple] - e[simple]).mean()), 3),
        "e33d_mae_comp": round(float(np.abs(gt[comp] - e[comp]).mean()), 3),
        "r1_mae_simple": round(float(np.abs(gt[simple] - r[simple]).mean()), 3),
        "r1_mae_comp": round(float(np.abs(gt[comp] - r[comp]).mean()), 3),
        "n_simple": int(simple.sum()), "n_comp": int(comp.sum()),
    }
    # teste pareado E33d vs R1 (|erro| por item)
    ae = np.abs(gt - e)
    ar = np.abs(gt - r)
    p = 1.0 if np.all(ae - ar == 0) else float(wilcoxon(ae, ar, zero_method="zsplit").pvalue)
    delta = float(cliffs_delta(ae, ar))
    agg["wilcoxon_p"] = f"{p:.2e}"
    agg["cliffs_delta"] = round(delta, 3)
    agg["melhor"] = "R1" if ar.mean() < ae.mean() else "E33d"

    out = BASE / "results" / "cosmic-casestudies"
    with (out / "external-validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with (out / "external-validation-per-item.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_item_all[0].keys()))
        w.writeheader()
        w.writerows(per_item_all)

    md = ["# Validação externa: case studies canônicos do COSMIC Consortium", "",
          "SA com IC95 bootstrap (10k, semente 42) por estudo e no agregado, para",
          "E33d (via simbólica, custo nulo) e R1 (COSMET/GPT-4-turbo). Baseline pela média.", "",
          "## Por estudo", "",
          "| Estudo | n | E33d SA [IC95] | R1 SA [IC95] |",
          "|---|--:|---|---|"]
    for r_ in rows_out:
        r1c = r_["r1_ci"] if r_["r1_sa"] == "" else f"{r_['r1_sa']:+.3f} {r_['r1_ci']}"
        md.append(f"| {r_['base']}{'' if r_['justo']=='sim' else ' (excl.)'} | {r_['n']} | "
                  f"{r_['e33d_sa']:+.3f} {r_['e33d_ci']} | {r1c} |")
    md += ["",
           f"## Agregado justo (C-REG + ACME + RestoSys, n={agg['n']})", "",
           f"- E33d: MAE {agg['e33d_mae']}, SA {agg['e33d_sa']:+.3f} {agg['e33d_ci']}.",
           f"- R1:   MAE {agg['r1_mae']}, SA {agg['r1_sa']:+.3f} {agg['r1_ci']}.",
           f"- Pareado E33d vs R1 (|erro|): Wilcoxon p={agg['wilcoxon_p']}, "
           f"Cliff's δ={agg['cliffs_delta']}, melhor={agg['melhor']}.",
           f"- Por complexidade: simples (CFP≤4, n={agg['n_simple']}) E33d MAE "
           f"{agg['e33d_mae_simple']} / R1 {agg['r1_mae_simple']}; complexo (CFP≥5, "
           f"n={agg['n_comp']}) E33d {agg['e33d_mae_comp']} / R1 {agg['r1_mae_comp']}.", "",
           "Nota: o agregado positivo apoia-se sobretudo no RestoSys (maior n, e o estudo",
           "cujo texto foi re-extraído da fonte de requisitos); o C-REG é levemente negativo,",
           "reportado sem omissão. O Web Advice é reportado à parte (texto esparso, teste",
           "pouco justo a qualquer medidor textual, como confirma também o R1)."]
    (out / "external-validation.md").write_text("\n".join(md) + "\n")

    for r_ in rows_out:
        r1s = "N/A" if r_["r1_sa"] == "" else f"{r_['r1_sa']:+.3f} {r_['r1_ci']}"
        print(f"{r_['base']:20s} n={r_['n']:2d} E33d {r_['e33d_sa']:+.3f} {r_['e33d_ci']}  R1 {r1s}")
    print(f"\nAGREGADO justo n={agg['n']}: E33d {agg['e33d_sa']:+.3f} {agg['e33d_ci']} | "
          f"R1 {agg['r1_sa']:+.3f} {agg['r1_ci']}")
    print(f"  pareado: Wilcoxon p={agg['wilcoxon_p']} δ={agg['cliffs_delta']} melhor={agg['melhor']}")
    print(f"  simples E33d {agg['e33d_mae_simple']}/R1 {agg['r1_mae_simple']} | "
          f"complexo E33d {agg['e33d_mae_comp']}/R1 {agg['r1_mae_comp']}")


if __name__ == "__main__":
    main()
