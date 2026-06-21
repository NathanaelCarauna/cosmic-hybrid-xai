#!/usr/bin/env python3
"""Validação externa out-of-sample no case study canônico C-REG (19 FPs, CFP 102).

Compara as vias NATIVAS de medição de texto de caso de uso:
  - E33d (via simbólica proposta): roda localmente, custo nulo.
  - R1-COSMET (GPT-4-turbo): run paga via cosmet_batch_runner (datasets/prepared/c-reg-usecases).

R2 (regressor sobre NOMES de caso de uso) e R3 (classificador within-dataset) não
são nativamente aplicáveis a texto de caso de uso completo; a matriz de
heterogeneidade já documenta seu colapso fora do formato de origem. E30f (via
neural local) é linha de base interna e depende do SLM com GPU.

Saída: results/cosmic-casestudies/c-reg-results.csv + .md
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))


def load_gt():
    return list(csv.DictReader((BASE / "datasets/cosmic-casestudies/c-reg.csv").open()))


def metrics(yt, yp):
    yt, yp = np.array(yt, float), np.array(yp, float)
    mae = float(np.abs(yt - yp).mean())
    base = float(np.abs(yt - yt.mean()).mean())
    sa = 1 - mae / base if base > 0 else float("nan")
    pred25 = float((np.abs(yt - yp) / np.maximum(yt, 1) <= 0.25).mean())
    return {"n": len(yt), "mae": round(mae, 3), "sa": round(float(sa), 3),
            "pred25": round(pred25, 3), "exact": int((yt == yp).sum())}


def run_e33d(rows):
    spec = importlib.util.spec_from_file_location(
        "nop", str(BASE / "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["nop"] = m
    spec.loader.exec_module(m)
    pipe = m.NerOntologyPipeline(use_ner=False)
    preds = {}
    for r in rows:
        preds[r["fp_id"]] = pipe.measure(r["text"], fp_id=r["fp_id"]).cfp
    return preds


def load_r1(rows):
    dirs = sorted((BASE / "results/cosmet").glob("c-reg-19-r1-gpt-4-turbo-*"))
    if not dirs:
        return None
    pr = {}
    for r in csv.DictReader((dirs[-1] / "batch_summary.csv").open()):
        if r["status"] == "ok":
            pr[r["file"].replace(".txt", "")] = float(r["cfp"])
    return pr


def main() -> None:
    rows = load_gt()
    gt = {r["fp_id"]: float(r["cfp_gt"]) for r in rows}
    ids = [r["fp_id"] for r in rows]

    results = {}
    e33d = run_e33d(rows)
    results["E33d (simbólico, proposto)"] = metrics(
        [gt[i] for i in ids], [e33d[i] for i in ids])
    r1 = load_r1(rows)
    if r1:
        keys = [i for i in ids if i in r1]
        results["R1-COSMET (GPT-4-turbo)"] = metrics(
            [gt[i] for i in keys], [r1[i] for i in keys])

    # por-item (para a tabela detalhada)
    out = BASE / "results" / "cosmic-casestudies"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "c-reg-per-item.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fp_id", "name", "cfp_gt", "e33d", "r1"])
        for r in rows:
            i = r["fp_id"]
            w.writerow([i, r["name"], int(gt[i]), e33d.get(i, ""),
                        int(r1[i]) if r1 and i in r1 else ""])

    with (out / "c-reg-results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "mae", "sa", "pred25", "exact"])
        for name, mt in results.items():
            w.writerow([name, mt["n"], mt["mae"], mt["sa"], mt["pred25"], mt["exact"]])

    md = ["# Validação externa: case study canônico C-REG (COSMIC Consortium)", "",
          "19 processos funcionais, formato de caso de uso, CFP total 102, medição",
          "canônica do COSMIC Measurement Practices Committee (ISO/IEC 19761).",
          "Conjunto genuinamente out-of-sample, GT padrão-ouro. SA com baseline pela média.", "",
          "| Método | n | MAE | SA | PRED(25) | exatos |",
          "|---|--:|--:|--:|--:|--:|"]
    for name, mt in results.items():
        md.append(f"| {name} | {mt['n']} | {mt['mae']} | {mt['sa']:+.3f} | "
                  f"{mt['pred25']:.3f} | {mt['exact']}/19 |")
    md += ["",
           "Referência (formato de origem De Vito): E33d MAE 0,933 / SA +0,555; "
           "R1 MAE 0,370 / SA +0,824.", "",
           "Leitura: na validação externa, a via simbólica acerta os processos CRUD simples "
           "mas subconta os complexos (geração de relatório, comunicação inter-sistema, "
           "iteração sobre grupos de dados), o que corrobora a limitação de subcontagem em "
           "processos grandes documentada na avaliação por movimento e puxa a SA agregada "
           "para perto de zero. O R1 (GPT-4-turbo), no seu formato de origem (caso de uso), "
           "generaliza melhor, como já observado em De Vito. A via simbólica preserva custo "
           "nulo, execução local e auditabilidade."]
    (out / "c-reg-results.md").write_text("\n".join(md) + "\n")

    for name, mt in results.items():
        print(f"{name:32s} n={mt['n']} MAE={mt['mae']} SA={mt['sa']:+.3f} "
              f"PRED25={mt['pred25']:.3f} exatos={mt['exact']}/19")
    print(f"-> {out}/c-reg-results.{{csv,md}}")


if __name__ == "__main__":
    main()
