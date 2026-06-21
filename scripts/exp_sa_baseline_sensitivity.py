#!/usr/bin/env python3
"""Sensibilidade da Standardized Accuracy (SA) à convenção de PREDITOR TRIVIAL.

A SA = 1 - MAE/MAE_baseline depende de qual preditor trivial define MAE_baseline.
Três convenções usuais:
  - média:    MAE_baseline = (1/n) Σ |y_i - ȳ|            (a adotada na dissertação)
  - mediana:  MAE_baseline = (1/n) Σ |y_i - mediana(y)|   (preditor constante ÓTIMO sob MAE)
  - aleatório: diferença média de Gini E|x-y|, x,y ~ gabarito
              (palpite aleatório de Shepperd & MacDonell 2012, o canônico)

Em distribuições largas as três coincidem; em distribuições estreitas divergem.
Reproduz a Tabela tab:sa-baseline (Resultados, Seção de sensibilidade).

Saída: results/sensitivity/sa_baseline_sensitivity.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent

CELLS = {
    "De Vito (casos de uso)": "results/ner-ontology/e33d_results.csv",
    "Zenodo (sentenças)": "results/ner-ontology/e33d_zenodo_results.csv",
    "Ochodek (nomes de UC)": "results/ner-ontology/ochodek_holdout_per_item.csv",
}


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((BASE / path).open()))
    yt = np.array([float(r["y_true"]) for r in rows])
    yp = np.array([float(r["y_pred"]) for r in rows])
    return yt, yp


def gini_md(y: np.ndarray) -> float:
    """Diferença média de Gini E|x-y| (= MAE esperado do palpite aleatório)."""
    y = np.sort(y)
    n = len(y)
    i = np.arange(1, n + 1)
    return float((2 * np.sum((2 * i - n - 1) * y)) / (n * n))


def sa(mae: float, base: float) -> float:
    return float(1 - mae / base) if base > 0 else float("nan")


def main() -> None:
    out_rows = []
    for name, path in CELLS.items():
        yt, yp = load(path)
        mae = float(np.abs(yt - yp).mean())
        b_mean = float(np.abs(yt - yt.mean()).mean())
        b_med = float(np.abs(yt - np.median(yt)).mean())
        b_rand = gini_md(yt)
        out_rows.append({
            "formato": name, "n": len(yt), "mae_modelo": round(mae, 3),
            "sa_media": round(sa(mae, b_mean), 3),
            "sa_mediana": round(sa(mae, b_med), 3),
            "sa_aleatorio": round(sa(mae, b_rand), 3),
        })

    out = BASE / "results" / "sensitivity" / "sa_baseline_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    hdr = f"{'formato':24s} {'n':>5s} {'MAE':>6s} {'SA(méd)':>8s} {'SA(med)':>8s} {'SA(ale)':>8s}"
    print(hdr)
    for r in out_rows:
        print(f"{r['formato']:24s} {r['n']:5d} {r['mae_modelo']:6.3f} "
              f"{r['sa_media']:+8.3f} {r['sa_mediana']:+8.3f} {r['sa_aleatorio']:+8.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
