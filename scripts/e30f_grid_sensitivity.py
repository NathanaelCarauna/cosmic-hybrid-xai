#!/usr/bin/env python3
"""T2 (Sprint 3) — Sensibilidade do grid de TieredCorrection (E30f).

Compara o grid completo (17.280 combinacoes) com um grid reduzido (~240x
menor, 72 combinacoes): ambos calibrados no train-89 De Vito, aplicados
congelados ao holdout-30 (braco correction-only, comparavel a ablacao).

Se o delta-MAE no holdout for pequeno, o grid denso nao esta overfitando o
espaco de hiperparametros; se grande, e limitacao a reportar.

Output: results/sensitivity/e30f-grid-sensitivity.csv
"""
from __future__ import annotations

import csv
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from shared.seeds import set_global_seeds
from bert_hybrid_improvements_v2 import (
    load_e28a_data, _grid_search_correction, _apply_tiered_correction,
)

TRAIN_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245"
HOLDOUT_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-holdout-devito-20260525T133453"

# Grid reduzido: 1*3*1*2*3*2*2 = 72 combos (240x menor que 17.280)
REDUCED = {
    "threshold": [12],
    "alpha": [0.05, 0.20, 0.30],
    "beta": [6],
    "gamma": [0.1, 0.3],
    "alpha_l": [0.3, 0.5, 0.7],
    "beta_l": [3, 5],
    "gamma_l": [0.4, 0.6],
}


def grid_search_reduced(y_true, y_raw, n_steps):
    best_params, best_mae = None, float("inf")
    for th, a, b, g, al, bl, gl in product(
            REDUCED["threshold"], REDUCED["alpha"], REDUCED["beta"],
            REDUCED["gamma"], REDUCED["alpha_l"], REDUCED["beta_l"],
            REDUCED["gamma_l"]):
        params = (a, b, g, al, bl, gl, th)
        preds = np.array([_apply_tiered_correction(y_raw[j], n_steps[j], params)
                          for j in range(len(y_true))])
        mae = np.mean(np.abs(y_true - preds))
        if mae < best_mae:
            best_mae, best_params = mae, params
    return best_params


def eval_holdout(params, hold):
    yt = np.array([d["y_true"] for d in hold], float)
    yp = np.array([max(0, round(_apply_tiered_correction(
        d["y_raw_slm"], d["n_steps"], params))) for d in hold], float)
    ae = np.abs(yt - yp)
    mad = np.abs(yt - yt.mean()).mean()
    return float(ae.mean()), float(1 - ae.mean() / mad)


def main() -> None:
    set_global_seeds()
    train = load_e28a_data(TRAIN_DIR)
    hold = load_e28a_data(HOLDOUT_DIR)
    print(f"train={len(train)}, holdout={len(hold)}")
    yt = np.array([d["y_true"] for d in train], float)
    yr = np.array([d["y_raw_slm"] for d in train], float)
    ns = np.array([d["n_steps"] for d in train], float)

    rows = []
    t0 = time.perf_counter()
    p_full = _grid_search_correction(yt, yr, ns)
    t_full = time.perf_counter() - t0
    mae_f, sa_f = eval_holdout(p_full, hold)
    rows.append({"grid": "completo (17280)", "params": str(p_full),
                 "search_s": round(t_full, 1),
                 "holdout_mae": round(mae_f, 4), "holdout_sa": round(sa_f, 4)})

    t0 = time.perf_counter()
    p_red = grid_search_reduced(yt, yr, ns)
    t_red = time.perf_counter() - t0
    mae_r, sa_r = eval_holdout(p_red, hold)
    rows.append({"grid": "reduzido (72)", "params": str(p_red),
                 "search_s": round(t_red, 1),
                 "holdout_mae": round(mae_r, 4), "holdout_sa": round(sa_r, 4)})

    out = BASE / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "e30f-grid-sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['grid']:18s} holdout MAE={r['holdout_mae']} SA={r['holdout_sa']} "
              f"(busca {r['search_s']}s) params={r['params']}")
    print(f"ΔMAE (reduzido − completo): {mae_r - mae_f:+.4f}")


if __name__ == "__main__":
    main()
