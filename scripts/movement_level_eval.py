#!/usr/bin/env python3
"""S4-T1 — Avaliação POR-MOVIMENTO contra os XLSX oficiais De Vito (119 FPs).

Motivação: acerto de total ≠ acerto de decomposição (caso FP41: total 9=9 com
2 falsos positivos + 2 perdidos que se cancelam). Um trace explicável só é
confiável se os MOVIMENTOS estão certos. Esta análise compara as contagens
por tipo (E/X/R/W) de cada método com o ground truth oficial por-movimento.

Métricas por tipo t e FP i (sobre contagens):
  match_i = min(pred_i, gt_i)  → micro: P = Σmatch/Σpred, R = Σmatch/Σgt,
  F1 = 2PR/(P+R); MAE_t = mean|pred_i − gt_i|; viés_t = mean(pred_i − gt_i).

Métodos: E33d (e33d_results.csv) e R1-COSMET (measurements por sistema).
Output: results/movement-level/movement_eval.csv + per_fp.csv + summary.md
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from shared.io_utils import extract_movements_from_csv

OUT = BASE / "results" / "movement-level"
OUT.mkdir(parents=True, exist_ok=True)

DS = BASE / "replications" / "cosmet-devito" / "datasets"
SYSTEMS = {
    "ALBERGATE": (DS / "MIS" / "ALBERGATE", "ALBERGATE-gpt-4-turbo-20260305T023001"),
    "K01719": (DS / "IoT_Telemedicine" / "K01719", "K01719-gpt-4-turbo-20260323T005355"),
    "K01720": (DS / "IoT_Telemedicine" / "K01720", "K01720-gpt-4-turbo-20260323T005553"),
    "K01726": (DS / "IoT_Telemedicine" / "K01726", "K01726-gpt-4-turbo-20260323T005703"),
    "U-CURE": (DS / "ML" / "U-CURE", "U-CURE-gpt-4-turbo-20260323T010152"),
    "AutomaticLineSwitching": (DS / "Real-time" / "AutomaticLineSwitching",
                               "AutomaticLineSwitching-gpt-4-turbo-20260323T010407"),
    "RiceCooker": (DS / "Real-time" / "RiceCooker", "RiceCooker-gpt-4-turbo-20260323T010514"),
}
TYPES = ["E", "X", "R", "W"]
DM_MAP = {"entry": "E", "exit": "X", "read": "R", "write": "W"}


def gt_counts(xlsx: Path) -> dict[str, int] | None:
    """Contagens GT por tipo a partir do XLSX oficial (soma da coluna CFP por
    DM normalizado — linhas 'counted only once' têm CFP=0 e não somam)."""
    try:
        df = pd.read_excel(xlsx)
    except Exception as e:
        print(f"  [warn] {xlsx.name}: {e}", file=sys.stderr)
        return None
    counts = {t: 0 for t in TYPES}
    for _, r in df.iterrows():
        dm = str(r.get("DM", "")).strip().lower()
        cfp = r.get("CFP")
        if pd.isna(cfp):
            continue
        for key, t in DM_MAP.items():
            if dm.startswith(key):
                counts[t] += int(cfp)
                break
    return counts


def main() -> None:
    # GT por FP
    gt: dict[tuple[str, str], dict] = {}
    for sysname, (root, _) in SYSTEMS.items():
        for xlsx in sorted((root / "ground truth").glob("FP*.xlsx")):
            c = gt_counts(xlsx)
            if c is not None:
                gt[(sysname, xlsx.stem)] = c
    print(f"GT por-movimento: {len(gt)} FPs")

    # E33d preds
    e33d: dict[tuple[str, str], dict] = {}
    with (BASE / "results/ner-ontology/e33d_results.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["system"], r["fp"])
            e33d[key] = {"E": int(r["E_pred"]), "R": int(r["R_pred"]),
                         "W": int(r["W_pred"]), "X": int(r["X_pred"])}

    # R1 preds (measurements por sistema)
    r1: dict[tuple[str, str], dict] = {}
    for sysname, (_, run_id) in SYSTEMS.items():
        meas = BASE / "results" / "cosmet" / run_id / "measurements"
        if not meas.exists():
            print(f"  [warn] sem measurements p/ {sysname}", file=sys.stderr)
            continue
        for fp_csv in meas.glob("*_measurement.csv"):
            num = re.match(r"(\d+)_", fp_csv.name)
            if not num:
                continue
            mv = extract_movements_from_csv(fp_csv)
            if mv:
                r1[(sysname, f"FP{num.group(1)}")] = {
                    "E": mv["entry"], "X": mv["exit"],
                    "R": mv["read"], "W": mv["write"]}

    methods = {"E33d": e33d, "R1-COSMET": r1}
    per_fp_rows, agg_rows = [], []
    for mname, preds in methods.items():
        common = sorted(k for k in gt if k in preds)
        print(f"{mname}: {len(common)} FPs pareados")
        sum_match = {t: 0 for t in TYPES}
        sum_pred = {t: 0 for t in TYPES}
        sum_gt = {t: 0 for t in TYPES}
        abs_err = {t: [] for t in TYPES}
        signed = {t: [] for t in TYPES}
        exact_decomp = 0
        exact_total_only = 0
        for key in common:
            g, p = gt[key], preds[key]
            all_types_ok = True
            for t in TYPES:
                sum_match[t] += min(p[t], g[t])
                sum_pred[t] += p[t]
                sum_gt[t] += g[t]
                abs_err[t].append(abs(p[t] - g[t]))
                signed[t].append(p[t] - g[t])
                if p[t] != g[t]:
                    all_types_ok = False
            tot_ok = sum(p.values()) == sum(g.values())
            if all_types_ok:
                exact_decomp += 1
            elif tot_ok:
                exact_total_only += 1
            per_fp_rows.append({
                "method": mname, "system": key[0], "fp": key[1],
                **{f"{t}_gt": g[t] for t in TYPES},
                **{f"{t}_pred": p[t] for t in TYPES},
                "total_gt": sum(g.values()), "total_pred": sum(p.values()),
                "decomp_exact": all_types_ok, "total_exact_only": (tot_ok and not all_types_ok),
            })
        for t in TYPES:
            P = sum_match[t] / sum_pred[t] if sum_pred[t] else float("nan")
            R = sum_match[t] / sum_gt[t] if sum_gt[t] else float("nan")
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            agg_rows.append({
                "method": mname, "movement": t, "n_fps": len(common),
                "gt_total": sum_gt[t], "pred_total": sum_pred[t],
                "precision": round(P, 4), "recall": round(R, 4),
                "f1": round(F1, 4),
                "mae_per_fp": round(float(np.mean(abs_err[t])), 4),
                "bias_per_fp": round(float(np.mean(signed[t])), 4),
            })
        agg_rows.append({
            "method": mname, "movement": "TOTAL-decomp",
            "n_fps": len(common), "gt_total": "", "pred_total": "",
            "precision": "", "recall": "",
            "f1": "",
            "mae_per_fp": f"decomp exata: {exact_decomp}/{len(common)}",
            "bias_per_fp": f"total exato c/ decomp errada: {exact_total_only}",
        })

    with (OUT / "per_fp.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fp_rows[0].keys()))
        w.writeheader(); w.writerows(per_fp_rows)
    with (OUT / "movement_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)

    print(f"\n{'método':10s} {'mov':>4s} {'P':>7s} {'R':>7s} {'F1':>7s} {'MAE':>6s} {'viés':>7s}")
    for r in agg_rows:
        if r["movement"] == "TOTAL-decomp":
            print(f"{r['method']:10s} {r['mae_per_fp']} | {r['bias_per_fp']}")
        else:
            print(f"{r['method']:10s} {r['movement']:>4s} {r['precision']:7.3f} "
                  f"{r['recall']:7.3f} {r['f1']:7.3f} {r['mae_per_fp']:6.3f} {r['bias_per_fp']:+7.3f}")
    print(f"\n-> {OUT}/movement_eval.csv + per_fp.csv")


if __name__ == "__main__":
    main()
