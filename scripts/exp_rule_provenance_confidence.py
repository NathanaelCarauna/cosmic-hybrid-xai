#!/usr/bin/env python3
"""E44 — Proveniência de regra como sinal de confiança (custo financeiro ZERO).

Hipótese: as 26 regras da ontologia não são equivalentes em certeza. Algumas
disparam com evidência sintática clara (R4_SYS_READ, R5_SYS_WRITE, R10_USER_ENTRY);
outras são a ontologia ADMITINDO que não sabe (R7_AMBIG_*, R14_*_FALLBACK,
R15_USER_FALLBACK, R_DEFAULT). Se a densidade de regras-de-baixa-certeza previr o
erro, obtém-se um escore de confiança de graça — e que é ele próprio explicável
("confiança baixa porque 3 dos 5 movimentos vieram de regras de fallback").

Isso é hibridização sem custo: nenhum modelo novo, nenhuma chamada de API, nenhum
treino. O dado já é computado a cada medição e hoje é descartado.

Uso:
    .venv/bin/python scripts/exp_rule_provenance_confidence.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ner_ontology_pipeline import load_mod, wrap_as_uc, fp_sort_key  # noqa: E402
from adapters.connextra_adapter import connextra_to_ucs  # noqa: E402

_PIPE_MOD = load_mod(
    "ner_ontology_pipeline",
    str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
)

# Regras que representam admissão de incerteza pela própria ontologia.
LOW_CONFIDENCE = {
    "R7_AMBIG_READ", "R7_AMBIG_INTERNAL", "R7_AMBIG_PROCESS",
    "R14_SYS_FALLBACK_READ", "R14_SYS_FALLBACK_EXIT", "R14_SYS_UNKNOWN",
    "R15_USER_FALLBACK", "R_DEFAULT",
}


def main() -> int:
    # use_ner=False: a ablação mostrou o NER decorativo (achado-ner-decorativo.md).
    pipeline = _PIPE_MOD.NerOntologyPipeline(use_ner=False)
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = {r["fp_id"]: int(r["cfp_gt"])
          for r in csv.DictReader((fp_dir / "ground_truth.csv").open(newline="", encoding="utf-8"))}

    rows = []
    rule_counter: Counter[str] = Counter()
    for fp_file in sorted(fp_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem)):
        fp_id = fp_file.stem
        if fp_id not in gt:
            continue
        raw = fp_file.read_text(encoding="utf-8").strip()
        ucs = connextra_to_ucs(raw, fp_id) or [wrap_as_uc(raw, fp_id)]

        movements = []
        for uc in ucs:
            movements.extend(pipeline.measure(uc, fp_id=fp_id).movements)

        rules = [getattr(m, "rule_id", "?") or "?" for m in movements]
        rule_counter.update(rules)
        n_low = sum(1 for r in rules if r in LOW_CONFIDENCE)
        y_pred, y_true = len(movements), gt[fp_id]
        rows.append({
            "fp": fp_id, "y_true": y_true, "y_pred": y_pred,
            "abs_error": abs(y_pred - y_true),
            "n_mov": len(movements), "n_low": n_low,
            "low_frac": n_low / len(movements) if movements else 0.0,
        })

    out = BASE / "results" / "rule-provenance"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "user_stories_provenance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    err = np.array([r["abs_error"] for r in rows], float)
    low = np.array([r["low_frac"] for r in rows], float)
    print(f"n={len(rows)}  MAE global={err.mean():.3f}")
    print(f"\nRegras mais frequentes: {rule_counter.most_common(8)}")

    clean = [r for r in rows if r["n_low"] == 0]
    dirty = [r for r in rows if r["n_low"] > 0]
    for lbl, g in (("SEM regra de fallback", clean), ("COM regra de fallback", dirty)):
        if not g:
            print(f"\n{lbl}: (vazio)")
            continue
        e = np.array([r["abs_error"] for r in g], float)
        exact = (e == 0).mean()
        print(f"\n{lbl}: n={len(g):3d} ({len(g)/len(rows):5.1%})  "
              f"MAE={e.mean():.3f}  exatos={exact:.1%}")

    if low.std() > 0:
        from scipy.stats import spearmanr
        rho, p = spearmanr(low, err)
        print(f"\nSpearman(fração de fallback, |erro|) = {rho:+.3f}  p={p:.3g}")
    else:
        print("\nfração de fallback é constante — sem sinal discriminante")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
