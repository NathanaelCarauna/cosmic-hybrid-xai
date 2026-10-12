#!/usr/bin/env python3
"""Avaliação única do porte português, com atribuição de causa por item.

Compara o CFP português contra o gabarito e contra o trace inglês do mesmo
item, classificando cada divergência em uma causa diagnosticável.

Uso:
    .venv/bin/python scripts/eval_pt_port.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.metrics import mae as MAE, standardized_accuracy as SA  # noqa: E402
from eval_ner_ontology_pipeline import load_mod, wrap_as_uc  # noqa: E402
from adapters.connextra_adapter import connextra_to_ucs  # noqa: E402

_PIPE = load_mod(
    "ner_ontology_pipeline",
    str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
)
_LP = load_mod(
    "language_profile",
    str(BASE / "models" / "hybrid-xai" / "ner_srl" / "language_profile.py"),
)

FALLBACK = {
    "R7_AMBIG_READ", "R7_AMBIG_INTERNAL", "R7_AMBIG_PROCESS",
    "R14_SYS_FALLBACK_READ", "R14_SYS_FALLBACK_EXIT", "R14_SYS_UNKNOWN",
    "R15_USER_FALLBACK", "R_DEFAULT",
}
SYSTEM_RULES = {"R3_SYS_REQUEST", "R3_SYS_EXIT", "R4_SYS_READ",
                "R5_SYS_WRITE", "R6_SYS_ENTRY"}


def atribuir(rs_en: list[str], rs_pt: list[str], adaptou_pt: bool,
             cfp_en: int | None = None, cfp_pt: int | None = None,
             y_true: int | None = None) -> str:
    """Classifica a causa provável de uma divergência.

    Antes de tentar atribuir a causa ao PORTE, separa erro HERDADO do motor:
    se o inglês já erra o MESMO valor que o português (cfp_pt == cfp_en !=
    y_true), o porte não teve chance nenhuma de acertar — o motor de
    ontologia erraria de qualquer forma, no idioma que fosse. Só quando o
    inglês acerta (ou erra para um valor DIFERENTE do português) é que a
    divergência é honestamente atribuível ao porte (traducao_ou_residual e
    as demais categorias abaixo). Requer os três argumentos opcionais; sem
    eles (chamada antiga), o comportamento anterior é preservado.
    """
    if cfp_en is not None and cfp_pt is not None and y_true is not None:
        if cfp_pt == cfp_en and cfp_en != y_true:
            return "motor_ingles_herdado"
    if not adaptou_pt:
        return "frame_connextra"
    if not rs_pt:
        return "rotulo_dependencia"
    if sum(1 for r in rs_pt if r in FALLBACK) > len(rs_pt) / 2:
        return "lexico"
    en_sys = sum(1 for r in rs_en if r in SYSTEM_RULES)
    pt_sys = sum(1 for r in rs_pt if r in SYSTEM_RULES)
    if en_sys > 0 and pt_sys == 0:
        return "system_names"
    return "traducao_ou_residual"


def _resumo(nome: str, rows: list[dict], ref: str, out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    yt = np.array([r["y_true"] for r in rows], float)
    yp = np.array([r["y_pred_pt"] for r in rows], float)
    print(f"{nome:28} n={len(rows):4}  MAE={MAE(yt, yp):.3f}  SA={SA(yt, yp):+.3f}")
    print(f"{'  referência inglesa':28} {ref}")
    # signed_error médio: revela viés direcional (super/sub-estimação
    # sistemática) que o abs_error, por construção, não distingue.
    vies = np.array([r["signed_error"] for r in rows], float).mean()
    print(f"{'  viés (signed_error médio)':28} {vies:+.3f}")
    causas = Counter(r["causa"] for r in rows if r["causa"])
    for causa, n in causas.most_common():
        print(f"    {causa:24} {n:3d}")
    print()


def avaliar_devito(pt, en, out_dir: Path) -> None:
    """De Vito: casos de uso. GT do próprio dataset, todos os 119."""
    import hybrid_xai_runner as R
    gt = R.load_devito_gt(scope="all")
    pt_dir = BASE / "datasets" / "pt-br" / "devito"
    rows = []
    for fp_id, texto_en, _sf in R.iter_devito_fps(
        BASE / "replications" / "cosmet-devito" / "datasets", scope="all"
    ):
        if fp_id not in gt:
            continue
        f = pt_dir / f"{fp_id.replace('/', '_')}.txt"
        if not f.exists():
            continue
        r_pt = pt.measure(f.read_text(encoding="utf-8").strip(), fp_id=fp_id)
        r_en = en.measure(texto_en, fp_id=fp_id)
        rs_pt = [getattr(m, "rule_id", "?") or "?" for m in r_pt.movements]
        rs_en = [getattr(m, "rule_id", "?") or "?" for m in r_en.movements]
        alvo = gt[fp_id]
        causa = "" if r_pt.cfp == alvo else atribuir(
            rs_en, rs_pt, True, cfp_en=r_en.cfp, cfp_pt=r_pt.cfp, y_true=alvo)
        rows.append({"fp": fp_id, "y_true": alvo, "y_pred_pt": r_pt.cfp,
                     "abs_error": abs(r_pt.cfp - alvo),
                     "signed_error": r_pt.cfp - alvo, "causa": causa})
    _resumo("DE VITO (PT)", rows, "MAE=0.933  SA=+0.555",
            out_dir / "devito_per_item.csv")


def avaliar_zenodo(pt, en, out_dir: Path) -> None:
    """Zenodo amostra-300: sentenças avulsas.

    A referência inglesa é RECOMPUTADA sobre os mesmos 300 — os valores
    canônicos (0,345 / +0,161) são sobre 2040 itens e não servem de
    comparação para este subconjunto.
    """
    gt_csv = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sample-300" / "ground_truth.csv"
    gt = {r["fp_file"].replace(".txt", ""): int(r["cfp_gt"])
          for r in csv.DictReader(gt_csv.open(newline="", encoding="utf-8"))}
    en_dir = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sample-300" / "FP_prompts"
    pt_dir = BASE / "datasets" / "pt-br" / "zenodo300"

    rows, yt_en, yp_en = [], [], []
    for fp_id, alvo in sorted(gt.items()):
        f_pt, f_en = pt_dir / f"{fp_id}.txt", en_dir / f"{fp_id}.txt"
        if not (f_pt.exists() and f_en.exists()):
            continue
        r_pt = pt.measure(f_pt.read_text(encoding="utf-8").strip(),
                          fp_id=fp_id, sentence_mode=True)
        r_en = en.measure(f_en.read_text(encoding="utf-8").strip(),
                          fp_id=fp_id, sentence_mode=True)
        yt_en.append(alvo); yp_en.append(r_en.cfp)
        rs_pt = [getattr(m, "rule_id", "?") or "?" for m in r_pt.movements]
        rs_en = [getattr(m, "rule_id", "?") or "?" for m in r_en.movements]
        causa = "" if r_pt.cfp == alvo else atribuir(
            rs_en, rs_pt, True, cfp_en=r_en.cfp, cfp_pt=r_pt.cfp, y_true=alvo)
        rows.append({"fp": fp_id, "y_true": alvo, "y_pred_pt": r_pt.cfp,
                     "abs_error": abs(r_pt.cfp - alvo),
                     "signed_error": r_pt.cfp - alvo, "causa": causa})

    a, b = np.array(yt_en, float), np.array(yp_en, float)
    ref = f"MAE={MAE(a, b):.3f}  SA={SA(a, b):+.3f}  (recomputada nos mesmos {len(a)})"
    _resumo("ZENODO-300 (PT)", rows, ref, out_dir / "zenodo300_per_item.csv")


def main() -> int:
    pt = _PIPE.NerOntologyPipeline(use_ner=False, profile=_LP.PORTUGUESE)
    en = _PIPE.NerOntologyPipeline(use_ner=False)

    gt_path = BASE / "results" / "cross-input" / "user-stories-98" / "ground_truth.csv"
    gt = {r["fp_id"]: int(r["cfp_gt"])
          for r in csv.DictReader(gt_path.open(newline="", encoding="utf-8"))}

    gt2 = BASE / "datasets" / "user-stories-100-ground-truth-FINAL-v2.csv"
    dev_us = {r["us_id"] for r in csv.DictReader(gt2.open(newline="", encoding="utf-8"))
              if r["kappa_subset"] == "True"}
    fpmap = {r["fp_id"]: r["us_id"]
             for r in csv.DictReader(gt_path.open(newline="", encoding="utf-8"))}

    out_dir = BASE / "results" / "pt-port"
    out_dir.mkdir(parents=True, exist_ok=True)

    avaliar_devito(pt, en, out_dir)
    avaliar_zenodo(pt, en, out_dir)

    rows, causas = [], Counter()
    pt_dir = BASE / "datasets" / "pt-br" / "user-stories"
    en_dir = BASE / "results" / "cross-input" / "user-stories-98"

    for fp_id in sorted(gt):
        if fpmap[fp_id] in dev_us:
            continue  # avaliação única: só o teste-78
        raw_pt = (pt_dir / f"{fp_id}.txt").read_text(encoding="utf-8").strip()
        raw_en = (en_dir / f"{fp_id}.txt").read_text(encoding="utf-8").strip()

        ucs_pt = connextra_to_ucs(raw_pt, fp_id, lang="pt")
        adaptou = ucs_pt is not None
        ucs_pt = ucs_pt or [wrap_as_uc(raw_pt, fp_id)]
        ucs_en = connextra_to_ucs(raw_en, fp_id) or [wrap_as_uc(raw_en, fp_id)]

        cfp_pt, rs_pt = 0, []
        for uc in ucs_pt:
            r = pt.measure(uc, fp_id=fp_id)
            cfp_pt += r.cfp
            rs_pt += [getattr(m, "rule_id", "?") or "?" for m in r.movements]
        cfp_en, rs_en = 0, []
        for uc in ucs_en:
            r = en.measure(uc, fp_id=fp_id)
            cfp_en += r.cfp
            rs_en += [getattr(m, "rule_id", "?") or "?" for m in r.movements]

        alvo = gt[fp_id]
        causa = "" if cfp_pt == alvo else atribuir(
            rs_en, rs_pt, adaptou, cfp_en=cfp_en, cfp_pt=cfp_pt, y_true=alvo)
        if causa:
            causas[causa] += 1
        rows.append({"fp": fp_id, "y_true": alvo, "y_pred_pt": cfp_pt,
                     "abs_error": abs(cfp_pt - alvo),
                     "signed_error": cfp_pt - alvo, "causa": causa})

    with (out_dir / "user_stories_per_item.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    yt = np.array([r["y_true"] for r in rows], float)
    yp = np.array([r["y_pred_pt"] for r in rows], float)
    print(f"USER STORIES teste-78 (PT)  n={len(rows)}  "
          f"MAE={MAE(yt, yp):.3f}  SA={SA(yt, yp):+.3f}")
    print("Referência inglesa:         n=78   MAE=0.500  SA=+0.261")
    vies = np.array([r["signed_error"] for r in rows], float).mean()
    print(f"Viés (signed_error médio):  {vies:+.3f}")
    print("\nAtribuição de causa das divergências:")
    for causa, n in causas.most_common():
        print(f"  {causa:24} {n:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
