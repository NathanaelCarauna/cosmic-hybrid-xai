#!/usr/bin/env python3
"""T1 (Sprint 3) — Consolida predições por-item de todas as células da matriz.

Saídas:
  results/error-analysis/per_item_all.csv  — method,dataset,item_id,y_true,
      y_pred,signed_error,abs_error,cfp_bucket,text
  results/error-analysis/top20-<method>-<dataset>.csv — 20 piores |erro|/célula

R2 não tem predições por-item persistidas (apenas agregados de fold) — fica de
fora do per-item; caracterização via confusion matrix/summary na taxonomia.
R3 idem (fold-level).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
OUT = BASE / "results" / "error-analysis"
OUT.mkdir(parents=True, exist_ok=True)


def bucket(cfp: float) -> str:
    if cfp <= 2:
        return "1-2"
    if cfp <= 4:
        return "3-4"
    if cfp <= 9:
        return "5-9"
    return "10+"


def load_texts() -> dict[tuple[str, str], str]:
    """(dataset, item_id) -> texto, das fontes disponíveis."""
    texts: dict[tuple[str, str], str] = {}
    # De Vito: manifest (id tipo "cosmet-devito-ALBERGATE-FP1"? checar col id)
    man = BASE / "datasets" / "manifest.csv"
    if man.exists():
        with man.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                src = r.get("source", "")
                if src.startswith("cosmet-devito"):
                    # id no formato PROJ/FPn usado pelos comparison: tenta ambos
                    texts[("devito", r.get("id", ""))] = r.get("text", "")
                elif src == "zenodo-heterogeneous":
                    texts[("zenodo-full", r.get("id", ""))] = r.get("text", "")
                elif src.startswith("ochodek"):
                    texts[("ochodek-437", r.get("id", ""))] = r.get("text", "")
    # cross-input dirs (FP*.txt)
    for ds, d in [("zenodo-200", "zenodo-200"), ("ochodek-200", "ochodek-200"),
                  ("user-stories", "user-stories-98")]:
        for fp in (BASE / "results" / "cross-input" / d).glob("FP*.txt"):
            texts[(ds, fp.stem)] = fp.read_text(encoding="utf-8").strip()
    # ochodek-200 usa ids O001 no GT csv
    gt200 = BASE / "datasets" / "cross-sample" / "ochodek-200-r1.csv"
    if gt200.exists():
        with gt200.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                texts[("ochodek-200-oid", r.get("id", ""))] = r.get("text", "")
    return texts


def rows_from(path: Path, method: str, dataset: str, idcol: str,
              tcol: str, pcol: str, textkey: str | None,
              texts: dict) -> list[dict]:
    out = []
    if not path.exists():
        print(f"  [skip] {path}", file=sys.stderr)
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                yt = float(r[tcol]); yp = float(r[pcol])
            except (KeyError, ValueError):
                continue
            iid = r.get(idcol, "")
            text = r.get("text", "") or (texts.get((textkey, iid), "") if textkey else "")
            out.append({
                "method": method, "dataset": dataset, "item_id": iid,
                "y_true": yt, "y_pred": yp,
                "signed_error": yp - yt, "abs_error": abs(yp - yt),
                "cfp_bucket": bucket(yt), "text": text[:300],
            })
    print(f"  [ok] {method}/{dataset}: {len(out)} itens")
    return out


def main() -> None:
    texts = load_texts()
    H = BASE / "results"
    all_rows: list[dict] = []

    # ---- E33d / E34a ----
    for ds, fn, tk in [("devito", "e33d_results.csv", "devito"),
                       ("zenodo", "e33d_zenodo_results.csv", "zenodo-full"),
                       ("ochodek", "e33d_ochodek_results.csv", "ochodek-437"),
                       ("user-stories", "e34a_user_stories_results.csv", "user-stories")]:
        method = "E34a" if "e34a" in fn else "E33d"
        all_rows += rows_from(H / "ner-ontology" / fn, method, ds,
                              "fp", "y_true", "y_pred", tk, texts)
    # E33d sem adaptador em US (para contraste do modo de falha)
    all_rows += rows_from(H / "ner-ontology" / "e33d_user_stories_results.csv",
                          "E33d-sem-adaptador", "user-stories",
                          "fp", "y_true", "y_pred", "user-stories", texts)

    # ---- E30f ----
    all_rows += rows_from(
        H / "hybrid-xai" / "holdout-final-devito" / "E30f-correction-ensemble" / "comparison_results.csv",
        "E30f", "devito", "fp", "y_true", "y_pred", "devito", texts)
    for ds, pat, tk in [("zenodo", "e30f-cross-zenodo-*", "zenodo-200"),
                        ("ochodek", "e30f-cross-ochodek-*", "ochodek-200-oid"),
                        ("user-stories", "e30f-cross-user-stories-20260610*", "user-stories")]:
        dirs = sorted((H / "hybrid-xai").glob(pat))
        if dirs:
            all_rows += rows_from(dirs[-1] / "comparison_results.csv",
                                  "E30f", ds, "fp_id", "y_true", "y_pred", tk, texts)

    # ---- R1 COSMET ----
    all_rows += rows_from(
        H / "cosmet" / "devito-119-combined-gpt-4-turbo-merged" / "comparison_results.csv",
        "R1-COSMET", "devito", "fp", "y_true", "y_pred", "devito", texts)
    for ds, pat, tk in [("zenodo", "zenodo-200-r1-*", "zenodo-200"),
                        ("ochodek", "ochodek-200-r1-*", "ochodek-200-oid"),
                        ("user-stories", "user-stories-98-r1-*", "user-stories")]:
        dirs = sorted((H / "cosmet").glob(pat))
        if dirs:
            all_rows += rows_from(dirs[-1] / "measurements" / "comparison_results.csv",
                                  "R1-COSMET", ds, "fp", "y_true", "y_pred", tk, texts)

    # ---- escreve consolidado ----
    fields = ["method", "dataset", "item_id", "y_true", "y_pred",
              "signed_error", "abs_error", "cfp_bucket", "text"]
    with (OUT / "per_item_all.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nConsolidado: {len(all_rows)} linhas -> {OUT/'per_item_all.csv'}")

    # ---- top-20 por célula ----
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in all_rows:
        cells.setdefault((r["method"], r["dataset"]), []).append(r)
    for (m, ds), rows in sorted(cells.items()):
        worst = sorted(rows, key=lambda r: -r["abs_error"])[:20]
        fn = OUT / f"top20-{m}-{ds}.csv"
        with fn.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(worst)
    print(f"Top-20 por célula: {len(cells)} arquivos")


if __name__ == "__main__":
    main()
