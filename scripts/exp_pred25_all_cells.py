"""PRED(25) para todas as células que a tese declara reportar.

A Metodologia declara MAE, MMRE e PRED(25) como métricas de acurácia, e o protocolo
da validação externa promete explicitamente "MAE, SA e PRED(25)". Até aqui a PRED(25)
só aparecia na célula de histórias de usuário. Este experimento fecha a lacuna.

Definição usada é a da tese (fundamentacao.tex):

    PRED(25) = (1/n) * sum( 1[ |y_hat - y| / max(y,1) <= 0.25 ] )

O guarda `max(y,1)` importa: sem ele, itens de gabarito 0 (que existem no Zenodo)
dividiriam por zero. Com ele, um item de gabarito 0 só entra na PRED(25) se o erro
absoluto for <= 0.25, isto é, na prática, se a predição também for 0.

Saída: results/pred25/pred25_all_cells.{csv,md}
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_ITEM = ROOT / "results" / "error-analysis" / "per_item_all.csv"
EXTERNAL = ROOT / "results" / "cosmic-casestudies" / "external-validation-per-item.csv"
GT_V2 = ROOT / "datasets" / "user-stories-100-ground-truth-FINAL-v2.csv"
# O Web Advice Module é excluído do agregado externo por reportar tamanhos elevados a
# partir de texto esparso (Seção sec:val-externa-dados); o agregado da tese é n=53.
EXCLUIDO_DO_AGREGADO = {"Web Advice Module"}
FP_MAP = ROOT / "results" / "cross-input" / "user-stories-98" / "ground_truth.csv"
OUT_DIR = ROOT / "results" / "pred25"


def pred25(pairs: list[tuple[float, float]]) -> float:
    """Fração de predições dentro de 25% do gabarito, com o guarda max(y,1)."""
    if not pairs:
        return float("nan")
    hits = sum(1 for y, yhat in pairs if abs(yhat - y) / max(y, 1.0) <= 0.25)
    return hits / len(pairs)


def mmre(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    return sum(abs(yhat - y) / max(y, 1.0) for y, yhat in pairs) / len(pairs)


def dev_ids() -> set[str]:
    """Os 20 itens cujo gabarito foi visto na calibração do adaptador Connextra."""
    if not GT_V2.exists():
        return set()
    with GT_V2.open() as fh:
        return {r["us_id"] for r in csv.DictReader(fh) if r["kappa_subset"] == "True"}


def fp_to_us() -> dict[str, str]:
    """Mapa fp_id -> us_id, o mesmo usado por `build_final_statistical_table.py`."""
    if not FP_MAP.exists():
        return {}
    with FP_MAP.open(newline="") as fh:
        return {r["fp_id"]: r["us_id"] for r in csv.DictReader(fh)}


def main() -> None:
    rows: list[dict] = []

    with PER_ITEM.open() as fh:
        per_item = list(csv.DictReader(fh))

    cells: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in per_item:
        key = (r["method"], r["dataset"])
        cells.setdefault(key, []).append((float(r["y_true"]), float(r["y_pred"])))

    for (method, dataset), pairs in sorted(cells.items()):
        rows.append(
            {
                "celula": f"{method} / {dataset}",
                "n": len(pairs),
                "pred25": round(pred25(pairs), 4),
                "mmre": round(mmre(pairs), 4),
            }
        )

    # E34a restrito ao conjunto de teste congelado (78 itens), o número citado na tese.
    dev = dev_ids()
    fpmap = fp_to_us()
    if dev and fpmap:
        # A célula E34a vem da CONFIGURAÇÃO IMPLANTADA (NER desligado), e não de
        # `per_item_all.csv`, que preserva a configuração pré-otimização (com NER).
        # As duas divergem em um item, FP064 (3 contra 2), que cai no dev-20: a
        # PRED(25) do teste-78 é idêntica nas duas, mas a do conjunto de 98 não.
        with (ROOT / "results" / "ner-ontology" / "e34a_user_stories_results.csv").open(
                newline="") as fh:
            us = [{"item_id": r["fp"], "y_true": r["y_true"], "y_pred": r["y_pred"]}
                  for r in csv.DictReader(fh)]
        # A junção TEM de passar pelo mapa: `per_item_all.csv` numera como FP### e o
        # gabarito como US###, mas a correspondência NÃO é a troca de prefixo. Duas
        # histórias foram excluídas do meio da coleção, de modo que a numeração desloca
        # a partir de FP028 (FP028 -> US029). Trocar o prefixo devolve 79 itens em vez
        # de 78 e desloca 71 dos 98 pareamentos, sem erro visível.
        test = [
            (float(r["y_true"]), float(r["y_pred"]))
            for r in us
            if fpmap[r["item_id"]] not in dev
        ]
        assert len(test) == 78, f"esperado 78 itens de teste, obtido {len(test)}"
        rows.append(
            {
                "celula": "E34a / user-stories (teste-78)",
                "n": len(test),
                "pred25": round(pred25(test), 4),
                "mmre": round(mmre(test), 4),
            }
        )

    # Validação externa: a Metodologia promete PRED(25) e a tabela só trazia SA.
    with EXTERNAL.open() as fh:
        ext = list(csv.DictReader(fh))

    for arm in ("e33d", "r1"):
        by_base: dict[str, list[tuple[float, float]]] = {}
        for r in ext:
            by_base.setdefault(r["base"], []).append(
                (float(r["cfp_gt"]), float(r[arm]))
            )
        todos: list[tuple[float, float]] = []
        for base, pairs in sorted(by_base.items()):
            if base not in EXCLUIDO_DO_AGREGADO:
                todos.extend(pairs)
            rows.append(
                {
                    "celula": f"{arm.upper()} / externa {base}",
                    "n": len(pairs),
                    "pred25": round(pred25(pairs), 4),
                    "mmre": round(mmre(pairs), 4),
                }
            )
        rows.append(
            {
                "celula": f"{arm.upper()} / externa AGREGADO",
                "n": len(todos),
                "pred25": round(pred25(todos), 4),
                "mmre": round(mmre(todos), 4),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "pred25_all_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["celula", "n", "pred25", "mmre"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# PRED(25) e MMRE por célula",
        "",
        "Definição da tese: `|y_hat - y| / max(y, 1) <= 0,25`.",
        "",
        "| Célula | n | PRED(25) | MMRE |",
        "|---|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(f"| {r['celula']} | {r['n']} | {r['pred25']:.4f} | {r['mmre']:.4f} |")
    (OUT_DIR / "pred25_all_cells.md").write_text("\n".join(lines) + "\n")

    for r in rows:
        print(f"{r['celula']:44s} n={r['n']:5d}  PRED(25)={r['pred25']:.4f}  MMRE={r['mmre']:.4f}")


if __name__ == "__main__":
    main()
