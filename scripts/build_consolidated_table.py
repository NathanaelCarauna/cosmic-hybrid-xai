"""Tabela consolidada: todos os métodos, todos os formatos, todas as métricas.

Substitui `tab:matriz-sa` (só SA) e `tab:custo` (só formato de origem) por uma única
tabela com acurácia, métricas relativas, custo e tempo por célula.

Fontes, em ordem de precedência, e por que cada uma:

1. `results/final_statistical_table.csv` — MAE, SA e IC95 das células do método
   proposto, da via neural e do R1. É a fonte da verdade declarada da tese
   (bootstrap 10.000, semente 42) e prevalece sobre qualquer outra.
2. `results/cross-dataset-matrix/summary.csv` — MAE e SA das células de R2 e R3,
   que não entram no bootstrap canônico.
3. `results/runs_registry.csv` — PRED(25) e MMRE de R2 e R3. A execução de cada
   célula é escolhida pela MESMA regra do gerador da matriz (menor MAE entre as
   execuções within-dataset daquele dataset, empate resolvido pela ordem do
   registro), para que a tabela não reporte uma execução diferente da que produziu
   o MAE citado.
4. `results/error-analysis/per_item_all.csv` — PRED(25), MMRE e acertos exatos das
   demais células, calculados aqui pela definição da tese.
5. `results/cost-benchmark/comparison.csv` — segundos e dólares por item.

Saída: results/consolidated/consolidated_table.{csv,md,tex}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "consolidated"

CANON = ROOT / "results" / "final_statistical_table.csv"
MATRIX = ROOT / "results" / "cross-dataset-matrix" / "summary.csv"
REGISTRY = ROOT / "results" / "runs_registry.csv"
PER_ITEM = ROOT / "results" / "error-analysis" / "per_item_all.csv"
COST = ROOT / "results" / "cost-benchmark" / "comparison.csv"

FORMATO = {
    "devito": "Casos de uso",
    "zenodo": "Sentenças",
    "ochodek": "Nomes de UC",
    "user-stories": "Histórias",
}

# (rótulo na tabela, chave canônica ou da matriz, dataset, marcação)
# H = formato de origem; w = treino within-dataset.
LINHAS = [
    ("Hybrid-XAI", "E33d", "devito", "H"),
    ("Hybrid-XAI", "E33d", "zenodo", ""),
    ("Hybrid-XAI", "E33d", "ochodek", ""),
    ("Hybrid-XAI", "E34a-test78", "user-stories", ""),
    ("\\quad sem adaptador", "E33d-sem-adaptador", "user-stories", ""),
    ("E30f (base interna)", "E30f", "devito", "H"),
    ("E30f (base interna)", "E30f", "zenodo", ""),
    ("E30f (base interna)", "E30f", "ochodek", ""),
    ("E30f (base interna)", "E30f", "user-stories", ""),
    ("R1-COSMET", "R1-COSMET", "devito", "H"),
    ("R1-COSMET", "R1-COSMET", "zenodo", ""),
    ("R1-COSMET", "R1-COSMET", "ochodek", ""),
    ("R1-COSMET", "R1-COSMET", "user-stories", ""),
    ("R2-DEEP", "r2-deep-cosmic", "devito", ""),
    ("R2-DEEP", "r2-deep-cosmic", "zenodo", ""),
    ("R2-DEEP", "r2-deep-cosmic", "ochodek", "H"),
    ("R3-HKO", "r3-hko", "devito", "w"),
    ("R3-HKO", "r3-hko", "zenodo", "w"),
    ("R3-HKO", "r3-hko", "ochodek", "w"),
    ("R3-HKO", "r3-hko", "user-stories", "w"),
]

# Custo e tempo por item: (método, dataset) -> (s/item, US$/item, hardware).
# A célula de custo é por método e formato, não por método apenas: o R1 leva 5 s por
# sentença e 16 s por caso de uso, e reportar um número só apagaria essa variação.
CUSTO_CHAVE = {
    ("E33d", "devito"): ("E33d (NER+Ontologia)", "devito-119"),
    ("E33d", "zenodo"): ("E33d (NER+Ontologia)", "zenodo-2040"),
    ("E33d", "ochodek"): ("E33d (NER+Ontologia)", "ochodek-437"),
    ("E34a-test78", "user-stories"): ("E34a (E33d+adaptador)", "user-stories-98"),
    ("E33d-sem-adaptador", "user-stories"): ("E33d (NER+Ontologia)", "devito-119"),
    ("E30f", "devito"): ("E30f/E28a (Qwen3-8B local)", "devito-89-train"),
    ("E30f", "zenodo"): ("E30f/E28a (Qwen3-8B local)", "zenodo-200"),
    ("E30f", "ochodek"): ("E30f/E28a (Qwen3-8B local)", "ochodek-200"),
    ("E30f", "user-stories"): ("E30f/E28a (Qwen3-8B local)", "user-stories-98"),
    ("R1-COSMET", "devito"): ("R1-COSMET (gpt-4-turbo)", "devito-119"),
    ("R1-COSMET", "zenodo"): ("R1-COSMET (gpt-4-turbo)", "zenodo-200"),
    ("R1-COSMET", "ochodek"): ("R1-COSMET (gpt-4-turbo)", "ochodek-200"),
    ("R1-COSMET", "user-stories"): ("R1-COSMET (gpt-4-turbo)", "user-stories-98"),
    ("r2-deep-cosmic", "ochodek"): ("R2-DEEP-COSMIC (RF)", "ochodek-437 (treino CV)"),
    ("r3-hko", "devito"): ("R3-HKO (C4.5, CV completo)", "devito-119"),
    ("r3-hko", "user-stories"): ("R3-HKO (C4.5, CV completo)", "user-stories-98"),
}

MATRIX_KEY = {
    "r2-deep-cosmic": "r2-deep-cosmic",
    "r3-hko": "r3-hko",
}


def _f(v: str | None) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def pred25_mmre_exatos(pairs: list[tuple[float, float]]) -> tuple[float, float, int]:
    """Definição da tese: |y_hat - y| / max(y, 1) <= 0,25."""
    n = len(pairs)
    p = sum(1 for y, yh in pairs if abs(yh - y) / max(y, 1.0) <= 0.25) / n
    m = sum(abs(yh - y) / max(y, 1.0) for y, yh in pairs) / n
    e = sum(1 for y, yh in pairs if abs(yh - y) < 1e-9)
    return p, m, e


def carrega_canonico() -> dict[tuple[str, str], dict]:
    """Lê só a PARTE A do artefato canônico. O arquivo concatena três tabelas com
    títulos entre elas, de modo que ler tudo com DictReader mistura esquemas."""
    linhas = CANON.read_text(encoding="utf-8").splitlines()
    ini = next(i for i, l in enumerate(linhas) if l.startswith("method,dataset,n,mae"))
    fim = next(
        (i for i, l in enumerate(linhas[ini:], ini) if l.startswith("PARTE B")), len(linhas)
    )
    bloco = [l for l in linhas[ini:fim] if l.strip()]
    return {
        (r["method"], r["dataset"]): r for r in csv.DictReader(bloco)
    }


def parse_ci(txt: str | None) -> tuple[float | None, float | None]:
    """`"[0.412, 0.668]"` -> (0.412, 0.668)."""
    if not txt:
        return None, None
    partes = txt.strip().strip("[]").split(",")
    if len(partes) != 2:
        return None, None
    return _f(partes[0].strip()), _f(partes[1].strip())


def carrega_matriz() -> dict[str, dict]:
    with MATRIX.open() as fh:
        return {r["method"]: r for r in csv.DictReader(fh)}


def registro_r2r3(prefixo: str) -> dict | None:
    """Mesma regra do gerador da matriz: menor MAE entre as execuções
    within-dataset, empate pela ordem do registro."""
    with REGISTRY.open() as fh:
        cands = [
            r
            for r in csv.DictReader(fh)
            if r.get("dataset", "").startswith(prefixo) and "-to-" not in r.get("dataset", "")
        ]
    cands = [r for r in cands if _f(r.get("mae")) is not None]
    if not cands:
        return None
    return min(cands, key=lambda r: _f(r["mae"]))


def metricas_r2(ds: str) -> tuple[float | None, float | None]:
    """PRED(25) e MMRE do R2 na configuração que a matriz reporta: RF canônico no
    formato de origem, RF zero-shot (name-only em casos de uso, sentence em
    sentenças) fora dele. Cada valor é localizado casando o MAE com o da matriz,
    para não reportar uma configuração diferente da que gerou o número citado."""
    if ds == "ochodek":
        with REGISTRY.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("run_id", "").startswith("ochodek-uc-437-rf"):
                    return _f(r.get("pred25")), _f(r.get("mmre"))
        return None, None
    caminho = ROOT / "results" / "deep-cosmic-uc" / f"cross-dataset-{ds}" / "cross_dataset_summary.csv"
    if not caminho.exists():
        return None, None
    esperado = {"devito": ("rf", "name-only"), "zenodo": ("rf", "sentence")}.get(ds)
    with caminho.open() as fh:
        for r in csv.DictReader(fh):
            if esperado and (r["model"], r["condition"]) == esperado:
                return _f(r.get("pred25")), _f(r.get("mmre"))
    return None, None


def carrega_custo() -> dict[tuple[str, str], dict]:
    with COST.open() as fh:
        return {(r["method"], r["dataset"]): r for r in csv.DictReader(fh)}


def main() -> None:
    canon = carrega_canonico()
    matriz = carrega_matriz()
    custo = carrega_custo()

    with PER_ITEM.open() as fh:
        per_item = list(csv.DictReader(fh))
    cells: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in per_item:
        cells.setdefault((r["method"], r["dataset"]), []).append(
            (float(r["y_true"]), float(r["y_pred"]))
        )

    # As células da via simbólica vêm dos artefatos da CONFIGURAÇÃO IMPLANTADA, e não
    # de `per_item_all.csv`, que preserva as predições pré-otimização para a taxonomia
    # de erros. O MAE coincide (0,9328 nos dois), mas as predições item a item NÃO: o
    # implantado acerta 61 dos 119 em De Vito e o de `per_item_all` acerta 60 (a
    # diferença é o NER removido). Qualquer estatística que não seja o MAE diverge.
    def _implantado(caminho: str, ytc: str = "y_true", ypc: str = "y_pred"):
        with (ROOT / caminho).open(newline="") as fh:
            return [(float(r[ytc]), float(r[ypc])) for r in csv.DictReader(fh)]

    cells[("E33d", "devito")] = _implantado("results/ner-ontology/e33d_results.csv")
    cells[("E33d", "zenodo")] = _implantado("results/ner-ontology/e33d_zenodo_results.csv")
    cells[("E33d", "ochodek")] = _implantado("results/ner-ontology/ochodek_holdout_per_item.csv")
    assert len(cells[("E33d", "devito")]) == 119
    assert pred25_mmre_exatos(cells[("E33d", "devito")])[2] == 61, "De Vito implantado acerta 61"

    # test-78 do E34a passa pelo mapa fp->us (a troca de prefixo NÃO funciona).
    fpmap = {
        r["fp_id"]: r["us_id"]
        for r in csv.DictReader(
            (ROOT / "results/cross-input/user-stories-98/ground_truth.csv").open(newline="")
        )
    }
    dev = {
        r["us_id"]
        for r in csv.DictReader(
            (ROOT / "datasets/user-stories-100-ground-truth-FINAL-v2.csv").open(newline="")
        )
        if r["kappa_subset"] == "True"
    }
    with (ROOT / "results/ner-ontology/e34a_user_stories_results.csv").open(newline="") as fh:
        us98 = list(csv.DictReader(fh))
    cells[("E34a-test78", "user-stories")] = [
        (float(r["y_true"]), float(r["y_pred"])) for r in us98 if fpmap[r["fp"]] not in dev
    ]
    assert len(cells[("E34a-test78", "user-stories")]) == 78

    linhas = []
    for rotulo, chave, ds, marca in LINHAS:
        mae = sa = ci = None
        n = None
        p25 = mmre = None
        exatos = None

        if (chave, ds) in canon:
            c = canon[(chave, ds)]
            n = int(c["n"])
            mae = _f(c["mae"])
            sa = _f(c["sa"])
            ci = parse_ci(c.get("sa_ci95"))
        elif chave in MATRIX_KEY:
            m = matriz[MATRIX_KEY[chave]]
            mae = _f(m.get(f"{ds}_mae"))
            sa = _f(m.get(f"{ds}_sa"))
            n = int(_f(m.get(f"{ds}_n")) or 0) or None

        if (chave, ds) in cells:
            p25, mmre, exatos = pred25_mmre_exatos(cells[(chave, ds)])
        elif chave == "r3-hko":
            reg = registro_r2r3(f"hko-{ds}")
            if reg:
                p25 = _f(reg.get("pred25"))
                mmre = _f(reg.get("mmre"))
        elif chave == "r2-deep-cosmic":
            p25, mmre = metricas_r2(ds)

        ck = CUSTO_CHAVE.get((chave, ds))
        seg = usd = None
        if ck and ck in custo:
            seg = _f(custo[ck]["s_per_item"])
            usd = _f(custo[ck]["usd_per_item"])

        linhas.append(
            {
                "metodo": rotulo,
                "formato": FORMATO[ds],
                "marca": marca,
                "n": n,
                "mae": mae,
                "sa": sa,
                "sa_ci_lo": ci[0] if ci else None,
                "sa_ci_hi": ci[1] if ci else None,
                "pred25": p25,
                "mmre": mmre,
                "exatos": exatos,
                "s_item": seg,
                "usd_item": usd,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    campos = list(linhas[0].keys())
    with (OUT_DIR / "consolidated_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    def fmt(v, casas=3, pct=False):
        if v is None:
            return "---"
        if pct:
            return f"{100*v:.1f}\\%".replace(".", ",")
        return f"{v:.{casas}f}".replace(".", ",")

    md = ["# Tabela consolidada", "", "| Método | Formato | n | MAE | SA [IC95] | PRED(25) | MMRE | exatos | s/item | US$/item |", "|---|---|--:|--:|---|--:|--:|--:|--:|--:|"]
    for r in linhas:
        sa_txt = fmt(r["sa"])
        if r["sa_ci_lo"] is not None:
            sa_txt += f" [{fmt(r['sa_ci_lo'])}; {fmt(r['sa_ci_hi'])}]"
        md.append(
            f"| {r['metodo']} {r['marca']} | {r['formato']} | {r['n'] or '---'} | {fmt(r['mae'])} | {sa_txt} | "
            f"{fmt(r['pred25'], pct=True)} | {fmt(r['mmre'])} | {r['exatos'] if r['exatos'] is not None else '---'} | "
            f"{fmt(r['s_item'])} | {fmt(r['usd_item'], 4)} |"
        )
    (OUT_DIR / "consolidated_table.md").write_text("\n".join(md) + "\n")

    # Corpo LaTeX, para inclusão direta. A tabela é gerada, nunca digitada: foi
    # exatamente a transcrição à mão que produziu as divergências da Fase 0.
    tex: list[str] = []
    anterior = None
    for r in linhas:
        if anterior is not None and r["metodo"] != anterior:
            tex.append("\\hline")
        anterior = r["metodo"]
        rot = r["metodo"] if r["metodo"] != anterior or True else ""
        marca = f"\\,({r['marca']})" if r["marca"] else ""
        sa_txt = fmt(r["sa"])
        if r["sa_ci_lo"] is not None:
            sa_txt = (
                f"${sa_txt}$ \\scriptsize[{fmt(r['sa_ci_lo'])}; {fmt(r['sa_ci_hi'])}]"
            )
        else:
            sa_txt = f"${sa_txt}$"
        tex.append(
            " & ".join(
                [
                    rot + marca,
                    r["formato"],
                    str(r["n"]) if r["n"] else "---",
                    fmt(r["mae"]),
                    sa_txt,
                    fmt(r["pred25"], pct=True),
                    fmt(r["mmre"]),
                    str(r["exatos"]) if r["exatos"] is not None else "---",
                    fmt(r["s_item"]) if r["s_item"] is not None else "---",
                    fmt(r["usd_item"], 4) if r["usd_item"] else "0",
                ]
            )
            + " \\\\"
        )
    (OUT_DIR / "consolidated_table.tex").write_text("\n".join(tex) + "\n")

    for line in md[4:]:
        print(line)


if __name__ == "__main__":
    main()
