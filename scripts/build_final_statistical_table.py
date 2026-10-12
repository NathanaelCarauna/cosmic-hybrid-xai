#!/usr/bin/env python3
"""S4-T2 — Tabela estatística canônica da dissertação.

Parte A: para cada célula (método × dataset) com dados por-item:
  n, MAE [IC95 bootstrap], SA [IC95 bootstrap] (10k reamostragens, seed 42).
Parte B: comparações pareadas (mesmos itens): Wilcoxon signed-rank sobre
  |erro| + Cliff's δ. Pares: De Vito E33d×R1 (119, via movement-level),
  Zenodo-200 E30f×R1, User Stories E34a×{R1, E30f, E33d-sem-adaptador} (98).

Output: results/final_statistical_table.csv + .md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)


def boot_ci(values_fn, yt, yp, n=10000):
    """IC95 percentílico + a distribuição reamostrada que o produziu.

    A distribuição é devolvida para que o p unilateral da SA saia das MESMAS
    reamostragens do intervalo, sem consumir sorteios adicionais de `rng` — o
    que deslocaria todas as células seguintes e quebraria a reprodutibilidade
    byte a byte do artefato.
    """
    idx = rng.integers(0, len(yt), size=(n, len(yt)))
    vals = np.array([values_fn(yt[i], yp[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi, vals


def mae_fn(yt, yp):
    return np.abs(yt - yp).mean()


def sa_fn(yt, yp):
    mad = np.abs(yt - yt.mean()).mean()
    return 1 - np.abs(yt - yp).mean() / mad if mad > 0 else np.nan


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def magnitude(d):
    d = abs(d)
    if d < 0.147:
        return "negligível"
    if d < 0.33:
        return "pequeno"
    if d < 0.474:
        return "médio"
    return "grande"


def main() -> None:
    # ---- células por-item ----
    cells = defaultdict(list)
    with (BASE / "results/error-analysis/per_item_all.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            cells[(r["method"], r["dataset"])].append(
                (r["item_id"], float(r["y_true"]), float(r["y_pred"])))

    # Reconciliação das células-manchete da via simbólica com a CONFIGURAÇÃO
    # IMPLANTADA citada na dissertação (matriz e leitura de IC), em vez dos preds
    # pré-otimização que per_item_all.csv preserva para a taxonomia de erros:
    #   - Zenodo: pipeline com a regra R_DISPLAY_IMPLICIT_READ ativa (held-out),
    #     MAE 0,345 / SA 0,161 — fonte: results/ner-ontology/e33d_zenodo_results.csv;
    #   - Nomes de UC (Ochodek): estimador linear por-formato em LOPO-CV
    #     [n_obj, op_proc], MAE 4,41 / SA 0,067 — fonte: ochodek_holdout_per_item.csv;
    #   - Histórias de usuário (E34a): adaptador Connextra com o NER DESLIGADO,
    #     MAE 0,469 / SA 0,349 — fonte: e34a_user_stories_results.csv. Sem este
    #     override a célula vinha da configuração COM NER, que a arquitetura final
    #     não usa (§4.3) e que diverge em um item (FP064, 3 contra 2). O item cai no
    #     dev-20, de modo que a célula de teste-78 é idêntica nas duas configurações
    #     — foi por isso que a discrepância sobreviveu até a revisão final.
    def _load_deployed(path, idc, ytc, ypc, sufixo=True):
        # `sufixo` desambigua ids repetidos entre sistemas (Zenodo/Ochodek). Para o
        # E34a ele precisa ficar DESLIGADO: o id é a chave do mapa fp->us que
        # recorta o conjunto de teste logo abaixo, e o sufixo quebraria a junção.
        with (BASE / path).open(newline="") as fh:
            return [(r[idc] + (f"#{k}" if sufixo else ""), float(r[ytc]), float(r[ypc]))
                    for k, r in enumerate(csv.DictReader(fh))]

    cells[("E33d", "zenodo")] = _load_deployed(
        "results/ner-ontology/e33d_zenodo_results.csv", "fp", "y_true", "y_pred")
    cells[("E33d", "ochodek")] = _load_deployed(
        "results/ner-ontology/ochodek_holdout_per_item.csv", "item_id", "y_true", "y_pred")
    cells[("E34a", "user-stories")] = _load_deployed(
        "results/ner-ontology/e34a_user_stories_results.csv", "fp", "y_true", "y_pred",
        sufixo=False)

    # E34a test-78 (dev-20 excluído)
    gt2 = {r["us_id"]: r for r in csv.DictReader(
        (BASE / "datasets/user-stories-100-ground-truth-FINAL-v2.csv").open(newline=""))}
    fpmap = {r["fp_id"]: r["us_id"] for r in csv.DictReader(
        (BASE / "results/cross-input/user-stories-98/ground_truth.csv").open(newline=""))}
    dev_ids = {u for u, r in gt2.items() if r["kappa_subset"] == "True"}
    e34a_all = cells[("E34a", "user-stories")]
    cells[("E34a-test78", "user-stories")] = [
        x for x in e34a_all if fpmap[x[0]] not in dev_ids]
    # Mesma restrição aplicada ao competidor, para viabilizar o pareado sobre os 78
    # (Parte C). Fica FORA da família de Holm: é reanálise de um subconjunto do par
    # "E34a vs R1-COSMET" da Parte B, não uma hipótese independente.
    cells[("R1-COSMET-test78", "user-stories")] = [
        x for x in cells[("R1-COSMET", "user-stories")] if fpmap[x[0]] not in dev_ids]

    rows_a = []
    for (m, ds), items in sorted(cells.items()):
        yt = np.array([x[1] for x in items])
        yp = np.array([x[2] for x in items])
        mae = mae_fn(yt, yp); sa = sa_fn(yt, yp)
        mlo, mhi, _ = boot_ci(mae_fn, yt, yp)
        slo, shi, sa_vals = boot_ci(sa_fn, yt, yp)
        # p unilateral: fração das reamostragens em que SA <= 0. Qualifica as
        # células cujo IC bilateral cruza zero por pouco — é o valor que o texto
        # cita para a célula de histórias de usuário (E34a-test78).
        p_le0 = float(np.mean(sa_vals[~np.isnan(sa_vals)] <= 0))
        rows_a.append({
            "method": m, "dataset": ds, "n": len(items),
            "mae": round(mae, 3), "mae_ci95": f"[{mlo:.3f}, {mhi:.3f}]",
            "sa": round(sa, 3), "sa_ci95": f"[{slo:.3f}, {shi:.3f}]",
            "sa_sig": ("positivo" if slo > 0 else
                       "negativo" if shi < 0 else "indistinguível de 0"),
            "p_sa_le_0": round(p_le0, 4),
        })

    # ---- pares ----
    pairs = []

    # De Vito: E33d × R1 via movement-level per_fp (mesma chave system+fp)
    mv = defaultdict(dict)
    with (BASE / "results/movement-level/per_fp.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["system"], r["fp"])
            mv[key][r["method"]] = (int(r["total_gt"]), int(r["total_pred"]))
    common = [k for k, v in mv.items() if len(v) == 2]
    e1 = np.array([abs(mv[k]["E33d"][1] - mv[k]["E33d"][0]) for k in common], float)
    e2 = np.array([abs(mv[k]["R1-COSMET"][1] - mv[k]["R1-COSMET"][0]) for k in common], float)
    pairs.append(("E33d vs R1-COSMET", "devito (119)", e1, e2))

    # Zenodo-200 e User Stories: pareamento por item_id dentro do dataset
    def get_errs(method, ds, ids):
        d = {x[0]: abs(x[2] - x[1]) for x in cells[(method, ds)]}
        return np.array([d[i] for i in ids], float)

    for m1, m2, ds in [("E30f", "R1-COSMET", "zenodo"),
                       ("E34a", "R1-COSMET", "user-stories"),
                       ("E34a", "E30f", "user-stories"),
                       ("E34a", "E33d-sem-adaptador", "user-stories")]:
        ids1 = {x[0] for x in cells[(m1, ds)]}
        ids2 = {x[0] for x in cells[(m2, ds)]}
        ids = sorted(ids1 & ids2)
        if len(ids) < 10:
            continue
        pairs.append((f"{m1} vs {m2}", f"{ds} ({len(ids)})",
                      get_errs(m1, ds, ids), get_errs(m2, ds, ids)))

    rows_b = []
    raw_p = []
    for name, ds, a, b in pairs:
        diff = a - b
        if np.all(diff == 0):
            p = 1.0
        else:
            p = wilcoxon(a, b, zero_method="zsplit").pvalue
        d = cliffs_delta(a, b)
        raw_p.append(p)
        rows_b.append({
            "comparison": name, "dataset": ds,
            "mae_1": round(a.mean(), 3), "mae_2": round(b.mean(), 3),
            "wilcoxon_p": f"{p:.2e}", "cliffs_delta": round(d, 3),
            "magnitude": magnitude(d),
            "melhor": name.split(" vs ")[0] if a.mean() < b.mean()
                      else name.split(" vs ")[1],
        })

    # Correção de Holm-Bonferroni sobre a FAMÍLIA dos testes pareados (controla o
    # erro tipo I familiar das comparações múltiplas). Procedimento step-down:
    # ordena p crescente, p_ajustado_(k) = max_{j<=k} (m-j+1)*p_(j), cap em 1.0.
    m = len(raw_p)
    order = sorted(range(m), key=lambda i: raw_p[i])
    holm = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min((m - rank) * raw_p[i], 1.0)
        running = max(running, adj)  # monotonicidade do step-down
        holm[i] = running
    ALPHA = 0.05
    for i, r in enumerate(rows_b):
        r["holm_p"] = f"{holm[i]:.2e}"
        r["holm_sig"] = "sim" if holm[i] < ALPHA else "não"

    # ---- Parte C: robustez da manchete à exclusão dos itens de desenvolvimento ----
    # O adaptador Connextra foi calibrado observando o gabarito de 20 histórias
    # (kappa_subset). Este par repete a comparação-manchete apenas sobre os 78 itens
    # restantes. Calculado FORA da família de Holm (ver nota na criação da célula).
    rows_c = []
    ids78 = sorted({x[0] for x in cells[("E34a-test78", "user-stories")]}
                   & {x[0] for x in cells[("R1-COSMET-test78", "user-stories")]})
    if len(ids78) >= 10:
        a78 = get_errs("E34a-test78", "user-stories", ids78)
        b78 = get_errs("R1-COSMET-test78", "user-stories", ids78)
        p78 = wilcoxon(a78, b78, zero_method="zsplit").pvalue
        d78 = cliffs_delta(a78, b78)
        rows_c.append({
            "comparison": "E34a vs R1-COSMET (dev-20 excluído)",
            "dataset": f"user-stories ({len(ids78)})",
            "mae_1": round(a78.mean(), 3), "mae_2": round(b78.mean(), 3),
            "wilcoxon_p": f"{p78:.2e}", "cliffs_delta": round(d78, 3),
            "magnitude": magnitude(d78),
            "melhor": "E34a" if a78.mean() < b78.mean() else "R1-COSMET",
        })

    # ---- escreve ----
    out_csv = BASE / "results/final_statistical_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PARTE A — células com bootstrap IC95 (10k, seed 42)"])
        w.writerow(list(rows_a[0].keys()))
        for r in rows_a:
            w.writerow(list(r.values()))
        w.writerow([])
        w.writerow(["PARTE B — comparações pareadas (Wilcoxon |erro| + Cliff's δ)"])
        w.writerow(list(rows_b[0].keys()))
        for r in rows_b:
            w.writerow(list(r.values()))
        if rows_c:
            w.writerow([])
            w.writerow(["PARTE C — robustez (dev-20 excluído; FORA da família de Holm)"])
            w.writerow(list(rows_c[0].keys()))
            for r in rows_c:
                w.writerow(list(r.values()))

    md = ["# Tabela Estatística Canônica (Sprint 4, T2)", "",
          "Bootstrap percentílico 10k (seed 42); Wilcoxon signed-rank sobre |erro| por item; Cliff's δ (limiar Romano et al.).", "",
          "## A — Células (MAE e SA com IC95)", "",
          "`p(SA<=0)` é o p unilateral das mesmas reamostragens do IC: fração dos "
          "bootstraps com SA não-positiva. Qualifica as células cujo IC bilateral cruza zero.", "",
          "| Método | Dataset | n | MAE [IC95] | SA [IC95] | SA | p(SA<=0) |",
          "|--------|---------|--:|------------|-----------|----|---------:|"]
    for r in rows_a:
        md.append(f"| {r['method']} | {r['dataset']} | {r['n']} | "
                  f"{r['mae']} {r['mae_ci95']} | {r['sa']} {r['sa_ci95']} | {r['sa_sig']} | "
                  f"{r['p_sa_le_0']} |")
    md += ["", "## B — Comparações pareadas", "",
           "Wilcoxon signed-rank sobre |erro| por item; *p* (Holm) = valor-p ajustado "
           f"por Holm-Bonferroni sobre a família dos {len(rows_b)} testes pareados (α=0,05).", "",
           "| Comparação | Dataset | MAE₁ | MAE₂ | p (Wilcoxon) | p (Holm) | Sig. (Holm) | Cliff's δ | Magnitude | Melhor |",
           "|------------|---------|-----:|-----:|--------------|----------|-------------|----------:|-----------|--------|"]
    for r in rows_b:
        md.append(f"| {r['comparison']} | {r['dataset']} | {r['mae_1']} | {r['mae_2']} | "
                  f"{r['wilcoxon_p']} | {r['holm_p']} | {r['holm_sig']} | "
                  f"{r['cliffs_delta']} | {r['magnitude']} | {r['melhor']} |")
    if rows_c:
        md += ["", "## C — Robustez da manchete à exclusão dos itens de desenvolvimento", "",
               "O adaptador Connextra foi calibrado observando o gabarito de 20 histórias "
               "(`kappa_subset`). Esta comparação repete o par da Parte B apenas sobre os 78 "
               "itens restantes. **Fora da família de Holm**: é reanálise de um subconjunto do "
               "mesmo par, não hipótese independente.", "",
               "| Comparação | Dataset | MAE₁ | MAE₂ | p (Wilcoxon) | Cliff's δ | Magnitude | Melhor |",
               "|------------|---------|-----:|-----:|--------------|----------:|-----------|--------|"]
        for r in rows_c:
            md.append(f"| {r['comparison']} | {r['dataset']} | {r['mae_1']} | {r['mae_2']} | "
                      f"{r['wilcoxon_p']} | {r['cliffs_delta']} | {r['magnitude']} | {r['melhor']} |")
    (BASE / "results/final_statistical_table.md").write_text("\n".join(md) + "\n")

    print(f"A: {len(rows_a)} células | B: {len(rows_b)} pares (Holm sobre {len(rows_b)} testes)")
    for r in rows_b:
        print(f"  {r['comparison']:38s} {r['dataset']:20s} p={r['wilcoxon_p']} "
              f"p_holm={r['holm_p']} sig={r['holm_sig']} δ={r['cliffs_delta']} melhor={r['melhor']}")


if __name__ == "__main__":
    main()
