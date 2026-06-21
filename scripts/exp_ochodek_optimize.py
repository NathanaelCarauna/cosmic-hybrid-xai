#!/usr/bin/env python3
"""Otimização do estimador de NOMES DE UC (tarefa #36), mantendo:
  - INTERPRETABILIDADE: modelo linear, coeficientes no sentido COSMIC.
  - VALIDAÇÃO HELD-OUT: leave-one-project-out CV (sem tunar+testar no mesmo dado).

Engenharia de features COSMIC-fundamentada. Ideia central: o tamanho de um nome
de UC ≈ número de PROCESSOS FUNCIONAIS que ele implica:
  n_proc ≈ n_objetos × n_operacoes   (CRUD conta como 4 processos: C/R/U/D)
e CFP ≈ base + cfp_por_processo · n_proc (+ outros termos).

Compara conjuntos de features por CV MAE/SA e imprime coeficientes (p/ checar
sinais). Alvo: aproximar do R2 (especialista, MAE 3.591) sem perder explicação.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OCHODEK = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"

# Operações explícitas de 1 processo (verbo de ação único).
_OPS_1 = frozenset({
    "edit", "update", "modify", "change", "delete", "remove", "erase",
    "add", "create", "insert", "register", "new", "view", "show", "display",
    "list", "browse", "search", "find", "filter", "assign", "approve",
    "reject", "accept", "import", "export", "download", "upload", "send",
    "generate", "print", "calculate", "compute", "select", "configure",
    "define", "set", "manage", "handle", "process", "execute", "run",
})
# marcadores que NÃO são objeto de dados (verbos/operações)
_MARKERS = _OPS_1 | {"crud"}


def featurize(title: str) -> dict:
    toks = title.strip().lower().split()
    objs = [t for t in toks if t not in _MARKERS]
    n_obj = max(1, len(objs))
    has_crud = 1 if "crud" in toks else 0
    # operações explícitas presentes (exceto crud)
    n_ops_expl = sum(1 for t in toks if t in _OPS_1)
    # processos por "operação": crud=4; senão soma das ops explícitas (min 1)
    op_proc = 4 if has_crud else max(1, n_ops_expl)
    n_proc = n_obj * op_proc
    return {
        "n_tokens": len(toks), "n_obj": n_obj, "has_crud": has_crud,
        "n_ops_expl": n_ops_expl, "op_proc": op_proc, "n_proc": n_proc,
        "crud_x_obj": has_crud * n_obj, "objs": objs,
    }


def load():
    data = []
    for r in csv.DictReader(open(OCHODEK)):
        f = featurize(r["TitleTokens"])
        f["cfp"] = int(r["Cfp"]); f["proj"] = r["ProjectID"]
        data.append(f)
    return data


def evalset(data, cols):
    y = np.array([d["cfp"] for d in data], float)
    base_mae = np.abs(y - y.mean()).mean()
    projs = sorted({d["proj"] for d in data})
    pred = np.zeros(len(data))
    for h in projs:
        tr = [i for i, d in enumerate(data) if d["proj"] != h]
        te = [i for i, d in enumerate(data) if d["proj"] == h]
        X = np.array([[1.0] + [data[i][c] for c in cols] for i in tr])
        coef, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        for i in te:
            pred[i] = np.array([1.0] + [data[i][c] for c in cols]) @ coef
    yp = np.round(np.clip(pred, 1, None))
    mae = np.abs(yp - y).mean()
    sa = 1 - mae / base_mae
    rng = np.random.default_rng(42); idx = np.arange(len(y)); sas = []
    for _ in range(5000):
        s = rng.choice(idx, len(idx), True); b = np.abs(y[s] - y[s].mean()).mean()
        sas.append(1 - np.abs(yp[s] - y[s]).mean() / b)
    lo, hi = np.percentile(sas, [2.5, 97.5])
    Xa = np.array([[1.0] + [d[c] for c in cols] for d in data])
    ca, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return {"cols": cols, "mae": round(float(mae), 3), "sa": round(float(sa), 3),
            "sa_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "coef": [round(float(c), 3) for c in ca]}


def main():
    data = load()
    sets = [
        ["has_crud", "n_obj"],                 # atual (referência)
        ["n_proc"],                            # único feature COSMIC
        ["n_proc", "n_obj"],
        ["n_proc", "has_crud"],
        ["n_obj", "op_proc"],
        ["n_obj", "crud_x_obj"],
        ["n_proc", "n_tokens"],
        ["n_proc", "n_obj", "n_ops_expl"],
    ]
    res = [evalset(data, c) for c in sets]
    res.sort(key=lambda r: r["mae"])
    print(f"{'features':38s} {'MAE':>6s} {'SA':>7s}  IC95          coef[intercept,...]")
    for r in res:
        print(f"{str(r['cols']):38s} {r['mae']:6.3f} {r['sa']:+7.3f}  "
              f"[{r['sa_ci95'][0]:+.3f},{r['sa_ci95'][1]:+.3f}]  {r['coef']}")
    out = {"referencia_R2_home": 3.591, "atual_2feat": 4.568, "resultados": res}
    (BASE / "results" / "ner-ontology" / "ochodek_optimize.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dump das predições fora-de-fold (LOPO-CV) do estimador IMPLANTADO para nomes
    # de UC — feature set vencedor [n_obj, op_proc] — para alimentar a tabela
    # estatística canônica (build_final_statistical_table.py). É esta a configuração
    # citada na dissertação (matriz e leitura de IC), e não o pipeline simbólico
    # bruto que per_item_all.csv mantém para a taxonomia de erros.
    WIN = ["n_obj", "op_proc"]
    y = np.array([d["cfp"] for d in data], float)
    projs = sorted({d["proj"] for d in data})
    pred = np.zeros(len(data))
    for h in projs:
        tr = [i for i, d in enumerate(data) if d["proj"] != h]
        te = [i for i, d in enumerate(data) if d["proj"] == h]
        X = np.array([[1.0] + [data[i][c] for c in WIN] for i in tr])
        coef, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        for i in te:
            pred[i] = np.array([1.0] + [data[i][c] for c in WIN]) @ coef
    yp = np.round(np.clip(pred, 1, None))
    import csv as _csv
    per_item = BASE / "results" / "ner-ontology" / "ochodek_holdout_per_item.csv"
    with per_item.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["item_id", "y_true", "y_pred"])
        for i in range(len(data)):
            w.writerow([f"UC{i + 1}", int(y[i]), int(yp[i])])


if __name__ == "__main__":
    main()
