"""Estimador per-formato para NOMES DE CASO DE USO (formato Ochodek).

Substitui a antiga tabela palavra-chave->CFP (in-sample, não validada) por um
modelo LINEAR transparente, COSMIC-fundamentado, calibrado e validado com
disciplina dev/teste (leave-one-project-out CV; ver scripts/exp_ochodek_*.py e
docs/.../resultados-ochodek-performat-holdout.md).

  CFP ≈ base + w_proc · op_proc + w_obj · n_objetos

onde op_proc ESTIMA O NÚMERO DE PROCESSOS FUNCIONAIS que o nome implica:
  - CRUD ou verbo de gestão (manage/administer/maintain) => 4 processos
    (Create/Read/Update/Delete — regra 4.6 do protocolo de anotação);
  - senão, número de operações explícitas (edit, delete, view, search, ...);
  e w_proc ≈ 3.2 CFP por processo funcional (interpretação COSMIC direta).

EXPLICABILIDADE: nomes de UC NÃO carregam os movimentos de dados (um nome de
2-4 palavras não diz quais Entry/Exit/Read/Write ocorrem), então NÃO há trace
regra-a-trecho nem diagrama de sequência possível — é um limite de informação
do formato, não do modelo. A explicação desce ao NÍVEL DA OPERAÇÃO/PROCESSO:
como o modelo é linear, a contribuição de cada feature é EXATA e fiel por
construção (ao contrário de LIME).

Desempenho held-out (leave-one-project-out CV, 437 nomes / 27 projetos):
  MAE 4.190, SA +0.113 [0.060, 0.161] (significativamente positivo). Vence R1
  (6.7) e a via neural E30f (7.4); atrás do R2 (3.591, especialista treinado no
  formato, caixa-preta). Coeficientes abaixo: ajuste em todos os 437.
"""
from __future__ import annotations

# Coeficientes ajustados em todos os 437 nomes do Ochodek (modelo de produção).
# Recalibrar com os nomes de UC do próprio produto ao implantar.
_COEF = {"base": 3.968, "op_proc": 3.197, "obj": 0.416}

# Operações explícitas de 1 processo funcional.
_OPS_1 = frozenset({
    "edit", "update", "modify", "change", "delete", "remove", "erase",
    "add", "create", "insert", "register", "new", "view", "show", "display",
    "list", "browse", "search", "find", "filter", "assign", "approve",
    "reject", "accept", "import", "export", "download", "upload", "send",
    "generate", "print", "calculate", "compute", "select", "configure",
    "define", "set", "handle", "process", "execute", "run",
})
# Verbos de gestão que implicam CRUD (4 processos) — protocolo, regra 4.6.
_CRUD_IMPLIED = frozenset({"manage", "administer", "maintain"})
_MARKERS = _OPS_1 | _CRUD_IMPLIED | {"crud"}


def featurize(title: str) -> dict:
    toks = title.strip().lower().split()
    tset = set(toks)
    objects = [t for t in toks if t not in _MARKERS]
    n_obj = max(1, len(objects))
    crud_like = ("crud" in tset) or bool(_CRUD_IMPLIED & tset)
    n_ops_expl = sum(1 for t in toks if t in _OPS_1)
    op_proc = 4 if crud_like else max(1, n_ops_expl)
    op_label = ("CRUD/gestão" if crud_like
                else (", ".join(t for t in toks if t in _OPS_1) or toks[0] if toks else ""))
    return {
        "tokens": toks, "operation": toks[0] if toks else "", "op_label": op_label,
        "objects": objects, "n_obj": n_obj, "crud_like": crud_like,
        "n_ops_expl": n_ops_expl, "op_proc": op_proc,
    }


def estimate(title: str) -> dict:
    """Retorna {cfp, contributions, explanation, features}. cfp é a soma EXATA
    das contribuições lineares (fiel por construção), arredondada e ≥ 1."""
    f = featurize(title)
    contrib = {
        "base": _COEF["base"],
        f"processos funcionais (×{f['op_proc']})": _COEF["op_proc"] * f["op_proc"],
        f"objetos (×{f['n_obj']})": _COEF["obj"] * f["n_obj"],
    }
    cfp = max(1, round(sum(contrib.values())))

    proc_txt = (f"{f['op_proc']} processos funcionais "
                + ("(padrão CRUD/gestão: Create/Read/Update/Delete)"
                   if f["crud_like"] else f"({f['op_label']})"))
    explanation = (
        f"Nome de caso de uso ('{f['operation']}'): implica {proc_txt} × "
        f"{_COEF['op_proc']:.1f} CFP/processo + {_COEF['obj']:.1f}×{f['n_obj']} "
        f"objeto(s)" + (f" [{', '.join(f['objects'])}]" if f["objects"] else "")
        + f" + base {_COEF['base']:.1f} = CFP ≈ {cfp}. "
        "Sem trace regra-a-trecho (o nome não carrega os movimentos de dados); "
        "explicação fiel ao nível da operação (modelo linear, contribuição exata)."
    )
    return {"cfp": cfp, "contributions": contrib, "explanation": explanation, "features": f}
