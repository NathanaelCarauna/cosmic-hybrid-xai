#!/usr/bin/env python3
"""Laço de ajuste do perfil português — RESTRITO AO DEV-20.

Reporta, para os 20 itens de desenvolvimento (kappa_subset das user stories),
o CFP medido contra o gabarito e quais movimentos caíram em regras de
fallback (sinal de léxico incompleto).

NUNCA aponte este script para o conjunto de teste (os 78 restantes) nem para
os datasets De Vito/Zenodo. Ajustar léxico olhando o teste invalida o
experimento de forma irreversível.

DESVIO DO BRIEF ORIGINAL (documentado, ver task-9-report.md): a versão do
brief chamava pipeline.measure() direto sobre o texto Connextra cru
("Como um Desenvolvedor, eu quero..."). Isso mede 0 CFP em TODOS os 20 itens
— _parse_uc não reconhece uma sentença Connextra sem cabeçalho UC (sem
"MAIN SCENARIO", sem linhas tab-separadas), então nenhum step é extraído e
nenhum léxico chega a ser exercitado, não importa o que se coloque nos
frozensets. O caminho real (mesmo usado por
`eval_ner_ontology_pipeline.py --dataset user-stories --adapter connextra`
para o inglês) é converter a US via `connextra_to_ucs(..., lang="pt")`
(scripts/adapters/connextra_adapter.py) em pseudo-UC(s) e então medir cada
uma, somando CFP quando o adaptador decompõe em >1 processo funcional
(regra 4.6, ex.: FP074 "criar, atualizar e excluir").

HISTÓRICO DE RODADAS (dev-20; ver task-9-report.md para o detalhe por item):
  r0 (léxicos vazios, script do brief sem adaptador, sem conectar ao
     connextra_to_ucs): CFP=0 em 20/20 — piso trivial e não informativo (ver
     nota acima sobre o desvio do brief); não comparável às rodadas
     seguintes, pois nenhum léxico chega a ser exercitado nesse caminho.
  r1 (léxicos amplos — tradução da amplitude inteira dos 6 léxicos + 9
     sub-léxicos ingleses, adaptador Connextra PT ligado):
     MAE dev-20 = 0.450, exatos = 13/20, fallback = 6/67 (9.0%).
  r2 (+ "esclarecer" em exit_verbs — FP004 caía em R14_SYS_FALLBACK_EXIT por
     coincidência, já batendo o GT; "esclarecer" ~ "clarify"/"explain" é
     tradução legítima para exit_verbs, não um hack para mudar o CFP):
     MAE dev-20 = 0.450 (inalterado — FP004 já batia via fallback), exatos =
     13/20, fallback = 5/67 (7.5%). Densidade de fallback caiu, MAE parado.
  PARADA em r2 (2 rodadas). Análise item-a-item dos 7 erros restantes (ver
  task-9-report.md) mostrou que NENHUM é lexicon-fixável sem violar a
  disciplina dev-only ou sem sobrepor decisões de outra camada:
    - FP013/FP064/FP065: verbo "ter" (have) — sem equivalente em NENHUM dos
      6 léxicos ingleses; deixado de propósito fora de qualquer léxico.
    - FP071/FP087/FP098: R7_AMBIG_INTERNAL/R7_AMBIG_PROCESS disparando como
      projetado — espelha EXATAMENTE a semântica do R7 inglês (verbos
      ambíguos sem marcador de persistência → sem movimento). Não é bug do
      léxico PT; forçar "verificar"/"gerenciar"/"realizar" para fora de
      ambiguous_verbs só para bater esses 3 itens quebraria a semântica R7
      e seria sobreajuste a 3 de 20 itens.
    - FP021: duplicação de "adicionar" (coordenação "X ou adicionar Y" que o
      connextra_adapter não absorve no objeto) gera 1 movimento extra —
      dívida do adaptador (Task 10), não do léxico.
    - FP038: o ramo genérico do adaptador PT (verbo fora de
      _WRITE_VERBS_PT/_QUERY_VERBS_PT) sempre gera 3 passos (Entry+ação+
      Exit); GT=4 exigiria um 4º movimento estrutural que o léxico não pode
      criar sozinho — dívida do adaptador (Task 10).
    - FP088: "disponibilizar" → R3_SYS_EXIT correto, mas
      inject_implicit_reads=False é o DEFAULT do pipeline (mesmo usado por
      eval_ner_ontology_pipeline.py para o inglês) — não é algo que o
      léxico PT controle.
    - FP048: "consulta" (que o léxico já cobre com AMBAS as formas
      consulta/consultar) é ocasionalmente marcado NOUN pelo
      pt_core_news_sm em frases com objeto plural/raro ("os referers... que
      usam links"), fazendo o extrator cair para o próximo VERBO da
      sentença ("usam"→usar, fora de qualquer léxico). Limitação do
      tagger do modelo pequeno, não do léxico (que já contém a forma
      correta) — ver task-9-report.md.

Uso:
    .venv/bin/python scripts/tune_pt_profile_on_dev.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def dev_ids() -> set[str]:
    gt = BASE / "datasets" / "user-stories-100-ground-truth-FINAL-v2.csv"
    us = {r["us_id"] for r in csv.DictReader(gt.open(newline="", encoding="utf-8"))
          if r["kappa_subset"] == "True"}
    fpmap = BASE / "results" / "cross-input" / "user-stories-98" / "ground_truth.csv"
    return {r["fp_id"] for r in csv.DictReader(fpmap.open(newline="", encoding="utf-8"))
            if r["us_id"] in us}


def measure_fp(pipeline, raw: str, fp_id: str) -> tuple[int, list]:
    """Mede um item via adaptador Connextra PT, somando sub-processos (regra
    4.6). Fallback para wrap_as_uc se o frame Connextra não casar (sentença
    fora do padrão "Como um X, eu quero...")."""
    ucs = connextra_to_ucs(raw, fp_id, lang="pt")
    if ucs is None:
        ucs = [wrap_as_uc(raw, fp_id)]
    cfp = 0
    movements = []
    for uc in ucs:
        res = pipeline.measure(uc, fp_id=fp_id)
        cfp += res.cfp
        movements.extend(res.movements)
    return cfp, movements


def main() -> int:
    ids = dev_ids()
    gt_path = BASE / "results" / "cross-input" / "user-stories-98" / "ground_truth.csv"
    gt = {r["fp_id"]: int(r["cfp_gt"])
          for r in csv.DictReader(gt_path.open(newline="", encoding="utf-8"))}

    pipeline = _PIPE.NerOntologyPipeline(use_ner=False, profile=_LP.PORTUGUESE)
    pt_dir = BASE / "datasets" / "pt-br" / "user-stories"

    erros, regras = [], Counter()
    print(f"{'item':8} {'pred':>4} {'gt':>4} {'erro':>5}  regras")
    for fp_id in sorted(ids):
        f = pt_dir / f"{fp_id}.txt"
        if not f.exists():
            print(f"{fp_id}: AUSENTE no corpus PT")
            continue
        raw = f.read_text(encoding="utf-8").strip()
        cfp, movements = measure_fp(pipeline, raw, fp_id)
        rs = [getattr(m, "rule_id", "?") or "?" for m in movements
              if m.movement_type is not None]
        regras.update(rs)
        e = cfp - gt[fp_id]
        erros.append(abs(e))
        marca = "".join("!" if r in FALLBACK else "." for r in rs)
        print(f"{fp_id:8} {cfp:4} {gt[fp_id]:4} {e:+5}  {marca}")

    if erros:
        print(f"\nMAE dev-20 = {sum(erros)/len(erros):.3f}  "
              f"exatos = {sum(1 for e in erros if e == 0)}/{len(erros)}")
    print("\nRegras mais frequentes:", regras.most_common(15))
    n_fb = sum(v for k, v in regras.items() if k in FALLBACK)
    n_total = sum(regras.values())
    print(f"Movimentos por fallback: {n_fb}/{n_total} "
          f"({n_fb/n_total*100 if n_total else 0:.1f}%, alto = léxico incompleto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
