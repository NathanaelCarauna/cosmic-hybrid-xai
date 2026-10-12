"""Módulo de explicabilidade: trace serializável + diagrama, sobre a via simbólica."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from eval_ner_ontology_pipeline import load_mod, wrap_as_uc  # noqa: E402

_PIPE = load_mod("ner_ontology_pipeline",
                 str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
_LP = load_mod("language_profile",
               str(BASE / "models" / "hybrid-xai" / "ner_srl" / "language_profile.py"))
_XAI = load_mod("xai_symbolic", str(BASE / "models" / "hybrid-xai" / "xai_symbolic.py"))


def _measure(text):
    # wrap_as_uc dá a estrutura mínima de UC (ID/Description/MAIN SCENARIO)
    # que o parser exige — measure() não aceita sentença nua sem cabeçalho,
    # mesmo com sentence_mode=True (confirmado empiricamente: sem isso, o
    # parser não encontra nenhum step e o trace fica vazio).
    pipe = _PIPE.NerOntologyPipeline(use_ner=False, profile=_LP.PORTUGUESE)
    uc = wrap_as_uc(text, "T1")
    return pipe.measure(uc, fp_id="T1", sentence_mode=True)


def test_trace_to_list_estrutura():
    res = _measure("O sistema exibe os dados do cliente.")
    trace = _XAI.trace_to_list(res)
    assert isinstance(trace, list) and trace, "trace vazio"
    m = trace[0]
    assert set(m) >= {"type", "data_group", "rule_id", "source_text", "reasoning"}
    assert m["type"] in ("E", "X", "R", "W")  # movimentos suprimidos não entram
    assert isinstance(m["rule_id"], str) and m["rule_id"]


def test_diagram_mermaid_bem_formado():
    res = _measure("O sistema exibe os dados do cliente.")
    d = _XAI.diagram_mermaid(res, label="Ver dados do cliente")
    assert d.startswith("sequenceDiagram")
    assert "autonumber" in d
    assert f"CFP {res.cfp}" in d  # a nota final cita a contagem


def test_trace_ignora_movimentos_suprimidos():
    # movimentos com movement_type None (regras de supressão) não entram no trace
    res = _measure("O sistema exibe os dados do cliente.")
    for m in _XAI.trace_to_list(res):
        assert m["type"] is not None


def test_diagram_mermaid_single_delega_ao_multi():
    # a compat de um processo é byte-idêntica ao multi com lista de um
    res = _measure("O sistema exibe os dados do cliente.")
    assert (_XAI.diagram_mermaid(res, label="Ver dados")
            == _XAI.diagram_mermaid_multi([res], label="Ver dados"))


def test_diagram_mermaid_multi_soma_cfp():
    # dois processos → um diagrama só; a nota final soma o CFP total
    a = _measure("O sistema exibe os dados do cliente.")
    b = _measure("O usuário cadastra um novo pedido.")
    d = _XAI.diagram_mermaid_multi([a, b], label="Dois processos")
    assert d.startswith("sequenceDiagram")
    assert f"CFP {a.cfp + b.cfp}" in d
