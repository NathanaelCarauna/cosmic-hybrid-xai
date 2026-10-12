"""Explicabilidade da via simbólica (M3): trace serializável + diagrama de sequência.

Extraído de scripts/m3_generate_materials.py para ser importável pelo serviço HTTP
e pelos scripts. Opera sobre PipelineResult da via simbólica (ner_srl), NÃO sobre
RuleEngineResult da via neural (o gerador de xai.py é para aquele caminho).
"""
from __future__ import annotations


def trace_to_list(result) -> list[dict]:
    """Serializa o trace regra→trecho para o contrato JSON.

    Só movimentos efetivos (movement_type não-None) entram — os suprimidos por
    regra (ex.: R9_SELECT_UI) não são CFP e não têm nota no diagrama.
    """
    out = []
    for m in result.movements:
        if m.movement_type is None:
            continue
        out.append({
            "type": str(m.movement_type).split(".")[-1],
            "data_group": str(m.data_group),
            "rule_id": m.rule_id,
            "source_text": m.source_text,
            "reasoning": m.reasoning,
        })
    return out


def diagram_mermaid(result, label: str, actor: str = "Usuário") -> str:
    """Diagrama de SEQUÊNCIA gerado do trace real do pipeline (um processo).

    Compat: delega ao multi com uma lista de um. Saída idêntica à de antes
    (caminho da tese, que mede um UC por vez).
    """
    return diagram_mermaid_multi([result], label, actor)


def diagram_mermaid_multi(results, label: str, actor: str = "Usuário") -> str:
    """Diagrama de SEQUÊNCIA combinando 1+ processos funcionais.

    Mapeamento COSMIC (ISO 19761) → lifelines:
      Entry (E): Ator → Sistema; Exit (X): Sistema → Ator;
      Read (R): Armazenamento → Sistema; Write (W): Sistema → Armazenamento.
    Com autonumber, a numeração das mensagens É a contagem de CFP.

    Um item roteado por adaptador de formato (regra 4.6: CRUD coordenado) vira
    vários PipelineResult; os movimentos de todos entram num diagrama só e a
    numeração total é a soma dos CFP. Com um único resultado, a saída é
    byte-idêntica à do produto/tese anteriores.
    """
    label = " ".join(label.replace("\t", " — ").split())[:60]
    lines = ["sequenceDiagram",
             "    autonumber",
             f"    actor U as 👤 {actor}",
             "    participant S as ⚙️ Sistema",
             "    participant DB as 🗄️ Armazenamento",
             f"    Note over U,DB: 📄 {label}"]
    arrow = {
        "E": ("U", "S", "->>"),
        "X": ("S", "U", "->>"),
        "R": ("DB", "S", "-->>"),
        "W": ("S", "DB", "->>"),
    }
    total = 0
    for result in results:
        for m in result.movements:
            mt = str(m.movement_type).split(".")[-1]
            if mt not in arrow:
                continue
            src, dst, arr = arrow[mt]
            dg = str(m.data_group).replace('"', "'").strip() or "(grupo ambíguo)"
            dg = " ".join(dg.split())[:38]
            lines.append(f"    {src}{arr}{dst}: {mt} · {dg} · {m.rule_id}")
        total += result.cfp
    lines.append(f"    Note over U,DB: Σ {total} movimentos = CFP {total}")
    return "\n".join(lines)
