"""Frame Connextra em português: reconhecimento e decomposição CRUD."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from adapters.connextra_adapter import connextra_to_ucs  # noqa: E402


def test_reconhece_frame_pt():
    ucs = connextra_to_ucs(
        "Como um gerente, quero consultar o relatório de vendas, "
        "para que eu possa acompanhar o desempenho.",
        "FP001", lang="pt",
    )
    assert ucs is not None and len(ucs) == 1


def test_frame_pt_sem_clausula_de_beneficio():
    ucs = connextra_to_ucs(
        "Como um analista, quero exportar os dados do cliente.",
        "FP002", lang="pt",
    )
    assert ucs is not None and len(ucs) == 1


def test_crud_enumerado_decompoe_em_processos():
    ucs = connextra_to_ucs(
        "Como administrador, quero criar, atualizar e excluir registros.",
        "FP003", lang="pt",
    )
    assert ucs is not None and len(ucs) == 3


def test_coordenacao_de_consulta_nao_decompoe():
    ucs = connextra_to_ucs(
        "Como um usuário, quero filtrar e ordenar a lista de pedidos.",
        "FP004", lang="pt",
    )
    assert ucs is not None and len(ucs) == 1


def test_texto_ingles_continua_funcionando():
    ucs = connextra_to_ucs(
        "As a manager, I want to view the sales report, so that I can track performance.",
        "FP005",
    )
    assert ucs is not None and len(ucs) == 1
