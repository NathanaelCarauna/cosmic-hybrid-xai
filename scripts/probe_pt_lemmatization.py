#!/usr/bin/env python3
"""Sonda: o lematizador do pt_core_news_sm reduz formas conjugadas ao lema?

Os léxicos de verbos da ontologia casam por LEMA. A morfologia verbal do
português é muito mais rica que a inglesa; se a lematização falhar, o léxico
não casa e a regra não dispara — falha silenciosa.

Uso:
    .venv/bin/python scripts/probe_pt_lemmatization.py
"""
from __future__ import annotations

import spacy

# Verbos candidatos ao léxico português, com formas conjugadas frequentes em
# texto de requisito (3a pessoa presente, 3a pessoa passado, gerúndio, futuro).
CASOS: dict[str, list[str]] = {
    "exibir":    ["exibe", "exibiu", "exibindo", "exibirá"],
    "mostrar":   ["mostra", "mostrou", "mostrando", "mostrará"],
    "informar":  ["informa", "informou", "informando", "informará"],
    "consultar": ["consulta", "consultou", "consultando", "consultará"],
    "buscar":    ["busca", "buscou", "buscando", "buscará"],
    "gravar":    ["grava", "gravou", "gravando", "gravará"],
    "salvar":    ["salva", "salvou", "salvando", "salvará"],
    "cadastrar": ["cadastra", "cadastrou", "cadastrando", "cadastrará"],
    "atualizar": ["atualiza", "atualizou", "atualizando", "atualizará"],
    "excluir":   ["exclui", "excluiu", "excluindo", "excluirá"],
    "selecionar":["seleciona", "selecionou", "selecionando", "selecionará"],
    "enviar":    ["envia", "enviou", "enviando", "enviará"],
    "validar":   ["valida", "validou", "validando", "validará"],
    "registrar": ["registra", "registrou", "registrando", "registrará"],
}


def main() -> int:
    nlp = spacy.load("pt_core_news_sm")
    total = acertos = 0
    falhas: list[tuple[str, str, str]] = []
    for lema, formas in CASOS.items():
        # Contexto mínimo de frase: lematizadores dependem de POS.
        for forma in formas:
            doc = nlp(f"O sistema {forma} os dados do cliente.")
            tok = next((t for t in doc if t.text == forma), None)
            obtido = tok.lemma_.lower() if tok is not None else "<não encontrado>"
            total += 1
            if obtido == lema:
                acertos += 1
            else:
                falhas.append((forma, lema, obtido))

    print(f"Taxa de lematização: {acertos}/{total} = {acertos/total:.1%}\n")
    if falhas:
        print("Falhas (forma -> esperado / obtido):")
        for forma, esperado, obtido in falhas:
            print(f"  {forma:16} -> {esperado:12} / {obtido}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
