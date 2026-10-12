#!/usr/bin/env python3
"""Verifica integridade estrutural do corpus traduzido (não a qualidade).

Confere: contagem por dataset, correspondência 1-1 de identificadores contra a
origem, ausência de arquivos vazios, e — nos casos de uso — presença dos
rótulos de seção traduzidos, sem os quais o parser não segmenta o documento.

Qualidade de tradução é revisão humana (amostra de 30, protocolo do spec),
não escopo deste script.

Uso:
    .venv/bin/python scripts/verify_translation_corpus.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PT = BASE / "datasets" / "pt-br"

# (nome, glob destino, n esperado). A origem dos ids é resolvida em _source_ids.
DATASETS = [
    ("user-stories", "FP*.txt", 98),
    ("zenodo300", "FP*.txt", 300),
    ("devito", "*.txt", 119),
]


def _source_ids(ds: str) -> set[str]:
    """Nomes de arquivo esperados por dataset, a partir da origem canônica.

    De Vito não tem diretório-fonte plano: os ids vêm do iterador
    iter_devito_fps (com '/' trocado por '_'), a mesma regra usada na tradução.
    """
    if ds == "user-stories":
        d = BASE / "results" / "cross-input" / "user-stories-98"
        return {p.name for p in d.glob("FP*.txt")}
    if ds == "zenodo300":
        d = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sample-300" / "FP_prompts"
        return {p.name for p in d.glob("FP*.txt")}
    if ds == "devito":
        import sys as _sys
        _sys.path.insert(0, str(BASE / "scripts"))
        import hybrid_xai_runner as R
        root = BASE / "replications" / "cosmet-devito" / "datasets"
        return {f"{fp.replace('/', '_')}.txt" for fp, _t, _s in R.iter_devito_fps(root, scope="all")}
    return set()
# Comparados em minúsculas: os templates do De Vito são heterogêneos — ALBERGATE
# usa "CENÁRIO PRINCIPAL" (caixa alta), os de IoT/ML usam "Cenário principal".
ROTULOS_UC = ("cenário principal", "fluxo principal")


def main() -> int:
    falhas: list[str] = []
    for ds, glob, n in DATASETS:
        dst = PT / ds
        if not dst.is_dir():
            falhas.append(f"{ds}: diretório de destino ausente")
            continue

        src_ids = _source_ids(ds)
        dst_ids = {p.name for p in dst.glob("*.txt")}

        if len(dst_ids) != n:
            falhas.append(f"{ds}: {len(dst_ids)} traduções, esperado {n}")

        faltando = src_ids - dst_ids
        if faltando:
            falhas.append(f"{ds}: {len(faltando)} sem tradução: {sorted(faltando)[:5]}")
        sobrando = dst_ids - src_ids
        if sobrando:
            falhas.append(f"{ds}: {len(sobrando)} tradução(ões) sem origem: {sorted(sobrando)[:5]}")

        vazios = [p.name for p in dst.glob("*.txt") if not p.read_text(encoding="utf-8").strip()]
        if vazios:
            falhas.append(f"{ds}: {len(vazios)} arquivos vazios: {vazios[:5]}")

        if ds == "devito":
            sem_rotulo = [
                p.name for p in dst.glob("*.txt")
                if not any(r in p.read_text(encoding="utf-8").lower() for r in ROTULOS_UC)
            ]
            if sem_rotulo:
                falhas.append(
                    f"devito: {len(sem_rotulo)} sem rótulo de seção traduzido: {sem_rotulo[:5]}"
                )

        # Âncora de conteúdo (Zenodo): a linha REQ-<n> deve conter o mesmo número
        # do nome FP<n>. Pega arquivo cujo conteúdo veio de outra fonte — falha
        # que o mapeamento de nome não detecta (colisão de scratchpad entre lotes).
        if ds == "zenodo300":
            trocados = []
            for p in dst.glob("*.txt"):
                num = re.match(r"FP(\d+)\.txt", p.name)
                if not num:
                    continue
                m = re.search(r"REQ-(\d+)", p.read_text(encoding="utf-8"))
                if m and m.group(1) != num.group(1):
                    trocados.append(f"{p.name}(REQ-{m.group(1)})")
            if trocados:
                falhas.append(
                    f"zenodo300: {len(trocados)} com conteúdo de fonte trocada: {trocados[:5]}"
                )

    if falhas:
        print("FALHAS DE INTEGRIDADE:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("Corpus íntegro: 98 + 300 + 119 = 517 itens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
