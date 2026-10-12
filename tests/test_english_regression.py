"""Regressão do motor inglês: igualdade EXATA contra o golden baseline.

Ver tests/golden/README.md. Falha aqui = o refactor mudou o inglês.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from build_golden_baseline import (  # noqa: E402
    _PIPE, devito_rows, zenodo_rows, user_story_rows,
)

GOLDEN_DIR = BASE / "tests" / "golden"


def load_golden(dataset: str) -> dict[str, int]:
    path = GOLDEN_DIR / f"{dataset}.csv"
    with path.open(newline="") as f:
        return {r["fp"]: int(r["cfp"]) for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def pipeline():
    return _PIPE.NerOntologyPipeline(use_ner=False)


@pytest.mark.parametrize("dataset,fn", [
    ("devito", devito_rows),
    ("zenodo300", zenodo_rows),
    ("user_stories", user_story_rows),
])
def test_english_cfp_unchanged(pipeline, dataset, fn):
    golden = load_golden(dataset)
    atual = dict(fn(pipeline))
    assert set(atual) == set(golden), (
        f"{dataset}: conjunto de itens mudou"
    )
    divergentes = {k: (golden[k], atual[k]) for k in golden if golden[k] != atual[k]}
    assert not divergentes, (
        f"{dataset}: {len(divergentes)} itens com CFP alterado "
        f"(golden, atual): {dict(list(divergentes.items())[:10])}"
    )
