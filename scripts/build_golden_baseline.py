#!/usr/bin/env python3
"""Gera o golden baseline do motor INGLÊS: CFP por item, por dataset.

Estes arquivos são a rede de segurança do porte para português. Qualquer
alteração neles significa que o refactor mudou o comportamento em inglês,
o que invalidaria os números canônicos da dissertação.

Uso:
    .venv/bin/python scripts/build_golden_baseline.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ner_ontology_pipeline import load_mod, wrap_as_uc, fp_sort_key  # noqa: E402
from adapters.connextra_adapter import connextra_to_ucs  # noqa: E402

_PIPE = load_mod(
    "ner_ontology_pipeline",
    str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
)

GOLDEN_DIR = BASE / "tests" / "golden"


def devito_rows(pipeline) -> list[tuple[str, int]]:
    import hybrid_xai_runner as R
    rows = []
    for fp_id, text, _sf in R.iter_devito_fps(
        BASE / "replications" / "cosmet-devito" / "datasets", scope="all"
    ):
        rows.append((fp_id, pipeline.measure(text, fp_id=fp_id).cfp))
    return rows


def zenodo_rows(pipeline) -> list[tuple[str, int]]:
    fp_dir = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sample-300" / "FP_prompts"
    rows = []
    for f in sorted(fp_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem)):
        text = f.read_text(encoding="utf-8").strip()
        rows.append((f.stem, pipeline.measure(text, fp_id=f.stem, sentence_mode=True).cfp))
    return rows


def user_story_rows(pipeline) -> list[tuple[str, int]]:
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    rows = []
    for f in sorted(fp_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem)):
        raw = f.read_text(encoding="utf-8").strip()
        ucs = connextra_to_ucs(raw, f.stem) or [wrap_as_uc(raw, f.stem)]
        cfp = sum(pipeline.measure(uc, fp_id=f.stem).cfp for uc in ucs)
        rows.append((f.stem, cfp))
    return rows


def main() -> int:
    pipeline = _PIPE.NerOntologyPipeline(use_ner=False)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("devito", devito_rows),
                     ("zenodo300", zenodo_rows),
                     ("user_stories", user_story_rows)):
        rows = fn(pipeline)
        out = GOLDEN_DIR / f"{name}.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fp", "cfp"])
            w.writerows(rows)
        print(f"{name}: {len(rows)} itens -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
