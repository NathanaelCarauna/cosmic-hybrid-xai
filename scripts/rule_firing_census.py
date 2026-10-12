#!/usr/bin/env python3
"""Censo de alcançabilidade das regras do M2 sobre o corpus de avaliação.

Conta quantas vezes cada `rule_id` aparece no trace da via simbólica (E33d),
por formato, e lista as regras que nunca disparam. É o artefato que sustenta a
afirmação de alcançabilidade da Seção 4.3 da dissertação — antes ela era um
número transcrito à mão, sem script que a regerasse.

O trace inclui as decisões de NÃO-movimento (rule_id com movement_type=None):
elas são parte da explicação e contam como disparo da regra.

Uso:
    python3 scripts/rule_firing_census.py
Saída:
    results/rule-provenance/rule_firing_census.csv
    results/rule-provenance/rule_firing_census.md
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "models" / "hybrid-xai"))
sys.path.insert(0, str(BASE / "scripts"))

OUT_DIR = BASE / "results" / "rule-provenance"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


evalmod = _load("eval_ner_ontology_pipeline",
                BASE / "scripts" / "eval_ner_ontology_pipeline.py")
from ner_srl.ner_ontology_pipeline import NerOntologyPipeline  # noqa: E402


def declared_rule_ids() -> tuple[list[str], list[str]]:
    """Extrai os identificadores declarados no código, sem executá-lo.

    Classificação = os que aparecem em `cosmic_ontology.py` dentro de `_apply_rules`
    (mk/mk_none); pós-processamento = os `rule_id=` construídos fora dela.
    """
    src = (BASE / "models" / "hybrid-xai" / "ner_srl" / "cosmic_ontology.py").read_text()
    pipe = (BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py").read_text()
    head, _, body = src.partition("def _apply_rules")
    import re
    classif = sorted(set(re.findall(r'"(R\d+_[A-Z_]+|R0_NEGATED|R_DEFAULT)"', body)))
    post = sorted(set(re.findall(r'rule_id="(R_[A-Z_]+)"', head + pipe)))
    return classif, post


def census() -> dict[str, Counter]:
    per_format: dict[str, Counter] = defaultdict(Counter)

    # A configuração canônica da tese é SEM NER (o default do script de avaliação
    # é o oposto — ver results/ner-ontology/ablation_ner.md).
    pipeline = NerOntologyPipeline(use_ner=False)

    # ── De Vito: casos de uso estruturados (119 FPs) ──────────────────────
    for sys_name, cfg in evalmod.DEVITO_SYSTEMS.items():
        gt = evalmod.load_devito_gt(cfg["gt_csv"])
        for fp_file in sorted(cfg["prompts"].glob("FP*.txt"),
                              key=lambda p: evalmod.fp_sort_key(p.stem)):
            if fp_file.stem not in gt:
                continue
            res = pipeline.measure(fp_file.read_text(encoding="utf-8"),
                                   fp_id=fp_file.stem)
            per_format["casos de uso (De Vito)"].update(
                m.rule_id for m in res.movements)

    # ── Zenodo: sentenças avulsas ─────────────────────────────────────────
    gt = evalmod.load_zenodo_gt()
    prompt_dir = evalmod.ZENODO_DIR / "FP_prompts"
    for fp_file in sorted(prompt_dir.glob("FP*.txt"),
                          key=lambda p: evalmod.fp_sort_key(p.stem)):
        if fp_file.stem not in gt:
            continue
        res = pipeline.measure(fp_file.read_text(encoding="utf-8"),
                               fp_id=fp_file.stem, sentence_mode=True)
        per_format["sentenças (Zenodo)"].update(m.rule_id for m in res.movements)

    # ── Ochodek: nomes de casos de uso (estimador per-formato) ────────────
    for item in evalmod.load_ochodek_items():
        res = pipeline.measure(item["title"], fp_id=item["uc"], uc_name=True)
        per_format["nomes de UC (Ochodek)"].update(m.rule_id for m in res.movements)

    # ── Dalpiaz: histórias de usuário, via adaptador Connextra (E34a) ─────
    from adapters.connextra_adapter import connextra_to_ucs
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    for fp_file in sorted(fp_dir.glob("FP*.txt"),
                          key=lambda p: evalmod.fp_sort_key(p.stem)):
        raw = fp_file.read_text(encoding="utf-8").strip()
        ucs = connextra_to_ucs(raw, fp_file.stem) or [evalmod.wrap_as_uc(raw, fp_file.stem)]
        for uc in ucs:
            res = pipeline.measure(uc, fp_id=fp_file.stem)
            per_format["histórias (Dalpiaz)"].update(m.rule_id for m in res.movements)

    return per_format


def main() -> int:
    classif, post = declared_rule_ids()
    print(f"Declaradas no código: {len(classif)} regras de classificação, "
          f"{len(post)} de pós-processamento com identificador")

    per_format = census()

    total = Counter()
    for c in per_format.values():
        total.update(c)

    formats = list(per_format.keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for rid in classif + post:
        rows.append({
            "rule_id": rid,
            "tipo": "classificação" if rid in classif else "pós-processamento",
            **{f: per_format[f].get(rid, 0) for f in formats},
            "total": total.get(rid, 0),
        })

    csv_path = OUT_DIR / "rule_firing_census.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    silent_c = [r["rule_id"] for r in rows
                if r["tipo"] == "classificação" and r["total"] == 0]
    fired_c = len(classif) - len(silent_c)

    lines = [
        "# Censo de alcançabilidade das regras do M2 (via simbólica E33d)",
        "",
        f"Gerado por `scripts/rule_firing_census.py`. Configuração canônica da tese "
        f"(`use_ner=False`).",
        "",
        f"**{fired_c} de {len(classif)}** regras de classificação disparam ao menos "
        f"uma vez no corpus de avaliação.",
        "",
        "| regra | tipo | " + " | ".join(formats) + " | total |",
        "|---|---|" + "---|" * (len(formats) + 1),
    ]
    for r in rows:
        lines.append("| `" + r["rule_id"] + "` | " + r["tipo"] + " | "
                     + " | ".join(str(r[f]) for f in formats)
                     + f" | {r['total']} |")
    lines += ["", "## Regras que nunca disparam", ""]
    lines += [f"- `{rid}`" for rid in silent_c] or ["- (nenhuma)"]
    lines.append("")

    md_path = OUT_DIR / "rule_firing_census.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  {fired_c}/{len(classif)} regras de classificação disparam")
    print(f"  nunca disparam: {', '.join(silent_c) or '(nenhuma)'}")
    print(f"  → {csv_path}")
    print(f"  → {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
