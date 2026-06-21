#!/usr/bin/env python3
"""B5 — Compute Cohen's κ between annotators for R4 user story annotation.

Usage:
    python scripts/compute_annotation_kappa.py \
        --annotator1 datasets/user-stories-100-annotated-a1.csv \
        --annotator2 datasets/user-stories-100-annotated-a2-subset.csv \
        --output results/r4-annotation/kappa.md

The two CSV files must share the us_id column. A2 typically covers only the
20-item kappa subset; the script aligns on common us_ids automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent


def _load_annotations(path: Path) -> dict[str, dict]:
    """Load annotation CSV → {us_id: row} dict."""
    with path.open(newline="", encoding="utf-8") as f:
        return {r["us_id"]: r for r in csv.DictReader(f)}


def cohen_kappa_linear(y1: list[int], y2: list[int]) -> float:
    """Linear-weighted Cohen's κ (suitable for ordinal CFP values)."""
    n = len(y1)
    if n == 0:
        return float("nan")
    categories = sorted(set(y1) | set(y2))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Observed agreement (linear weight)
    w_obs = 0.0
    w_exp = 0.0
    counts = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        counts[cat_idx[a], cat_idx[b]] += 1

    w = np.ones((k, k)) - np.abs(
        np.subtract.outer(np.arange(k), np.arange(k))
    ) / (k - 1 if k > 1 else 1)

    p_obs = np.sum(w * counts) / n
    marginal_a = counts.sum(axis=1) / n
    marginal_b = counts.sum(axis=0) / n
    p_exp = np.sum(w * np.outer(marginal_a, marginal_b))

    if p_exp == 1.0:
        return 1.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def exact_kappa(y1: list[int], y2: list[int]) -> float:
    """Unweighted (exact-match) Cohen's κ."""
    n = len(y1)
    if n == 0:
        return float("nan")
    p_obs = sum(a == b for a, b in zip(y1, y2)) / n
    categories = sorted(set(y1) | set(y2))
    from collections import Counter
    c1 = Counter(y1)
    c2 = Counter(y2)
    p_exp = sum(c1[c] / n * c2[c] / n for c in categories)
    if p_exp == 1.0:
        return 1.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def per_movement_kappa(a1: dict, a2: dict, common_ids: list[str]) -> dict[str, float]:
    """Compute κ per movement type (Entry/Exit/Read/Write)."""
    result = {}
    for col in ("count_entry", "count_exit", "count_read", "count_write"):
        y1, y2 = [], []
        for us_id in common_ids:
            r1 = a1[us_id]
            r2 = a2[us_id]
            try:
                y1.append(int(r1.get(col, 0) or 0))
                y2.append(int(r2.get(col, 0) or 0))
            except ValueError:
                pass
        if y1:
            result[col.replace("count_", "")] = cohen_kappa_linear(y1, y2)
    return result


def _interp_kappa(k: float) -> str:
    if math.isnan(k):
        return "N/A"
    if k < 0:
        return "Poor (< 0)"
    if k < 0.20:
        return "Slight (0–0.20)"
    if k < 0.40:
        return "Fair (0.21–0.40)"
    if k < 0.60:
        return "Moderate (0.41–0.60)"
    if k < 0.80:
        return "Substantial (0.61–0.80)"
    return "Almost perfect (> 0.80)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's κ between two annotators for R4 user story annotation"
    )
    parser.add_argument("--annotator1", type=Path, required=True)
    parser.add_argument("--annotator2", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "results" / "r4-annotation" / "kappa.md",
    )
    args = parser.parse_args()

    if not args.annotator1.exists():
        print(f"ERROR: {args.annotator1} not found", file=sys.stderr)
        sys.exit(1)
    if not args.annotator2.exists():
        print(f"ERROR: {args.annotator2} not found", file=sys.stderr)
        sys.exit(1)

    a1 = _load_annotations(args.annotator1)
    a2 = _load_annotations(args.annotator2)

    common_ids = sorted(set(a1) & set(a2))
    n = len(common_ids)

    if n == 0:
        print("ERROR: no common us_id between the two files", file=sys.stderr)
        sys.exit(1)

    print(f"Common items: {n} (A1={len(a1)}, A2={len(a2)})")

    cfp1 = [int(a1[i].get("cfp_total", 0) or 0) for i in common_ids]
    cfp2 = [int(a2[i].get("cfp_total", 0) or 0) for i in common_ids]

    k_exact = exact_kappa(cfp1, cfp2)
    k_linear = cohen_kappa_linear(cfp1, cfp2)
    k_per_mov = per_movement_kappa(a1, a2, common_ids)

    # Discordances table
    discordances = [
        (us_id, a1[us_id].get("cfp_total"), a2[us_id].get("cfp_total"),
         abs(int(a1[us_id].get("cfp_total", 0) or 0) - int(a2[us_id].get("cfp_total", 0) or 0)))
        for us_id in common_ids
        if a1[us_id].get("cfp_total") != a2[us_id].get("cfp_total")
    ]
    discordances.sort(key=lambda x: -x[3])

    # Build markdown report
    lines = [
        "# Relatório de Concordância entre Anotadores — R4 User Stories",
        "",
        f"**Anotador 1:** `{args.annotator1.name}`  |  n={len(a1)} itens",
        f"**Anotador 2:** `{args.annotator2.name}`  |  n={len(a2)} itens",
        f"**Itens em comum (κ subset):** {n}",
        "",
        "---",
        "",
        "## 1. κ do CFP Total",
        "",
        f"| Métrica | Valor | Interpretação |",
        "|---------|-------|---------------|",
        f"| κ exato (sem peso) | {k_exact:.4f} | {_interp_kappa(k_exact)} |",
        f"| κ ponderado linear | {k_linear:.4f} | {_interp_kappa(k_linear)} |",
        "",
        "---",
        "",
        "## 2. κ por Tipo de Movimento",
        "",
        "| Movimento | κ (linear) | Interpretação |",
        "|-----------|-----------|---------------|",
    ]
    for mv, k in k_per_mov.items():
        lines.append(f"| {mv.capitalize()} | {k:.4f} | {_interp_kappa(k)} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Discordâncias",
        "",
        f"**Itens com CFP divergente:** {len(discordances)} / {n} "
        f"({100*len(discordances)/n:.1f}%)",
        "",
        "| us_id | CFP A1 | CFP A2 | Δ |",
        "|-------|--------|--------|---|",
    ]
    for us_id, c1, c2, delta in discordances[:20]:
        lines.append(f"| {us_id} | {c1} | {c2} | {delta} |")

    if len(discordances) > 20:
        lines.append(f"| *(+ {len(discordances)-20} outras)* | | | |")

    lines += [
        "",
        "---",
        "",
        "## 4. Protocolo de Resolução",
        "",
        "Para itens com |Δ| ≥ 2, realizar reunião de consenso entre anotadores.",
        "Para itens com |Δ| = 1, o anotador principal decide com justificativa.",
        "Registrar decisão na coluna `notes` do CSV consolidado.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  κ exact:  {k_exact:.4f}  ({_interp_kappa(k_exact)})")
    print(f"  κ linear: {k_linear:.4f}  ({_interp_kappa(k_linear)})")
    print(f"  Discordances: {len(discordances)}/{n}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
