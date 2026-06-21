#!/usr/bin/env python3
"""Build pilot input dirs for cost-projection runs of R1 COSMET on Zenodo + Ochodek.

Samples 10 stratified items per dataset and emits one .txt per item under
results/cost-pilot/{dataset}/FP{N:02d}.txt, plus a manifest with ground truth.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "cost-pilot"

random.seed(42)


def stratified_sample(rows: list[dict], key_fn, target_per_bucket: dict[str, int]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        b = key_fn(r)
        buckets.setdefault(b, []).append(r)
    out: list[dict] = []
    for b, n in target_per_bucket.items():
        pool = buckets.get(b, [])
        pick = random.sample(pool, min(n, len(pool)))
        out.extend(pick)
    return out


def build_zenodo() -> None:
    src = BASE / "datasets" / "raw" / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
    rows = list(csv.DictReader(src.open()))
    # CFP distribution: ~3% CFP=0, ~71% CFP=1, ~25% CFP=2, ~1% CFP≥3
    targets = {"0": 1, "1": 6, "2": 2, "3+": 1}

    def key(r: dict) -> str:
        c = int(r["CFP"])
        if c >= 3:
            return "3+"
        return str(c)

    picks = stratified_sample(rows, key, targets)
    out_dir = OUT / "zenodo-10"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i, r in enumerate(picks, 1):
        fp = f"FP{i:02d}.txt"
        (out_dir / fp).write_text(r["requirement_en"].strip() + "\n", encoding="utf-8")
        manifest_rows.append({
            "fp": fp,
            "cfp_gt": r["CFP"],
            "text": r["requirement_en"].strip(),
        })
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "cfp_gt", "text"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"[zenodo] wrote {len(picks)} .txt to {out_dir}")


def build_ochodek() -> None:
    src = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
    rows = list(csv.DictReader(src.open()))
    # CFP varies widely; bucket as small (≤5), med (6-15), large (16+)
    targets = {"small": 4, "med": 4, "large": 2}

    def key(r: dict) -> str:
        c = int(r["Cfp"])
        if c <= 5:
            return "small"
        if c <= 15:
            return "med"
        return "large"

    picks = stratified_sample(rows, key, targets)
    out_dir = OUT / "ochodek-10"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i, r in enumerate(picks, 1):
        fp = f"FP{i:02d}.txt"
        (out_dir / fp).write_text(r["TitleTokens"].strip() + "\n", encoding="utf-8")
        manifest_rows.append({
            "fp": fp,
            "cfp_gt": r["Cfp"],
            "uc_id": r["UC"],
            "text": r["TitleTokens"].strip(),
        })
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fp", "cfp_gt", "uc_id", "text"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"[ochodek] wrote {len(picks)} .txt to {out_dir}")


if __name__ == "__main__":
    build_zenodo()
    build_ochodek()
