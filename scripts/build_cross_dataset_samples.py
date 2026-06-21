#!/usr/bin/env python3
"""Build stratified sample CSVs for cross-dataset R1 COSMET runs (A8/A9).

Samples 200 items from Zenodo and 200 from Ochodek with seed=42 before any
API calls are made (sample lock-in protocol). Commit the CSV files to git and
tag the commit before running cosmet_batch_runner.

Outputs (lock-in, commit to git):
  datasets/cross-sample/zenodo-200-r1.csv      — 200 Zenodo sentences (GT)
  datasets/cross-sample/ochodek-200-r1.csv     — 200 Ochodek UC names (GT)

Outputs (FP*.txt input dirs for cosmet_batch_runner, not committed):
  results/cross-input/zenodo-200/FP001.txt … FP200.txt
  results/cross-input/ochodek-200/FP001.txt … FP200.txt
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "datasets" / "cross-sample"
SEED = 42


def _stratified_sample(
    rows: list[dict],
    key_fn,
    target_per_bucket: dict[str, int],
    seed: int = SEED,
) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        b = key_fn(r)
        buckets.setdefault(b, []).append(r)
    out: list[dict] = []
    for b, n in target_per_bucket.items():
        pool = buckets.get(b, [])
        pick = rng.sample(pool, min(n, len(pool)))
        out.extend(pick)
    return out


def build_zenodo() -> None:
    src = BASE / "datasets" / "raw" / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("requirement_en", "").strip()]

    def cfp_bucket(r: dict) -> str:
        c = int(r["CFP"])
        return "0" if c == 0 else "1" if c == 1 else "2" if c == 2 else "3+"

    # CFP distribution: ~3% 0, ~71% 1, ~25% 2, ~1% ≥3 → sample proportionally × 200
    targets = {"0": 6, "1": 142, "2": 48, "3+": 4}

    picks = _stratified_sample(rows, cfp_bucket, targets)
    OUT.mkdir(parents=True, exist_ok=True)

    out_path = OUT / "zenodo-200-r1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "CFP", "entry", "read", "write", "exit"])
        w.writeheader()
        for i, r in enumerate(picks, 1):
            w.writerow({
                "id": f"Z{i:03d}",
                "text": r["requirement_en"].strip(),
                "CFP": r["CFP"],
                "entry": r.get("entry", ""),
                "read": r.get("read", ""),
                "write": r.get("write", ""),
                "exit": r.get("exit", ""),
            })

    manifest = {
        "seed": SEED,
        "n_total": len(rows),
        "n_sample": len(picks),
        "cfp_distribution": {b: sum(1 for p in picks if cfp_bucket(p) == b) for b in targets},
        "source": "datasets/raw/zenodo-heterogeneous/heterogeneous_dataset.csv",
    }
    (OUT / "zenodo-200-r1-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[zenodo] {len(picks)} items → {out_path}")
    dist = manifest["cfp_distribution"]
    print(f"  CFP distribution: {dist}")

    # Write FP*.txt input directory for cosmet_batch_runner
    fp_dir = BASE / "results" / "cross-input" / "zenodo-200"
    fp_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(picks, 1):
        (fp_dir / f"FP{i:03d}.txt").write_text(
            r["requirement_en"].strip() + "\n", encoding="utf-8"
        )
    print(f"  FP*.txt → {fp_dir}")


def build_ochodek() -> None:
    src = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("TitleTokens", "").strip()]

    def cfp_bucket(r: dict) -> str:
        c = int(r["Cfp"])
        return "small" if c <= 5 else "medium" if c <= 15 else "large"

    # Ochodek CFP: mix of ranges; sample 200 stratified across size buckets
    targets = {"small": 67, "medium": 100, "large": 33}

    picks = _stratified_sample(rows, cfp_bucket, targets)
    OUT.mkdir(parents=True, exist_ok=True)

    out_path = OUT / "ochodek-200-r1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "CFP", "uc_id", "project_id"])
        w.writeheader()
        for i, r in enumerate(picks, 1):
            w.writerow({
                "id": f"O{i:03d}",
                "text": r["TitleTokens"].strip(),
                "CFP": r["Cfp"],
                "uc_id": r.get("UC", ""),
                "project_id": r.get("ProjectID", ""),
            })

    manifest = {
        "seed": SEED,
        "n_total": len(rows),
        "n_sample": len(picks),
        "cfp_distribution": {b: sum(1 for p in picks if cfp_bucket(p) == b) for b in targets},
        "source": "datasets/raw/ochodek-uc-437/use-cases.csv",
    }
    (OUT / "ochodek-200-r1-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ochodek] {len(picks)} items → {out_path}")
    dist = manifest["cfp_distribution"]
    print(f"  CFP distribution: {dist}")

    # Write FP*.txt input directory for cosmet_batch_runner
    fp_dir = BASE / "results" / "cross-input" / "ochodek-200"
    fp_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(picks, 1):
        (fp_dir / f"FP{i:03d}.txt").write_text(
            r["TitleTokens"].strip() + "\n", encoding="utf-8"
        )
    print(f"  FP*.txt → {fp_dir}")


def main() -> None:
    print("Building cross-dataset samples for R1 COSMET API runs...")
    print("Seed:", SEED)
    print()
    build_zenodo()
    build_ochodek()
    print()
    print("IMPORTANT: Commit these files to git before running R1 API calls!")
    print("  git add datasets/cross-sample/")
    print("  git commit -m 'data: lock cross-dataset samples for R1 (zenodo-200, ochodek-200)'")
    print("  git tag r1-cross-zenodo-canonical")
    print("  git tag r1-cross-ochodek-canonical")


if __name__ == "__main__":
    main()
