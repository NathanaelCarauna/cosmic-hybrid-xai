#!/usr/bin/env python3
"""T7 — Build stratified hold-out splits for De Vito (30 FPs) and Zenodo (50 sentences).

Stratification:
  De Vito: project × cfp_bucket {small:2-4, medium:5-9, large:10+}
           Singletons (<2 members) are merged to project-only stratum (logged).
  Zenodo:  cfp_bucket {1, 2, 3+}

Outputs (committed before any re-calibration runs):
  datasets/holdout/devito_holdout.csv
  datasets/holdout/devito_train.csv
  datasets/holdout/zenodo_holdout.csv
  datasets/holdout/zenodo_train.csv
  datasets/holdout/manifest_split.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MANIFEST = BASE / "datasets" / "manifest.csv"
OUT_DIR = BASE / "datasets" / "holdout"

SEED = 42
DEVITO_HOLDOUT_N = 30
ZENODO_HOLDOUT_N = 50


def cfp_bucket_devito(cfp: int) -> str:
    if cfp <= 4:
        return "small"
    if cfp <= 9:
        return "medium"
    return "large"


def cfp_bucket_zenodo(cfp: int) -> str:
    if cfp == 1:
        return "1"
    if cfp == 2:
        return "2"
    return "3+"


def stratified_split(
    ids: list[str],
    strata: list[str],
    n_test: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) with approximately proportional strata."""
    from sklearn.model_selection import train_test_split

    test_ratio = n_test / len(ids)
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_ratio,
        stratify=strata,
        random_state=seed,
    )
    return list(train_ids), list(test_ids)


def build_devito_split(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    devito = [r for r in rows if r["source"].startswith("cosmet-devito")]
    ids = [r["id"] for r in devito]

    raw_strata = [
        f"{r['project']}|{cfp_bucket_devito(int(r['cfp']))}" for r in devito
    ]
    stratum_counts = Counter(raw_strata)

    # Find all projects that have at least one singleton bucket — collapse whole project
    projects_with_singletons = {
        s.split("|")[0] for s, c in stratum_counts.items() if c < 2
    }
    singleton_merges: dict[str, str] = {
        s: s.split("|")[0]
        for s in stratum_counts
        if s.split("|")[0] in projects_with_singletons
    }

    final_strata = [
        singleton_merges.get(raw, raw)
        for raw, row in zip(raw_strata, devito)
    ]

    if singleton_merges:
        print("  [build_holdout] De Vito strata collapsed to project-only (had singleton bucket):")
        collapsed = defaultdict(list)
        for k, v in singleton_merges.items():
            collapsed[v].append(f"{k}(n={stratum_counts[k]})")
        for proj, buckets in sorted(collapsed.items()):
            print(f"    {proj}: {', '.join(buckets)}")

    merged_counts = Counter(final_strata)
    still_singletons = [s for s, c in merged_counts.items() if c < 2]
    if still_singletons:
        print(f"  [build_holdout] WARNING: still-singleton strata after merge: {still_singletons}")
        print("  Falling back to project-only stratification for De Vito.")
        final_strata = [r["project"] for r in devito]

    train_ids, test_ids = stratified_split(ids, final_strata, DEVITO_HOLDOUT_N, SEED)

    meta = {
        "seed": SEED,
        "n_total": len(devito),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "stratify_by": "project×cfp_bucket" if not still_singletons else "project",
        "singleton_merges": singleton_merges,
        "strata_distribution": dict(Counter(final_strata)),
        "test_strata_distribution": dict(
            Counter(
                final_strata[ids.index(i)] for i in test_ids
            )
        ),
    }
    return train_ids, test_ids, meta


def build_zenodo_split(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    zenodo = [r for r in rows if r["source"] == "zenodo-heterogeneous"]
    ids = [r["id"] for r in zenodo]
    strata = [cfp_bucket_zenodo(int(r["cfp"])) for r in zenodo]

    train_ids, test_ids = stratified_split(ids, strata, ZENODO_HOLDOUT_N, SEED)

    meta = {
        "seed": SEED,
        "n_total": len(zenodo),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "stratify_by": "cfp_bucket",
        "strata_distribution": dict(Counter(strata)),
        "test_strata_distribution": dict(
            Counter(strata[ids.index(i)] for i in test_ids)
        ),
    }
    return train_ids, test_ids, meta


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    print(f"Reading {MANIFEST} ...")
    rows = list(csv.DictReader(MANIFEST.open()))
    id_to_row = {r["id"]: r for r in rows}
    print(f"  Total rows: {len(rows)}")

    print("\nBuilding De Vito hold-out split ...")
    devito_train_ids, devito_test_ids, devito_meta = build_devito_split(rows)
    print(f"  Train: {len(devito_train_ids)}, Test: {len(devito_test_ids)}")

    print("\nBuilding Zenodo hold-out split ...")
    zenodo_train_ids, zenodo_test_ids, zenodo_meta = build_zenodo_split(rows)
    print(f"  Train: {len(zenodo_train_ids)}, Test: {len(zenodo_test_ids)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUT_DIR / "devito_holdout.csv", [id_to_row[i] for i in devito_test_ids])
    write_csv(OUT_DIR / "devito_train.csv", [id_to_row[i] for i in devito_train_ids])
    write_csv(OUT_DIR / "zenodo_holdout.csv", [id_to_row[i] for i in zenodo_test_ids])
    write_csv(OUT_DIR / "zenodo_train.csv", [id_to_row[i] for i in zenodo_train_ids])

    split_manifest = {
        "devito": devito_meta,
        "zenodo": zenodo_meta,
    }
    with open(OUT_DIR / "manifest_split.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"\nOutputs written to {OUT_DIR}/")
    print("  devito_holdout.csv  devito_train.csv")
    print("  zenodo_holdout.csv  zenodo_train.csv")
    print("  manifest_split.json")
    print("\nDe Vito strata distribution:")
    for k, v in sorted(devito_meta["strata_distribution"].items()):
        test_v = devito_meta["test_strata_distribution"].get(k, 0)
        print(f"  {k}: total={v}, test={test_v}")
    print("\nZenodo strata distribution:")
    for k, v in sorted(zenodo_meta["strata_distribution"].items()):
        test_v = zenodo_meta["test_strata_distribution"].get(k, 0)
        print(f"  CFP={k}: total={v}, test={test_v}")


if __name__ == "__main__":
    main()
