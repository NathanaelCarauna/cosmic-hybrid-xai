#!/usr/bin/env python3
"""T10 — Reconcile runs_registry.csv with all metrics.json in results/.

Scans results/{hybrid-xai,cosmet,deep-cosmic-uc,hko-hussain,ner-ontology}/**/metrics.json.
For each unregistered directory, extracts available fields and adds them.
Incomplete entries go to results/runs_registry_orphans.csv for manual review.

Usage:
  python3 scripts/reconcile_registry.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
REGISTRY = RESULTS / "runs_registry.csv"
ORPHANS = RESULTS / "runs_registry_orphans.csv"

REGISTRY_FIELDS = [
    "run_id", "method", "method_type", "dataset", "llm_model",
    "timestamp", "n_fps", "n_errors", "total_cfp_pred",
    "mae", "mmre", "pred25", "wall_clock_s", "total_tokens",
    "total_cost_usd", "throughput_fp_min", "status",
]

# Additional fields from hybrid-xai metrics that aren't in legacy registry
EXTRA_FIELDS = ["mdae", "sa", "r2", "exact_match", "pred_25"]

SCAN_DIRS = [
    "hybrid-xai",
    "cosmet",
    "deep-cosmic-uc",
    "hko-hussain",
    "ner-ontology",
]

# Regex to extract timestamp from directory name
_TS_RE = re.compile(r"(\d{8}T\d{6})")
# Regex to extract experiment ID (E3, E28a, E30f, etc.)
_EXP_RE = re.compile(r"\bE(\d+[a-z]?)\b", re.IGNORECASE)


def _infer_run_info(metrics_path: Path) -> dict:
    """Infer run metadata from path and metrics.json content."""
    # Build a run_id as relative path from results/
    rel = metrics_path.parent.relative_to(RESULTS)
    run_id = str(rel).replace(os.sep, "/")

    dirname = metrics_path.parent.name
    parent_name = metrics_path.parent.parent.name

    # Infer timestamp
    ts_match = _TS_RE.search(str(rel))
    if ts_match:
        raw_ts = ts_match.group(1)
        try:
            timestamp = datetime.strptime(raw_ts, "%Y%m%dT%H%M%S").isoformat()
        except ValueError:
            timestamp = raw_ts
    else:
        mtime = metrics_path.stat().st_mtime
        timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    # Infer dataset from path
    path_str = str(rel).lower()
    if "ochodek" in path_str:
        dataset = "ochodek-437"
    elif "zenodo" in path_str:
        dataset = "zenodo-2040"
    elif "119fp" in path_str or "119-" in path_str:
        dataset = "devito-119"
    elif "devito" in path_str:
        dataset = "devito-119"
    elif "ner-ontology" in path_str:
        dataset = "devito-119"
    else:
        dataset = ""

    # Infer method
    first_dir = rel.parts[0] if rel.parts else ""
    if first_dir == "hybrid-xai":
        # Distinguish sub-methods
        if "bert" in path_str or "e30" in path_str or "e31" in path_str:
            method = "hybrid-xai-bert"
        elif "ner" in path_str or "ontology" in path_str:
            method = "hybrid-xai-ner"
        elif "gbr" in path_str:
            method = "hybrid-xai-gbr"
        else:
            method = "hybrid-xai"
    elif first_dir == "cosmet":
        method = "cosmet"
    elif first_dir == "deep-cosmic-uc":
        method = "deep-cosmic-uc"
    elif first_dir == "hko-hussain":
        method = "hko-hussain"
    elif first_dir == "ner-ontology":
        method = "hybrid-xai-ner"
    else:
        method = first_dir

    # Infer LLM model from path
    llm_model = ""
    if "qwen3-8b" in path_str or "qwen3_8b" in path_str:
        llm_model = "Qwen3-8B"
    elif "qwen25-14b" in path_str:
        llm_model = "Qwen2.5-14B"
    elif "qwen3b" in path_str or "3b" in path_str:
        llm_model = "Qwen3-3B"
    elif "7b" in path_str:
        llm_model = "Qwen2.5-7B"
    elif "turbo" in path_str:
        llm_model = "gpt-4-turbo"
    elif "mini" in path_str:
        llm_model = "gpt-4o-mini"
    elif "bert" in path_str:
        llm_model = "bert-base-uncased"

    # Parse metrics.json
    try:
        m = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        m = {}

    n_fps = m.get("n_fps") or m.get("n_with_gt") or ""
    mae = m.get("mae", "")
    mmre = m.get("mmre", "")
    pred25 = m.get("pred_25") or m.get("pred25") or ""
    wall_clock_s = m.get("total_time_s", "")

    # Format floats
    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if isinstance(v, float) else str(v) if v != "" else ""

    row = {
        "run_id": run_id,
        "method": method,
        "method_type": "hybrid-xai" if "hybrid" in method or "ner" in method else "replication",
        "dataset": dataset,
        "llm_model": llm_model,
        "timestamp": timestamp,
        "n_fps": str(n_fps) if n_fps != "" else "",
        "n_errors": "",
        "total_cfp_pred": "",
        "mae": fmt(mae),
        "mmre": fmt(mmre),
        "pred25": fmt(pred25),
        "wall_clock_s": fmt(wall_clock_s, 1),
        "total_tokens": "",
        "total_cost_usd": "",
        "throughput_fp_min": "",
        "status": "ok" if mae != "" else "incomplete",
        # Extra hybrid-xai fields
        "mdae": fmt(m.get("mdae", "")),
        "sa": fmt(m.get("sa", "")),
        "r2": fmt(m.get("r2", "")),
        "exact_match": fmt(m.get("exact_match", "")),
    }
    return row


def load_registry() -> tuple[list[dict], set[str]]:
    if not REGISTRY.exists():
        return [], set()
    rows = list(csv.DictReader(REGISTRY.open()))
    registered = {r["run_id"] for r in rows}
    return rows, registered


def find_all_metrics() -> list[Path]:
    paths = []
    for subdir in SCAN_DIRS:
        d = RESULTS / subdir
        if d.exists():
            paths.extend(d.rglob("metrics.json"))
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing")
    args = parser.parse_args()

    existing_rows, registered_ids = load_registry()
    all_metrics = find_all_metrics()

    print(f"Registry: {len(existing_rows)} entries")
    print(f"Found metrics.json: {len(all_metrics)}")

    new_rows = []
    orphan_rows = []

    for mp in all_metrics:
        rel = mp.parent.relative_to(RESULTS)
        run_id = str(rel).replace(os.sep, "/")

        if run_id in registered_ids:
            continue

        info = _infer_run_info(mp)

        is_complete = bool(info.get("mae") and info.get("n_fps"))
        if is_complete:
            new_rows.append(info)
        else:
            orphan_rows.append(info)

    print(f"New entries to add: {len(new_rows)}")
    print(f"Orphans (incomplete): {len(orphan_rows)}")

    if args.dry_run:
        print("\n[DRY RUN] Would add:")
        for r in new_rows[:10]:
            print(f"  {r['run_id']:60s} mae={r['mae']}")
        if len(new_rows) > 10:
            print(f"  ... and {len(new_rows) - 10} more")
        print("\n[DRY RUN] Orphans:")
        for r in orphan_rows[:5]:
            print(f"  {r['run_id']}")
        return

    # Merge all fields: registry base fields + extra hybrid-xai fields
    all_fields = REGISTRY_FIELDS + [f for f in EXTRA_FIELDS if f not in REGISTRY_FIELDS]

    # Pad existing rows with empty extra fields
    padded_existing = []
    for row in existing_rows:
        padded = {f: row.get(f, "") for f in all_fields}
        padded_existing.append(padded)

    # Pad new rows
    padded_new = [{f: r.get(f, "") for f in all_fields} for r in new_rows]

    # Write updated registry
    all_updated = padded_existing + padded_new
    with open(REGISTRY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(all_updated)
    print(f"\nRegistry updated: {len(all_updated)} entries → {REGISTRY}")

    # Write orphans
    if orphan_rows:
        padded_orphans = [{f: r.get(f, "") for f in all_fields} for r in orphan_rows]
        with open(ORPHANS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            w.writerows(padded_orphans)
        print(f"Orphans written: {len(orphan_rows)} entries → {ORPHANS}")
    else:
        print("No orphans.")

    print("\nSample new entries:")
    for r in new_rows[:5]:
        print(f"  {r['run_id']:55s} mae={r['mae']:6s} n={r['n_fps']}")


if __name__ == "__main__":
    main()
