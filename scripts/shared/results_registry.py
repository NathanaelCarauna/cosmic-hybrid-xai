"""Runs registry — append-only CSV tracking every experiment execution."""
from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

WIB = timezone(timedelta(hours=7))

FIELDS = [
    "run_id",
    "method",
    "method_type",
    "dataset",
    "llm_model",
    "timestamp",
    "n_fps",
    "n_errors",
    "total_cfp_pred",
    "mae",
    "mmre",
    "pred25",
    "wall_clock_s",
    "total_tokens",
    "total_cost_usd",
    "throughput_fp_min",
    "status",
]


def _registry_path(results_dir: Path) -> Path:
    return results_dir / "runs_registry.csv"


def _ensure_header(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def generate_run_id(dataset: str, llm_model: str) -> str:
    """Produce a deterministic run_id: {dataset}-{model_short}-{timestamp}."""
    model_short = llm_model.replace("/", "-")
    ts = datetime.now(WIB).strftime("%Y%m%dT%H%M%S")
    return f"{dataset}-{model_short}-{ts}"


def register_run(results_dir: Path, row: dict) -> None:
    """Append a new run entry to the registry (file-lock safe)."""
    path = _registry_path(results_dir)
    _ensure_header(path)
    safe_row = {k: row.get(k, "") for k in FIELDS}
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(safe_row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update_run(results_dir: Path, run_id: str, updates: dict) -> None:
    """Update fields of an existing run entry identified by run_id."""
    path = _registry_path(results_dir)
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str)
    mask = df["run_id"] == run_id
    if mask.sum() == 0:
        return
    for k, v in updates.items():
        if k in df.columns:
            df.loc[mask, k] = str(v)
    df.to_csv(path, index=False)


def load_registry(results_dir: Path) -> pd.DataFrame:
    """Load the full registry as a DataFrame."""
    path = _registry_path(results_dir)
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path)
