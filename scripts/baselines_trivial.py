#!/usr/bin/env python3
"""D2 — Trivial baselines on De Vito and Zenodo holdout sets.

Trains 3 trivial baselines on each train split and evaluates on the
corresponding holdout. Results are saved to a summary CSV and registered
in the global runs_registry.

Baselines:
  median-train  — always predicts median CFP of training set
  linear-tokens — LinearRegression on n_tokens (word count)
  tree-shallow  — DecisionTreeRegressor(max_depth=4) on text features

Datasets:
  De Vito  train:   datasets/holdout/devito_train.csv   (89 FPs)
           holdout: datasets/holdout/devito_holdout.csv (30 FPs)
  Zenodo   train:   datasets/holdout/zenodo_train.csv
           holdout: datasets/holdout/zenodo_holdout.csv

Output:
  results/baselines/trivial_summary.csv

Usage:
  python scripts/baselines_trivial.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ── Repo root & shared imports ─────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.results_registry import register_run

WIB = timezone(timedelta(hours=7))

# ── Paths ──────────────────────────────────────────────────────────────
DATASETS = {
    "devito": {
        "train": BASE / "datasets/holdout/devito_train.csv",
        "holdout": BASE / "datasets/holdout/devito_holdout.csv",
    },
    "zenodo": {
        "train": BASE / "datasets/holdout/zenodo_train.csv",
        "holdout": BASE / "datasets/holdout/zenodo_holdout.csv",
    },
}

OUTPUT_DIR = BASE / "results/baselines"
OUTPUT_CSV = OUTPUT_DIR / "trivial_summary.csv"
RESULTS_DIR = BASE / "results"

OUTPUT_CSV_FIELDS = [
    "baseline",
    "train_dataset",
    "eval_dataset",
    "n_train",
    "n_eval",
    "mae",
    "pred25",
    "sa",
    "r2",
    "note",
]


# ── Data loading ───────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X has columns [n_tokens, text_len, n_lines]."""
    X_list = []
    y_list = []
    for row in rows:
        text = row.get("text", "")
        n_tokens = len(text.split())
        text_len = len(text)
        n_lines = text.count("\n") + 1
        X_list.append([n_tokens, text_len, n_lines])
        y_list.append(float(row["cfp"]))
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)


# ── Metrics ────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))

    re = ae / np.maximum(y_true, 1)
    pred25 = float(np.mean(re <= 0.25))

    mean_gt = float(np.mean(y_true))
    random_mae = float(np.mean(np.abs(y_true - mean_gt)))
    sa = 1.0 - mae / random_mae if random_mae > 0 else 0.0

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_gt) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"mae": mae, "pred25": pred25, "sa": sa, "r2": r2}


# ── Baseline runners ───────────────────────────────────────────────────

def run_median_train(
    train_rows: list[dict], holdout_rows: list[dict]
) -> tuple[dict, str]:
    """Always predicts the median CFP of the training set."""
    y_train = np.array([float(r["cfp"]) for r in train_rows])
    y_holdout = np.array([float(r["cfp"]) for r in holdout_rows])

    median_cfp = float(np.median(y_train))
    y_pred = np.full(len(y_holdout), median_cfp)

    m = _metrics(y_holdout, y_pred)
    note = f"Median train CFP = {median_cfp:.1f}"
    return m, note


def run_linear_tokens(
    train_rows: list[dict], holdout_rows: list[dict]
) -> tuple[dict, str]:
    """LinearRegression on n_tokens (word count of text) only."""
    X_train, y_train = _extract_features(train_rows)
    X_holdout, y_holdout = _extract_features(holdout_rows)

    # Use only the first column: n_tokens
    X_tr = X_train[:, 0:1]
    X_ho = X_holdout[:, 0:1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression()
        reg.fit(X_tr, y_train)

    y_pred = np.maximum(1, reg.predict(X_ho))
    m = _metrics(y_holdout, y_pred)
    note = (
        f"LinearRegression on n_tokens; "
        f"coef={reg.coef_[0]:.4f} intercept={reg.intercept_:.4f}"
    )
    return m, note


def run_tree_shallow(
    train_rows: list[dict], holdout_rows: list[dict]
) -> tuple[dict, str]:
    """DecisionTreeRegressor(max_depth=4) on n_tokens, text_len, n_lines."""
    X_train, y_train = _extract_features(train_rows)
    X_holdout, y_holdout = _extract_features(holdout_rows)

    tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)

    y_pred = np.maximum(1, np.round(tree.predict(X_holdout)))
    m = _metrics(y_holdout, y_pred)
    note = (
        f"DecisionTreeRegressor(max_depth=4) on [n_tokens, text_len, n_lines]; "
        f"n_leaves={tree.get_n_leaves()}"
    )
    return m, note


# ── Registry helper ───────────────────────────────────────────────────

def _register(
    baseline: str,
    train_name: str,
    eval_name: str,
    n_train: int,
    n_eval: int,
    m: dict,
) -> None:
    ts = datetime.now(WIB).strftime("%Y%m%dT%H%M%S")
    run_id = f"{eval_name}-{baseline}-{ts}"
    row = {
        "run_id": run_id,
        "method": baseline,
        "method_type": "trivial-baseline",
        "dataset": eval_name,
        "llm_model": "none",
        "timestamp": datetime.now(WIB).isoformat(),
        "n_fps": n_eval,
        "n_errors": 0,
        "total_cfp_pred": "",
        "mae": round(m["mae"], 4),
        "mmre": "",
        "pred25": round(m["pred25"], 4),
        "wall_clock_s": 0,
        "total_tokens": 0,
        "total_cost_usd": 0,
        "throughput_fp_min": "",
        "status": "done",
    }
    register_run(RESULTS_DIR, row)


# ── Output ────────────────────────────────────────────────────────────

def _save_summary(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary saved to: {OUTPUT_CSV}")


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("TRIVIAL BASELINES — MAE / PRED25 / SA / R²")
    print("=" * 90)
    header = (
        f"{'Baseline':<20} {'Dataset':<10} {'n_tr':>5} {'n_ev':>5} "
        f"{'MAE':>6} {'PRED25':>7} {'SA':>7} {'R²':>7}"
    )
    print(header)
    print("-" * 90)
    for row in rows:
        print(
            f"{row['baseline']:<20} {row['eval_dataset']:<10} "
            f"{int(row['n_train']):>5} {int(row['n_eval']):>5} "
            f"{row['mae']:>6.3f} {row['pred25']*100:>6.1f}% "
            f"{row['sa']*100:>6.1f}% {row['r2']:>7.3f}"
        )
    print("-" * 90)


# ── Main ──────────────────────────────────────────────────────────────

BASELINES = [
    ("median-train", run_median_train),
    ("linear-tokens", run_linear_tokens),
    ("tree-shallow", run_tree_shallow),
]


def main() -> None:
    print("=" * 72)
    print("D2 — Trivial baselines on De Vito and Zenodo holdout sets")
    print("=" * 72)

    summary_rows: list[dict] = []

    for dataset_name, paths in DATASETS.items():
        train_path = paths["train"]
        holdout_path = paths["holdout"]

        # Validate both files exist
        if not train_path.exists():
            print(f"\n[WARN] Train CSV not found, skipping {dataset_name}: {train_path}")
            continue
        if not holdout_path.exists():
            print(f"\n[WARN] Holdout CSV not found, skipping {dataset_name}: {holdout_path}")
            continue

        print(f"\nDataset: {dataset_name}")
        train_rows = _load_csv(train_path)
        holdout_rows = _load_csv(holdout_path)
        print(f"  Train: {len(train_rows)} rows | Holdout: {len(holdout_rows)} rows")

        for baseline_name, runner_fn in BASELINES:
            print(f"\n  [{baseline_name}] on {dataset_name}...")
            try:
                m, note = runner_fn(train_rows, holdout_rows)
            except Exception as exc:
                print(f"    [ERROR] {exc}")
                continue

            print(
                f"    MAE={m['mae']:.3f} PRED25={m['pred25']*100:.1f}% "
                f"SA={m['sa']*100:.1f}% R²={m['r2']:.3f}"
            )
            print(f"    Note: {note}")

            summary_rows.append({
                "baseline": baseline_name,
                "train_dataset": dataset_name,
                "eval_dataset": dataset_name + "-holdout",
                "n_train": len(train_rows),
                "n_eval": len(holdout_rows),
                "mae": round(m["mae"], 4),
                "pred25": round(m["pred25"], 4),
                "sa": round(m["sa"], 4),
                "r2": round(m["r2"], 4),
                "note": note,
            })

            _register(
                baseline=baseline_name,
                train_name=dataset_name,
                eval_name=dataset_name + "-holdout",
                n_train=len(train_rows),
                n_eval=len(holdout_rows),
                m=m,
            )

    if not summary_rows:
        print("\n[WARN] No results to save — check dataset paths.")
        return

    _save_summary(summary_rows)
    _print_table(summary_rows)


if __name__ == "__main__":
    main()
