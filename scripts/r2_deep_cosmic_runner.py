#!/usr/bin/env python3
"""R2 DEEP-COSMIC-UC Replication Runner.

Trains and evaluates CNN (P37), RF, SVR, and BERT (P13) on the Ochodek
437-UC dataset using the pre-built 10×10-fold cross-validation splits.

Usage:
    python scripts/r2_deep_cosmic_runner.py --model rf
    python scripts/r2_deep_cosmic_runner.py --model all
    python scripts/r2_deep_cosmic_runner.py --model cnn --epochs 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.metrics import (
    cliffs_delta,
    error_distribution,
    mae,
    mdae,
    mmre,
    nmae,
    pearson_corr,
    pred_x,
    r_squared,
    spearman_corr,
    standardized_accuracy,
)
from shared.r2_models import MODEL_REGISTRY, create_model
from shared.results_registry import WIB, generate_run_id, register_run, update_run
from shared.seeds import set_global_seeds

BASE = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
DEFAULT_FOLDS = BASE / "replications" / "cosmet-devito" / "deepCOSMIC"
DEFAULT_RESULTS = BASE / "results"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    assert "TitleTokens" in df.columns and "Cfp" in df.columns
    return df


def load_folds(folds_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(folds_dir / "10-fold-train-full.csv", index_col=0)
    val = pd.read_csv(folds_dir / "10-fold-val-full.csv", index_col=0)
    return train, val


def parse_fold_indices(row: pd.Series) -> list[int]:
    """Extract integer indices from a fold-split row (ignoring run/k cols)."""
    vals = row.drop(labels=["run", "k"], errors="ignore").dropna()
    return [int(v) for v in vals]


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "mdae": round(mdae(y_true, y_pred), 4),
        "mmre": round(mmre(y_true, y_pred), 4),
        "pred25": round(pred_x(y_true, y_pred, 0.25), 4),
        "r_squared": round(r_squared(y_true, y_pred), 4),
        "sa": round(standardized_accuracy(y_true, y_pred), 4),
        "nmae": round(nmae(y_true, y_pred), 4),
        "spearman_rho": round(spearman_corr(y_true, y_pred)["rho"], 4),
        "pearson_r": round(pearson_corr(y_true, y_pred)["r"], 4),
        **{f"err_{k}": round(v, 4)
           for k, v in error_distribution(y_true, y_pred).items()},
    }


# ------------------------------------------------------------------
# Cross-validation engine
# ------------------------------------------------------------------

def run_cv(
    model_name: str,
    df: pd.DataFrame,
    train_folds: pd.DataFrame,
    val_folds: pd.DataFrame,
    *,
    model_kwargs: dict | None = None,
    verbose: bool = True,
    max_runs: int | None = None,
) -> dict:
    """Run 10×10-fold CV and return structured results."""

    model_kwargs = model_kwargs or {}
    texts = df["TitleTokens"].tolist()
    cfps = df["Cfp"].values.astype(float)

    n_runs_total = int(train_folds["run"].max()) + 1
    n_runs = min(n_runs_total, max_runs) if max_runs else n_runs_total
    n_folds = int(train_folds["k"].max()) + 1

    cv_rows: list[dict] = []
    fold_metrics: list[dict] = []
    wall_t0 = time.perf_counter()
    fold_count = 0

    for run_idx in range(n_runs):
        for k_idx in range(n_folds):
            fold_count += 1
            mask = (train_folds["run"] == run_idx) & (train_folds["k"] == k_idx)
            tr_row = train_folds[mask].iloc[0]
            vl_row = val_folds[
                (val_folds["run"] == run_idx) & (val_folds["k"] == k_idx)
            ].iloc[0]

            tr_idx = parse_fold_indices(tr_row)
            vl_idx = parse_fold_indices(vl_row)

            X_tr = [texts[i] for i in tr_idx]
            y_tr = cfps[tr_idx]
            X_vl = [texts[i] for i in vl_idx]
            y_vl = cfps[vl_idx]

            run_kwargs = dict(model_kwargs)
            if model_name == "cnn":
                run_kwargs["seed"] = model_kwargs.get("seed", 102329) + run_idx
            model = create_model(model_name, **run_kwargs)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_vl)
            dur = time.perf_counter() - t0

            fold_mae = mae(y_vl, preds)
            fold_sa = standardized_accuracy(y_vl, preds)

            fold_metrics.append({
                "run": run_idx, "fold": k_idx,
                "mae": round(fold_mae, 4),
                "sa": round(fold_sa, 4),
                "duration_s": round(dur, 3),
            })

            for j, vi in enumerate(vl_idx):
                cv_rows.append({
                    "run": run_idx, "fold": k_idx,
                    "uc_idx": vi,
                    "y_true": float(y_vl[j]),
                    "y_pred": round(float(preds[j]), 4),
                })

            if verbose:
                pct = fold_count / (n_runs * n_folds) * 100
                print(
                    f"  [{fold_count:3d}/{n_runs*n_folds}] "
                    f"run={run_idx} fold={k_idx}  "
                    f"MAE={fold_mae:.3f}  SA={fold_sa:.3f}  "
                    f"({dur:.1f}s)  [{pct:.0f}%]"
                )

    wall_clock = time.perf_counter() - wall_t0

    cv_df = pd.DataFrame(cv_rows)
    avg_preds = cv_df.groupby("uc_idx").agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
    ).reset_index()

    overall = compute_all_metrics(
        avg_preds["y_true"].values, avg_preds["y_pred"].values,
    )

    fold_maes = [fm["mae"] for fm in fold_metrics]
    overall["mean_fold_mae"] = round(float(np.mean(fold_maes)), 4)
    overall["std_fold_mae"] = round(float(np.std(fold_maes, ddof=1)), 4)

    return {
        "cv_predictions": cv_df,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "comparison_results": avg_preds,
        "overall": overall,
        "wall_clock_s": round(wall_clock, 3),
        "n_folds": len(fold_metrics),
    }


# ------------------------------------------------------------------
# Output writing & registry integration
# ------------------------------------------------------------------

def save_results(
    results: dict,
    model_name: str,
    dataset: str,
    results_dir: Path,
) -> Path:
    run_id = generate_run_id(dataset, model_name)
    run_dir = results_dir / "deep-cosmic-uc" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results["cv_predictions"].to_csv(
        run_dir / "cv_predictions.csv", index=False,
    )
    results["fold_metrics"].to_csv(
        run_dir / "fold_metrics.csv", index=False,
    )
    results["comparison_results"].to_csv(
        run_dir / "comparison_results.csv", index=False,
    )

    metrics_json = {
        "meta": {
            "run_id": run_id,
            "method": "deep-cosmic-uc",
            "method_type": "replication",
            "dataset": dataset,
            "llm_model": model_name,
            "timestamp": datetime.now(WIB).isoformat(),
            "n_fps": int(results["comparison_results"].shape[0]),
            "n_errors": 0,
            "total_retries": 0,
            "cv_folds": results["n_folds"],
        },
        "tokens": {
            "total_input": 0, "total_output": 0, "total": 0,
            "mean_input_per_fp": 0, "mean_output_per_fp": 0,
        },
        "cost": {"total_usd": 0.0, "mean_per_fp_usd": 0.0},
        "performance": {
            "wall_clock_s": results["wall_clock_s"],
            "total_api_time_s": results["wall_clock_s"],
            "fp_per_min": round(
                results["n_folds"] / max(results["wall_clock_s"], 0.01) * 60, 2,
            ),
        },
        "accuracy": results["overall"],
    }

    with open(run_dir / "run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    register_run(results_dir, {
        "run_id": run_id,
        "method": "deep-cosmic-uc",
        "method_type": "replication",
        "dataset": dataset,
        "llm_model": model_name,
        "timestamp": datetime.now(WIB).isoformat(),
        "n_fps": int(results["comparison_results"].shape[0]),
        "n_errors": 0,
        "total_cfp_pred": int(results["comparison_results"]["y_pred"].sum()),
        "mae": results["overall"]["mae"],
        "mmre": results["overall"]["mmre"],
        "pred25": results["overall"]["pred25"],
        "wall_clock_s": results["wall_clock_s"],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "throughput_fp_min": round(
            results["n_folds"] / max(results["wall_clock_s"], 0.01) * 60, 2,
        ),
        "status": "ok",
    })

    return run_dir


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    set_global_seeds()
    parser = argparse.ArgumentParser(
        description="R2 DEEP-COSMIC-UC Replication — CV Runner",
    )
    parser.add_argument(
        "--model", required=True,
        choices=[*MODEL_REGISTRY.keys(), "all"],
        help="Model to run (cnn/rf/svr/bert/all)",
    )
    parser.add_argument(
        "--dataset", default="ochodek-uc-437",
        help="Dataset label for registry",
    )
    parser.add_argument(
        "--data-csv", type=Path, default=DEFAULT_DATA,
        help="Path to use-cases.csv",
    )
    parser.add_argument(
        "--folds-dir", type=Path, default=DEFAULT_FOLDS,
        help="Directory containing 10-fold-{train,val}-full.csv",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Root results directory",
    )
    parser.add_argument("--epochs", type=int, default=400, help="CNN epochs")
    parser.add_argument("--runs", type=int, default=None,
                        help="Limit number of CV runs (default: all in fold file)")
    parser.add_argument("--seed", type=int, default=102329, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("R2 DEEP-COSMIC-UC Replication")
    print("=" * 60)

    df = load_dataset(args.data_csv)
    train_folds, val_folds = load_folds(args.folds_dir)
    print(f"Dataset: {len(df)} UCs from {args.data_csv.name}")
    print(f"CV: {int(train_folds['run'].max())+1} runs × "
          f"{int(train_folds['k'].max())+1} folds")

    models_to_run = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    all_results: dict[str, dict] = {}

    for model_name in models_to_run:
        print(f"\n{'─'*60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'─'*60}")

        kwargs: dict = {}
        if model_name == "cnn":
            print("Loading GloVe embeddings...")
            import gensim.downloader as api
            glove = api.load("glove-wiki-gigaword-300")
            kwargs["glove_vectors"] = glove
            kwargs["epochs"] = args.epochs
            kwargs["seed"] = args.seed
        elif model_name == "rf":
            kwargs["seed"] = args.seed

        results = run_cv(
            model_name, df, train_folds, val_folds,
            model_kwargs=kwargs,
            max_runs=args.runs,
        )

        run_dir = save_results(results, model_name, args.dataset, args.results_dir)
        all_results[model_name] = results

        print(f"\n  Results → {run_dir}")
        print(f"  Overall metrics:")
        for k, v in results["overall"].items():
            if not k.startswith("err_"):
                print(f"    {k:20s} = {v}")

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-model comparison")
        print(f"{'='*60}")
        _print_comparison(all_results)


def _print_comparison(all_results: dict[str, dict]) -> None:
    from scipy import stats as sp_stats

    header = f"{'Metric':<20s}"
    for name in all_results:
        header += f"  {name:>10s}"
    print(header)
    print("─" * len(header))

    for metric in ["mae", "mean_fold_mae", "std_fold_mae", "sa", "mmre", "pred25",
                    "r_squared", "nmae", "spearman_rho"]:
        row = f"{metric:<20s}"
        for name in all_results:
            val = all_results[name]["overall"].get(metric, "")
            row += f"  {val:>10}"
        print(row)

    names = list(all_results.keys())
    if len(names) >= 2:
        print(f"\nPairwise Wilcoxon (fold MAEs):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = all_results[names[i]]["fold_metrics"]["mae"].values
                b = all_results[names[j]]["fold_metrics"]["mae"].values
                stat, p = sp_stats.wilcoxon(a, b)
                d = cliffs_delta(a, b)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {names[i]} vs {names[j]}: "
                      f"p={p:.4f} ({sig}), Cliff's d={d:.3f}")


if __name__ == "__main__":
    main()
