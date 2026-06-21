#!/usr/bin/env python3
"""R2 Cross-Dataset Evaluation: ML models trained on Ochodek-437.

Trains R2 models (CNN, RF, SVR, BERT) on Ochodek-437, then evaluates on
a target dataset in the following conditions:
  devito  — 119 De Vito UCs (name-only / full-flow conditions)
  zenodo  — Zenodo 2040 heterogeneous sentences (sentence condition)

Compares against R1 COSMET results on the same eval set.

Usage:
    python scripts/r2_cross_dataset_eval.py --model all
    python scripts/r2_cross_dataset_eval.py --model rf
    python scripts/r2_cross_dataset_eval.py --model rf --eval-target zenodo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
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
from shared.results_registry import WIB, generate_run_id, register_run

BASE = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
DEFAULT_RESULTS = BASE / "results"

DEVITO_DATASETS = {
    "ALBERGATE": "MIS/ALBERGATE",
    "K01719": "IoT_Telemedicine/K01719",
    "K01720": "IoT_Telemedicine/K01720",
    "K01726": "IoT_Telemedicine/K01726",
    "U-CURE": "ML/U-CURE",
    "AutomaticLineSwitching": "Real-time/AutomaticLineSwitching",
    "RiceCooker": "Real-time/RiceCooker",
}


def extract_devito_ucs(devito_base: Path, r1_results: Path) -> pd.DataFrame:
    """Extract UC names, full text, and CFP ground truth from De Vito datasets."""
    rows = []
    for ds_name, ds_path in DEVITO_DATASETS.items():
        fp_dir = devito_base / ds_path / "FP_prompts"
        r1_matches = list(r1_results.glob(
            f"{ds_name}-*/measurements/comparison_results.csv"
        ))
        gt_dict = {}
        if r1_matches:
            gt_df = pd.read_csv(r1_matches[0])
            gt_dict = dict(zip(gt_df["fp"], gt_df["y_true"]))

        for f in sorted(fp_dir.glob("FP*.txt"),
                        key=lambda p: int(re.search(r"(\d+)", p.stem).group())):
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            lines = text.split("\n")
            name = _extract_uc_name(lines)
            cfp = gt_dict.get(f.stem)
            if cfp is None:
                continue

            rows.append({
                "dataset": ds_name,
                "fp_id": f.stem,
                "uc_name": name,
                "uc_full_text": text,
                "cfp": int(cfp),
            })

    return pd.DataFrame(rows)


def _extract_uc_name(lines: list[str]) -> str:
    first = lines[0].strip()
    if first == "ID" and len(lines) > 1:
        parts = lines[1].split("\t")
        if len(parts) >= 2:
            return parts[1].strip()
    for sep in ["–", "—", "-"]:
        if sep in first:
            candidate = first.split(sep, 1)[-1].strip()
            if len(candidate) > 2:
                return candidate
    return first


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
    }


def run_eval(
    model_name: str,
    train_texts: list[str],
    train_y: np.ndarray,
    test_texts: list[str],
    test_y: np.ndarray,
    condition: str,
    model_kwargs: dict | None = None,
) -> dict:
    """Train on Ochodek-437, predict on De Vito UCs."""
    model_kwargs = model_kwargs or {}
    model = create_model(model_name, **model_kwargs)

    t0 = time.perf_counter()
    model.fit(train_texts, train_y)
    preds = model.predict(test_texts)
    duration = time.perf_counter() - t0

    metrics = compute_metrics(test_y, preds)
    metrics["duration_s"] = round(duration, 3)
    metrics["condition"] = condition

    return {"metrics": metrics, "predictions": preds}


def load_zenodo(sample_csv: Path | None = None) -> pd.DataFrame:
    """Load Zenodo heterogeneous sentences, optionally from a stratified sample CSV."""
    if sample_csv and sample_csv.exists():
        df = pd.read_csv(sample_csv)
        return pd.DataFrame({
            "dataset": "zenodo",
            "fp_id": df["id"].values if "id" in df.columns else [f"Z{i}" for i in range(len(df))],
            "uc_name": df["text"].values,
            "uc_full_text": df["text"].values,
            "cfp": df["CFP"].astype(int).values if "CFP" in df.columns else df["cfp"].astype(int).values,
        })
    src = BASE / "datasets" / "raw" / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
    df = pd.read_csv(src)
    df = df.dropna(subset=["requirement_en", "CFP"])
    return pd.DataFrame({
        "dataset": "zenodo",
        "fp_id": [f"Z{i}" for i in range(len(df))],
        "uc_name": df["requirement_en"].values,
        "uc_full_text": df["requirement_en"].values,
        "cfp": df["CFP"].astype(int).values,
    })


def load_r1_results(r1_results: Path) -> pd.DataFrame:
    """Load R1 COSMET predictions for the same 119 UCs."""
    rows = []
    for ds_name in DEVITO_DATASETS:
        matches = list(r1_results.glob(
            f"{ds_name}-*/measurements/comparison_results.csv"
        ))
        if matches:
            df = pd.read_csv(matches[0])
            df["dataset"] = ds_name
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R2 Cross-Dataset Evaluation — Ochodek-437 train, configurable eval"
    )
    parser.add_argument(
        "--model", required=True,
        choices=[*MODEL_REGISTRY.keys(), "all"],
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=102329)
    parser.add_argument(
        "--eval-target", default="devito", choices=["devito", "zenodo"],
        help="Evaluation dataset: devito (119 UCs, default) or zenodo (2040 sentences)",
    )
    parser.add_argument(
        "--zenodo-sample-csv", type=Path, default=None,
        help="Optional pre-sampled Zenodo CSV (from cross-sample/). Uses full dataset if omitted.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"R2 Cross-Dataset Evaluation — train=Ochodek-437  eval={args.eval_target}")
    print("=" * 60)

    train_df = pd.read_csv(args.train_csv, index_col=0)
    train_texts = train_df["TitleTokens"].tolist()
    train_y = train_df["Cfp"].values.astype(float)
    print(f"Training set: {len(train_df)} UCs (Ochodek-437)")

    if args.eval_target == "zenodo":
        test_df = load_zenodo(args.zenodo_sample_csv)
        eval_label = "zenodo"
        conditions = [("sentence", "uc_full_text")]
        print(f"Test set: {len(test_df)} sentences (Zenodo heterogeneous)")
    else:
        devito_base = BASE / "replications" / "cosmet-devito" / "datasets"
        test_df = extract_devito_ucs(devito_base, args.results_dir / "cosmet")
        eval_label = "devito"
        conditions = [("name-only", "uc_name"), ("full-flow", "uc_full_text")]
        print(f"Test set: {len(test_df)} UCs (De Vito, 7 datasets)")

    print(f"CFP range: {test_df['cfp'].min()}–{test_df['cfp'].max()}, "
          f"mean={test_df['cfp'].mean():.2f}")

    models_to_run = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    out_dir = args.results_dir / "deep-cosmic-uc" / f"cross-dataset-{eval_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for model_name in models_to_run:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'─' * 60}")

        kwargs: dict = {}
        if model_name == "cnn":
            print("  Loading GloVe embeddings...")
            import gensim.downloader as api
            glove = api.load("glove-wiki-gigaword-300")
            kwargs["glove_vectors"] = glove
            kwargs["epochs"] = args.epochs
            kwargs["seed"] = args.seed
        elif model_name == "rf":
            kwargs["seed"] = args.seed

        for condition, text_col in conditions:
            test_texts = test_df[text_col].tolist()
            test_y = test_df["cfp"].values.astype(float)

            # Ochodek training set only has UC titles, so train_input is
            # identical for both conditions. Test texts vary (name vs full flow).
            train_input = train_texts

            result = run_eval(
                model_name, train_input, train_y,
                test_texts, test_y, condition,
                model_kwargs=kwargs,
            )

            m = result["metrics"]
            print(f"\n  [{condition}] MAE={m['mae']:.3f}  SA={m['sa']:.3f}  "
                  f"MMRE={m['mmre']:.3f}  PRED25={m['pred25']:.3f}  "
                  f"({m['duration_s']:.1f}s)")

            all_results.append({
                "model": model_name,
                "condition": condition,
                **m,
            })

    if eval_label == "devito":
        r1_df = load_r1_results(args.results_dir / "cosmet")
        if not r1_df.empty:
            r1_y_true = r1_df["y_true"].values.astype(float)
            r1_y_pred = r1_df["y_pred"].values.astype(float)
            r1_m = compute_metrics(r1_y_true, r1_y_pred)
            all_results.append({
                "model": "cosmet-gpt4-turbo",
                "condition": "full-flow",
                **r1_m,
                "duration_s": 0,
            })
            print(f"\n{'─' * 60}")
            print(f"R1 COSMET (reference)")
            print(f"{'─' * 60}")
            print(f"  [full-flow] MAE={r1_m['mae']:.3f}  SA={r1_m['sa']:.3f}  "
                  f"MMRE={r1_m['mmre']:.3f}  PRED25={r1_m['pred25']:.3f}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_dir / "cross_dataset_summary.csv", index=False)

    print(f"\n{'=' * 60}")
    print("CONSOLIDATED COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<25s} {'Condition':<12s} {'MAE':>6s} {'SA':>8s} "
          f"{'MMRE':>8s} {'PRED25':>8s} {'R²':>8s}")
    print("─" * 80)
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<25s} {r['condition']:<12s} "
              f"{r['mae']:>6.3f} {r['sa']:>8.4f} "
              f"{r['mmre']:>8.4f} {r['pred25']:>8.4f} "
              f"{r.get('r_squared', ''):>8}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
