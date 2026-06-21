"""R3 HKO Replication — Hussain, Kosseim & Ormandjieva (2013).

Classifies functional processes into COSMIC size classes using 13 linguistic
features and a C4.5 decision tree.  Adapted replication on De Vito (119 FPs)
and Zenodo (2040 sentences) since the original 61-FP dataset is unavailable.

Usage examples (within-dataset CV):
    python scripts/r3_hko_runner.py --dataset devito --classifier c45
    python scripts/r3_hko_runner.py --dataset zenodo --classifier all
    python scripts/r3_hko_runner.py --dataset devito --classifier all --n-classes 4

Usage examples (cross-dataset transfer):
    python scripts/r3_hko_runner.py --train-dataset ochodek --eval-dataset devito --classifier c45
    python scripts/r3_hko_runner.py --train-dataset ochodek --eval-dataset devito --classifier all --n-classes 4

Legacy / backward-compat:
    python scripts/r3_hko_runner.py --dataset devito --classifier c45 --legacy-mode
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score, precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.seeds import set_global_seeds
from shared.r3_features import (
    build_keyword_lists, extract_feature_matrix,
    ALL_FEATURE_NAMES, SYNTACTIC_FEATURE_NAMES,
    extract_syntactic_matrix, extract_pos_tokens_batch,
    keyword_features_from_cache, assemble_features,
)
from shared.results_registry import WIB, generate_run_id, register_run
from shared.metrics import mae, standardized_accuracy, mmre, pred_x

BASE = Path(__file__).resolve().parent.parent
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

CLASSIFIER_REGISTRY = {
    "c45": lambda: DecisionTreeClassifier(
        criterion="entropy", min_samples_leaf=6, random_state=42,
    ),
    "nb": lambda: GaussianNB(),
    "logistic": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ),
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

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


def load_devito(results_dir: Path) -> pd.DataFrame:
    """Load De Vito 119 FPs with full text and CFP ground truth."""
    devito_base = BASE / "replications" / "cosmet-devito" / "datasets"
    rows = []
    for ds_name, ds_path in DEVITO_DATASETS.items():
        fp_dir = devito_base / ds_path / "FP_prompts"
        r1_matches = list((results_dir / "cosmet").glob(
            f"{ds_name}-*/measurements/comparison_results.csv"
        ))
        gt_dict = {}
        if r1_matches:
            gt_df = pd.read_csv(r1_matches[0])
            gt_dict = dict(zip(gt_df["fp"], gt_df["y_true"]))

        for f in sorted(fp_dir.glob("FP*.txt"),
                        key=lambda p: int(re.search(r"(\d+)", p.stem).group())):
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            cfp = gt_dict.get(f.stem)
            if cfp is None:
                continue
            rows.append({
                "id": f"{ds_name}_{f.stem}",
                "text": text,
                "cfp": int(cfp),
                "dataset_source": ds_name,
            })
    return pd.DataFrame(rows)


def load_zenodo() -> pd.DataFrame:
    """Load Zenodo heterogeneous dataset (2040+ sentences)."""
    csv_path = BASE / "datasets" / "raw" / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["requirement_en", "CFP"])
    return pd.DataFrame({
        "id": [f"Z{i}" for i in range(len(df))],
        "text": df["requirement_en"].values,
        "cfp": df["CFP"].astype(int).values,
        "dataset_source": "zenodo",
    })


def load_ochodek() -> pd.DataFrame:
    """Load Ochodek-437 UC names (short titles, ~3 tokens each)."""
    csv_path = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
    df = pd.read_csv(csv_path)
    return pd.DataFrame({
        "id": df["UC"].values,
        "text": df["TitleTokens"].values,
        "cfp": df["Cfp"].astype(int).values,
        "dataset_source": "ochodek",
    })


def load_user_stories() -> pd.DataFrame:
    """Load R4 user-stories-98 (Dalpiaz) with adjudicated COSMIC ground truth.

    HKO (Hussain) was designed for agile user stories, so this is a home
    dataset for R3. Reads the FP*.txt inputs + adjudicated ground_truth.csv.
    """
    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = pd.read_csv(fp_dir / "ground_truth.csv")
    return pd.DataFrame({
        "id": gt["us_id"].values,
        "text": gt["text"].values,
        "cfp": gt["cfp_gt"].astype(int).values,
        "dataset_source": "user-stories",
    })


# ------------------------------------------------------------------
# Box-plot class annotation (HKO Section 4.2)
# ------------------------------------------------------------------

CLASS_ORDER_4 = ["Small", "Medium", "Large", "Complex"]
CLASS_ORDER_3 = ["Small", "Medium", "Large"]
CLASS_ORDER_2 = ["Small", "Large"]


def annotate_classes(cfp_values: np.ndarray, n_classes: int = 4) -> tuple[
    np.ndarray, dict[str, dict]
]:
    """Assign CFP values to size classes using box-plot quartiles.

    Returns:
        labels: array of class label strings
        class_data: dict mapping class name to {median, min, max, range}
    """
    q1 = np.percentile(cfp_values, 25)
    median = np.percentile(cfp_values, 50)
    q3 = np.percentile(cfp_values, 75)

    if n_classes == 4:
        boundaries = [
            ("Small", 0, q1),
            ("Medium", q1, median),
            ("Large", median, q3),
            ("Complex", q3, float("inf")),
        ]
    elif n_classes == 3:
        boundaries = [
            ("Small", 0, q1),
            ("Medium", q1, q3),
            ("Large", q3, float("inf")),
        ]
    else:
        boundaries = [
            ("Small", 0, median),
            ("Large", median, float("inf")),
        ]

    labels = np.empty(len(cfp_values), dtype=object)
    for name, lo, hi in boundaries:
        mask = (cfp_values >= lo) & (cfp_values < hi)
        labels[mask] = name

    # Edge case: items exactly at boundaries (e.g., == q1) may be unassigned
    # if lo == hi for adjacent classes. Also, the top class must capture items
    # exactly equal to its lower bound.
    unassigned = labels == None  # noqa: E711
    if unassigned.any():
        for name, lo, hi in reversed(boundaries):
            still_na = labels == None  # noqa: E711
            mask = still_na & (cfp_values >= lo)
            labels[mask] = name

    class_data = {}
    for name, lo, hi in boundaries:
        cls_vals = cfp_values[labels == name]
        if len(cls_vals) > 0:
            class_data[name] = {
                "median": float(np.median(cls_vals)),
                "min": float(cls_vals.min()),
                "max": float(cls_vals.max()),
                "count": int(len(cls_vals)),
                "range": [lo, hi],
            }
        else:
            class_data[name] = {
                "median": float((lo + min(hi, cfp_values.max())) / 2),
                "min": float(lo),
                "max": float(min(hi, cfp_values.max())),
                "count": 0,
                "range": [lo, hi],
            }

    return labels, class_data


# ------------------------------------------------------------------
# Cross-validation engine
# ------------------------------------------------------------------

def run_cv(
    texts: list[str],
    cfp_values: np.ndarray,
    labels: np.ndarray,
    class_data: dict[str, dict],
    classifier_name: str,
    *,
    n_folds: int = 10,
    n_classes: int = 4,
    verbose: bool = True,
    syntactic_cache: np.ndarray | None = None,
    pos_tokens_cache: list | None = None,
) -> dict:
    """Run stratified 10-fold CV and return structured results.

    Pass syntactic_cache and pos_tokens_cache to avoid redundant NLP calls
    across folds and classifiers (the expensive spaCy parsing is done once).
    """
    min_class_size = min(Counter(labels).values())
    effective_folds = min(n_folds, min_class_size)
    if effective_folds < n_folds and verbose:
        print(f"  [!] Reducing folds from {n_folds} to {effective_folds} "
              f"(smallest class has {min_class_size} samples)")

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_cfp_true = []
    all_cfp_pred = []
    total_time = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        val_cfp = cfp_values[val_idx]

        t0 = time.perf_counter()

        keyword_lists = build_keyword_lists(
            train_texts, train_labels.tolist(), top_k=20,
        )

        if syntactic_cache is not None and pos_tokens_cache is not None:
            train_kw = keyword_features_from_cache(
                [pos_tokens_cache[i] for i in train_idx], keyword_lists)
            val_kw = keyword_features_from_cache(
                [pos_tokens_cache[i] for i in val_idx], keyword_lists)
            X_train = assemble_features(syntactic_cache[train_idx], train_kw)
            X_val = assemble_features(syntactic_cache[val_idx], val_kw)
        else:
            val_texts = [texts[i] for i in val_idx]
            X_train = extract_feature_matrix(train_texts, keyword_lists)
            X_val = extract_feature_matrix(val_texts, keyword_lists)

        clf = CLASSIFIER_REGISTRY[classifier_name]()
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_val)

        dur = time.perf_counter() - t0
        total_time += dur

        cfp_pred = np.array([
            class_data.get(p, {}).get("median", 0) for p in preds
        ])

        fold_acc = accuracy_score(val_labels, preds)
        fold_kappa = cohen_kappa_score(val_labels, preds)

        fold_results.append({
            "fold": fold_idx,
            "accuracy": round(fold_acc, 4),
            "kappa": round(fold_kappa, 4),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "duration_s": round(dur, 3),
            "cfp_mae": round(float(np.mean(np.abs(val_cfp - cfp_pred))), 4),
        })

        all_y_true.extend(val_labels.tolist())
        all_y_pred.extend(preds.tolist())
        all_cfp_true.extend(val_cfp.tolist())
        all_cfp_pred.extend(cfp_pred.tolist())

        if verbose:
            print(f"  Fold {fold_idx:2d}: acc={fold_acc:.3f}  "
                  f"kappa={fold_kappa:.3f}  CFP-MAE={fold_results[-1]['cfp_mae']:.2f}  "
                  f"({dur:.1f}s)")

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    cfp_true_arr = np.array(all_cfp_true, dtype=float)
    cfp_pred_arr = np.array(all_cfp_pred, dtype=float)

    class_order = (
        CLASS_ORDER_4 if n_classes == 4
        else CLASS_ORDER_3 if n_classes == 3
        else CLASS_ORDER_2
    )
    present_classes = [c for c in class_order if c in set(all_y_true) | set(all_y_pred)]

    overall_acc = accuracy_score(y_true_arr, y_pred_arr)
    overall_kappa = cohen_kappa_score(y_true_arr, y_pred_arr)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=present_classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=present_classes, zero_division=0,
    )

    per_class = {}
    for i, cls_name in enumerate(present_classes):
        per_class[cls_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    cfp_approx_mae = mae(cfp_true_arr, cfp_pred_arr)
    cfp_approx_sa = standardized_accuracy(cfp_true_arr, cfp_pred_arr)

    return {
        "classifier": classifier_name,
        "n_classes": n_classes,
        "n_samples": len(texts),
        "n_folds": effective_folds,
        "overall": {
            "accuracy": round(overall_acc, 4),
            "kappa": round(overall_kappa, 4),
            "mean_precision": round(float(np.mean(prec)), 4),
            "mean_recall": round(float(np.mean(rec)), 4),
            "mean_f1": round(float(np.mean(f1)), 4),
        },
        "cfp_approximation": {
            "mae": round(cfp_approx_mae, 4),
            "sa": round(cfp_approx_sa, 4),
            "mmre": round(mmre(cfp_true_arr, cfp_pred_arr), 4),
            "pred25": round(pred_x(cfp_true_arr, cfp_pred_arr, 0.25), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present_classes,
            "matrix": cm.tolist(),
        },
        "fold_results": fold_results,
        "wall_clock_s": round(total_time, 3),
        "class_data": class_data,
    }


# ------------------------------------------------------------------
# Transfer (cross-dataset) engine
# ------------------------------------------------------------------

def run_transfer(
    train_texts: list[str],
    train_cfp: np.ndarray,
    eval_texts: list[str],
    eval_cfp: np.ndarray,
    eval_labels: np.ndarray,
    eval_class_data: dict[str, dict],
    classifier_name: str,
    n_classes: int,
    verbose: bool = True,
    train_label: str = "",
    eval_label: str = "",
) -> dict:
    """Fit classifier on entire train set, evaluate on a separate eval set.

    HKO box-plot boundaries are computed independently per dataset —
    train uses its own distribution to annotate classes (for fitting),
    and eval uses its own distribution (for evaluation).

    Returns a dict with the same shape as run_cv so save_results works,
    but n_folds=0, fold_results=[], and transfer_mode=True.
    """
    t0 = time.perf_counter()

    # Annotate train classes using train-set CFP distribution
    train_labels, _ = annotate_classes(train_cfp, n_classes)

    # Build features for the full train set
    keyword_lists = build_keyword_lists(
        train_texts, train_labels.tolist(), top_k=20,
    )
    X_train = extract_feature_matrix(train_texts, keyword_lists)
    X_eval = extract_feature_matrix(eval_texts, keyword_lists)

    # Fit and predict
    clf = CLASSIFIER_REGISTRY[classifier_name]()
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_eval)

    dur = time.perf_counter() - t0

    # Convert predicted class labels → CFP via eval_class_data medians
    # (eval dataset's own class boundaries are the reference for CFP approx)
    cfp_pred_arr = np.array([
        eval_class_data.get(p, {}).get("median", 0) for p in preds
    ], dtype=float)
    cfp_true_arr = eval_cfp.astype(float)

    if verbose:
        print(f"  Transfer fit+predict: {dur:.1f}s  "
              f"(train={len(train_texts)}, eval={len(eval_texts)})")

    overall_acc = accuracy_score(eval_labels, preds)
    overall_kappa = cohen_kappa_score(eval_labels, preds)

    class_order = (
        CLASS_ORDER_4 if n_classes == 4
        else CLASS_ORDER_3 if n_classes == 3
        else CLASS_ORDER_2
    )
    present_classes = [
        c for c in class_order if c in set(eval_labels) | set(preds)
    ]

    cm = confusion_matrix(eval_labels, preds, labels=present_classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        eval_labels, preds, labels=present_classes, zero_division=0,
    )

    per_class = {}
    for i, cls_name in enumerate(present_classes):
        per_class[cls_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    cfp_approx_mae = mae(cfp_true_arr, cfp_pred_arr)
    cfp_approx_sa = standardized_accuracy(cfp_true_arr, cfp_pred_arr)

    return {
        "classifier": classifier_name,
        "n_classes": n_classes,
        "n_samples": len(eval_texts),
        "n_folds": 0,
        "transfer_mode": True,
        "train_dataset": train_label,
        "eval_dataset": eval_label,
        "overall": {
            "accuracy": round(overall_acc, 4),
            "kappa": round(overall_kappa, 4),
            "mean_precision": round(float(np.mean(prec)), 4),
            "mean_recall": round(float(np.mean(rec)), 4),
            "mean_f1": round(float(np.mean(f1)), 4),
        },
        "cfp_approximation": {
            "mae": round(cfp_approx_mae, 4),
            "sa": round(cfp_approx_sa, 4),
            "mmre": round(mmre(cfp_true_arr, cfp_pred_arr), 4),
            "pred25": round(pred_x(cfp_true_arr, cfp_pred_arr, 0.25), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present_classes,
            "matrix": cm.tolist(),
        },
        "fold_results": [],
        "wall_clock_s": round(dur, 3),
        "class_data": eval_class_data,
    }


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------

def save_results(
    results: dict,
    dataset_label: str,
    results_dir: Path,
) -> Path:
    """Persist results to disk and register in runs_registry."""
    tag = f"{results['classifier']}-{results['n_classes']}cls"
    run_id = generate_run_id(f"hko-{dataset_label}", tag)
    out_dir = results_dir / "hko-hussain" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cm_labels = results["confusion_matrix"]["labels"]
    cm_df = pd.DataFrame(
        results["confusion_matrix"]["matrix"],
        index=cm_labels, columns=cm_labels,
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    pd.DataFrame(results["fold_results"]).to_csv(
        out_dir / "fold_results.csv", index=False,
    )

    is_transfer = results.get("transfer_mode", False)
    # throughput_fp_min: for CV use n_folds, for transfer use n_samples
    if is_transfer:
        throughput = round(
            results["n_samples"] / max(results["wall_clock_s"], 0.01) * 60, 2,
        )
    else:
        throughput = round(
            results["n_folds"] / max(results["wall_clock_s"], 0.01) * 60, 2,
        )

    register_run(results_dir, {
        "run_id": run_id,
        "method": "hko-hussain",
        "method_type": "replication",
        "dataset": f"hko-{dataset_label}",
        "llm_model": results["classifier"],
        "timestamp": datetime.now(WIB).isoformat(),
        "n_fps": results["n_samples"],
        "n_errors": 0,
        "total_cfp_pred": 0,
        "mae": results["cfp_approximation"]["mae"],
        "mmre": results["cfp_approximation"]["mmre"],
        "pred25": results["cfp_approximation"]["pred25"],
        "wall_clock_s": results["wall_clock_s"],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "throughput_fp_min": throughput,
        "train_dataset": results.get("train_dataset", ""),
        "eval_dataset": results.get("eval_dataset", ""),
        "status": "ok",
    })

    return out_dir


# ------------------------------------------------------------------
# Dataset loading helpers
# ------------------------------------------------------------------

_DATASET_LABELS = {
    "devito": "devito-119",
    "ochodek": "ochodek-437",
    "zenodo": "zenodo-2040",
    "user-stories": "user-stories-98",
}


def _load_dataset(name: str, results_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load a named dataset and return (df, label)."""
    if name == "devito":
        return load_devito(results_dir), "devito-119"
    elif name == "ochodek":
        return load_ochodek(), "ochodek-437"
    elif name == "user-stories":
        return load_user_stories(), "user-stories-98"
    else:
        return load_zenodo(), "zenodo-2040"


def _print_dataset_stats(label: str, df: pd.DataFrame, cfp_values: np.ndarray) -> None:
    print(f"  Dataset: {label} ({len(df)} items)")
    print(f"  CFP range: {cfp_values.min():.0f}–{cfp_values.max():.0f}, "
          f"mean={cfp_values.mean():.2f}, median={np.median(cfp_values):.1f}")
    print(f"  Q1={np.percentile(cfp_values, 25):.1f}, "
          f"Q3={np.percentile(cfp_values, 75):.1f}")


def _print_results_block(results: dict) -> None:
    """Print overall + CFP + confusion matrix + per-class for any result dict."""
    o = results["overall"]
    c = results["cfp_approximation"]
    print(f"\n  OVERALL: acc={o['accuracy']:.4f}  "
          f"kappa={o['kappa']:.4f}  "
          f"mean_f1={o['mean_f1']:.4f}")
    print(f"  CFP APPROX: MAE={c['mae']:.3f}  "
          f"SA={c['sa']:.4f}  "
          f"MMRE={c['mmre']:.4f}  "
          f"PRED25={c['pred25']:.4f}")

    cm_labels = results["confusion_matrix"]["labels"]
    cm_matrix = np.array(results["confusion_matrix"]["matrix"])
    print(f"\n  Confusion Matrix:")
    header = "  {:>10s}".format("") + "".join(f" {lbl:>8s}" for lbl in cm_labels)
    print(header)
    for i, lbl in enumerate(cm_labels):
        row_str = "  {:>10s}".format(lbl) + "".join(
            f" {cm_matrix[i, j]:>8d}" for j in range(len(cm_labels))
        )
        print(row_str)

    print(f"\n  Per-class metrics:")
    for cls_name, pc in results["per_class"].items():
        print(f"    {cls_name:>10s}: P={pc['precision']:.3f}  "
              f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
              f"n={pc['support']}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    set_global_seeds()
    parser = argparse.ArgumentParser(
        description="R3 HKO Replication — COSMIC Size Class Classification",
    )
    # Original flag — kept for backward compatibility
    parser.add_argument(
        "--dataset", default=None, choices=["devito", "zenodo", "ochodek", "user-stories"],
        help="Dataset for within-dataset CV (legacy mode). "
             "Required unless --train-dataset/--eval-dataset are used.",
    )
    # Cross-dataset transfer flags
    parser.add_argument(
        "--train-dataset", default=None, choices=["devito", "zenodo", "ochodek", "user-stories"],
        help="Dataset to fit the classifier on (cross-dataset mode).",
    )
    parser.add_argument(
        "--eval-dataset", default=None, choices=["devito", "zenodo", "ochodek", "user-stories"],
        help="Dataset to evaluate on (cross-dataset mode).",
    )
    parser.add_argument(
        "--legacy-mode", action="store_true",
        help="Force legacy CV mode even when --train-dataset/--eval-dataset are provided.",
    )
    parser.add_argument(
        "--classifier", required=True,
        choices=[*CLASSIFIER_REGISTRY.keys(), "all"],
        help="Classifier to use (c45/nb/logistic/all)",
    )
    parser.add_argument(
        "--n-classes", type=int, default=0,
        help="Number of size classes (2/3/4, default: run all three variants)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Root results directory",
    )
    parser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of CV folds (legacy mode only)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine mode
    # ------------------------------------------------------------------
    cross_dataset_mode = (
        not args.legacy_mode
        and args.train_dataset is not None
        and args.eval_dataset is not None
    )
    legacy_mode = not cross_dataset_mode

    if legacy_mode and args.dataset is None:
        parser.error(
            "Either --dataset (legacy/CV mode) or both --train-dataset and "
            "--eval-dataset (cross-dataset mode) must be provided."
        )

    classifiers = (
        list(CLASSIFIER_REGISTRY.keys()) if args.classifier == "all"
        else [args.classifier]
    )
    class_variants = (
        [args.n_classes] if args.n_classes in (2, 3, 4)
        else [4, 3, 2]
    )

    print("=" * 60)
    print("R3 HKO Replication — COSMIC Size Class Classification")
    print("=" * 60)

    # ==================================================================
    # CROSS-DATASET TRANSFER MODE
    # ==================================================================
    if cross_dataset_mode:
        train_df, train_label = _load_dataset(args.train_dataset, args.results_dir)
        eval_df, eval_label = _load_dataset(args.eval_dataset, args.results_dir)

        train_texts = train_df["text"].tolist()
        train_cfp = train_df["cfp"].values.astype(float)
        eval_texts = eval_df["text"].tolist()
        eval_cfp = eval_df["cfp"].values.astype(float)

        dataset_label = f"{train_label}-to-{eval_label}"

        print(f"Mode: CROSS-DATASET TRANSFER  ({train_label} → {eval_label})")
        print(f"Train dataset:")
        _print_dataset_stats(train_label, train_df, train_cfp)
        print(f"Eval dataset:")
        _print_dataset_stats(eval_label, eval_df, eval_cfp)

        all_summaries: list[dict] = []

        for n_cls in class_variants:
            # Each dataset uses its own CFP distribution to set class boundaries
            _, eval_class_data = annotate_classes(eval_cfp, n_cls)
            eval_labels, _ = annotate_classes(eval_cfp, n_cls)
            eval_class_dist = Counter(eval_labels)

            print(f"\n{'=' * 60}")
            print(f"  {n_cls}-CLASS VARIANT  [train={train_label}, eval={eval_label}]")
            print(f"{'=' * 60}")
            print(f"  Eval class distribution: {dict(eval_class_dist)}")
            for cls_name, cd in eval_class_data.items():
                print(f"    {cls_name}: n={cd['count']}, "
                      f"median_cfp={cd['median']:.1f}, "
                      f"range=[{cd['min']:.0f}, {cd['max']:.0f}]")

            for clf_name in classifiers:
                print(f"\n  ── Classifier: {clf_name.upper()} ──")

                results = run_transfer(
                    train_texts=train_texts,
                    train_cfp=train_cfp,
                    eval_texts=eval_texts,
                    eval_cfp=eval_cfp,
                    eval_labels=eval_labels,
                    eval_class_data=eval_class_data,
                    classifier_name=clf_name,
                    n_classes=n_cls,
                    verbose=True,
                    train_label=train_label,
                    eval_label=eval_label,
                )

                out_dir = save_results(results, dataset_label, args.results_dir)
                _print_results_block(results)

                o = results["overall"]
                c = results["cfp_approximation"]
                all_summaries.append({
                    "n_classes": n_cls,
                    "classifier": clf_name,
                    "accuracy": o["accuracy"],
                    "kappa": o["kappa"],
                    "mean_f1": o["mean_f1"],
                    "cfp_mae": c["mae"],
                    "cfp_sa": c["sa"],
                    "wall_clock_s": results["wall_clock_s"],
                    "train_dataset": train_label,
                    "eval_dataset": eval_label,
                })
                print(f"  Saved to: {out_dir}")

        summary_dir = args.results_dir / "hko-hussain"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(all_summaries)
        summary_path = summary_dir / f"r3_summary_{dataset_label}.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 60}")
        print("CONSOLIDATED SUMMARY (TRANSFER)")
        print(f"{'=' * 60}")
        print(f"\n{'Cls':>3s} {'Classifier':<12s} {'Acc':>7s} {'Kappa':>7s} "
              f"{'F1':>7s} {'CFP-MAE':>8s} {'CFP-SA':>8s} {'Time':>7s}")
        print("─" * 65)
        for _, row in summary_df.iterrows():
            print(f"{int(row['n_classes']):>3d} {row['classifier']:<12s} "
                  f"{row['accuracy']:>7.4f} {row['kappa']:>7.4f} "
                  f"{row['mean_f1']:>7.4f} {row['cfp_mae']:>8.3f} "
                  f"{row['cfp_sa']:>8.4f} {row['wall_clock_s']:>6.1f}s")
        print(f"\nSummary saved to: {summary_path}")

        return  # Done — skip legacy section below

    # ==================================================================
    # LEGACY / WITHIN-DATASET CV MODE
    # ==================================================================
    df, dataset_label = _load_dataset(args.dataset, args.results_dir)
    texts = df["text"].tolist()
    cfp_values = df["cfp"].values.astype(float)

    print(f"Mode: WITHIN-DATASET CV")
    _print_dataset_stats(dataset_label, df, cfp_values)

    print("\nPre-computing linguistic features (one-time NLP pass)...")
    t_feat = time.perf_counter()
    syn_cache = extract_syntactic_matrix(texts)
    pos_cache = extract_pos_tokens_batch(texts)
    feat_time = time.perf_counter() - t_feat
    print(f"  Feature extraction: {feat_time:.1f}s for {len(texts)} texts")

    all_summaries: list[dict] = []

    for n_cls in class_variants:
        labels, class_data = annotate_classes(cfp_values, n_cls)
        class_dist = Counter(labels)

        print(f"\n{'=' * 60}")
        print(f"  {n_cls}-CLASS VARIANT")
        print(f"{'=' * 60}")
        print(f"  Class distribution: {dict(class_dist)}")
        for cls_name, cd in class_data.items():
            print(f"    {cls_name}: n={cd['count']}, "
                  f"median_cfp={cd['median']:.1f}, "
                  f"range=[{cd['min']:.0f}, {cd['max']:.0f}]")

        for clf_name in classifiers:
            print(f"\n  ── Classifier: {clf_name.upper()} ──")

            results = run_cv(
                texts, cfp_values, labels, class_data,
                clf_name, n_folds=args.n_folds, n_classes=n_cls,
                syntactic_cache=syn_cache, pos_tokens_cache=pos_cache,
            )

            out_dir = save_results(results, dataset_label, args.results_dir)
            _print_results_block(results)

            o = results["overall"]
            c = results["cfp_approximation"]
            all_summaries.append({
                "n_classes": n_cls,
                "classifier": clf_name,
                "accuracy": o["accuracy"],
                "kappa": o["kappa"],
                "mean_f1": o["mean_f1"],
                "cfp_mae": c["mae"],
                "cfp_sa": c["sa"],
                "wall_clock_s": results["wall_clock_s"],
            })
            print(f"  Saved to: {out_dir}")

    summary_dir = args.results_dir / "hko-hussain"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_summaries)
    summary_path = summary_dir / f"r3_summary_{dataset_label}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("CONSOLIDATED SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Cls':>3s} {'Classifier':<12s} {'Acc':>7s} {'Kappa':>7s} "
          f"{'F1':>7s} {'CFP-MAE':>8s} {'CFP-SA':>8s} {'Time':>7s}")
    print("─" * 65)
    for _, row in summary_df.iterrows():
        print(f"{int(row['n_classes']):>3d} {row['classifier']:<12s} "
              f"{row['accuracy']:>7.4f} {row['kappa']:>7.4f} "
              f"{row['mean_f1']:>7.4f} {row['cfp_mae']:>8.3f} "
              f"{row['cfp_sa']:>8.4f} {row['wall_clock_s']:>6.1f}s")

    print(f"\nSummary saved to: {summary_path}")

    print("\n── HKO Benchmark Comparison ──")
    print("  Original (Hussain 2013, 61 FPs):")
    print("    4-class: acc=67.21%, kappa=0.5485")
    print("    3-class: kappa=0.575")
    print("    2-class: kappa=0.606")


if __name__ == "__main__":
    main()
