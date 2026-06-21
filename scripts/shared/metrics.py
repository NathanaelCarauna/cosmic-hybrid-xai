"""Statistical metrics for COSMIC measurement evaluation.

Covers all metrics used across the RSL (P11–P51) plus methodological
best-practices (MBRE/MIBRE per Shepperd & MacDonell 2012, Cliff's delta).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mdae(y_true, y_pred):
    """Median Absolute Error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.median(np.abs(y_true - y_pred)))


def mmre(y_true, y_pred):
    """Mean Magnitude of Relative Error (ignores zero-true entries)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def pred_x(y_true, y_pred, threshold: float = 0.25):
    """PRED(x): fraction of predictions within *threshold* relative error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    mre = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(mre <= threshold))


def accuracy(y_true, y_pred):
    """Exact-match accuracy."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(y_true == y_pred))


def r_squared(y_true, y_pred):
    """Coefficient of determination (R^2)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def standardized_accuracy(y_true, y_pred):
    """SA = 1 - (MAE / MAE_baseline), where baseline predicts mean(y_true).

    Used by Ochodek (P37, P25) and Yokogawa (P13) as primary metric.
    SA > 0 means the model beats the mean-predictor baseline.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    if mae_baseline == 0:
        return float("nan")
    return float(1.0 - mae_model / mae_baseline)


def nmae(y_true, y_pred):
    """Normalized MAE = MAE / mean(y_true).

    Primary metric in Ünlü et al. (P17). Scale-independent; enables
    cross-dataset comparison when CFP ranges differ.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mean_true = np.mean(y_true)
    if mean_true == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / mean_true)


def bre(y_true, y_pred):
    """Balanced Relative Error = |pred - true| / max(pred, true).

    Used by Ochodek (P19). Symmetric alternative to MRE that avoids
    penalising overestimation more than underestimation.
    Returns mean BRE across all items.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.maximum(y_true, y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def mbre(y_true, y_pred):
    """Mean Balanced Relative Error = mean(|pred-true| / true).

    Equivalent to MMRE but named per Shepperd & MacDonell (2012) taxonomy.
    Included for completeness; numerically identical to mmre().
    """
    return mmre(y_true, y_pred)


def mibre(y_true, y_pred):
    """Mean Inverted Balanced Relative Error = mean(|pred-true| / pred).

    Complementary to MBRE. While MBRE penalises overestimation,
    MIBRE penalises underestimation. Reporting both reveals
    systematic bias direction (Shepperd & MacDonell 2012).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_pred != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_pred[mask]))


def mean_signed_error(y_true, y_pred):
    """Mean Signed Error (bias) = mean(y_pred - y_true).

    Positive → systematic overestimation; negative → underestimation.
    Essential for understanding directionality of prediction errors.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(y_pred - y_true))


def spearman_corr(y_true, y_pred):
    """Spearman rank correlation coefficient and p-value.

    Rank-based; robust to non-normal distributions and outliers.
    Returns dict with 'rho' and 'p_value'.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 3:
        return {"rho": float("nan"), "p_value": float("nan")}
    rho, p = sp_stats.spearmanr(y_true, y_pred)
    return {"rho": float(rho), "p_value": float(p)}


def pearson_corr(y_true, y_pred):
    """Pearson linear correlation coefficient and p-value."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 3:
        return {"r": float("nan"), "p_value": float("nan")}
    r, p = sp_stats.pearsonr(y_true, y_pred)
    return {"r": float(r), "p_value": float(p)}


def cohens_kappa(y_true, y_pred):
    """Cohen's Kappa for classification agreement.

    Used by P19/P35/P49 (HKO) for size-class classification and by
    P11 (De Vito) for inter-rater agreement. Works on discrete labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    if n == 0:
        return float("nan")
    observed_agree = np.sum(y_true == y_pred) / n
    expected_agree = sum(
        (np.sum(y_true == lbl) / n) * (np.sum(y_pred == lbl) / n)
        for lbl in labels
    )
    if expected_agree == 1.0:
        return float("nan")
    return float((observed_agree - expected_agree) / (1.0 - expected_agree))


def cliffs_delta(x, y):
    """Cliff's delta effect size (non-parametric).

    Better than Cohen's d for non-normal distributions. Ranges [-1, 1].
    |d| < 0.147 → negligible, < 0.33 → small, < 0.474 → medium, else large.
    Used for statistical comparison between methods on same test set.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return float((more - less) / (nx * ny))


def error_distribution(y_true, y_pred):
    """Descriptive statistics of absolute errors.

    Returns percentiles (p5, p25, p50, p75, p95), mean, std, min, max.
    Enables box plots and violin plots for the dissertation.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    errors = np.abs(y_true - y_pred)
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "min": float(np.min(errors)),
        "p5": float(np.percentile(errors, 5)),
        "p25": float(np.percentile(errors, 25)),
        "p50": float(np.percentile(errors, 50)),
        "p75": float(np.percentile(errors, 75)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def movement_f1(y_true_movements: list[set[str]], y_pred_movements: list[set[str]]):
    """Micro-averaged F1 over sets of data movement types per sub-process.

    Each element is a set of movement strings, e.g. {"Entry", "Read", "Write"}.
    Comparable to De Vito's BERTScore/BLEU on data movement identification.
    """
    tp = fp = fn = 0
    for gt_set, pred_set in zip(y_true_movements, y_pred_movements):
        gt_norm = {m.strip().lower() for m in gt_set}
        pred_norm = {m.strip().lower() for m in pred_set}
        tp += len(gt_norm & pred_norm)
        fp += len(pred_norm - gt_norm)
        fn += len(gt_norm - pred_norm)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
