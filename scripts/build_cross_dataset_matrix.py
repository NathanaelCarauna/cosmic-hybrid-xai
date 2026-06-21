#!/usr/bin/env python3
"""
build_cross_dataset_matrix.py
------------------------------
Aggregate results from multiple experiment runs and emit a 5×4 heterogeneity
matrix (methods × datasets) as CSV, markdown table, and a degradation bar chart.

Methods (rows):
  1. hybrid-xai-e30f-clean  — E30f Sprint-1 corrected
  2. hybrid-xai-e33d        — NER+Ontologia pipeline
  3. r1-cosmet              — COSMET gpt-4-turbo replication
  4. r2-deep-cosmic         — DEEP-COSMIC-UC ML models
  5. r3-hko                 — HKO Hussain classification

Datasets (columns):
  1. devito       — De Vito 119 use cases
  2. zenodo       — Zenodo 2040 heterogeneous sentences
  3. ochodek      — Ochodek 437 UC names
  4. user-stories — 100 annotated user stories
"""

import csv
import json
import sys
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent  # software-size-estimator/

RESULTS = BASE / "results"
REGISTRY_CSV = RESULTS / "runs_registry.csv"

OUT_DIR = RESULTS / "cross-dataset-matrix"
FIGURES_DIR = BASE / "figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    """Load a JSON file, returning None and printing a warning on failure."""
    if not path.exists():
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not parse {path}: {exc}", file=sys.stderr)
        return None


def _get(d, *keys, default=None):
    """Nested dict getter with fallback."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def make_cell(mae=None, sa=None, n=None, note=""):
    return {
        "mae": mae,
        "sa": sa,
        "n": n,
        "note": note,
    }


def empty_cell(note="TBD"):
    return make_cell(note=note)


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

def load_registry():
    """Return list[dict] from runs_registry.csv."""
    rows = []
    if not REGISTRY_CSV.exists():
        print(f"[WARN] Registry not found: {REGISTRY_CSV}", file=sys.stderr)
        return rows
    with REGISTRY_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val, default=None):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------

def cell_e30f_devito():
    """
    E30f-clean / devito: report holdout-30 as canonical, LOOCV-119 in note.
    """
    holdout_path = (
        RESULTS / "hybrid-xai" / "holdout-final-devito"
        / "E30f-correction-ensemble" / "metrics.json"
    )
    loocv_path = (
        RESULTS / "hybrid-xai" / "bert-improvements-clean-loocv"
        / "E30f-correction-ensemble" / "metrics.json"
    )

    holdout = load_json(holdout_path)
    loocv = load_json(loocv_path)

    mae = _get(holdout, "mae") if holdout else None
    sa = _get(holdout, "sa") if holdout else None
    n_holdout = _get(holdout, "n_with_gt") if holdout else None

    loocv_mae = _get(loocv, "mae") if loocv else None
    loocv_n = _get(loocv, "n_with_gt") if loocv else None

    note_parts = []
    if n_holdout is not None:
        note_parts.append(f"holdout-{n_holdout}")
    if loocv_mae is not None and loocv_n is not None:
        note_parts.append(f"LOOCV-{loocv_n}={loocv_mae:.3f}")
    note = " / ".join(note_parts) if note_parts else "holdout"

    return make_cell(
        mae=round(mae, 4) if mae is not None else None,
        sa=round(sa, 4) if sa is not None else None,
        n=int(n_holdout) if n_holdout is not None else None,
        note=note,
    )


def cell_e30f_cross(dataset_label: str):
    """
    E30f-clean applied to cross-dataset (frozen hparams from De Vito train).
    Looks for the latest e30f-cross-<dataset>-* directory with metrics.json.
    """
    pattern = f"e30f-cross-{dataset_label}-*"
    candidates = sorted(
        (RESULTS / "hybrid-xai").glob(pattern),
        key=lambda p: p.name,
        reverse=True,  # latest timestamp first
    )
    if not candidates:
        return empty_cell(f"TBD (run apply_e30f_cross_dataset.py {dataset_label})")

    d = load_json(candidates[0] / "metrics.json")
    if d is None:
        return empty_cell(f"TBD (metrics.json missing in {candidates[0].name})")

    return make_cell(
        mae=_get(d, "mae"),
        sa=_get(d, "sa"),
        n=_get(d, "n_with_gt"),
        note=f"cross-dataset (frozen hparams, n={_get(d, 'n_with_gt')})",
    )


def cell_e33d_devito():
    path = RESULTS / "ner-ontology" / "e33d_summary.json"
    d = load_json(path)
    if d is None:
        return empty_cell()
    return make_cell(
        mae=_get(d, "mae"),
        sa=_get(d, "sa"),
        n=_get(d, "n_fps"),
        note="home",
    )


def cell_e33d_zenodo():
    path = RESULTS / "ner-ontology" / "e33d_zenodo_summary.json"
    d = load_json(path)
    if d is None:
        return empty_cell()
    return make_cell(
        mae=_get(d, "mae"),
        sa=_get(d, "sa"),
        n=_get(d, "n_fps"),
    )


def cell_e33d_ochodek():
    path = RESULTS / "ner-ontology" / "e33d_ochodek_summary.json"
    d = load_json(path)
    if d is None:
        return empty_cell()
    return make_cell(
        mae=_get(d, "mae"),
        sa=_get(d, "sa"),
        n=_get(d, "n_fps"),
        note="cross-dataset (NER/SRL fails on short UC names)",
    )


def cell_e33d_user_stories():
    path = RESULTS / "ner-ontology" / "e33d_user_stories_summary.json"
    d = load_json(path)
    if d is None:
        return empty_cell()
    return make_cell(
        mae=_get(d, "mae"),
        sa=_get(d, "sa"),
        n=_get(d, "n_fps"),
        note="cross-dataset (US wrapped as pseudo-UC; under-counts implicit E/R)",
    )


# E34a (adaptador Connextra) — métricas do TEST split (78 US nunca vistas;
# dev-20 kappa_subset usado para calibrar o adaptador). Valores computados
# de e34a_user_stories_results.csv excluindo o dev; ver
# results/r4-annotation/three-annotator-analysis.md e auditoria §19.12.
E34A_TEST_MAE = 0.500
E34A_TEST_SA = 0.261
E34A_TEST_N = 78


def cell_e34a_user_stories():
    path = RESULTS / "ner-ontology" / "e34a_user_stories_summary.json"
    d = load_json(path)
    if d is None:
        return empty_cell()
    return make_cell(
        mae=E34A_TEST_MAE,
        sa=E34A_TEST_SA,
        n=E34A_TEST_N,
        note="test-78 (adaptador Connextra calibrado em dev-20; "
             f"full-98: MAE={_get(d, 'mae')}, SA={_get(d, 'sa')})",
    )


def _hko_best_for_dataset(registry, dataset_prefix):
    """
    From registry rows with method=hko-hussain and dataset starting with
    dataset_prefix, pick within-dataset LOOCV runs only (no cross-dataset
    transfers, which contain '-to-' in the dataset name).
    """
    candidates = [
        r for r in registry
        if r.get("method") == "hko-hussain"
        and r.get("dataset", "").startswith(dataset_prefix)
        and "-to-" not in r.get("dataset", "")
    ]
    if not candidates:
        return None, None, None

    # Sort by timestamp descending (latest first), then pick best mae
    # Strategy: pick the run with lowest MAE among all variants for this dataset
    def _mae(r):
        v = _float(r.get("mae"))
        return v if v is not None else float("inf")

    best = min(candidates, key=_mae)

    # Read actual run_metrics.json for sa (not in registry)
    run_id = best.get("run_id", "")
    run_dir = RESULTS / "hko-hussain" / run_id
    metrics = load_json(run_dir / "run_metrics.json")
    sa = None
    if metrics:
        sa = _get(metrics, "cfp_approximation", "sa")

    return _float(best.get("mae")), sa, _int(best.get("n_fps"))


def cells_hko(registry):
    """Return dict of cells for R3 HKO across devito, zenodo, ochodek."""
    cells = {}

    mae_d, sa_d, n_d = _hko_best_for_dataset(registry, "hko-devito")
    cells["devito"] = make_cell(
        mae=round(mae_d, 4) if mae_d is not None else None,
        sa=round(sa_d, 4) if sa_d is not None else None,
        n=n_d,
        note="LOOCV (UC-flow parsed as sentences)",
    )

    mae_z, sa_z, n_z = _hko_best_for_dataset(registry, "hko-zenodo")
    cells["zenodo"] = make_cell(
        mae=round(mae_z, 4) if mae_z is not None else None,
        sa=round(sa_z, 4) if sa_z is not None else None,
        n=n_z,
    )

    mae_o, sa_o, n_o = _hko_best_for_dataset(registry, "hko-ochodek")
    cells["ochodek"] = make_cell(
        mae=round(mae_o, 4) if mae_o is not None else None,
        sa=round(sa_o, 4) if sa_o is not None else None,
        n=n_o,
        note="home",
    )

    mae_u, sa_u, n_u = _hko_best_for_dataset(registry, "hko-user-stories")
    cells["user-stories"] = make_cell(
        mae=round(mae_u, 4) if mae_u is not None else None,
        sa=round(sa_u, 4) if sa_u is not None else None,
        n=n_u,
        note="home (Hussain designed for user stories)",
    )
    return cells


def cell_r2_ochodek(registry):
    """
    R2 / ochodek: pick best (lowest mae) RF run from runs_registry.
    RF + name-only is canonical for ochodek home dataset.
    """
    candidates = [
        r for r in registry
        if r.get("method") == "deep-cosmic-uc"
        and "ochodek" in r.get("dataset", "").lower()
        and r.get("llm_model", "") == "rf"
    ]
    if not candidates:
        # fall back to any deep-cosmic-uc ochodek
        candidates = [
            r for r in registry
            if r.get("method") == "deep-cosmic-uc"
            and "ochodek" in r.get("dataset", "").lower()
        ]

    if not candidates:
        return empty_cell()

    def _mae(r):
        v = _float(r.get("mae"))
        return v if v is not None else float("inf")

    best = min(candidates, key=_mae)

    # Get sa from run_metrics.json
    run_id = best.get("run_id", "")
    run_dir = RESULTS / "deep-cosmic-uc" / run_id
    metrics = load_json(run_dir / "run_metrics.json")
    sa = None
    if metrics:
        sa = _get(metrics, "accuracy", "sa")

    return make_cell(
        mae=_float(best.get("mae")),
        sa=round(sa, 4) if sa is not None else None,
        n=_int(best.get("n_fps")),
        note="home (RF canonical)",
    )


def cell_r2_cross(eval_label):
    """R2 cross-dataset eval: train=Ochodek, eval=devito or zenodo."""
    summary_path = RESULTS / "deep-cosmic-uc" / f"cross-dataset-{eval_label}" / "cross_dataset_summary.csv"
    if not summary_path.exists():
        return empty_cell(f"TBD (run r2_cross_dataset_eval.py --eval-target {eval_label})")
    rows = []
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Pick RF, best condition (lowest MAE)
    rf_rows = [r for r in rows if r.get("model") == "rf"]
    if not rf_rows:
        rf_rows = rows  # fallback to any model
    best = min(rf_rows, key=lambda r: _float(r.get("mae"), default=float("inf")))
    mae = _float(best.get("mae"))
    sa = _float(best.get("sa"))
    n_rows = len([r for r in rows if r.get("model") == "rf"])
    return make_cell(
        mae=round(mae, 4) if mae is not None else None,
        sa=round(sa, 4) if sa is not None else None,
        n=None,
        note=f"zero-shot (train=Ochodek, RF {best.get('condition','')})",
    )


def cell_r1_cosmet_devito(registry):
    """
    R1 COSMET / devito:
    1. Prefer devito-119-combined aggregated metrics if available.
    2. Otherwise compute weighted-average MAE from per-project runs.
    """
    # First check the merged directory
    merged_path = (
        RESULTS / "cosmet" / "devito-119-combined-gpt-4-turbo-merged"
        / "comparison_metrics.json"
    )
    merged = load_json(merged_path)
    if merged and merged.get("mae") is not None:
        return make_cell(
            mae=_get(merged, "mae"),
            sa=_get(merged, "sa"),
            n=_get(merged, "n_with_gt"),
            note="home (gpt-4-turbo, per-project weighted avg)",
        )

    # Fallback: per-project rows from registry
    per_project_datasets = {
        "ALBERGATE", "K01719", "K01720", "K01726",
        "U-CURE", "AutomaticLineSwitching", "RiceCooker",
    }
    rows = [
        r for r in registry
        if r.get("method") == "cosmet"
        and r.get("llm_model") == "gpt-4-turbo"
        and r.get("dataset") in per_project_datasets
        and _float(r.get("mae")) is not None
        and _int(r.get("n_fps")) is not None
    ]

    if not rows:
        return empty_cell("home")

    total_fps = sum(_int(r["n_fps"]) for r in rows)
    weighted_mae = sum(_float(r["mae"]) * _int(r["n_fps"]) for r in rows) / total_fps
    n_projects = len(rows)

    return make_cell(
        mae=round(weighted_mae, 4),
        sa=None,  # not available per-project in registry
        n=total_fps,
        note=f"home (gpt-4-turbo, {n_projects} projects weighted avg)",
    )


def cell_r1_cosmet_cross(registry, dataset_key: str):
    """
    R1 COSMET on a cross-dataset (zenodo-200-r1 or ochodek-200-r1).
    Reads MAE/SA from runs_registry; falls back to computing from
    measurements/comparison_results.csv if SA missing.
    """
    candidates = [
        r for r in registry
        if r.get("method") == "cosmet"
        and r.get("llm_model") == "gpt-4-turbo"
        and r.get("dataset", "").startswith(dataset_key)
        and _float(r.get("mae")) is not None
    ]
    if not candidates:
        return empty_cell(f"TBD (run cosmet_batch_runner on {dataset_key})")

    # Pick latest run (by timestamp)
    best = max(candidates, key=lambda r: r.get("timestamp", ""))
    mae = _float(best.get("mae"))
    sa = _float(best.get("sa"))
    n = _int(best.get("n_fps"))

    # If SA not in registry, try to load from comparison_results.csv
    if sa is None:
        run_id = best.get("run_id", "")
        cmp_path = RESULTS / "cosmet" / run_id / "measurements" / "comparison_results.csv"
        if cmp_path.exists():
            import numpy as np
            rows_cmp = []
            with cmp_path.open(newline="") as f:
                reader = csv.DictReader(f)
                rows_cmp = list(reader)
            if rows_cmp:
                yt = np.array([_float(r.get("y_true")) for r in rows_cmp])
                yp = np.array([_float(r.get("y_pred")) for r in rows_cmp])
                ae = np.abs(yt - yp)
                mean_gt = np.mean(yt)
                random_mae = np.mean(np.abs(yt - mean_gt))
                sa = float(1 - np.mean(ae) / random_mae) if random_mae > 0 else 0.0

    note = f"cross-dataset (n={n}, gpt-4-turbo)"
    return make_cell(
        mae=round(mae, 4) if mae is not None else None,
        sa=round(sa, 4) if sa is not None else None,
        n=n,
        note=note,
    )


# ---------------------------------------------------------------------------
# Main matrix builder
# ---------------------------------------------------------------------------

METHODS = [
    "hybrid-xai-e30f-clean",
    "hybrid-xai-e33d",
    "hybrid-xai-e33d-adapters",
    "r1-cosmet",
    "r2-deep-cosmic",
    "r3-hko",
]

DATASETS = ["devito", "zenodo", "ochodek", "user-stories"]

# Home dataset mapping
HOME_DATASETS = {
    "hybrid-xai-e30f-clean": {"devito"},
    "hybrid-xai-e33d": {"devito"},
    "hybrid-xai-e33d-adapters": {"devito"},
    "r1-cosmet": {"devito"},
    "r2-deep-cosmic": {"ochodek"},
    "r3-hko": {"ochodek", "user-stories"},
}

METHOD_LABELS = {
    "hybrid-xai-e30f-clean": "E30f-clean (Hybrid-XAI)",
    "hybrid-xai-e33d": "E33d (NER+Ontologia)",
    "hybrid-xai-e33d-adapters": "E33d+E34a (com adaptador Connextra)",
    "r1-cosmet": "R1-COSMET (gpt-4-turbo)",
    "r2-deep-cosmic": "R2-DEEP-COSMIC (RF)",
    "r3-hko": "R3-HKO (Hussain)",
}

DATASET_LABELS = {
    "devito": "De Vito (119)",
    "zenodo": "Zenodo (2040)",
    "ochodek": "Ochodek (437)",
    "user-stories": "User Stories (100)",
}


def build_matrix():
    registry = load_registry()
    hko_cells = cells_hko(registry)

    matrix = {}

    # ---- E30f-clean ----
    matrix["hybrid-xai-e30f-clean"] = {
        "devito": cell_e30f_devito(),
        "zenodo": cell_e30f_cross("zenodo"),
        "ochodek": cell_e30f_cross("ochodek"),
        "user-stories": cell_e30f_cross("user-stories"),
    }

    # ---- E33d ----
    matrix["hybrid-xai-e33d"] = {
        "devito": cell_e33d_devito(),
        "zenodo": cell_e33d_zenodo(),
        "ochodek": cell_e33d_ochodek(),
        "user-stories": cell_e33d_user_stories(),
    }

    # ---- E33d + adaptadores de formato (E34a Connextra p/ user stories) ----
    # Demais células idênticas ao E33d (mesmo pipeline; o adaptador só atua
    # quando o input casa com o frame Connextra).
    matrix["hybrid-xai-e33d-adapters"] = {
        "devito": cell_e33d_devito(),
        "zenodo": cell_e33d_zenodo(),
        "ochodek": cell_e33d_ochodek(),
        "user-stories": cell_e34a_user_stories(),
    }

    # ---- R1 COSMET ----
    r1_devito = cell_r1_cosmet_devito(registry)
    matrix["r1-cosmet"] = {
        "devito": r1_devito,
        "zenodo": cell_r1_cosmet_cross(registry, "zenodo-200-r1"),
        "ochodek": cell_r1_cosmet_cross(registry, "ochodek-200-r1"),
        "user-stories": cell_r1_cosmet_cross(registry, "user-stories-98-r1"),
    }

    # ---- R2 DEEP-COSMIC ----
    matrix["r2-deep-cosmic"] = {
        "devito": cell_r2_cross("devito"),
        "zenodo": cell_r2_cross("zenodo"),
        "ochodek": cell_r2_ochodek(registry),
        "user-stories": empty_cell("N/A (format)"),
    }

    # ---- R3 HKO ----
    matrix["r3-hko"] = hko_cells

    return matrix


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv(matrix, out_path: Path):
    fieldnames = ["method"]
    for ds in DATASETS:
        for field in ["mae", "sa", "n", "note"]:
            fieldnames.append(f"{ds}_{field}")

    rows = []
    for method in METHODS:
        row = {"method": method}
        for ds in DATASETS:
            cell = matrix[method][ds]
            row[f"{ds}_mae"] = cell["mae"]
            row[f"{ds}_sa"] = cell["sa"]
            row[f"{ds}_n"] = cell["n"]
            row[f"{ds}_note"] = cell["note"]
        rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV written: {out_path}")


# ---------------------------------------------------------------------------
# Output: Markdown table
# ---------------------------------------------------------------------------

def _fmt_mae(cell):
    """Format MAE cell for markdown."""
    mae = cell.get("mae")
    note = cell.get("note", "")
    n = cell.get("n")

    if mae is None:
        return note if note else "TBD"

    val = f"{mae:.3f}"
    annotations = []
    if note:
        annotations.append(note)
    if n is not None:
        annotations.append(f"n={n}")

    if annotations:
        return f"{val} *({', '.join(annotations)})*"
    return val


def write_markdown(matrix, out_path: Path):
    lines = []
    lines.append("# Cross-Dataset Heterogeneity Matrix")
    lines.append("")
    lines.append("MAE values. Lower is better. *(home)* = method trained/validated on this dataset.")
    lines.append("")

    # Header
    header = ["Method"] + [DATASET_LABELS[ds] for ds in DATASETS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for method in METHODS:
        label = METHOD_LABELS[method]
        cells = []
        for ds in DATASETS:
            cell = matrix[method][ds]
            cells.append(_fmt_mae(cell))
        lines.append("| " + " | ".join([label] + cells) + " |")

    lines.append("")
    lines.append("**Legend:** TBD = not yet evaluated; N/A (format) = dataset format incompatible with method.")
    lines.append("")

    # SA table
    lines.append("## Standardized Accuracy (SA)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for method in METHODS:
        label = METHOD_LABELS[method]
        cells = []
        for ds in DATASETS:
            cell = matrix[method][ds]
            sa = cell.get("sa")
            note = cell.get("note", "")
            if sa is None:
                cells.append(note if note else "TBD")
            else:
                cells.append(f"{sa:.3f}")
        lines.append("| " + " | ".join([label] + cells) + " |")

    lines.append("")

    content = "\n".join(lines)
    with out_path.open("w") as f:
        f.write(content)

    print(f"[OK] Markdown written: {out_path}")


# ---------------------------------------------------------------------------
# Output: Bar chart
# ---------------------------------------------------------------------------

def make_figure(matrix, pdf_path: Path, png_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available — skipping figure generation.", file=sys.stderr)
        return

    # Collect data points: (method, dataset, mae, is_home)
    plot_data = {}  # method -> list of (dataset_label, mae, is_home)
    for method in METHODS:
        entries = []
        for ds in DATASETS:
            cell = matrix[method][ds]
            mae = cell.get("mae")
            if mae is None:
                continue
            is_home = ds in HOME_DATASETS.get(method, set())
            entries.append((DATASET_LABELS[ds], mae, is_home))
        if entries:
            plot_data[method] = entries

    if not plot_data:
        print("[WARN] No data to plot.", file=sys.stderr)
        return

    # Layout: one group per method, bars per dataset evaluated
    fig, ax = plt.subplots(figsize=(12, 6))

    home_color = "#2196F3"     # blue — home dataset
    cross_color = "#FF5722"    # orange-red — cross-dataset
    bar_width = 0.18
    group_gap = 0.8

    x_positions = []
    x_labels = []
    x_cursor = 0.0

    for method in METHODS:
        entries = plot_data.get(method)
        if not entries:
            x_cursor += group_gap
            continue

        group_start = x_cursor
        for i, (ds_label, mae, is_home) in enumerate(entries):
            color = home_color if is_home else cross_color
            xpos = x_cursor + i * (bar_width + 0.02)
            ax.bar(xpos, mae, width=bar_width, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3)
            ax.text(xpos, mae + 0.05, f"{mae:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=45, zorder=4)

        group_center = (group_start + x_cursor + (len(entries) - 1) * (bar_width + 0.02)) / 2
        x_positions.append(group_center)
        x_labels.append(METHOD_LABELS[method])
        x_cursor += len(entries) * (bar_width + 0.02) + group_gap

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (CFP)", fontsize=11)
    ax.set_title("Cross-Dataset Heterogeneity: MAE by Method and Evaluation Dataset", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=home_color, label="Home / training dataset"),
        mpatches.Patch(color=cross_color, label="Cross-dataset evaluation"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure written: {pdf_path}")
    print(f"[OK] Figure written: {png_path}")


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_summary(matrix):
    col_w = 22
    header = f"{'Method':<30}" + "".join(f"{DATASET_LABELS[ds]:>{col_w}}" for ds in DATASETS)
    print()
    print("=" * (30 + col_w * len(DATASETS)))
    print("CROSS-DATASET HETEROGENEITY MATRIX — MAE")
    print("=" * (30 + col_w * len(DATASETS)))
    print(header)
    print("-" * (30 + col_w * len(DATASETS)))
    for method in METHODS:
        row_str = f"{METHOD_LABELS[method]:<30}"
        for ds in DATASETS:
            cell = matrix[method][ds]
            mae = cell.get("mae")
            is_home = ds in HOME_DATASETS.get(method, set())
            if mae is None:
                val = cell.get("note", "TBD")[:10]
            else:
                suffix = " (H)" if is_home else ""
                val = f"{mae:.3f}{suffix}"
            row_str += f"{val:>{col_w}}"
        print(row_str)
    print("-" * (30 + col_w * len(DATASETS)))
    print("(H) = home / training dataset")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()

    write_csv(matrix, OUT_DIR / "summary.csv")
    write_markdown(matrix, OUT_DIR / "summary.md")
    make_figure(
        matrix,
        FIGURES_DIR / "heterogeneity-degradation.pdf",
        FIGURES_DIR / "heterogeneity-degradation.png",
    )
    print_summary(matrix)


if __name__ == "__main__":
    main()
