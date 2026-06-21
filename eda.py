#!/usr/bin/env python3
"""
Exploratory Data Analysis — COSMIC Software Size Estimation Datasets
=====================================================================
Loads every dataset from datasets/raw/, profiles structure and
distributions, and writes a consolidated report to stdout + PNG charts
into output/charts/.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from collections import Counter
from xml.etree import ElementTree as ET

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
RAW = BASE / "datasets" / "raw"
CHARTS = BASE / "output" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

DIVIDER = "=" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def describe_df(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{textwrap.indent(df.dtypes.to_string(), '    ')}")
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"  Nulls:\n{textwrap.indent(nulls[nulls > 0].to_string(), '    ')}")
    else:
        print("  Nulls: none")


def save_fig(name: str) -> None:
    path = CHARTS / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [chart] {path.relative_to(BASE)}")


# ---------------------------------------------------------------------------
# 1. COSMET / deepCOSMIC Use-Cases (with COSMIC CFP)
# ---------------------------------------------------------------------------
def load_cosmet() -> pd.DataFrame:
    p = RAW / "cosmet-deepcosmic" / "COSMET deepCOSMIC use-cases.csv"
    df = pd.read_csv(p, index_col=0)
    return df


def analyse_cosmet(df: pd.DataFrame) -> None:
    section("1  COSMET / deepCOSMIC — Use-Case level COSMIC")
    describe_df(df, "cosmet-deepcosmic")
    print(f"\n  Projects: {df['ProjectID'].nunique()}")
    print(f"  Use-cases total: {len(df)}")
    print(f"\n  CFP statistics:\n{textwrap.indent(df['Cfp'].describe().to_string(), '    ')}")

    trans_flat = [t for row in df["TransTypes"].dropna() for t in row.split("|")]
    tc = Counter(trans_flat)
    print(f"\n  Transaction-type frequency (from TransTypes):")
    for k, v in tc.most_common():
        print(f"    {k}: {v}")

    uc_flat = [t for row in df["UCType"].dropna() for t in row.split("|")]
    uc = Counter(uc_flat)
    print(f"\n  UCType frequency:")
    for k, v in uc.most_common():
        print(f"    {k}: {v}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(df["Cfp"], bins=30, edgecolor="black", linewidth=0.5)
    axes[0].set_title("CFP distribution")
    axes[0].set_xlabel("CFP")
    axes[0].set_ylabel("Count")

    proj_cfp = df.groupby("ProjectID")["Cfp"].sum().sort_values(ascending=False)
    axes[1].barh(proj_cfp.index, proj_cfp.values)
    axes[1].set_title("Total CFP per project")
    axes[1].set_xlabel("CFP")
    axes[1].invert_yaxis()

    proj_cnt = df.groupby("ProjectID").size().sort_values(ascending=False)
    axes[2].barh(proj_cnt.index, proj_cnt.values)
    axes[2].set_title("Use-cases per project")
    axes[2].set_xlabel("# use-cases")
    axes[2].invert_yaxis()

    save_fig("01_cosmet_overview")


# ---------------------------------------------------------------------------
# 2. Zenodo — Heterogeneous COSMIC Dataset (requirement-level)
# ---------------------------------------------------------------------------
def load_zenodo() -> pd.DataFrame:
    p = RAW / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
    return pd.read_csv(p)


def load_zenodo_metrics() -> dict[str, pd.DataFrame]:
    metrics = {}
    folder = RAW / "zenodo-heterogeneous"
    for f in sorted(folder.glob("*.csv")):
        if f.name == "heterogeneous_dataset.csv":
            continue
        metrics[f.stem] = pd.read_csv(f)
    return metrics


def analyse_zenodo(df: pd.DataFrame, metrics: dict[str, pd.DataFrame]) -> None:
    section("2  Zenodo — Heterogeneous COSMIC dataset (requirement level)")
    describe_df(df, "heterogeneous_dataset")

    movement_cols = ["entry", "read", "write", "exit"]
    agg_cols = ["CFP", "interaction", "process", "communication", "MicroM"]

    print(f"\n  Movement columns stats:")
    print(textwrap.indent(df[movement_cols].describe().to_string(), "    "))

    print(f"\n  Aggregate columns stats:")
    print(textwrap.indent(df[agg_cols].describe().to_string(), "    "))

    total_e = df["entry"].sum()
    total_r = df["read"].sum()
    total_w = df["write"].sum()
    total_x = df["exit"].sum()
    total_m = total_e + total_r + total_w + total_x
    print(f"\n  Total movements: {total_m}  (E={total_e}, R={total_r}, W={total_w}, X={total_x})")
    print(f"  Total CFP (sum): {df['CFP'].sum()}")

    non_zero = (df[movement_cols] > 0).sum()
    print(f"\n  Requirements with at least one movement of each type:")
    for c in movement_cols:
        print(f"    {c}: {non_zero[c]}  ({100*non_zero[c]/len(df):.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(df["CFP"], bins=range(0, int(df["CFP"].max()) + 2), edgecolor="black", linewidth=0.5)
    axes[0, 0].set_title("CFP per requirement")
    axes[0, 0].set_xlabel("CFP")

    for i, col in enumerate(movement_cols):
        ax = axes[(i + 1) // 3, (i + 1) % 3]
        vc = df[col].value_counts().sort_index()
        ax.bar(vc.index, vc.values, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{col} count distribution")
        ax.set_xlabel(col)

    axes[1, 2].bar(movement_cols, [total_e, total_r, total_w, total_x],
                   color=sns.color_palette("muted", 4), edgecolor="black", linewidth=0.5)
    axes[1, 2].set_title("Total movements by type")
    axes[1, 2].set_ylabel("Count")

    save_fig("02_zenodo_movements")

    if metrics:
        print(f"\n  Benchmark metrics (from zenodo CSVs):")
        for name, mdf in metrics.items():
            print(f"\n  -- {name}.csv --")
            print(textwrap.indent(mdf.to_string(index=False), "    "))


# ---------------------------------------------------------------------------
# 3. Archive — FR / NFR Classification
# ---------------------------------------------------------------------------
def load_archive_fr_nfr() -> dict[str, pd.DataFrame]:
    folder = RAW / "archive-fr-nfr"
    result = {}
    for fname in ["train.txt", "test.txt", "traind.txt", "testd.txt"]:
        p = folder / fname
        df = pd.read_csv(p, header=None, names=["type", "requirement", "subtype"])
        result[fname] = df
    return result


def analyse_archive(splits: dict[str, pd.DataFrame]) -> None:
    section("3  Archive — FR / NFR Classification (train/test)")

    for name, df in splits.items():
        describe_df(df, name)
        tc = df["type"].value_counts()
        print(f"  Type distribution:\n{textwrap.indent(tc.to_string(), '    ')}")
        if "subtype" in df.columns:
            sc = df["subtype"].value_counts()
            print(f"  Subtype distribution (top 15):\n{textwrap.indent(sc.head(15).to_string(), '    ')}")

    train = splits["train.txt"]
    test = splits["test.txt"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, df) in zip(axes, [("train", train), ("test", test)]):
        vc = df["type"].value_counts()
        ax.bar(vc.index, vc.values, color=sns.color_palette("muted", len(vc)), edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name} — FR vs NFR")
        ax.set_ylabel("Count")

    save_fig("03_archive_fr_nfr")


# ---------------------------------------------------------------------------
# 4. FR_NFR_Dataset.xlsx
# ---------------------------------------------------------------------------
def load_fr_nfr_xlsx() -> pd.DataFrame:
    p = RAW / "fr-nfr-xlsx" / "FR_NFR_Dataset.xlsx"
    df = pd.read_excel(p, engine="openpyxl")
    df = df[["Requirement Text", "Type"]].dropna(subset=["Requirement Text"])
    return df


def analyse_fr_nfr_xlsx(df: pd.DataFrame) -> None:
    section("4  FR_NFR_Dataset.xlsx")
    describe_df(df, "FR_NFR_Dataset")
    tc = df["Type"].value_counts()
    print(f"\n  Type distribution:\n{textwrap.indent(tc.to_string(), '    ')}")
    avg_len = df["Requirement Text"].str.split().str.len()
    print(f"\n  Requirement text length (words):\n{textwrap.indent(avg_len.describe().to_string(), '    ')}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(tc.index, tc.values, color=sns.color_palette("muted", len(tc)), edgecolor="black", linewidth=0.5)
    axes[0].set_title("FR vs NFR")
    axes[0].set_ylabel("Count")

    axes[1].hist(avg_len.dropna(), bins=40, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Requirement length (words)")
    axes[1].set_xlabel("# words")

    save_fig("04_fr_nfr_xlsx")


# ---------------------------------------------------------------------------
# 5. XML Requirements (structured SRS)
# ---------------------------------------------------------------------------
def analyse_xml(folder: Path | None = None) -> None:
    section("5  XML Requirements (structured SRS docs)")
    folder = folder or RAW / "xml-requirements"
    files = sorted(folder.glob("*.xml"))
    print(f"  Files found: {len(files)}")

    rows = []
    for f in files:
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
            text_bodies = root.iter(f"{ns}text_body") if ns else root.iter("text_body")
            texts = [tb.text.strip() for tb in text_bodies if tb.text and tb.text.strip()]
            rows.append({"file": f.name, "sections": len(texts), "total_chars": sum(len(t) for t in texts)})
        except ET.ParseError as e:
            rows.append({"file": f.name, "sections": -1, "total_chars": 0})
            print(f"  ⚠ Parse error in {f.name}: {e}")

    df = pd.DataFrame(rows)
    print(textwrap.indent(df.to_string(index=False), "    "))
    print(f"\n  Total sections: {df[df['sections']>0]['sections'].sum()}")
    print(f"  Total chars: {df['total_chars'].sum():,}")


# ---------------------------------------------------------------------------
# 6. req/ — PDF/DOC/HTML Requirement Documents
# ---------------------------------------------------------------------------
def analyse_req_docs() -> None:
    section("6  req/ — Requirement document files (PDF, DOC, HTML)")
    folder = RAW / "req-documents" / "req"
    if not folder.exists():
        print("  Folder not found.")
        return

    files = sorted(folder.iterdir())
    extensions = Counter(f.suffix.lower() for f in files)
    total_size = sum(f.stat().st_size for f in files)

    print(f"  Files: {len(files)}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Extensions:")
    for ext, cnt in extensions.most_common():
        print(f"    {ext}: {cnt}")

    sizes = [(f.name, f.stat().st_size / 1024) for f in files]
    sizes.sort(key=lambda x: -x[1])
    print(f"\n  Largest 10 files:")
    for name, kb in sizes[:10]:
        print(f"    {name}: {kb:.0f} KB")

    print(f"\n  Smallest 5 files:")
    for name, kb in sizes[-5:]:
        print(f"    {name}: {kb:.0f} KB")


# ---------------------------------------------------------------------------
# 7. Cross-dataset summary
# ---------------------------------------------------------------------------
def cross_summary(
    cosmet: pd.DataFrame,
    zenodo: pd.DataFrame,
    archive: dict[str, pd.DataFrame],
    fr_nfr: pd.DataFrame,
) -> None:
    section("7  Cross-Dataset Summary")

    rows = [
        {
            "dataset": "COSMET/deepCOSMIC",
            "granularity": "use-case",
            "rows": len(cosmet),
            "has_cosmic": True,
            "has_req_text": True,
            "format": "CSV",
        },
        {
            "dataset": "Zenodo heterogeneous",
            "granularity": "requirement",
            "rows": len(zenodo),
            "has_cosmic": True,
            "has_req_text": True,
            "format": "CSV",
        },
        {
            "dataset": "Archive FR/NFR (train+test)",
            "granularity": "requirement",
            "rows": len(archive["train.txt"]) + len(archive["test.txt"]),
            "has_cosmic": False,
            "has_req_text": True,
            "format": "TXT/CSV",
        },
        {
            "dataset": "FR_NFR_Dataset.xlsx",
            "granularity": "requirement",
            "rows": len(fr_nfr),
            "has_cosmic": False,
            "has_req_text": True,
            "format": "XLSX",
        },
        {
            "dataset": "XML Requirements",
            "granularity": "document",
            "rows": 18,
            "has_cosmic": False,
            "has_req_text": True,
            "format": "XML",
        },
        {
            "dataset": "req/ documents",
            "granularity": "document",
            "rows": 79,
            "has_cosmic": False,
            "has_req_text": True,
            "format": "PDF/DOC/HTML",
        },
    ]

    summary = pd.DataFrame(rows)
    print(textwrap.indent(summary.to_string(index=False), "    "))

    cosmic_total = len(cosmet) + len(zenodo)
    non_cosmic = sum(r["rows"] for r in rows if not r["has_cosmic"])
    print(f"\n  Items with COSMIC annotation: {cosmic_total}")
    print(f"  Items without COSMIC (req text only): {non_cosmic}")
    print(f"\n  >>> Datasets directly usable for COSMIC estimation training/evaluation:")
    print(f"      COSMET/deepCOSMIC: {len(cosmet)} use-cases with CFP")
    print(f"      Zenodo heterogeneous: {len(zenodo)} requirements with E/R/W/X + CFP")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(DIVIDER)
    print("  COSMIC Software Size Estimation — Exploratory Data Analysis")
    print(DIVIDER)

    cosmet = load_cosmet()
    analyse_cosmet(cosmet)

    zenodo = load_zenodo()
    zenodo_metrics = load_zenodo_metrics()
    analyse_zenodo(zenodo, zenodo_metrics)

    archive = load_archive_fr_nfr()
    analyse_archive(archive)

    fr_nfr = load_fr_nfr_xlsx()
    analyse_fr_nfr_xlsx(fr_nfr)

    analyse_xml()
    analyse_req_docs()

    cross_summary(cosmet, zenodo, archive, fr_nfr)

    print(f"\n{DIVIDER}")
    print(f"  Charts saved to: {CHARTS.relative_to(BASE)}/")
    print(DIVIDER)


if __name__ == "__main__":
    main()
