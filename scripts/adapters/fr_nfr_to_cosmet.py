#!/usr/bin/env python3
"""
FR/NFR Datasets -> COSMET format adapter.

Extracts Functional Requirements (FR) from the archive and XLSX FR/NFR
datasets and converts them into COSMET-compatible FP_prompts/*.txt.

These datasets have NO COSMIC ground truth, so only FP_prompts are
generated (no ground_truth.csv).  Useful for exploratory COSMIC
measurement or for augmenting training data.

Sources:
  - datasets/raw/archive-fr-nfr/train.txt + test.txt
  - datasets/raw/fr-nfr-xlsx/FR_NFR_Dataset.xlsx

Usage:
    python scripts/adapters/fr_nfr_to_cosmet.py [--source archive|xlsx|all]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "datasets" / "raw"
OUT = BASE / "datasets" / "prepared" / "fr-nfr-cosmet"


def load_archive_frs() -> pd.DataFrame:
    """Load FRs from archive train.txt + test.txt."""
    rows = []
    for fname in ["train.txt", "test.txt"]:
        p = RAW / "archive-fr-nfr" / fname
        df = pd.read_csv(p, header=None, names=["type", "requirement", "subtype"])
        frs = df[df["type"].str.upper() == "FR"]
        for _, row in frs.iterrows():
            rows.append({"source": f"archive/{fname}", "requirement": row["requirement"].strip()})
    return pd.DataFrame(rows)


def load_xlsx_frs() -> pd.DataFrame:
    """Load FRs from FR_NFR_Dataset.xlsx."""
    p = RAW / "fr-nfr-xlsx" / "FR_NFR_Dataset.xlsx"
    df = pd.read_excel(p, engine="openpyxl")
    df = df[["Requirement Text", "Type"]].dropna(subset=["Requirement Text", "Type"])
    frs = df[df["Type"].str.upper() == "FR"]
    rows = [{"source": "xlsx", "requirement": row["Requirement Text"].strip()}
            for _, row in frs.iterrows()]
    return pd.DataFrame(rows)


def generate_fp_prompts(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = out_dir / "FP_prompts"
    fp_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, row in df.iterrows():
        fp_num = idx + 1
        req = row["requirement"]
        content = (
            f"ID\n"
            f"{prefix}-{fp_num:04d}\t{req[:60]}\n"
            f"Description\t{req}\n"
            f"MAIN SCENARIO\n"
            f"1\tUser/System\t{req}\n"
        )
        fp_path = fp_dir / f"FP{fp_num}.txt"
        fp_path.write_text(content, encoding="utf-8")
        manifest.append({"fp_file": f"FP{fp_num}.txt", "source": row["source"], "requirement": req})

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fp_file", "source", "requirement"])
        w.writeheader()
        w.writerows(manifest)

    print(f"  {len(manifest)} FP files -> {fp_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  NOTE: No COSMIC ground truth available for this dataset.")


def main():
    parser = argparse.ArgumentParser(description="FR/NFR -> COSMET adapter")
    parser.add_argument("--source", choices=["archive", "xlsx", "all"], default="all")
    args = parser.parse_args()

    if args.source in ("archive", "all"):
        print("Loading archive FR/NFR ...")
        df = load_archive_frs()
        print(f"  {len(df)} FRs extracted from archive")
        generate_fp_prompts(df, OUT / "archive", "FR-A")

    if args.source in ("xlsx", "all"):
        print("Loading XLSX FR/NFR ...")
        df = load_xlsx_frs()
        print(f"  {len(df)} FRs extracted from xlsx")
        generate_fp_prompts(df, OUT / "xlsx", "FR-X")


if __name__ == "__main__":
    main()
