#!/usr/bin/env python3
"""
Zenodo Heterogeneous Dataset -> COSMET format adapter.

The Zenodo dataset contains 2040 individual requirement sentences, each
annotated with Entry/Read/Write/Exit counts and CFP.  COSMET expects
full use-case descriptions.

This adapter provides two strategies:
  --mode sentence : One .txt per sentence (direct per-sentence comparison)
  --mode grouped  : Heuristic grouping into pseudo-use-cases (closer to
                    COSMET's intended input)

Both strategies generate:
  <output>/FP_prompts/*.txt          — input files for cosmet_batch_runner.py
  <output>/ground_truth.csv          — ground truth for cosmet_compare.py

Usage:
    python scripts/adapters/zenodo_to_cosmet.py [--mode sentence|grouped]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "datasets" / "raw" / "zenodo-heterogeneous" / "heterogeneous_dataset.csv"
OUT = BASE / "datasets" / "prepared" / "zenodo-cosmet"


def _is_new_process_boundary(prev: str, curr: str) -> bool:
    """Heuristic: detect when a new functional process likely starts."""
    curr_low = curr.lower().strip()
    boundary_patterns = [
        r"^(the )?(user|admin|voter|student|candidate|operator|manager|doctor|actor|system admin)",
        r"^(he|she|they) (choose|select|click|log|enter|open|start|initiate)",
        r"(logs? in|sign[s ]? in|choose[s ]? the option|select[s ]? the)",
    ]
    if not prev:
        return True
    for pat in boundary_patterns:
        if re.search(pat, curr_low):
            if not re.search(pat, prev.lower().strip()):
                return True
    return False


def generate_sentence_mode(df: pd.DataFrame, out_dir: Path) -> None:
    """One FP_prompt per requirement sentence."""
    fp_dir = out_dir / "FP_prompts"
    fp_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = []
    for idx, row in df.iterrows():
        fp_num = idx + 1
        req = row["requirement_en"]
        fp_path = fp_dir / f"FP{fp_num}.txt"

        content = (
            f"ID\n"
            f"REQ-{fp_num}\t{req[:60]}\n"
            f"Description\t{req}\n"
            f"MAIN SCENARIO\n"
            f"1\tUser/System\t{req}\n"
        )
        fp_path.write_text(content, encoding="utf-8")

        gt_rows.append({
            "fp_file": f"FP{fp_num}.txt",
            "fp_num": fp_num,
            "requirement": req,
            "entry_gt": row["entry"],
            "read_gt": row["read"],
            "write_gt": row["write"],
            "exit_gt": row["exit"],
            "cfp_gt": row["CFP"],
        })

    gt_path = out_dir / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_rows[0].keys())
        w.writeheader()
        w.writerows(gt_rows)

    print(f"  Sentence mode: {len(gt_rows)} FP files -> {fp_dir}")
    print(f"  Ground truth: {gt_path}")


def generate_grouped_mode(df: pd.DataFrame, out_dir: Path, max_group: int = 12) -> None:
    """Group consecutive sentences into pseudo-use-cases."""
    fp_dir = out_dir / "FP_prompts"
    fp_dir.mkdir(parents=True, exist_ok=True)

    groups: list[list[int]] = []
    current: list[int] = []

    for idx, row in df.iterrows():
        req = row["requirement_en"]
        prev = df.iloc[idx - 1]["requirement_en"] if idx > 0 else ""

        if current and (_is_new_process_boundary(prev, req) or len(current) >= max_group):
            groups.append(current)
            current = []
        current.append(idx)

    if current:
        groups.append(current)

    gt_rows = []
    for g_idx, group_indices in enumerate(groups, 1):
        sub_df = df.iloc[group_indices]
        lines = []
        for step, (_, row) in enumerate(sub_df.iterrows(), 1):
            lines.append(f"{step}\tUser/System\t{row['requirement_en']}")

        first_req = sub_df.iloc[0]["requirement_en"]
        uc_name = first_req[:60].rstrip(".")

        content = (
            f"ID\n"
            f"UC-Z-{g_idx:04d}\t{uc_name}\n"
            f"Description\tGrouped requirements ({len(group_indices)} steps)\n"
            f"MAIN SCENARIO\n"
            + "\n".join(lines)
            + "\n"
        )

        fp_path = fp_dir / f"FP{g_idx}.txt"
        fp_path.write_text(content, encoding="utf-8")

        gt_rows.append({
            "fp_file": f"FP{g_idx}.txt",
            "fp_num": g_idx,
            "num_steps": len(group_indices),
            "entry_gt": int(sub_df["entry"].sum()),
            "read_gt": int(sub_df["read"].sum()),
            "write_gt": int(sub_df["write"].sum()),
            "exit_gt": int(sub_df["exit"].sum()),
            "cfp_gt": int(sub_df["CFP"].sum()),
            "step_indices": ",".join(str(i) for i in group_indices),
        })

    gt_path = out_dir / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gt_rows[0].keys())
        w.writeheader()
        w.writerows(gt_rows)

    print(f"  Grouped mode: {len(groups)} groups ({df.shape[0]} sentences) -> {fp_dir}")
    print(f"  Ground truth: {gt_path}")


def main():
    parser = argparse.ArgumentParser(description="Zenodo -> COSMET adapter")
    parser.add_argument("--mode", choices=["sentence", "grouped"], default="sentence",
                        help="sentence = 1 FP per requirement; grouped = heuristic use-case blocks")
    parser.add_argument("--max-group", type=int, default=12,
                        help="Max sentences per group in grouped mode (default: 12)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output directory (default: {OUT}/<mode>)")
    args = parser.parse_args()

    out_dir = args.output or OUT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW} ...")
    df = pd.read_csv(RAW)
    print(f"  {df.shape[0]} rows loaded")

    if args.mode == "sentence":
        generate_sentence_mode(df, out_dir)
    else:
        generate_grouped_mode(df, out_dir, args.max_group)


if __name__ == "__main__":
    main()
