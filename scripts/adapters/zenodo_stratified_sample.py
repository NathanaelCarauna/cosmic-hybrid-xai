#!/usr/bin/env python3
"""Create a stratified random sample of the Zenodo dataset for cost-efficient evaluation.

Produces a subdirectory with symlinked FP_prompts and a filtered ground_truth.csv,
ready for cosmet_batch_runner.py. Already-processed FPs (from a partial run) are
force-included so nothing is wasted.

Usage:
    python scripts/adapters/zenodo_stratified_sample.py \
        --full-dir datasets/prepared/zenodo-cosmet/sentence \
        --out-dir  datasets/prepared/zenodo-cosmet/sample-300 \
        --n 300 \
        --existing-run results/cosmet/ZENODO-sentence-gpt-4-turbo-20260323T011209 \
        --seed 42
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified sample of Zenodo FP_prompts")
    parser.add_argument("--full-dir", type=Path, required=True,
                        help="Full zenodo-cosmet/sentence directory (with FP_prompts/ and ground_truth.csv)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for the sample")
    parser.add_argument("--n", type=int, default=300,
                        help="Target sample size (default: 300)")
    parser.add_argument("--existing-run", type=Path, default=None,
                        help="Path to an existing run dir whose FPs should be force-included")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gt = pd.read_csv(args.full_dir / "ground_truth.csv")
    print(f"Full dataset: {len(gt)} items")
    print(f"CFP distribution:\n{gt['cfp_gt'].value_counts().sort_index()}\n")

    forced_fps: set[int] = set()
    if args.existing_run:
        meas_dir = args.existing_run / "measurements"
        if meas_dir.exists():
            for f in meas_dir.glob("*_measurement.csv"):
                m = re.search(r"(\d+)", f.stem)
                if m:
                    forced_fps.add(int(m.group(1)))
        print(f"Force-including {len(forced_fps)} already-processed FPs: {sorted(forced_fps)[:10]}...")

    forced_mask = gt["fp_num"].isin(forced_fps)
    forced_df = gt[forced_mask].copy()
    pool = gt[~forced_mask].copy()

    remaining_n = max(0, args.n - len(forced_df))
    print(f"Forced: {len(forced_df)}, remaining to sample: {remaining_n}")

    rng = np.random.default_rng(args.seed)

    if remaining_n > 0:
        strata_col = "cfp_gt"
        strata_counts = pool[strata_col].value_counts()
        total_pool = len(pool)

        sampled_parts = []
        for stratum, count in strata_counts.items():
            stratum_df = pool[pool[strata_col] == stratum]
            alloc = max(1, round(count / total_pool * remaining_n))
            alloc = min(alloc, len(stratum_df))
            chosen = stratum_df.sample(n=alloc, random_state=rng.integers(0, 2**31))
            sampled_parts.append(chosen)
            print(f"  CFP={stratum}: pool={count}, sampled={alloc}")

        sampled = pd.concat(sampled_parts)
        if len(sampled) > remaining_n:
            sampled = sampled.sample(n=remaining_n, random_state=rng.integers(0, 2**31))
    else:
        sampled = pd.DataFrame(columns=gt.columns)

    sample = pd.concat([forced_df, sampled]).drop_duplicates(subset="fp_num").sort_values("fp_num")
    print(f"\nFinal sample: {len(sample)} items")
    print(f"Sample CFP distribution:\n{sample['cfp_gt'].value_counts().sort_index()}\n")

    out_prompts = args.out_dir / "FP_prompts"
    out_prompts.mkdir(parents=True, exist_ok=True)

    src_prompts = args.full_dir / "FP_prompts"
    n_copied = 0
    for _, row in sample.iterrows():
        src = src_prompts / row["fp_file"]
        dst = out_prompts / row["fp_file"]
        if src.exists():
            shutil.copy2(src, dst)
            n_copied += 1

    sample.to_csv(args.out_dir / "ground_truth.csv", index=False)

    print(f"Copied {n_copied} FP_prompts to {out_prompts}")
    print(f"Ground truth saved to {args.out_dir / 'ground_truth.csv'}")

    pop_dist = gt["cfp_gt"].value_counts(normalize=True).sort_index()
    samp_dist = sample["cfp_gt"].value_counts(normalize=True).sort_index()
    print(f"\nStratification check:")
    print(f"  {'CFP':>4s}  {'Pop%':>6s}  {'Sample%':>8s}")
    for cfp in sorted(gt["cfp_gt"].unique()):
        p = pop_dist.get(cfp, 0) * 100
        s = samp_dist.get(cfp, 0) * 100
        print(f"  {cfp:>4d}  {p:>5.1f}%  {s:>7.1f}%")


if __name__ == "__main__":
    main()
