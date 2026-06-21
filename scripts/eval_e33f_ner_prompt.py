#!/usr/bin/env python3
"""E33f — Evaluate SLM with NER entity hints in prompt.

Runs the SLM normalizer with NER pre-annotation injected into the prompt.
Compares against E28a baseline (same SLM without hints).

Usage:
  python3 scripts/eval_e33f_ner_prompt.py \
    --model-path /path/to/qwen3-8b-q4.gguf \
    --output-dir results/ner-ontology/e33f
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DATASETS = BASE / "replications" / "cosmet-devito" / "datasets"
RESULTS_COSMET = BASE / "results" / "cosmet"

SYSTEMS = {
    "ALBERGATE": {
        "prompts": DATASETS / "MIS" / "ALBERGATE" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv",
    },
    "K01719": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01719" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01719-gpt-4-turbo-20260323T005355" / "measurements" / "comparison_results.csv",
    },
    "K01720": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01720" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01720-gpt-4-turbo-20260323T005553" / "measurements" / "comparison_results.csv",
    },
    "K01726": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01726" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01726-gpt-4-turbo-20260323T005703" / "measurements" / "comparison_results.csv",
    },
    "U-CURE": {
        "prompts": DATASETS / "ML" / "U-CURE" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "U-CURE-gpt-4-turbo-20260323T010152" / "measurements" / "comparison_results.csv",
    },
    "AutomaticLineSwitching": {
        "prompts": DATASETS / "Real-time" / "AutomaticLineSwitching" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "AutomaticLineSwitching-gpt-4-turbo-20260323T010407" / "measurements" / "comparison_results.csv",
    },
    "RiceCooker": {
        "prompts": DATASETS / "Real-time" / "RiceCooker" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "RiceCooker-gpt-4-turbo-20260323T010514" / "measurements" / "comparison_results.csv",
    },
}


def load_gt(csv_path: Path) -> dict[str, dict]:
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row["fp"]] = {"y_true": int(row["y_true"])}
            if "entry_gt" in row:
                gt[row["fp"]].update({
                    "entry_gt": int(row["entry_gt"]),
                    "read_gt": int(row["read_gt"]),
                    "write_gt": int(row["write_gt"]),
                    "exit_gt": int(row["exit_gt"]),
                })
    return gt


def fp_sort_key(stem):
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/ner-ontology/e33f")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=8192)
    args = parser.parse_args()

    sys.path.insert(0, str(BASE))

    from models.hybrid_xai.normalizers.slm_normalizer import SLMNormalizer
    from models.hybrid_xai.normalizers.ner_enhanced_normalizer import NEREnhancedNormalizer
    from models.hybrid_xai.schema import SourceFormat

    print("Loading SLM...")
    base_normalizer = SLMNormalizer(
        args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
    )
    ner_normalizer = NEREnhancedNormalizer(base_normalizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_time = 0.0

    for sys_name, cfg in SYSTEMS.items():
        gt = load_gt(cfg["gt_csv"])
        prompt_dir = cfg["prompts"]

        fp_files = sorted(prompt_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem))

        for fp_file in fp_files:
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue

            text = fp_file.read_text(encoding="utf-8")
            print(f"  Processing {sys_name}/{fp_id}...", flush=True)

            t0 = time.time()
            try:
                nr = ner_normalizer.normalize(text, SourceFormat.USE_CASE, fp_id=fp_id)
                y_pred = nr.total_cfp
            except Exception as e:
                print(f"    ERROR: {e}")
                y_pred = 0
            elapsed = time.time() - t0
            total_time += elapsed

            y_true = gt[fp_id]["y_true"]

            all_results.append({
                "system": sys_name,
                "fp": fp_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": y_pred - y_true,
                "abs_error": abs(y_pred - y_true),
                "time_s": round(elapsed, 2),
            })

            status = "✅" if y_pred == y_true else "⚠️"
            print(f"    {status} pred={y_pred} gt={y_true} err={y_pred-y_true:+d} ({elapsed:.1f}s)")

    n = len(all_results)
    errors = [r["abs_error"] for r in all_results]
    mae = sum(errors) / n
    pred25 = sum(1 for r in all_results if abs(r["y_pred"] - r["y_true"]) / max(r["y_true"], 1) <= 0.25) / n
    mean_mag = sum(r["y_true"] for r in all_results) / n
    sa = 1 - mae / mean_mag

    print(f"\n{'='*60}")
    print(f"  E33f — SLM + NER Entity Hints")
    print(f"{'='*60}")
    print(f"  FPs: {n}")
    print(f"  MAE: {mae:.3f}")
    print(f"  PRED(25): {pred25*100:.1f}%")
    print(f"  SA: {sa*100:.1f}%")
    print(f"  Exact: {sum(1 for e in errors if e == 0)}/{n}")
    print(f"  Total time: {total_time:.0f}s ({total_time/n:.1f}s/FP)")

    csv_path = output_dir / "e33f_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "fp", "y_true", "y_pred", "error", "abs_error", "time_s"])
        writer.writeheader()
        writer.writerows(all_results)

    summary = {
        "experiment": "E33f",
        "description": "SLM (Qwen3-8B) + NER entity hints in prompt",
        "n_fps": n,
        "mae": round(mae, 3),
        "pred25": round(pred25, 3),
        "sa": round(sa, 3),
        "total_time_s": round(total_time, 1),
        "avg_time_per_fp_s": round(total_time / n, 1),
        "exact_matches": sum(1 for e in errors if e == 0),
    }
    with open(output_dir / "e33f_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
