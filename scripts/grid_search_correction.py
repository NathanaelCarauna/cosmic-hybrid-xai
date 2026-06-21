#!/usr/bin/env python3
"""Grid search LOOCV for TieredCorrectionNormalizer parameters.

Re-uses saved IRs from E20 (strips correction_* movements to recover raw
SLM CFP), then simulates tiered correction with different parameter combos.
No LLM inference required — pure arithmetic.

Usage:
    python scripts/grid_search_correction.py
"""
from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

# Dynamic load of hybrid-xai modules (hyphen in dir name)
pkg_base = BASE / "models" / "hybrid-xai"
for mod_name, mod_path in [
    ("models.hybrid_xai", pkg_base / "__init__.py"),
    ("models.hybrid_xai.schema", pkg_base / "schema.py"),
    ("models.hybrid_xai.normalizers", pkg_base / "normalizers" / "__init__.py"),
    ("models.hybrid_xai.normalizers.base", pkg_base / "normalizers" / "base.py"),
]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
for mod_name in list(sys.modules):
    if mod_name.startswith("models.hybrid_xai") and hasattr(sys.modules[mod_name], "__spec__"):
        if sys.modules[mod_name].__spec__ and sys.modules[mod_name].__spec__.loader:
            sys.modules[mod_name].__spec__.loader.exec_module(sys.modules[mod_name])

from models.hybrid_xai.normalizers.base import BaseNormalizer

E20_DIR = BASE / "results" / "hybrid-xai" / "devito-E20-correction-gamma05-20260327T151833"
FP_PROMPTS_DIR = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"


def count_steps(text: str) -> int:
    """Count total steps (main + alt + exc) using the same parser as the normalizer."""
    fields = BaseNormalizer.parse_uc_fields(text)
    return len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])


def load_data() -> list[dict]:
    """Load ground truth, raw SLM CFP (stripped of corrections), and n_steps."""
    gt_path = E20_DIR / "comparison_results.csv"
    gt: dict[str, int] = {}
    with open(gt_path) as f:
        for row in csv.DictReader(f):
            gt[row["fp"]] = int(row["y_true"])

    data = []
    for fp_id, gt_cfp in sorted(gt.items()):
        ir_path = E20_DIR / f"{fp_id}_ir.json"
        if not ir_path.exists():
            continue

        ir = json.loads(ir_path.read_text())
        raw_cfp = 0
        for fp in ir.get("functional_processes", []):
            for dm in fp.get("data_movements", []):
                dg = dm.get("data_group", "")
                if not dg.startswith("correction_"):
                    raw_cfp += 1

        fp_text_path = FP_PROMPTS_DIR / f"{fp_id}.txt"
        if not fp_text_path.exists():
            continue
        text = fp_text_path.read_text(encoding="utf-8", errors="replace")
        n_steps = count_steps(text)

        data.append({
            "fp_id": fp_id,
            "gt_cfp": gt_cfp,
            "raw_cfp": raw_cfp,
            "n_steps": n_steps,
        })

    return data


def apply_tiered_correction(
    raw_cfp: int,
    n_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    alpha_large: float,
    beta_large: float,
    gamma_large: float,
    large_threshold: int,
) -> int:
    """Simulate TieredCorrectionNormalizer arithmetic."""
    if n_steps >= large_threshold:
        a, b, g = alpha_large, beta_large, gamma_large
    else:
        a, b, g = alpha, beta, gamma

    raw_correction = a * max(0, n_steps - b)
    correction = min(raw_correction, raw_cfp * g)

    if correction > 0.5:
        return raw_cfp + round(correction)
    return raw_cfp


def compute_mae(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("inf")
    return mean(abs(t - p) for t, p in zip(y_true, y_pred))


def compute_pred25(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    count = sum(1 for t, p in zip(y_true, y_pred) if t > 0 and abs(p - t) / t <= 0.25)
    return count / len(y_true)


def loocv_mae(data: list[dict], params: dict) -> tuple[float, float]:
    """Leave-one-out cross-validation MAE and PRED(25)."""
    y_true, y_pred = [], []
    for item in data:
        pred = apply_tiered_correction(
            item["raw_cfp"], item["n_steps"], **params
        )
        y_true.append(item["gt_cfp"])
        y_pred.append(pred)
    return compute_mae(y_true, y_pred), compute_pred25(y_true, y_pred)


def main():
    data = load_data()
    print(f"Loaded {len(data)} FPs with raw SLM IRs")
    print(f"  Example: {data[0]}")

    alphas = [round(x, 2) for x in [i * 0.05 for i in range(4, 13)]]           # 0.20..0.60
    betas = list(range(4, 11))                                                     # 4..10
    gammas = [round(x, 2) for x in [i * 0.1 for i in range(4, 11)]]             # 0.4..1.0
    large_thresholds = list(range(10, 17))                                         # 10..16
    alphas_large = [round(x, 2) for x in [i * 0.05 for i in range(6, 17)]]      # 0.30..0.80
    betas_large = list(range(3, 9))                                                # 3..8
    gammas_large = [round(x, 2) for x in [i * 0.1 for i in range(5, 13)]]       # 0.5..1.2

    total_combos = (
        len(alphas) * len(betas) * len(gammas)
        * len(large_thresholds)
        * len(alphas_large) * len(betas_large) * len(gammas_large)
    )
    print(f"Grid size: {total_combos:,} combinations")

    if total_combos > 2_000_000:
        print("Grid too large — using two-stage optimization (medium tier, then large tier)")
        best_mae = float("inf")
        best_params = {}

        print("\n--- Stage 1: Optimize medium-tier params (fix large-tier at defaults) ---")
        stage1_count = 0
        for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
            params = {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "alpha_large": 0.50, "beta_large": 6.0, "gamma_large": 0.8,
                "large_threshold": 13,
            }
            mae_val, pred25 = loocv_mae(data, params)
            stage1_count += 1
            if mae_val < best_mae:
                best_mae = mae_val
                best_params = {**params, "pred25": pred25}

        print(f"  Evaluated {stage1_count} combos")
        print(f"  Best medium-tier: MAE={best_mae:.3f}, params={best_params}")

        print("\n--- Stage 2: Optimize large-tier params (fix medium-tier at best) ---")
        stage2_count = 0
        fixed_medium = {
            "alpha": best_params["alpha"],
            "beta": best_params["beta"],
            "gamma": best_params["gamma"],
        }
        best_mae2 = float("inf")
        best_params2 = {}

        for al, bl, gl, lt in itertools.product(
            alphas_large, betas_large, gammas_large, large_thresholds,
        ):
            params = {
                **fixed_medium,
                "alpha_large": al, "beta_large": bl,
                "gamma_large": gl, "large_threshold": lt,
            }
            mae_val, pred25 = loocv_mae(data, params)
            stage2_count += 1
            if mae_val < best_mae2:
                best_mae2 = mae_val
                best_params2 = {**params, "pred25": pred25}

        print(f"  Evaluated {stage2_count} combos")
        print(f"  Best overall: MAE={best_mae2:.3f}")
        best_params = best_params2
        best_mae = best_mae2

        print("\n--- Stage 3: Joint refinement around optimum ---")
        a0 = best_params["alpha"]
        b0 = best_params["beta"]
        g0 = best_params["gamma"]
        al0 = best_params["alpha_large"]
        bl0 = best_params["beta_large"]
        gl0 = best_params["gamma_large"]
        lt0 = best_params["large_threshold"]

        refine_alphas = [round(a0 + d, 3) for d in [-0.05, -0.025, 0, 0.025, 0.05] if 0.1 <= a0 + d <= 0.8]
        refine_gammas = [round(g0 + d, 3) for d in [-0.1, -0.05, 0, 0.05, 0.1] if 0.2 <= g0 + d <= 1.2]
        refine_al = [round(al0 + d, 3) for d in [-0.05, -0.025, 0, 0.025, 0.05] if 0.2 <= al0 + d <= 1.0]
        refine_gl = [round(gl0 + d, 3) for d in [-0.1, -0.05, 0, 0.05, 0.1] if 0.3 <= gl0 + d <= 1.5]
        refine_lt = [lt0 - 1, lt0, lt0 + 1]

        stage3_count = 0
        for ra, rg, ral, rgl, rlt in itertools.product(
            refine_alphas, refine_gammas, refine_al, refine_gl, refine_lt,
        ):
            params = {
                "alpha": ra, "beta": b0, "gamma": rg,
                "alpha_large": ral, "beta_large": bl0,
                "gamma_large": rgl, "large_threshold": rlt,
            }
            mae_val, pred25 = loocv_mae(data, params)
            stage3_count += 1
            if mae_val < best_mae:
                best_mae = mae_val
                best_params = {**params, "pred25": pred25}

        print(f"  Evaluated {stage3_count} combos")

    else:
        best_mae = float("inf")
        best_params = {}
        checked = 0
        for alpha, beta, gamma, lt, al, bl, gl in itertools.product(
            alphas, betas, gammas, large_thresholds,
            alphas_large, betas_large, gammas_large,
        ):
            params = {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "alpha_large": al, "beta_large": bl,
                "gamma_large": gl, "large_threshold": lt,
            }
            mae_val, pred25 = loocv_mae(data, params)
            checked += 1
            if mae_val < best_mae:
                best_mae = mae_val
                best_params = {**params, "pred25": pred25}
            if checked % 100000 == 0:
                print(f"  ... {checked:,} / {total_combos:,} checked, best MAE={best_mae:.3f}")

    print(f"\n{'='*60}")
    print(f"OPTIMAL PARAMETERS (LOOCV on {len(data)} ALBERGATE FPs)")
    print(f"{'='*60}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  LOOCV MAE: {best_mae:.3f}")
    print(f"  LOOCV PRED(25): {best_params.get('pred25', 'N/A'):.3f}")

    current_params = {
        "alpha": 0.35, "beta": 8.0, "gamma": 0.6,
        "alpha_large": 0.50, "beta_large": 6.0,
        "gamma_large": 0.8, "large_threshold": 13,
    }
    current_mae, current_pred25 = loocv_mae(data, current_params)
    print(f"\n  Current E24a MAE: {current_mae:.3f}, PRED(25): {current_pred25:.3f}")
    improvement = ((current_mae - best_mae) / current_mae) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    out_path = BASE / "results" / "hybrid-xai" / "grid_search_optimal_params.json"
    out_path.write_text(json.dumps({
        "optimal": best_params,
        "optimal_mae": round(best_mae, 4),
        "current_e24a_mae": round(current_mae, 4),
        "improvement_pct": round(improvement, 2),
        "n_fps": len(data),
        "dataset": "ALBERGATE",
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
