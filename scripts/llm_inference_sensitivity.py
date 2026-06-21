#!/usr/bin/env python3
"""T3 (Sprint 3) — Sensibilidade de inferência LLM (E28a no holdout-30).

Varia: temperature {0, 0.3, 0.7} × seeds {42, 123, 7} + 5 permutações da
ordem dos few-shots limpos (temp=0, seed=42). Modelo carregado UMA vez;
temperature/seed são parâmetros por chamada; ordem de few-shots troca apenas
o prompt (atributo _fewshot_text).

Output: results/sensitivity/llm-inference-sensitivity.csv
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import hybrid_xai_runner as hxr  # registra módulos dinâmicos
from models.hybrid_xai.rule_engine import CosmicRuleEngine

slm_mod = sys.modules["models.hybrid_xai.normalizers.slm_normalizer"]
SlmNormalizer = slm_mod.SLMNormalizer
Metadata = hxr.Metadata
SourceFormat = hxr.SourceFormat

MODEL = BASE / "models" / "gguf" / "Qwen3-8B-Q4_K_M.gguf"
HOLDOUT = BASE / "datasets" / "holdout" / "devito_holdout.csv"
FEWSHOTS_DIR = Path(slm_mod._FEWSHOTS_DIR)


def load_holdout():
    """Carrega os 30 FPs do holdout com os TEXTOS ORIGINAIS dos arquivos FP
    (via iter_devito_fps), não o campo text do manifest (formatação degradada
    — tabs/estrutura de UC perdidos, o que infla o MAE do SLM)."""
    import re
    rows = list(csv.DictReader(HOLDOUT.open(newline="", encoding="utf-8")))
    cfp_by_runner_id = {}
    for r in rows:
        mid = r["id"]
        mid = mid[len("COSMET-"):] if mid.startswith("COSMET-") else mid
        runner_id = re.sub(r"-FP(\d+)$", r"/FP\1", mid)
        cfp_by_runner_id[runner_id] = int(r["cfp"])
    items = []
    input_dir = BASE / "replications" / "cosmet-devito" / "datasets" / "MIS" / "ALBERGATE" / "FP_prompts"
    for fp_id, text, _fmt in hxr.iter_devito_fps(input_dir, scope="all"):
        if fp_id in cfp_by_runner_id:
            items.append((fp_id, text, cfp_by_runner_id[fp_id]))
    assert len(items) == len(rows), f"esperava {len(rows)}, achei {len(items)}"
    return items


def fewshot_text(order: list[int]) -> str:
    files = sorted(FEWSHOTS_DIR.glob("example_*.txt"))
    return "\n\n".join(files[i].read_text(encoding="utf-8").strip()
                       for i in order)


def run_config(norm, engine, items, label):
    t0 = time.perf_counter()
    yt, yp = [], []
    for fp_id, text, cfp in items:
        nr = norm.normalize(text, SourceFormat.USE_CASE, fp_id=fp_id,
                            metadata=Metadata(source_file=fp_id))
        res = engine.measure(nr)
        yt.append(cfp); yp.append(res.total_cfp)
    yt, yp = np.array(yt, float), np.array(yp, float)
    mae = float(np.abs(yt - yp).mean())
    mad = float(np.abs(yt - yt.mean()).mean())
    dur = time.perf_counter() - t0
    print(f"  {label:34s} MAE={mae:.4f} SA={1-mae/mad:+.4f} ({dur:.0f}s)",
          flush=True)
    return {"config": label, "mae": round(mae, 4),
            "sa": round(1 - mae / mad, 4), "wall_s": round(dur, 1)}


def main() -> None:
    items = load_holdout()
    print(f"holdout: {len(items)} FPs", flush=True)
    engine = CosmicRuleEngine()
    norm = SlmNormalizer(MODEL, n_gpu_layers=35, temperature=0.0, seed=42)
    base_fs = norm._fewshot_text

    rows = []
    # braço 1: temperatura × seed
    for t in [0.0, 0.3, 0.7]:
        for s in [42, 123, 7]:
            norm.temperature, norm.seed = t, s
            norm._fewshot_text = base_fs
            rows.append(run_config(norm, engine, items, f"temp={t} seed={s}"))

    # braço 2: ordem dos few-shots (temp=0, seed=42)
    norm.temperature, norm.seed = 0.0, 42
    rng = random.Random(42)
    for k in range(5):
        order = list(range(6))
        rng.shuffle(order)
        norm._fewshot_text = fewshot_text(order)
        rows.append(run_config(norm, engine, items,
                               f"fewshot-order={''.join(map(str, order))}"))

    out = BASE / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "llm-inference-sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "mae", "sa", "wall_s"])
        w.writeheader(); w.writerows(rows)

    maes = [r["mae"] for r in rows]
    temp_maes = [r["mae"] for r in rows if r["config"].startswith("temp")]
    ord_maes = [r["mae"] for r in rows if r["config"].startswith("fewshot")]
    print(f"\nGeral: MAE {min(maes):.3f}–{max(maes):.3f}")
    print(f"σ(temperatura×seed) = {np.std(temp_maes):.4f}")
    print(f"σ(ordem few-shots)  = {np.std(ord_maes):.4f}")


if __name__ == "__main__":
    main()
