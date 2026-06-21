#!/usr/bin/env python3
"""E33f — NER entity hints injected into SLM prompt.

Runs the SLM (Qwen3-8B) with NER pre-analysis hints appended to the
requirement text. Compares against baseline E28a (no hints).

Usage:
  python3 scripts/eval_ner_prompt_injection.py \
    --model-path models/gguf/Qwen3-8B-Q4_K_M.gguf \
    --n-gpu-layers 99
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

# Reuse the hybrid-xai module loader from hybrid_xai_runner
def _load_hybrid_xai():
    pkg_base = BASE / "models" / "hybrid-xai"
    mod_list = [
        ("models.hybrid_xai", pkg_base / "__init__.py"),
        ("models.hybrid_xai.schema", pkg_base / "schema.py"),
        ("models.hybrid_xai.rule_engine", pkg_base / "rule_engine.py"),
        ("models.hybrid_xai.normalizers", pkg_base / "normalizers" / "__init__.py"),
        ("models.hybrid_xai.normalizers.base", pkg_base / "normalizers" / "base.py"),
        ("models.hybrid_xai.normalizers.nlp_normalizer", pkg_base / "normalizers" / "nlp_normalizer.py"),
        ("models.hybrid_xai.normalizers.slm_normalizer", pkg_base / "normalizers" / "slm_normalizer.py"),
    ]
    for name, path in mod_list:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod

    pkg = sys.modules["models.hybrid_xai"]
    pkg.schema = sys.modules["models.hybrid_xai.schema"]
    pkg.rule_engine = sys.modules["models.hybrid_xai.rule_engine"]
    pkg.normalizers = sys.modules["models.hybrid_xai.normalizers"]
    norms = sys.modules["models.hybrid_xai.normalizers"]
    norms.base = sys.modules["models.hybrid_xai.normalizers.base"]

    for name, _ in mod_list:
        if name in sys.modules:
            sys.modules[name].__spec__.loader.exec_module(sys.modules[name])


_load_hybrid_xai()

from models.hybrid_xai.schema import SourceFormat
from models.hybrid_xai.normalizers.slm_normalizer import SLMNormalizer

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
    return gt


def load_ner_pipeline():
    mod_path = BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"
    spec = importlib.util.spec_from_file_location("ner_ontology_pipeline", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod.NerOntologyPipeline()


def fp_sort_key(stem):
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=8192)
    args = parser.parse_args()

    print("Loading SLM...")
    normalizer = SLMNormalizer(
        args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    print("Loading NER+Ontology pipeline...")
    ner_pipeline = load_ner_pipeline()

    import spacy
    try:
        ner_nlp = spacy.load(str(BASE / "models" / "cosmic-ner" / "best"))
    except OSError:
        ner_nlp = None
        print("  ⚠️ Custom NER model not found, using ontology only")

    output_dir = BASE / "results" / "ner-ontology" / "e33f"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_time = 0.0

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n  === {sys_name} ===")
        gt = load_gt(cfg["gt_csv"])
        fp_files = sorted(cfg["prompts"].glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem))

        for fp_file in fp_files:
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue

            text = fp_file.read_text(encoding="utf-8")

            ner_entities = {}
            if ner_nlp:
                doc = ner_nlp(text)
                for ent in doc.ents:
                    ner_entities.setdefault(ent.label_, set()).add(ent.text)

            ner_result = ner_pipeline.measure(text)
            bd = ner_result.movement_breakdown

            hint_lines = [
                "\n--- NER Pre-Analysis (use as reference, not as final answer) ---",
            ]
            fu = sorted(ner_entities.get("FUNC_USER", set()))
            do_list = sorted(ner_entities.get("DATA_OBJ", set()))
            ps = sorted(ner_entities.get("PERSIST_STORE", set()))
            ui = sorted(ner_entities.get("UI_ELEMENT", set()))
            if fu:
                hint_lines.append(f"Functional Users: {', '.join(fu)}")
            if do_list:
                hint_lines.append(f"Data Objects: {', '.join(do_list)}")
            if ps:
                hint_lines.append(f"Persistent Storage: {', '.join(ps)}")
            if ui:
                hint_lines.append(f"UI Elements (0 CFP): {', '.join(ui)}")
            hint_lines.append(
                f"NER estimate: ~{bd['E']}E {bd['R']}R {bd['W']}W {bd['X']}X = ~{ner_result.cfp} CFP"
            )
            hint_lines.append("--- End NER Pre-Analysis ---\n")
            enhanced_text = text + "\n".join(hint_lines)

            t0 = time.time()
            try:
                nr = normalizer.normalize(
                    enhanced_text,
                    SourceFormat.USE_CASE,
                    fp_id=f"{sys_name}/{fp_id}",
                )
                y_pred = nr.total_cfp
            except Exception as e:
                print(f"  ⚠️ {sys_name}/{fp_id}: ERROR {e}")
                y_pred = ner_result.cfp
            elapsed = time.time() - t0
            total_time += elapsed

            y_true = gt[fp_id]["y_true"]
            status = "✅" if y_pred == y_true else "⚠️"
            print(f"  {status} {fp_id}: pred={y_pred} gt={y_true} ner={ner_result.cfp} [{elapsed:.1f}s]")

            ir_path = output_dir / f"{sys_name}_{fp_id}_ir.json"
            try:
                with open(ir_path, "w") as f:
                    json.dump(nr.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            all_results.append({
                "system": sys_name,
                "fp": fp_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "ner_hint_cfp": ner_result.cfp,
                "error": y_pred - y_true,
                "abs_error": abs(y_pred - y_true),
                "time_s": round(elapsed, 2),
            })

    n = len(all_results)
    if n == 0:
        print("No results!")
        return

    errors = [r["abs_error"] for r in all_results]
    mae = sum(errors) / n
    pred25 = sum(1 for r in all_results if abs(r["y_pred"] - r["y_true"]) / max(r["y_true"], 1) <= 0.25) / n
    mean_mag = sum(r["y_true"] for r in all_results) / n
    sa = 1 - mae / mean_mag

    print(f"\n{'='*60}")
    print(f"  E33f — NER Hint Injection into SLM")
    print(f"{'='*60}")
    print(f"  FPs: {n}")
    print(f"  MAE: {mae:.3f}")
    print(f"  PRED(25): {pred25*100:.1f}%")
    print(f"  SA: {sa*100:.1f}%")
    print(f"  Exact: {sum(1 for e in errors if e == 0)}/{n}")
    print(f"  Time: {total_time:.0f}s ({total_time/n:.1f}s/FP)")

    csv_path = output_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "fp", "y_true", "y_pred",
                                           "ner_hint_cfp", "error", "abs_error", "time_s"])
        w.writeheader()
        w.writerows(all_results)

    summary = {
        "experiment": "E33f",
        "description": "SLM + NER entity hints injected into prompt",
        "model": "Qwen3-8B-Q4_K_M",
        "n_fps": n,
        "mae": round(mae, 3),
        "pred25": round(pred25, 3),
        "sa": round(sa, 3),
        "exact_matches": sum(1 for e in errors if e == 0),
        "total_time_s": round(total_time, 1),
        "avg_time_per_fp_s": round(total_time / n, 1),
        "cost_usd": 0.0,
    }
    with open(output_dir / "e33f_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {output_dir}")


if __name__ == "__main__":
    main()
