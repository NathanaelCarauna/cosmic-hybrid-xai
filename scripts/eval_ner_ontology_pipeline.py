#!/usr/bin/env python3
"""Evaluate the NER+SRL+Ontology pipeline on multiple datasets.

Supports: devito (119 FPs), zenodo (2237 sentences), ochodek (437 UC names).

Usage:
  python3 scripts/eval_ner_ontology_pipeline.py [--dataset devito|zenodo|ochodek] [--no-ner]
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
OUTPUT_DIR = BASE / "results" / "ner-ontology"

DEVITO_DATASETS = BASE / "replications" / "cosmet-devito" / "datasets"
DEVITO_RESULTS = BASE / "results" / "cosmet"
ZENODO_DIR = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sentence"
OCHODEK_CSV = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"

DEVITO_SYSTEMS = {
    "ALBERGATE": {
        "prompts": DEVITO_DATASETS / "MIS" / "ALBERGATE" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv",
    },
    "K01719": {
        "prompts": DEVITO_DATASETS / "IoT_Telemedicine" / "K01719" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "K01719-gpt-4-turbo-20260323T005355" / "measurements" / "comparison_results.csv",
    },
    "K01720": {
        "prompts": DEVITO_DATASETS / "IoT_Telemedicine" / "K01720" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "K01720-gpt-4-turbo-20260323T005553" / "measurements" / "comparison_results.csv",
    },
    "K01726": {
        "prompts": DEVITO_DATASETS / "IoT_Telemedicine" / "K01726" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "K01726-gpt-4-turbo-20260323T005703" / "measurements" / "comparison_results.csv",
    },
    "U-CURE": {
        "prompts": DEVITO_DATASETS / "ML" / "U-CURE" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "U-CURE-gpt-4-turbo-20260323T010152" / "measurements" / "comparison_results.csv",
    },
    "AutomaticLineSwitching": {
        "prompts": DEVITO_DATASETS / "Real-time" / "AutomaticLineSwitching" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "AutomaticLineSwitching-gpt-4-turbo-20260323T010407" / "measurements" / "comparison_results.csv",
    },
    "RiceCooker": {
        "prompts": DEVITO_DATASETS / "Real-time" / "RiceCooker" / "FP_prompts",
        "gt_csv": DEVITO_RESULTS / "RiceCooker-gpt-4-turbo-20260323T010514" / "measurements" / "comparison_results.csv",
    },
}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fp_sort_key(stem: str) -> int:
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def load_devito_gt(csv_path: Path) -> dict[str, dict]:
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fp_id = row["fp"]
            entry = {"y_true": int(row["y_true"])}
            if "entry_gt" in row:
                entry["entry_gt"] = int(row["entry_gt"])
                entry["read_gt"] = int(row["read_gt"])
                entry["write_gt"] = int(row["write_gt"])
                entry["exit_gt"] = int(row["exit_gt"])
            gt[fp_id] = entry
    return gt


def load_zenodo_gt() -> dict[str, dict]:
    """Load Zenodo ground truth: fp_file -> {y_true, entry_gt, ...}."""
    gt = {}
    gt_csv = ZENODO_DIR / "ground_truth.csv"
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            fp_id = row["fp_file"].replace(".txt", "")
            gt[fp_id] = {
                "y_true": int(row["cfp_gt"]),
                "entry_gt": int(row["entry_gt"]),
                "read_gt": int(row["read_gt"]),
                "write_gt": int(row["write_gt"]),
                "exit_gt": int(row["exit_gt"]),
            }
    return gt


def load_ochodek_items() -> list[dict]:
    """Load Ochodek-437: each row -> {uc, title, cfp}."""
    items = []
    with open(OCHODEK_CSV) as f:
        for row in csv.DictReader(f):
            items.append({
                "uc": row["UC"],
                "title": row["TitleTokens"].strip(),
                "cfp": int(row["Cfp"]),
                "project": row["ProjectID"],
            })
    return items


def wrap_as_uc(text: str, fp_id: str) -> str:
    """Wrap a single requirement/title into COSMET-like UC format."""
    return (
        f"ID\n"
        f"{fp_id}\t{text[:60]}\n"
        f"Description\t{text}\n"
        f"MAIN SCENARIO\n"
        f"1\tUser/System\t{text}\n"
    )


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    errors = [r["abs_error"] for r in results]
    mae = sum(errors) / n
    pred25 = sum(
        1 for r in results
        if abs(r["y_pred"] - r["y_true"]) / max(r["y_true"], 1) <= 0.25
    ) / n
    # SA = 1 - MAE/MAE_baseline (baseline prediz a média do GT) — definição de
    # Shepperd & MacDonell, igual a shared.metrics.standardized_accuracy.
    # Errata 2026-06-10: a versão anterior usava 1 - MAE/mean(GT) (= 1-NMAE),
    # que infla o valor em distribuições estreitas e não é comparável às demais
    # linhas da matriz cross-dataset.
    mean_gt = sum(r["y_true"] for r in results) / n
    mae_baseline = sum(abs(r["y_true"] - mean_gt) for r in results) / n
    sa = 1 - mae / mae_baseline if mae_baseline > 0 else float("nan")
    nmae_val = mae / mean_gt if mean_gt > 0 else float("nan")
    exact = sum(1 for e in errors if e == 0)
    mse = sum(r["error"] for r in results) / n
    return {
        "n": n, "mae": mae, "pred25": pred25, "sa": sa, "nmae": nmae_val,
        "exact": exact, "mean_signed_error": mse,
    }


def print_report(dataset_name: str, results: list[dict], total_time: float,
                 group_key: str = "system", group_order: list[str] | None = None):
    m = compute_metrics(results)
    n = m["n"]
    print(f"\n{'='*60}")
    print(f"  NER + SRL + Ontology Pipeline — {dataset_name}")
    print(f"{'='*60}")
    print(f"  FPs evaluated: {n}")
    print(f"  MAE: {m['mae']:.3f}")
    print(f"  PRED(25): {m['pred25']*100:.1f}%")
    print(f"  SA: {m['sa']*100:.1f}%")
    print(f"  Total time: {total_time:.1f}s ({total_time/n:.3f}s per FP)")
    print(f"  Exact matches: {m['exact']}/{n} ({m['exact']/n*100:.1f}%)")
    print(f"  Mean signed error: {m['mean_signed_error']:+.3f}")

    groups = group_order or sorted(set(r[group_key] for r in results))
    if len(groups) > 1:
        print(f"\n  Per-{group_key} breakdown:")
        for g in groups:
            gr = [r for r in results if r[group_key] == g]
            if not gr:
                continue
            gm = sum(r["abs_error"] for r in gr) / len(gr)
            ge = sum(1 for r in gr if r["abs_error"] == 0)
            print(f"    {g:25s}: MAE={gm:.3f}, exact={ge}/{len(gr)}")


def save_results(tag: str, dataset_name: str, results: list[dict], total_time: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(results)
    n = m["n"]

    csv_path = OUTPUT_DIR / f"{tag}_results.csv"
    fieldnames = ["system", "fp", "y_true", "y_pred", "error", "abs_error",
                  "E_pred", "R_pred", "W_pred", "X_pred",
                  "E_gt", "R_gt", "W_gt", "X_gt", "time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {csv_path}")

    summary = {
        "experiment": tag.upper(),
        "description": f"NER+SRL+Ontology pipeline — {dataset_name}",
        "n_fps": n,
        "mae": round(m["mae"], 3),
        "pred25": round(m["pred25"], 3),
        "sa": round(m["sa"], 3),
        "total_time_s": round(total_time, 1),
        "avg_time_per_fp_s": round(total_time / n, 3),
        "exact_matches": m["exact"],
        "mean_signed_error": round(m["mean_signed_error"], 3),
        "cost_usd": 0.0,
    }
    summary_path = OUTPUT_DIR / f"{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")


def measure_item(pipeline, text: str, fp_id: str, system: str,
                 uc_name: bool = False,
                 sentence_mode: bool = False) -> tuple[dict, float]:
    t0 = time.time()
    result = pipeline.measure(text, fp_id=fp_id, uc_name=uc_name,
                              sentence_mode=sentence_mode)
    elapsed = time.time() - t0
    return {
        "system": system,
        "fp": fp_id,
        "y_pred": result.cfp,
        "E_pred": result.n_entries,
        "R_pred": result.n_reads,
        "W_pred": result.n_writes,
        "X_pred": result.n_exits,
        "time_s": round(elapsed, 3),
    }, elapsed


def eval_devito(pipeline) -> int:
    all_results = []
    total_time = 0.0

    for sys_name, cfg in DEVITO_SYSTEMS.items():
        gt = load_devito_gt(cfg["gt_csv"])
        fp_files = sorted(cfg["prompts"].glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem))

        for fp_file in fp_files:
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue
            text = fp_file.read_text(encoding="utf-8")
            rec, elapsed = measure_item(pipeline, text, fp_id, sys_name)
            total_time += elapsed

            y_true = gt[fp_id]["y_true"]
            rec["y_true"] = y_true
            rec["error"] = rec["y_pred"] - y_true
            rec["abs_error"] = abs(rec["error"])
            if "entry_gt" in gt[fp_id]:
                rec["E_gt"] = gt[fp_id]["entry_gt"]
                rec["R_gt"] = gt[fp_id]["read_gt"]
                rec["W_gt"] = gt[fp_id]["write_gt"]
                rec["X_gt"] = gt[fp_id]["exit_gt"]
            all_results.append(rec)

    print_report("E33d — De Vito (119 FPs)", all_results, total_time,
                 group_order=list(DEVITO_SYSTEMS.keys()))
    save_results("e33d", "De Vito 119 FPs", all_results, total_time)
    return 0


def eval_zenodo(pipeline) -> int:
    gt = load_zenodo_gt()
    prompt_dir = ZENODO_DIR / "FP_prompts"
    fp_files = sorted(prompt_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem))

    all_results = []
    total_time = 0.0
    processed = 0

    for fp_file in fp_files:
        fp_id = fp_file.stem
        if fp_id not in gt:
            continue
        text = fp_file.read_text(encoding="utf-8")
        rec, elapsed = measure_item(pipeline, text, fp_id, "ZENODO",
                                    sentence_mode=True)
        total_time += elapsed

        y_true = gt[fp_id]["y_true"]
        rec["y_true"] = y_true
        rec["error"] = rec["y_pred"] - y_true
        rec["abs_error"] = abs(rec["error"])
        rec["E_gt"] = gt[fp_id]["entry_gt"]
        rec["R_gt"] = gt[fp_id]["read_gt"]
        rec["W_gt"] = gt[fp_id]["write_gt"]
        rec["X_gt"] = gt[fp_id]["exit_gt"]
        all_results.append(rec)
        processed += 1
        if processed % 500 == 0:
            print(f"  ... processed {processed}/{len(gt)} FPs ({total_time:.1f}s)")

    print_report("E33d-zenodo — Zenodo (heterogeneous sentences)", all_results, total_time)
    save_results("e33d_zenodo", f"Zenodo {len(all_results)} sentences", all_results, total_time)
    return 0


def eval_ochodek(pipeline) -> int:
    items = load_ochodek_items()
    all_results = []
    total_time = 0.0

    for i, item in enumerate(items):
        fp_id = f"UC{i+1}"
        # Roteamento por formato (M1): nomes de UC vão ao estimador per-formato.
        rec, elapsed = measure_item(pipeline, item["title"], fp_id,
                                    item["project"], uc_name=True)
        total_time += elapsed

        y_true = item["cfp"]
        rec["y_true"] = y_true
        rec["error"] = rec["y_pred"] - y_true
        rec["abs_error"] = abs(rec["error"])
        all_results.append(rec)

        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{len(items)} UCs ({total_time:.1f}s)")

    print_report("E33d-ochodek — Ochodek-437 (UC names)", all_results, total_time,
                 group_key="system")
    save_results("e33d_ochodek", f"Ochodek {len(all_results)} UCs", all_results, total_time)
    return 0


def eval_user_stories(pipeline, use_connextra_adapter: bool = False) -> int:
    """E33d on R4 user-stories-98 (Dalpiaz, COSMIC ground truth adjudicado).

    use_connextra_adapter=True ativa o normalizador de formato Connextra
    (E34a): a US é convertida em pseudo-UC(s) canônicos seguindo as regras do
    protocolo de anotação; processos decompostos (regra 4.6) são medidos
    separadamente e somados. Calibrado APENAS no dev (20 kappa_subset).
    """
    if use_connextra_adapter:
        from adapters.connextra_adapter import connextra_to_ucs

    fp_dir = BASE / "results" / "cross-input" / "user-stories-98"
    gt = {}
    with (fp_dir / "ground_truth.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["fp_id"]] = {"y_true": int(r["cfp_gt"])}

    fp_files = sorted(fp_dir.glob("FP*.txt"), key=lambda p: fp_sort_key(p.stem))
    all_results = []
    total_time = 0.0
    n_fallback = 0

    for fp_file in fp_files:
        fp_id = fp_file.stem
        if fp_id not in gt:
            continue
        raw = fp_file.read_text(encoding="utf-8").strip()
        if use_connextra_adapter:
            ucs = connextra_to_ucs(raw, fp_id)
            if ucs is None:
                # fallback: frame Connextra não reconhecido → wrapper antigo
                ucs = [wrap_as_uc(raw, fp_id)]
                n_fallback += 1
            rec = None
            elapsed_total = 0.0
            for uc in ucs:
                sub, elapsed = measure_item(pipeline, uc, fp_id, "USER-STORIES")
                elapsed_total += elapsed
                if rec is None:
                    rec = sub
                else:
                    for k in ("y_pred", "E_pred", "R_pred", "W_pred", "X_pred"):
                        rec[k] += sub[k]
            rec["time_s"] = round(elapsed_total, 3)
            elapsed = elapsed_total
        else:
            # Wrap as pseudo-UC (same fair treatment used for Ochodek UC
            # names): the full story becomes the single MAIN SCENARIO step.
            text = wrap_as_uc(raw, fp_id)
            rec, elapsed = measure_item(pipeline, text, fp_id, "USER-STORIES")
        total_time += elapsed

        y_true = gt[fp_id]["y_true"]
        rec["y_true"] = y_true
        rec["error"] = rec["y_pred"] - y_true
        rec["abs_error"] = abs(rec["error"])
        all_results.append(rec)

    label = "E34a-user-stories (adaptador Connextra)" if use_connextra_adapter \
        else "E33d-user-stories — R4 User Stories (98)"
    prefix = "e34a_user_stories" if use_connextra_adapter else "e33d_user_stories"
    if use_connextra_adapter and n_fallback:
        print(f"  [INFO] {n_fallback} US sem frame Connextra (fallback wrapper)")
    print_report(label, all_results, total_time)
    save_results(prefix, f"User Stories {len(all_results)}", all_results, total_time)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NER+SRL+Ontology pipeline on COSMIC datasets")
    parser.add_argument("--dataset", choices=["devito", "zenodo", "ochodek", "user-stories"],
                        default="devito", help="Dataset to evaluate (default: devito)")
    parser.add_argument("--no-ner", action="store_true",
                        help="Run without custom NER (use ontology only)")
    parser.add_argument("--adapter", choices=["connextra"], default=None,
                        help="Format adapter (user-stories only): 'connextra' "
                             "ativa o normalizador E34a")
    args = parser.parse_args()

    pipe_mod = load_mod(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"),
    )

    # Correção (2026-06-16): --no-ner agora desliga o NER de fato (use_ner=False).
    pipeline = pipe_mod.NerOntologyPipeline(use_ner=not args.no_ner)

    if args.dataset == "user-stories":
        return eval_user_stories(pipeline,
                                 use_connextra_adapter=args.adapter == "connextra")
    dispatch = {"devito": eval_devito, "zenodo": eval_zenodo, "ochodek": eval_ochodek}
    return dispatch[args.dataset](pipeline)


if __name__ == "__main__":
    sys.exit(main())
