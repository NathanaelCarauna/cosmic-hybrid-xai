#!/usr/bin/env python3
"""
COSMET Batch Runner — Headless replication of De Vito et al.'s COSMET tool.

Reads FP_prompts/*.txt, runs the COSMET pipeline (sentence split + COSMIC
measurement via OpenAI GPT-4 or a local GGUF model), and saves per-FP
measurement results.

Prompts are imported verbatim from the cloned COSMET repository to ensure
faithful replication.

Usage (default — structured output to results/<method>/<run_id>/):
    python scripts/cosmet_batch_runner.py <input_dir> \\
        [--dataset NAME] [--model MODEL] [--results-dir results/] [--dry-run]

Usage (local model):
    python scripts/cosmet_batch_runner.py <input_dir> \\
        --backend local --model-path models/gguf/Qwen3-8B-Q4_K_M.gguf \\
        --n-gpu-layers 99 --n-ctx 8192

Usage (legacy — flat output, no registry):
    python scripts/cosmet_batch_runner.py <input_dir> --legacy <output_dir> [--model MODEL]

Requires:
    OPENAI_API_KEY environment variable (unless --dry-run or --backend local).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.base_runner import BaseRunner, ChatResult
from shared.io_utils import find_fp_files, save_measurement_csv, save_measurement_xlsx
from shared.pricing import estimate_cost
from shared.results_registry import generate_run_id, register_run, WIB
from shared.seeds import set_global_seeds
from shared.timing import compute_throughput, compute_timing_stats

BASE = Path(__file__).resolve().parent.parent
COSMET_REPO = BASE / "replications" / "cosmet-devito"
PROMPT_FILE = COSMET_REPO / "COSMET" / "cosmet" / "cosmic" / "prompt.py"


# ---------------------------------------------------------------------------
# Load prompts verbatim from the cloned repo
# ---------------------------------------------------------------------------
def _load_prompts() -> dict[str, str]:
    """Exec prompt.py in an isolated namespace and return the prompt strings."""
    ns: dict = {}
    exec(PROMPT_FILE.read_text(encoding="utf-8"), ns)
    return {
        "SYSTEM_SPLIT_UC": ns["SYSTEM_SPLIT_UC"],
        "USER_SPLIT": ns["USER_SPLIT"],
        "SYSTEM_COSMIC_MEASURE": ns["SYSTEM_COSMIC_MEASURE"],
        "USER_COSMIC_MEASURE": ns["USER_COSMIC_MEASURE"],
    }


PROMPTS = _load_prompts()


# ---------------------------------------------------------------------------
# Global state for local model backend
# ---------------------------------------------------------------------------
_LOCAL_LLM = None
_BACKEND = "openai"


def _init_local_model(model_path: str, n_gpu_layers: int, n_ctx: int) -> None:
    """Initialise the local llama.cpp model (singleton)."""
    global _LOCAL_LLM
    if _LOCAL_LLM is not None:
        return
    from llama_cpp import Llama
    _LOCAL_LLM = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=6,
        verbose=False,
    )


def _clean_local_output(text: str) -> str:
    """Remove <think> blocks and normalise unicode dashes to ASCII hyphens."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def _chat_local(system: str, user: str) -> ChatResult:
    """Call the local llama.cpp model with the same interface as _chat_openai."""
    assert _LOCAL_LLM is not None, "Local model not initialised — call _init_local_model first"

    max_retries = 3
    retries_used = 0
    total_wait = 0.0

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = _LOCAL_LLM.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + " /no_think"},
                ],
                temperature=0,
                max_tokens=4096,
            )
            dur = time.perf_counter() - t0
            text = _clean_local_output(resp["choices"][0]["message"]["content"] or "")
            usage = resp.get("usage", {})
            inp_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            return ChatResult(
                text=text,
                input_tokens=inp_tok,
                output_tokens=out_tok,
                duration_s=dur,
                retries=retries_used,
                wait_time_s=total_wait,
            )
        except Exception as exc:
            retries_used += 1
            wait = 2 ** attempt
            print(f"  [local retry {attempt+1}/{max_retries}] {exc}")
            time.sleep(wait)
            total_wait += wait

    raise RuntimeError("Local model failed after max retries")


def _chat_dispatch(model: str, system: str, user: str) -> ChatResult:
    """Route to OpenAI or local backend depending on global _BACKEND."""
    if _BACKEND == "local":
        return _chat_local(system, user)
    return _chat_openai(model, system, user)


# ---------------------------------------------------------------------------
# OpenAI wrapper — returns ChatResult with retry tracking
# ---------------------------------------------------------------------------
def _chat_openai(model: str, system: str, user: str) -> ChatResult:
    """Call the OpenAI Chat API with exponential back-off."""
    import openai

    max_retries = 5
    wait = 5
    total_wait = 0.0
    retries_used = 0

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )
                text = resp.choices[0].message.content
                inp_tok = resp.usage.prompt_tokens
                out_tok = resp.usage.completion_tokens
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )
                text = resp["choices"][0]["message"]["content"]
                inp_tok = resp["usage"]["prompt_tokens"]
                out_tok = resp["usage"]["completion_tokens"]

            dur = time.perf_counter() - t0
            return ChatResult(
                text=text,
                input_tokens=inp_tok,
                output_tokens=out_tok,
                duration_s=dur,
                retries=retries_used,
                wait_time_s=total_wait,
            )

        except Exception as exc:
            retries_used += 1
            print(f"  [retry {attempt+1}/{max_retries}] {exc}")
            time.sleep(wait)
            total_wait += wait
            wait *= 2

    raise RuntimeError("OpenAI API failed after max retries")


# ---------------------------------------------------------------------------
# COSMET pipeline
# ---------------------------------------------------------------------------
def split_sentences(content: str, model: str) -> ChatResult:
    content = content.strip("\n")
    user_msg = PROMPTS["USER_SPLIT"].format(uc=content)
    return _chat_dispatch(model, PROMPTS["SYSTEM_SPLIT_UC"], user_msg)


def cosmic_analyze(uc_name: str, uc_content: str, model: str) -> dict:
    """Run the full COSMET pipeline with granular timing per phase."""
    split_result = split_sentences(uc_content, model)

    user_msg = PROMPTS["USER_COSMIC_MEASURE"].format(
        uc_name=uc_name, uc=split_result.text
    )
    measure_result = _chat_dispatch(model, PROMPTS["SYSTEM_COSMIC_MEASURE"], user_msg)

    table, trg_event, fu = _parse_measurement(measure_result.text)

    return {
        "raw_response": measure_result.text,
        "split_text": split_result.text,
        "table": table,
        "triggering_event": trg_event,
        "functional_user": fu,
        "duration": split_result.duration_s + measure_result.duration_s,
        "split_duration": split_result.duration_s,
        "analyze_duration": measure_result.duration_s,
        "input_tokens": split_result.input_tokens + measure_result.input_tokens,
        "output_tokens": split_result.output_tokens + measure_result.output_tokens,
        "split_input_tokens": split_result.input_tokens,
        "split_output_tokens": split_result.output_tokens,
        "analyze_input_tokens": measure_result.input_tokens,
        "analyze_output_tokens": measure_result.output_tokens,
        "retries": split_result.retries + measure_result.retries,
        "wait_time": split_result.wait_time_s + measure_result.wait_time_s,
    }


# ---------------------------------------------------------------------------
# Parser (faithful replica of print_analysis + parseGPTResponse)
# ---------------------------------------------------------------------------
def _parse_gpt_line(res: str) -> list[str]:
    FU = DG = OOI = DM = ""
    res = res.lstrip()
    if res.startswith("FU (") or res.startswith("- FU ("):
        if res.startswith("- FU"):
            res = res[2:]
        try:
            i = res.find(")")
            FU = res[4:i]
            j = res.find("DG (")
            i = res.find(")", j)
            DG = res[j + 4 : i]
            j = res.find("OOI (")
            i = res.find(")", j)
            OOI = res[j + 5 : i]
            j = i
            i = res.find(".", j)
            DM = res[j + 4 : i] if i >= 0 else res[j + 4 :]
            if DM.startswith("D"):
                DM = "Write"
        except Exception:
            pass
    elif res.startswith("DG (") or res.startswith("- DG ("):
        if res.startswith("- DG ("):
            res = res[2:]
        try:
            j = res.find("DG (")
            i = res.find(")", j)
            DG = res[j + 4 : i]
            j = res.find("OOI (")
            i = res.find(")", j)
            OOI = res[j + 5 : i]
            j = res.find(") - ", j)
            i = res.find(".", j)
            DM = res[j + 4 : i] if i >= 0 else res[j + 4 :]
            if DM.startswith("D"):
                DM = "Write"
            if res.find("Exit. It must be counted only once.") != -1:
                DM = "Exit. It must be counted only once."
        except Exception:
            pass
    elif res.startswith("No data movement.") or res.startswith("- No data movement"):
        DM = "No data movement"
    return [FU, DG, OOI, DM]


def _parse_measurement(raw: str) -> tuple[dict, str, str]:
    table = {"FU": [], "Sub-process": [], "DG": [], "OOI": [], "DM": [], "CFP": []}
    lines = raw.split("\n")

    if len(lines) > 3:
        if lines[0].lstrip().startswith('FP "'):
            del lines[0]
        if lines and len(lines[0].lstrip()) == 0:
            del lines[0]
    while lines and not lines[0].startswith("Triggering Event: "):
        del lines[0]

    trg_event = ""
    init_fu = ""
    if len(lines) > 3 and lines[0].startswith("Triggering Event: "):
        trg_event = lines[0].replace("Triggering Event:", "").strip()
        if lines[1].startswith("Functional User: "):
            init_fu = lines[1].replace("Functional User:", "").strip()
        lines_r = lines[3:]
    else:
        lines_r = lines

    sub_process = ""
    cont = 0
    messages = 0
    is_msg = False
    total = 0

    for line in lines_r:
        if line.strip().startswith("### Explanation of Data Movements"):
            break
        parsed = _parse_gpt_line(line)
        if line.endswith("Exit. It must be counted only once."):
            messages += 1
            is_msg = True

        if len(parsed[3]) > 0:
            table["FU"].append(parsed[0])
            table["Sub-process"].append(sub_process)
            table["DG"].append(parsed[1])
            table["OOI"].append(parsed[2])
            table["DM"].append(parsed[3])
            if len(parsed[1]) > 0:
                if is_msg:
                    if messages < 2:
                        table["CFP"].append("1")
                        total += 1
                    else:
                        table["CFP"].append("0")
                else:
                    tot = 0
                    dm_lower = parsed[3].lower()
                    if "entry" in dm_lower:
                        tot += 1
                    if "exit" in dm_lower:
                        tot += 1
                    if "read" in dm_lower:
                        tot += 1
                    if "write" in dm_lower:
                        tot += 1
                    table["CFP"].append(str(tot))
                    total += tot
            else:
                table["CFP"].append("0")
            cont = 0
        else:
            if cont == 1:
                table["FU"].append("")
                table["Sub-process"].append(sub_process)
                table["DG"].append("")
                table["OOI"].append("")
                table["DM"].append("")
                table["CFP"].append("")
            sub_process = line
            cont = 1

    table["FU"].append("")
    table["Sub-process"].append("")
    table["DG"].append("")
    table["OOI"].append("")
    table["DM"].append("TOTAL")
    table["CFP"].append(str(total))

    return table, trg_event, init_fu


# ---------------------------------------------------------------------------
# Structured runner (inherits BaseRunner)
# ---------------------------------------------------------------------------
class CosmetRunner(BaseRunner):
    """COSMET replication runner using the BaseRunner contract."""

    def process_fp(self, fp_file: Path) -> dict[str, Any]:
        content = fp_file.read_text(encoding="utf-8", errors="replace")
        uc_name = self.extract_uc_name(content, fp_file.name)
        return cosmic_analyze(uc_name, content, self.llm_model)


# ---------------------------------------------------------------------------
# Legacy batch (flat output_dir, no registry)
# ---------------------------------------------------------------------------
def run_batch_legacy(
    input_dir: Path, output_dir: Path, model: str, dry_run: bool
) -> None:
    """Original flat-output batch for backward compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = find_fp_files(input_dir)

    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(files)} FP files in {input_dir}")
    summary: list[dict] = []

    for idx, fp_file in enumerate(files, 1):
        content = fp_file.read_text(encoding="utf-8", errors="replace")
        uc_name_line = content.split("\n")
        uc_name = ""
        for line in uc_name_line:
            line = line.strip()
            if not line or line == "ID":
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                uc_name = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
                break
            uc_name = line
            break
        if not uc_name:
            uc_name = fp_file.stem

        fp_num = re.search(r"(\d+)", fp_file.stem)
        fp_id = fp_num.group(1) if fp_num else str(idx)

        print(f"  [{idx}/{len(files)}] {fp_file.name} -> \"{uc_name}\"", end="")

        if dry_run:
            print(" [DRY-RUN: skipped API call]")
            summary.append({
                "file": fp_file.name, "uc_name": uc_name, "cfp": "N/A",
                "status": "dry-run", "duration_s": "", "split_duration_s": "",
                "analyze_duration_s": "", "input_tokens": "", "output_tokens": "",
                "split_input_tokens": "", "split_output_tokens": "",
                "analyze_input_tokens": "", "analyze_output_tokens": "",
                "retries": "", "wait_time_s": "", "estimated_cost_usd": "",
            })
            continue

        try:
            result = cosmic_analyze(uc_name, content, model)
            cfp = result["table"]["CFP"][-1] if result["table"]["CFP"] else "0"

            save_measurement_xlsx(result, output_dir / f"{fp_id}_measurement.xlsx")
            save_measurement_csv(result, output_dir / f"{fp_id}_measurement.csv")
            raw_path = output_dir / f"{fp_id}_raw_response.txt"
            raw_path.write_text(result["raw_response"], encoding="utf-8")
            split_path = output_dir / f"{fp_id}_split_response.txt"
            split_path.write_text(result.get("split_text", ""), encoding="utf-8")

            dur = result["duration"]
            inp_tok = result["input_tokens"]
            out_tok = result["output_tokens"]
            cost = estimate_cost(inp_tok, out_tok, model)

            print(f" -> CFP={cfp} ({dur:.1f}s, {inp_tok}+{out_tok} tok, ${cost:.4f})")
            summary.append({
                "file": fp_file.name, "uc_name": uc_name, "cfp": cfp, "status": "ok",
                "duration_s": round(dur, 3),
                "split_duration_s": round(result.get("split_duration", 0), 3),
                "analyze_duration_s": round(result.get("analyze_duration", 0), 3),
                "input_tokens": inp_tok, "output_tokens": out_tok,
                "split_input_tokens": result.get("split_input_tokens", 0),
                "split_output_tokens": result.get("split_output_tokens", 0),
                "analyze_input_tokens": result.get("analyze_input_tokens", 0),
                "analyze_output_tokens": result.get("analyze_output_tokens", 0),
                "retries": result.get("retries", 0),
                "wait_time_s": round(result.get("wait_time", 0.0), 3),
                "estimated_cost_usd": round(cost, 6),
            })

        except Exception as exc:
            print(f" -> ERROR: {exc}")
            summary.append({
                "file": fp_file.name, "uc_name": uc_name, "cfp": "ERROR",
                "status": str(exc), "duration_s": "", "split_duration_s": "",
                "analyze_duration_s": "", "input_tokens": "", "output_tokens": "",
                "split_input_tokens": "", "split_output_tokens": "",
                "analyze_input_tokens": "", "analyze_output_tokens": "",
                "retries": "", "wait_time_s": "", "estimated_cost_usd": "",
            })

    from shared.base_runner import SUMMARY_FIELDS
    summary_path = output_dir / "batch_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary written to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    set_global_seeds()
    global _BACKEND

    parser = argparse.ArgumentParser(
        description="COSMET Batch Runner — headless replication",
        epilog=(
            "Default: structured output to results/<method>/<run_id>/.\n"
            "Use --legacy <output_dir> for flat output without registry.\n"
            "Use --backend local for GPU-based inference via llama.cpp."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="Directory with FP_prompts/*.txt files")
    parser.add_argument("--model", default=None, help="OpenAI model (default: gpt-4-turbo)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling API")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (inferred from input_dir if omitted)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root results directory (default: results/)")
    parser.add_argument("--method", default="cosmet", help="Method name (default: cosmet)")
    parser.add_argument("--method-type", default="replication",
                        choices=["replication", "own"], help="Method type (default: replication)")
    parser.add_argument("--resume", default=None, metavar="RUN_ID",
                        help="Resume a previous run by its run_id (skips already-processed FPs)")
    parser.add_argument("--legacy", type=Path, default=None, metavar="OUTPUT_DIR",
                        help="(Legacy) Flat output dir without registry/metrics")
    parser.add_argument("--backend", default="openai", choices=["openai", "local"],
                        help="LLM backend: openai (API) or local (llama.cpp GGUF)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to GGUF model file (required for --backend local)")
    parser.add_argument("--n-gpu-layers", type=int, default=99,
                        help="Number of layers to offload to GPU (default: 99 = all)")
    parser.add_argument("--n-ctx", type=int, default=8192,
                        help="Context window size for local model (default: 8192)")
    args = parser.parse_args()

    _BACKEND = args.backend

    if _BACKEND == "local":
        if not args.model_path:
            sys.exit("ERROR: --model-path is required when --backend local")
        if not args.model_path.exists():
            sys.exit(f"ERROR: Model file not found: {args.model_path}")
        model = f"local:{args.model_path.stem}"
        print(f"Initialising local model: {args.model_path.name} "
              f"(n_gpu_layers={args.n_gpu_layers}, n_ctx={args.n_ctx})")
        _init_local_model(str(args.model_path), args.n_gpu_layers, args.n_ctx)
        print("Local model ready.\n")
    else:
        model = args.model or os.getenv("MODEL", "gpt-4-turbo")

    if _BACKEND == "openai" and not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set. Use --dry-run to validate without API calls.")

    if args.legacy:
        print(f"Model: {model}")
        print(f"Input: {args.input_dir}")
        print(f"Output: {args.legacy}")
        if args.dry_run:
            print("Mode: DRY-RUN (no API calls)")
        print()
        run_batch_legacy(args.input_dir, args.legacy, model, args.dry_run)
    else:
        dataset = args.dataset or args.input_dir.name
        runner = CosmetRunner(
            method=args.method,
            method_type=args.method_type,
            dataset=dataset,
            llm_model=model,
            results_dir=args.results_dir,
            dry_run=args.dry_run,
            resume_run_id=args.resume,
        )
        runner.run_batch(args.input_dir)


if __name__ == "__main__":
    main()
