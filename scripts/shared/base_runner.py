"""BaseRunner — abstract contract for all experiment runners.

Every runner (replication or own model) inherits from this class to ensure
identical output structure, registry integration, and metrics collection.
"""
from __future__ import annotations

import abc
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from .pricing import estimate_cost
from .results_registry import WIB, generate_run_id, register_run
from .timing import compute_throughput, compute_timing_stats


class ChatResult(NamedTuple):
    """Structured return type for LLM API calls."""
    text: str
    input_tokens: int
    output_tokens: int
    duration_s: float
    retries: int
    wait_time_s: float


SUMMARY_FIELDS = [
    "file",
    "uc_name",
    "cfp",
    "status",
    "duration_s",
    "split_duration_s",
    "analyze_duration_s",
    "input_tokens",
    "output_tokens",
    "split_input_tokens",
    "split_output_tokens",
    "analyze_input_tokens",
    "analyze_output_tokens",
    "retries",
    "wait_time_s",
    "estimated_cost_usd",
]


class BaseRunner(abc.ABC):
    """Abstract batch runner with standardised output and metrics."""

    def __init__(
        self,
        *,
        method: str,
        method_type: str,
        dataset: str,
        llm_model: str,
        results_dir: Path = Path("results"),
        dry_run: bool = False,
        resume_run_id: str | None = None,
    ) -> None:
        self.method = method
        self.method_type = method_type
        self.dataset = dataset
        self.llm_model = llm_model
        self.results_dir = results_dir
        self.dry_run = dry_run
        self.run_id = resume_run_id or generate_run_id(dataset, llm_model)
        self.run_dir = results_dir / method / self.run_id
        self.measurements_dir = self.run_dir / "measurements"
        self._resuming = resume_run_id is not None

    def setup(self) -> None:
        """Create output directories."""
        self.measurements_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def process_fp(self, fp_file: Path) -> dict[str, Any]:
        """Process a single FP file and return a result dict.

        Must return at minimum::

            {
                "table": {...},
                "triggering_event": str,
                "functional_user": str,
                "raw_response": str,
                "split_text": str,
                "duration": float,
                "split_duration": float,
                "analyze_duration": float,
                "input_tokens": int,
                "output_tokens": int,
                "split_input_tokens": int,
                "split_output_tokens": int,
                "analyze_input_tokens": int,
                "analyze_output_tokens": int,
                "retries": int,
                "wait_time": float,
            }
        """

    def extract_uc_name(self, content: str, filename: str) -> str:
        """Extract use-case name from content or fall back to filename."""
        import re
        for line in content.split("\n"):
            line = line.strip()
            if not line or line == "ID":
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                return parts[-1].strip() if len(parts) > 1 else parts[0].strip()
            return line
        return Path(filename).stem

    def _already_processed(self, fp_id: str) -> bool:
        """Check if this FP was already processed in a previous run."""
        return (self.measurements_dir / f"{fp_id}_measurement.csv").exists()

    def _load_existing_summary(self) -> list[dict]:
        """Load rows from a previous batch_summary.csv on resume."""
        path = self.run_dir / "batch_summary.csv"
        if not path.exists():
            return []
        import pandas as pd
        df = pd.read_csv(path, dtype=str)
        return df.to_dict("records")

    def _append_summary_row(self, row: dict) -> None:
        """Append a single row to batch_summary.csv (incremental write)."""
        path = self.run_dir / "batch_summary.csv"
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def run_batch(self, input_dir: Path) -> None:
        """Iterate over FP files, collect metrics, produce standard output.

        Supports --resume: skips FPs that already have measurement output,
        and appends new results to the existing batch_summary.csv.
        """
        import re
        from .io_utils import find_fp_files

        self.setup()
        files = find_fp_files(input_dir)
        if not files:
            print(f"No .txt files found in {input_dir}")
            return

        print(f"Method:  {self.method} ({self.method_type})")
        print(f"Model:   {self.llm_model}")
        print(f"Dataset: {self.dataset}")
        print(f"Run ID:  {self.run_id}")
        print(f"Output:  {self.run_dir}")
        if self._resuming:
            print("Mode:    RESUME")
        if self.dry_run:
            print("Mode:    DRY-RUN")
        print(f"\nFound {len(files)} FP files in {input_dir}")

        n_skipped = 0

        existing_summary: list[dict] = []
        if self._resuming:
            existing_summary = self._load_existing_summary()

        summary: list[dict] = []
        durations: list[float] = []
        split_durations: list[float] = []
        analyze_durations: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_retries = 0
        n_errors = 0
        total_cfp = 0

        existing_by_file: dict[str, dict] = {r["file"]: r for r in existing_summary}

        wall_t0 = time.perf_counter()

        for idx, fp_file in enumerate(files, 1):
            content = fp_file.read_text(encoding="utf-8", errors="replace")
            uc_name = self.extract_uc_name(content, fp_file.name)
            fp_num_m = re.search(r"(\d+)", fp_file.stem)
            fp_id = fp_num_m.group(1) if fp_num_m else str(idx)

            if self._resuming and self._already_processed(fp_id):
                prev = existing_by_file.get(fp_file.name)
                if prev and prev.get("status") == "ok":
                    summary.append(prev)
                    try:
                        total_cfp += int(prev.get("cfp", 0))
                        total_input_tokens += int(prev.get("input_tokens") or 0)
                        total_output_tokens += int(prev.get("output_tokens") or 0)
                        durations.append(float(prev.get("duration_s") or 0))
                        split_durations.append(float(prev.get("split_duration_s") or 0))
                        analyze_durations.append(float(prev.get("analyze_duration_s") or 0))
                    except (ValueError, TypeError):
                        pass
                n_skipped += 1
                continue

            print(f"  [{idx}/{len(files)}] {fp_file.name} -> \"{uc_name}\"", end="")

            if self.dry_run:
                print(" [DRY-RUN]")
                row = self._dry_row(fp_file.name, uc_name)
                summary.append(row)
                self._append_summary_row(row)
                continue

            try:
                result = self.process_fp(fp_file)
                cfp = result["table"]["CFP"][-1] if result["table"]["CFP"] else "0"
                cfp_int = int(cfp) if cfp.isdigit() else 0
                total_cfp += cfp_int

                self._save_fp_output(fp_id, result)

                dur = result["duration"]
                split_dur = result.get("split_duration", 0.0)
                analyze_dur = result.get("analyze_duration", 0.0)
                inp_tok = result["input_tokens"]
                out_tok = result["output_tokens"]
                retries = result.get("retries", 0)

                durations.append(dur)
                split_durations.append(split_dur)
                analyze_durations.append(analyze_dur)
                total_input_tokens += inp_tok
                total_output_tokens += out_tok
                total_retries += retries

                cost = estimate_cost(inp_tok, out_tok, self.llm_model)

                print(f" -> CFP={cfp} ({dur:.1f}s, {inp_tok}+{out_tok} tok, ${cost:.4f})")

                split_in = result.get("split_input_tokens", 0)
                split_out = result.get("split_output_tokens", 0)
                analyze_in = result.get("analyze_input_tokens", 0)
                analyze_out = result.get("analyze_output_tokens", 0)
                wait_time = result.get("wait_time", 0.0)

                row = {
                    "file": fp_file.name,
                    "uc_name": uc_name,
                    "cfp": cfp,
                    "status": "ok",
                    "duration_s": round(dur, 3),
                    "split_duration_s": round(split_dur, 3),
                    "analyze_duration_s": round(analyze_dur, 3),
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                    "split_input_tokens": split_in,
                    "split_output_tokens": split_out,
                    "analyze_input_tokens": analyze_in,
                    "analyze_output_tokens": analyze_out,
                    "retries": retries,
                    "wait_time_s": round(wait_time, 3),
                    "estimated_cost_usd": round(cost, 6),
                }
                summary.append(row)
                self._append_summary_row(row)

            except Exception as exc:
                print(f" -> ERROR: {exc}")
                n_errors += 1
                row = {
                    "file": fp_file.name,
                    "uc_name": uc_name,
                    "cfp": "ERROR",
                    "status": str(exc),
                    "duration_s": "",
                    "split_duration_s": "",
                    "analyze_duration_s": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "split_input_tokens": "",
                    "split_output_tokens": "",
                    "analyze_input_tokens": "",
                    "analyze_output_tokens": "",
                    "retries": "",
                    "wait_time_s": "",
                    "estimated_cost_usd": "",
                }
                summary.append(row)
                self._append_summary_row(row)

        wall_clock = time.perf_counter() - wall_t0

        if n_skipped:
            print(f"\n  Resumed: {n_skipped} FPs skipped (already processed)")

        self._write_summary(summary)

        n_total_fps = len(files)
        if not self.dry_run:
            self._write_run_metrics(
                durations=durations,
                split_durations=split_durations,
                analyze_durations=analyze_durations,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_retries=total_retries,
                n_errors=n_errors,
                total_cfp=total_cfp,
                wall_clock=wall_clock,
                n_fps=n_total_fps,
            )

            self._register(
                n_fps=n_total_fps,
                n_errors=n_errors,
                total_cfp=total_cfp,
                wall_clock=wall_clock,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )

        print(f"\nRun complete: {self.run_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dry_row(filename: str, uc_name: str) -> dict:
        return {
            "file": filename,
            "uc_name": uc_name,
            "cfp": "N/A",
            "status": "dry-run",
            "duration_s": "",
            "split_duration_s": "",
            "analyze_duration_s": "",
            "input_tokens": "",
            "output_tokens": "",
            "split_input_tokens": "",
            "split_output_tokens": "",
            "analyze_input_tokens": "",
            "analyze_output_tokens": "",
            "retries": "",
            "wait_time_s": "",
            "estimated_cost_usd": "",
        }

    def _save_fp_output(self, fp_id: str, result: dict) -> None:
        from .io_utils import save_measurement_csv, save_measurement_xlsx

        save_measurement_xlsx(result, self.measurements_dir / f"{fp_id}_measurement.xlsx")
        save_measurement_csv(result, self.measurements_dir / f"{fp_id}_measurement.csv")
        raw_path = self.measurements_dir / f"{fp_id}_raw_response.txt"
        raw_path.write_text(result.get("raw_response", ""), encoding="utf-8")
        split_path = self.measurements_dir / f"{fp_id}_split_response.txt"
        split_path.write_text(result.get("split_text", ""), encoding="utf-8")

    def _write_summary(self, summary: list[dict]) -> None:
        path = self.run_dir / "batch_summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(summary)
        print(f"\nSummary: {path}")

    def _write_run_metrics(
        self,
        *,
        durations: list[float],
        split_durations: list[float],
        analyze_durations: list[float],
        total_input_tokens: int,
        total_output_tokens: int,
        total_retries: int,
        n_errors: int,
        total_cfp: int,
        wall_clock: float,
        n_fps: int,
    ) -> None:
        total_tokens = total_input_tokens + total_output_tokens
        cost = estimate_cost(total_input_tokens, total_output_tokens, self.llm_model)
        throughput = compute_throughput(n_fps, total_tokens, wall_clock)

        metrics = {
            "meta": {
                "run_id": self.run_id,
                "method": self.method,
                "method_type": self.method_type,
                "dataset": self.dataset,
                "llm_model": self.llm_model,
                "timestamp": datetime.now(WIB).isoformat(),
                "n_fps": n_fps,
                "n_errors": n_errors,
                "total_retries": total_retries,
            },
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
                "total": total_tokens,
                "mean_input_per_fp": round(total_input_tokens / max(n_fps, 1), 1),
                "mean_output_per_fp": round(total_output_tokens / max(n_fps, 1), 1),
            },
            "cost": {
                "total_usd": round(cost, 6),
                "mean_per_fp_usd": round(cost / max(n_fps, 1), 6),
            },
            "performance": {
                "wall_clock_s": round(wall_clock, 3),
                "total_api_time_s": round(sum(durations), 3),
                **{f"fp_{k}": v for k, v in compute_timing_stats(durations).items()},
                "mean_split_duration_s": round(
                    sum(split_durations) / max(len(split_durations), 1), 3
                ),
                "mean_analyze_duration_s": round(
                    sum(analyze_durations) / max(len(analyze_durations), 1), 3
                ),
                **throughput,
            },
            "accuracy": {
                "total_cfp_pred": total_cfp,
            },
        }

        path = self.run_dir / "run_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics: {path}")

    def _register(
        self,
        *,
        n_fps: int,
        n_errors: int,
        total_cfp: int,
        wall_clock: float,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> None:
        total_tokens = total_input_tokens + total_output_tokens
        cost = estimate_cost(total_input_tokens, total_output_tokens, self.llm_model)
        throughput = compute_throughput(n_fps, total_tokens, wall_clock)

        register_run(
            self.results_dir,
            {
                "run_id": self.run_id,
                "method": self.method,
                "method_type": self.method_type,
                "dataset": self.dataset,
                "llm_model": self.llm_model,
                "timestamp": datetime.now(WIB).isoformat(),
                "n_fps": n_fps,
                "n_errors": n_errors,
                "total_cfp_pred": total_cfp,
                "wall_clock_s": round(wall_clock, 3),
                "total_tokens": total_tokens,
                "total_cost_usd": round(cost, 6),
                "throughput_fp_min": throughput["fp_per_min"],
                "status": "ok" if n_errors == 0 else "partial",
            },
        )
