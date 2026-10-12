#!/usr/bin/env python3
"""Hybrid XAI Experiment Runner — runs E1-E6 across all datasets.

Usage:
    # E1: NLP-only (spaCy)
    python scripts/hybrid_xai_runner.py <input_dir> --normalizer nlp

    # E2: SLM Qwen 7B (IR-Full)
    python scripts/hybrid_xai_runner.py <input_dir> --normalizer slm \
        --model-path models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

    # E3: SLM Qwen 3B (IR-Full)
    python scripts/hybrid_xai_runner.py <input_dir> --normalizer slm \
        --model-path models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf

    # E5: Hybrid (spaCy + SLM)
    python scripts/hybrid_xai_runner.py <input_dir> --normalizer hybrid \
        --model-path models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf

    # Full dataset scan (De Vito)
    python scripts/hybrid_xai_runner.py \
        replications/cosmet-devito/datasets/MIS/ALBERGATE/FP_prompts \
        --normalizer nlp --dataset devito --source-format use_case

    # Zenodo sentences
    python scripts/hybrid_xai_runner.py \
        replications/cosmet-devito/datasets/Zenodo \
        --normalizer slm --dataset zenodo --source-format sentence \
        --model-path models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE / "models" / "hybrid-xai" / "schema.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.seeds import set_global_seeds

# ---------------------------------------------------------------------------
# Dynamic module loading for hybrid-xai package (hyphen in dir name)
# ---------------------------------------------------------------------------

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
        ("models.hybrid_xai.normalizers.hybrid_normalizer", pkg_base / "normalizers" / "hybrid_normalizer.py"),
        ("models.hybrid_xai.normalizers.stepwise_normalizer", pkg_base / "normalizers" / "stepwise_normalizer.py"),
        ("models.hybrid_xai.normalizers.keyword_normalizer", pkg_base / "normalizers" / "keyword_normalizer.py"),
        ("models.hybrid_xai.normalizers.ensemble_normalizer", pkg_base / "normalizers" / "ensemble_normalizer.py"),
        ("models.hybrid_xai.normalizers.regression_normalizer", pkg_base / "normalizers" / "regression_normalizer.py"),
        ("models.hybrid_xai.normalizers.pipeline_normalizer", pkg_base / "normalizers" / "pipeline_normalizer.py"),
        ("models.hybrid_xai.normalizers.rewrite_normalizer", pkg_base / "normalizers" / "rewrite_normalizer.py"),
        ("models.hybrid_xai.normalizers.openai_normalizer", pkg_base / "normalizers" / "openai_normalizer.py"),
        ("models.hybrid_xai.normalizers.format_aware_normalizer", pkg_base / "normalizers" / "format_aware_normalizer.py"),
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
    norms.nlp_normalizer = sys.modules["models.hybrid_xai.normalizers.nlp_normalizer"]
    if "models.hybrid_xai.normalizers.slm_normalizer" in sys.modules:
        norms.slm_normalizer = sys.modules["models.hybrid_xai.normalizers.slm_normalizer"]
    if "models.hybrid_xai.normalizers.hybrid_normalizer" in sys.modules:
        norms.hybrid_normalizer = sys.modules["models.hybrid_xai.normalizers.hybrid_normalizer"]

    for name, _ in mod_list:
        if name in sys.modules:
            sys.modules[name].__spec__.loader.exec_module(sys.modules[name])


_load_hybrid_xai()

from models.hybrid_xai.schema import (
    Metadata,
    NormalizedRequirement,
    SourceFormat,
)
from models.hybrid_xai.rule_engine import CosmicRuleEngine
from models.hybrid_xai.normalizers.nlp_normalizer import NLPNormalizer
from models.hybrid_xai.normalizers.base import BaseNormalizer


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

_DEVITO_GT_SOURCES: dict[str, str] = {
    "ALBERGATE": "ALBERGATE-gpt-4-turbo-20260305T023001",
    "K01719": "K01719-gpt-4-turbo-20260323T005355",
    "K01720": "K01720-gpt-4-turbo-20260323T005553",
    "K01726": "K01726-gpt-4-turbo-20260323T005703",
    "U-CURE": "U-CURE-gpt-4-turbo-20260323T010152",
    "AutomaticLineSwitching": "AutomaticLineSwitching-gpt-4-turbo-20260323T010407",
    "RiceCooker": "RiceCooker-gpt-4-turbo-20260323T010514",
}

_DEVITO_FP_DIRS: dict[str, str] = {
    "ALBERGATE": "MIS/ALBERGATE/FP_prompts",
    "K01719": "IoT_Telemedicine/K01719/FP_prompts",
    "K01720": "IoT_Telemedicine/K01720/FP_prompts",
    "K01726": "IoT_Telemedicine/K01726/FP_prompts",
    "U-CURE": "ML/U-CURE/FP_prompts",
    "AutomaticLineSwitching": "Real-time/AutomaticLineSwitching/FP_prompts",
    "RiceCooker": "Real-time/RiceCooker/FP_prompts",
}


def load_devito_gt(scope: str = "albergate") -> dict[str, int]:
    """Load ground truth for De Vito dataset.

    scope='albergate' -> 62 FPs, backward-compatible flat keys (FP1, FP2, ...).
    scope='all'       -> 119 FPs, composite keys (ALBERGATE/FP1, K01719/FP3, ...).
    """
    if scope == "albergate":
        gt_path = (
            BASE / "results" / "cosmet"
            / "ALBERGATE-gpt-4-turbo-20260305T023001"
            / "measurements" / "comparison_results.csv"
        )
        gt: dict[str, int] = {}
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                gt[row["fp"]] = int(row["y_true"])
        return gt

    gt = {}
    for system, run_id in _DEVITO_GT_SOURCES.items():
        csv_path = (
            BASE / "results" / "cosmet" / run_id
            / "measurements" / "comparison_results.csv"
        )
        if not csv_path.exists():
            print(f"  WARNING: GT not found for {system}: {csv_path}")
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                gt[f"{system}/{row['fp']}"] = int(row["y_true"])
    return gt


def load_zenodo_gt() -> dict[str, int]:
    gt_path = BASE / "datasets" / "prepared" / "zenodo-cosmet" / "sentence" / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    gt: dict[str, int] = {}
    with open(gt_path) as f:
        for row in csv.DictReader(f):
            fp_id = row.get("fp_file", "").replace(".txt", "")
            if not fp_id:
                fp_id = f"FP{row.get('fp_num', row.get('id', ''))}"
            gt[fp_id] = int(row["cfp_gt"])
    return gt


def load_ochodek_gt() -> dict[str, int]:
    csv_path = BASE / "datasets" / "raw" / "ochodek-uc-437" / "use-cases.csv"
    if not csv_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    gt: dict[str, int] = {}
    for _, row in df.iterrows():
        gt[row["UC"]] = int(row["Cfp"])
    return gt


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def iter_devito_fps(input_dir: Path, scope: str = "albergate"):
    """Yield (fp_id, raw_text, source_format) for De Vito FPs.

    scope='albergate' -> iterate only the provided input_dir (62 FPs).
    scope='all'       -> iterate all 7 systems (119 FPs), composite keys.
    """
    if scope == "albergate":
        for fp_file in sorted(input_dir.glob("FP*.txt")):
            fp_id = fp_file.stem
            text = fp_file.read_text(encoding="utf-8", errors="replace")
            yield fp_id, text, SourceFormat.USE_CASE
        return

    datasets_root = BASE / "replications" / "cosmet-devito" / "datasets"
    for system, rel_path in sorted(_DEVITO_FP_DIRS.items()):
        fp_dir = datasets_root / rel_path
        if not fp_dir.exists():
            print(f"  WARNING: FP dir not found for {system}: {fp_dir}")
            continue
        fps = sorted(fp_dir.glob("FP*.txt"))
        for fp_file in fps:
            fp_id = f"{system}/{fp_file.stem}"
            text = fp_file.read_text(encoding="utf-8", errors="replace")
            yield fp_id, text, SourceFormat.USE_CASE


def iter_zenodo_fps(input_dir: Path):
    """Yield (fp_id, raw_text, source_format) for Zenodo sentences."""
    fp_dir = input_dir / "FP_prompts"
    if fp_dir.exists():
        input_dir = fp_dir

    fps = sorted(input_dir.glob("FP*.txt"))
    if not fps:
        gt_csv = input_dir / "ground_truth.csv"
        if gt_csv.exists():
            with open(gt_csv) as f:
                for row in csv.DictReader(f):
                    fp_id = row.get("fp_file", "").replace(".txt", "")
                    text = row.get("requirement", "")
                    if text:
                        yield fp_id, text, SourceFormat.SENTENCE
            return

    for fp_file in fps:
        fp_id = fp_file.stem
        text = fp_file.read_text(encoding="utf-8", errors="replace").strip()
        yield fp_id, text, SourceFormat.SENTENCE


def iter_ochodek_fps(input_dir: Path):
    """Yield (fp_id, raw_text, source_format) for Ochodek UC names."""
    if input_dir.is_file() and input_dir.suffix == ".csv":
        csv_path = input_dir
    else:
        csv_path = input_dir / "use-cases.csv"
        if not csv_path.exists():
            csv_path = input_dir / "ochodek_437.csv"
        if not csv_path.exists():
            for candidate in input_dir.glob("*.csv"):
                csv_path = candidate
                break
    if not csv_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        fp_id = str(row.get("UC", row.get("id", f"O{row.name}")))
        text = str(row.get("TitleTokens", row.get("use_case_name", row.get("text", ""))))
        yield fp_id, text, SourceFormat.UC_NAME


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def create_normalizer(
    norm_type: str, model_path: str | None, n_threads: int,
    n_gpu_layers: int = 0, chain_of_thought: bool = False,
    n_ctx: int = 8192,
    **kwargs,
):
    def _slm(**overrides):
        from models.hybrid_xai.normalizers.slm_normalizer import SLMNormalizer
        if not model_path:
            raise ValueError("--model-path required for SLM normalizer")
        kw = dict(n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
        kw.update(overrides)
        return SLMNormalizer(model_path, **kw)

    if norm_type == "nlp":
        return NLPNormalizer()
    elif norm_type == "slm":
        return _slm(chain_of_thought=chain_of_thought)
    elif norm_type == "hybrid":
        from models.hybrid_xai.normalizers.hybrid_normalizer import HybridNormalizer
        return HybridNormalizer(model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    elif norm_type == "stepwise":
        from models.hybrid_xai.normalizers.stepwise_normalizer import StepwiseNormalizer
        return StepwiseNormalizer(model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    elif norm_type == "keyword":
        from models.hybrid_xai.normalizers.keyword_normalizer import KeywordNormalizer
        return KeywordNormalizer()
    elif norm_type == "ensemble":
        from models.hybrid_xai.normalizers.keyword_normalizer import KeywordNormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import EnsembleNormalizer
        return EnsembleNormalizer([NLPNormalizer(), KeywordNormalizer(), _slm()])
    elif norm_type == "adaptive":
        from models.hybrid_xai.normalizers.stepwise_normalizer import StepwiseNormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import AdaptiveNormalizer
        return AdaptiveNormalizer(
            simple_normalizer=NLPNormalizer(),
            complex_normalizer=StepwiseNormalizer(
                model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers,
            ),
            step_threshold=8,
        )
    elif norm_type == "correction":
        from models.hybrid_xai.normalizers.ensemble_normalizer import CorrectionNormalizer
        return CorrectionNormalizer(base_normalizer=_slm())
    elif norm_type == "regression":
        from models.hybrid_xai.normalizers.regression_normalizer import RegressionNormalizer
        return RegressionNormalizer()
    elif norm_type == "pipeline":
        from models.hybrid_xai.normalizers.pipeline_normalizer import PipelineNormalizer
        return PipelineNormalizer(slm_normalizer=_slm(), nlp_normalizer=NLPNormalizer())
    elif norm_type == "pipeline-nodecomp":
        from models.hybrid_xai.normalizers.pipeline_normalizer import PipelineNormalizer
        return PipelineNormalizer(
            slm_normalizer=_slm(), nlp_normalizer=NLPNormalizer(), decompose=False,
        )
    elif norm_type == "rewrite":
        from models.hybrid_xai.normalizers.rewrite_normalizer import RewriteNormalizer
        return RewriteNormalizer(_slm())
    elif norm_type == "enhanced-nlp":
        from models.hybrid_xai.normalizers.nlp_normalizer import EnhancedNLPNormalizer
        return EnhancedNLPNormalizer()
    elif norm_type == "rewrite-token":
        from models.hybrid_xai.normalizers.rewrite_normalizer import RewriteTokenNormalizer
        return RewriteTokenNormalizer(_slm())
    elif norm_type == "rewrite-v2":
        from models.hybrid_xai.normalizers.rewrite_normalizer import RewriteV2Normalizer
        return RewriteV2Normalizer(_slm())
    elif norm_type == "tiered-correction":
        from models.hybrid_xai.normalizers.ensemble_normalizer import TieredCorrectionNormalizer
        return TieredCorrectionNormalizer(base_normalizer=_slm())
    elif norm_type == "smart-ensemble":
        from models.hybrid_xai.normalizers.rewrite_normalizer import RewriteV2Normalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import SmartEnsembleNormalizer
        slm = _slm()
        return SmartEnsembleNormalizer(
            primary_normalizer=slm, rewrite_normalizer=RewriteV2Normalizer(slm),
        )
    elif norm_type == "tiered-correction-opt":
        from models.hybrid_xai.normalizers.ensemble_normalizer import TieredCorrectionNormalizer
        return TieredCorrectionNormalizer(
            base_normalizer=_slm(),
            alpha=0.2, beta=7.0, gamma=0.4,
            alpha_large=0.45, beta_large=3.0, gamma_large=0.6,
            large_threshold=16,
        )
    elif norm_type == "openai-slm":
        from models.hybrid_xai.normalizers.openai_normalizer import OpenAINormalizer
        return OpenAINormalizer(model=kwargs.get("openai_model", "gpt-4o-mini"))
    elif norm_type == "openai-fmt":
        from models.hybrid_xai.normalizers.format_aware_normalizer import (
            FormatAwareOpenAINormalizer,
        )
        return FormatAwareOpenAINormalizer(
            model=kwargs.get("openai_model", "gpt-4o-mini"),
            force_bucket=kwargs.get("fewshot_bucket"),
        )
    elif norm_type == "openai-tiered-correction":
        from models.hybrid_xai.normalizers.openai_normalizer import OpenAINormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import TieredCorrectionNormalizer
        return TieredCorrectionNormalizer(
            base_normalizer=OpenAINormalizer(model=kwargs.get("openai_model", "gpt-4o-mini")),
        )
    elif norm_type == "openai-strict-tiered":
        from models.hybrid_xai.normalizers.openai_normalizer import OpenAINormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import TieredCorrectionNormalizer
        return TieredCorrectionNormalizer(
            base_normalizer=OpenAINormalizer(
                model=kwargs.get("openai_model", "gpt-4o-mini"),
                use_strict_schema=True,
            ),
        )
    elif norm_type == "self-consistency":
        from models.hybrid_xai.normalizers.ensemble_normalizer import (
            SelfConsistencyNormalizer, TieredCorrectionNormalizer,
        )
        return TieredCorrectionNormalizer(
            base_normalizer=SelfConsistencyNormalizer(_slm(), n_samples=3),
        )
    elif norm_type == "openai-self-consistency":
        from models.hybrid_xai.normalizers.openai_normalizer import OpenAINormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import (
            SelfConsistencyNormalizer, TieredCorrectionNormalizer,
        )
        base = OpenAINormalizer(model=kwargs.get("openai_model", "gpt-4o-mini"))
        return TieredCorrectionNormalizer(
            base_normalizer=SelfConsistencyNormalizer(base, n_samples=5),
        )
    elif norm_type == "hierarchical":
        from models.hybrid_xai.normalizers.ensemble_normalizer import HierarchicalDecompositionNormalizer
        return HierarchicalDecompositionNormalizer(_slm())
    elif norm_type == "hierarchical-tiered":
        from models.hybrid_xai.normalizers.ensemble_normalizer import (
            HierarchicalDecompositionNormalizer, TieredCorrectionNormalizer,
        )
        return TieredCorrectionNormalizer(
            base_normalizer=HierarchicalDecompositionNormalizer(_slm()),
            alpha=0.2, beta=7.0, gamma=0.4,
            alpha_large=0.45, beta_large=3.0, gamma_large=0.6,
            large_threshold=16,
        )
    elif norm_type == "hierarchical-sbert":
        from models.hybrid_xai.normalizers.ensemble_normalizer import HierarchicalDecompositionNormalizer
        return HierarchicalDecompositionNormalizer(_slm(), sbert_dedup=True, sbert_sim_threshold=0.75)
    elif norm_type == "hierarchical-sbert-tiered":
        from models.hybrid_xai.normalizers.ensemble_normalizer import (
            HierarchicalDecompositionNormalizer, TieredCorrectionNormalizer,
        )
        return TieredCorrectionNormalizer(
            base_normalizer=HierarchicalDecompositionNormalizer(
                _slm(), sbert_dedup=True, sbert_sim_threshold=0.75,
            ),
            alpha=0.2, beta=7.0, gamma=0.4,
            alpha_large=0.45, beta_large=3.0, gamma_large=0.6,
            large_threshold=16,
        )
    elif norm_type == "split-tiered-correction":
        from models.hybrid_xai.normalizers.split_preprocessor import SplitPreprocessor
        from models.hybrid_xai.normalizers.ensemble_normalizer import SplitNormalizer
        slm = _slm()
        splitter = SplitPreprocessor(backend="local", llm=slm._llm)
        return SplitNormalizer(
            split_preprocessor=splitter,
            base_normalizer=slm,
            apply_correction=True,
        )
    elif norm_type == "split-openai-tiered":
        from models.hybrid_xai.normalizers.split_preprocessor import SplitPreprocessor
        from models.hybrid_xai.normalizers.openai_normalizer import OpenAINormalizer
        from models.hybrid_xai.normalizers.ensemble_normalizer import SplitNormalizer
        oai_model = kwargs.get("openai_model", "gpt-4o-mini")
        splitter = SplitPreprocessor(backend="openai", openai_model=oai_model)
        return SplitNormalizer(
            split_preprocessor=splitter,
            base_normalizer=OpenAINormalizer(model=oai_model),
            apply_correction=True,
        )
    else:
        raise ValueError(f"Unknown normalizer: {norm_type}")


def run_experiment(
    normalizer,
    engine: CosmicRuleEngine,
    fp_iter,
    gt: dict[str, int],
    *,
    output_dir: Path,
    max_fps: int | None = None,
) -> dict[str, Any]:
    """Run normalizer + engine on all FPs and collect metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    timings: list[float] = []

    count = 0
    for fp_id, text, source_format in fp_iter:
        if max_fps and count >= max_fps:
            break

        t0 = time.time()
        nr = normalizer.normalize(
            text, source_format, fp_id=fp_id,
            metadata=Metadata(source_file=fp_id),
        )
        engine_result = engine.measure(nr)
        elapsed = time.time() - t0

        pred_cfp = engine_result.total_cfp
        gt_cfp = gt.get(fp_id)

        row = {
            "fp": fp_id,
            "y_pred": pred_cfp,
            "y_true": gt_cfp if gt_cfp is not None else "",
            "n_processes": len(nr.functional_processes),
            "n_movements": sum(
                len(pr.movements) for pr in engine_result.process_results
            ),
            "time_s": round(elapsed, 2),
            "source_format": source_format.value,
        }
        results.append(row)

        if gt_cfp is not None:
            y_true.append(gt_cfp)
            y_pred.append(pred_cfp)

        timings.append(elapsed)
        count += 1

        status = ""
        if gt_cfp is not None:
            err = pred_cfp - gt_cfp
            status = f"err={err:+d}" if err != 0 else "OK"

        print(f"  [{count:>4}] {fp_id:>8} | pred={pred_cfp:>3} | gt={gt_cfp or '?':>3} | {elapsed:.1f}s | {status}")

        safe_name = fp_id.replace("/", "_")
        ir_path = output_dir / f"{safe_name}_ir.json"
        ir_path.write_text(nr.to_json(indent=2), encoding="utf-8")

    # Write results CSV
    csv_path = output_dir / "comparison_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # Compute metrics
    metrics: dict[str, Any] = {
        "n_fps": count,
        "total_time_s": round(sum(timings), 1),
        "avg_time_s": round(sum(timings) / max(count, 1), 2),
    }

    if y_true and y_pred:
        from shared.metrics import mae, mdae, mmre, pred_x, standardized_accuracy as sa, accuracy

        metrics.update({
            "mae": round(mae(y_true, y_pred), 3),
            "mdae": round(mdae(y_true, y_pred), 3),
            "mmre": round(mmre(y_true, y_pred), 3),
            "pred_25": round(pred_x(y_true, y_pred, 0.25), 3),
            "sa": round(sa(y_true, y_pred), 3),
            "accuracy": round(accuracy(y_true, y_pred), 3),
            "n_with_gt": len(y_true),
        })

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main():
    set_global_seeds()
    parser = argparse.ArgumentParser(
        description="Hybrid XAI Experiment Runner"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--normalizer", required=True,
                        choices=["nlp", "slm", "hybrid", "stepwise", "keyword",
                                 "ensemble", "adaptive", "correction", "regression",
                                 "pipeline", "pipeline-nodecomp",
                                 "rewrite", "enhanced-nlp", "rewrite-token",
                                 "rewrite-v2", "tiered-correction",
                                 "tiered-correction-opt",
                                 "smart-ensemble",
                                 "openai-slm", "openai-fmt", "openai-tiered-correction",
                                 "openai-strict-tiered",
                                 "self-consistency", "openai-self-consistency",
                                 "hierarchical", "hierarchical-tiered",
                                 "hierarchical-sbert", "hierarchical-sbert-tiered",
                                 "split-tiered-correction", "split-openai-tiered"])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", default=None, choices=["devito", "zenodo", "ochodek"])
    parser.add_argument("--source-format", default=None, choices=["use_case", "sentence", "uc_name"])
    parser.add_argument("--results-dir", type=Path, default=BASE / "results" / "hybrid-xai")
    parser.add_argument("--max-fps", type=int, default=None)
    parser.add_argument("--fewshot-bucket", default=None,
                        help="Override few-shot bucket for --normalizer openai-fmt "
                             "(use_case|sentence|connextra). Needed because "
                             "--source-format has no user_story value.")
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="GPU layers for Vulkan/ROCm (0=CPU, 99=all on GPU)")
    parser.add_argument("--n-ctx", type=int, default=8192,
                        help="Context window size for SLM (default: 8192)")
    parser.add_argument("--chain-of-thought", action="store_true",
                        help="Use two-pass chain-of-thought for SLM")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="OpenAI model name for openai-* normalizers (default: gpt-4o-mini)")
    parser.add_argument("--devito-scope", default="albergate",
                        choices=["albergate", "all"],
                        help="De Vito scope: 'albergate' (62 FPs) or 'all' (119 FPs, 7 systems)")
    parser.add_argument("--fp-ids-file", type=Path, default=None,
                        help="CSV with 'id' column — restrict run to these FP IDs only")
    args = parser.parse_args()

    dataset = args.dataset
    if dataset is None:
        path_lower = str(args.input_dir).lower()
        if "albergate" in path_lower or "devito" in path_lower:
            dataset = "devito"
        elif "zenodo" in path_lower:
            dataset = "zenodo"
        elif "ochodek" in path_lower:
            dataset = "ochodek"
        else:
            dataset = args.input_dir.name

    model_name = ""
    if args.normalizer.startswith("openai-"):
        model_name = args.openai_model
    elif args.model_path:
        model_name = args.model_path.stem.split("-q4")[0].replace("-instruct", "")

    exp_id = args.experiment_id or f"{args.normalizer}-{model_name or 'cpu'}"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    scope_tag = "119fp" if (dataset == "devito" and args.devito_scope == "all") else dataset
    run_name = f"{scope_tag}-{exp_id}-{timestamp}"
    output_dir = args.results_dir / run_name

    print(f"{'='*60}")
    print(f"Hybrid XAI Experiment: {run_name}")
    print(f"  Normalizer: {args.normalizer}")
    print(f"  Model: {args.model_path or 'N/A (NLP only)'}")
    print(f"  Dataset: {dataset}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    normalizer = create_normalizer(
        args.normalizer, str(args.model_path) if args.model_path else None,
        args.n_threads, args.n_gpu_layers, args.chain_of_thought,
        n_ctx=args.n_ctx, openai_model=args.openai_model,
        fewshot_bucket=args.fewshot_bucket,
    )
    engine = CosmicRuleEngine()

    if dataset == "devito":
        gt = load_devito_gt(scope=args.devito_scope)
        fp_iter = iter_devito_fps(args.input_dir, scope=args.devito_scope)
    elif dataset == "zenodo":
        gt = load_zenodo_gt()
        fp_iter = iter_zenodo_fps(args.input_dir)
    elif dataset == "ochodek":
        gt = load_ochodek_gt()
        fp_iter = iter_ochodek_fps(args.input_dir)
    else:
        gt = {}
        fp_iter = iter_devito_fps(args.input_dir)

    if args.fp_ids_file:
        import csv as _csv
        import re as _re

        def _to_runner_id(manifest_id: str) -> str:
            # "COSMET-ALBERGATE-FP32" → "ALBERGATE/FP32"; "COSMET-U-CURE-FP1" → "U-CURE/FP1"
            mid = manifest_id[len("COSMET-"):] if manifest_id.startswith("COSMET-") else manifest_id
            return _re.sub(r"-FP(\d+)$", r"/FP\1", mid)

        raw_ids = {r["id"] for r in _csv.DictReader(args.fp_ids_file.open())}
        allowed_ids = {_to_runner_id(i) for i in raw_ids} | raw_ids
        print(f"  Filtering to {len(raw_ids)} FP IDs from {args.fp_ids_file}")
        fp_iter = (item for item in fp_iter if item[0] in allowed_ids)

    metrics = run_experiment(
        normalizer, engine, fp_iter, gt,
        output_dir=output_dir,
        max_fps=args.max_fps,
    )

    inner = None
    if hasattr(normalizer, "get_usage_summary"):
        inner = normalizer
    elif hasattr(normalizer, "base") and hasattr(normalizer.base, "get_usage_summary"):
        inner = normalizer.base
    elif hasattr(normalizer, "base_normalizer") and hasattr(normalizer.base_normalizer, "get_usage_summary"):
        inner = normalizer.base_normalizer
    if inner:
        usage = inner.get_usage_summary()
        (output_dir / "openai_usage.json").write_text(
            json.dumps(usage, indent=2), encoding="utf-8"
        )
        metrics["openai_usage"] = usage

    print(f"\n{'='*60}")
    print(f"Results: {run_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
