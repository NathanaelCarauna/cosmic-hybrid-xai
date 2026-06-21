"""Hybrid XAI Pipeline — end-to-end COSMIC measurement from raw text.

Orchestrates Module 1 (normalization), Module 2 (rule engine), and
Module 3 (XAI trace generation) into a single callable pipeline.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

# Dynamic import helper for hyphenated directory
_PKG_DIR = Path(__file__).resolve().parent


def _ensure_modules():
    """Register hybrid-xai submodules if not already loaded."""
    if "models.hybrid_xai.schema" in sys.modules:
        return

    mod_list = [
        ("models.hybrid_xai", _PKG_DIR / "__init__.py"),
        ("models.hybrid_xai.schema", _PKG_DIR / "schema.py"),
        ("models.hybrid_xai.rule_engine", _PKG_DIR / "rule_engine.py"),
        ("models.hybrid_xai.xai", _PKG_DIR / "xai.py"),
        ("models.hybrid_xai.normalizers", _PKG_DIR / "normalizers" / "__init__.py"),
        ("models.hybrid_xai.normalizers.base", _PKG_DIR / "normalizers" / "base.py"),
        ("models.hybrid_xai.normalizers.nlp_normalizer", _PKG_DIR / "normalizers" / "nlp_normalizer.py"),
        ("models.hybrid_xai.normalizers.slm_normalizer", _PKG_DIR / "normalizers" / "slm_normalizer.py"),
        ("models.hybrid_xai.normalizers.hybrid_normalizer", _PKG_DIR / "normalizers" / "hybrid_normalizer.py"),
    ]
    for name, path in mod_list:
        if not path.exists() or name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod

    pkg = sys.modules.get("models.hybrid_xai")
    if pkg:
        pkg.schema = sys.modules.get("models.hybrid_xai.schema")
        pkg.rule_engine = sys.modules.get("models.hybrid_xai.rule_engine")
        norms = sys.modules.get("models.hybrid_xai.normalizers")
        if norms:
            pkg.normalizers = norms
            norms.base = sys.modules.get("models.hybrid_xai.normalizers.base")
            norms.nlp_normalizer = sys.modules.get("models.hybrid_xai.normalizers.nlp_normalizer")
            if "models.hybrid_xai.normalizers.slm_normalizer" in sys.modules:
                norms.slm_normalizer = sys.modules["models.hybrid_xai.normalizers.slm_normalizer"]
            if "models.hybrid_xai.normalizers.hybrid_normalizer" in sys.modules:
                norms.hybrid_normalizer = sys.modules["models.hybrid_xai.normalizers.hybrid_normalizer"]

    for name, _ in mod_list:
        if name in sys.modules and hasattr(sys.modules[name], "__spec__"):
            try:
                sys.modules[name].__spec__.loader.exec_module(sys.modules[name])
            except Exception:
                pass


class HybridXAIPipeline:
    """End-to-end COSMIC measurement pipeline.

    Args:
        normalizer: "nlp", "slm", or "hybrid"
        model_path: Path to GGUF model (required for slm/hybrid)
        n_threads: CPU threads for inference
        n_gpu_layers: GPU layers for llama.cpp
    """

    def __init__(
        self,
        normalizer: str = "nlp",
        model_path: Optional[str | Path] = None,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
    ) -> None:
        _ensure_modules()

        from .rule_engine import CosmicRuleEngine
        self.engine = CosmicRuleEngine()

        if normalizer == "nlp":
            from .normalizers.nlp_normalizer import NLPNormalizer
            self.normalizer = NLPNormalizer()
        elif normalizer == "slm":
            from .normalizers.slm_normalizer import SLMNormalizer
            self.normalizer = SLMNormalizer(
                model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers
            )
        elif normalizer == "hybrid":
            from .normalizers.hybrid_normalizer import HybridNormalizer
            self.normalizer = HybridNormalizer(
                model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers
            )
        else:
            raise ValueError(f"Unknown normalizer: {normalizer}")

    def analyze(
        self,
        text: str,
        *,
        fp_id: str = "",
        source_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze a requirement and return COSMIC measurement.

        Returns:
            dict with keys: cfp, movements, trace_report, mermaid,
            ir (NormalizedRequirement), result (RuleEngineResult)
        """
        from .schema import Metadata, SourceFormat
        from .normalizers.base import BaseNormalizer
        from .xai import generate_trace_report, generate_mermaid

        if source_format:
            sf = SourceFormat(source_format)
        else:
            sf = BaseNormalizer.detect_source_format(text)

        nr = self.normalizer.normalize(
            text, sf, fp_id=fp_id,
            metadata=Metadata(source_file=fp_id),
        )

        result = self.engine.measure(nr)

        trace = generate_trace_report(nr, result)
        mermaid = generate_mermaid(nr, result)

        return {
            "cfp": result.total_cfp,
            "movements": [
                {
                    "type": m.type.value,
                    "data_group": m.data_group,
                    "object_of_interest": m.object_of_interest,
                    "functional_user": m.functional_user,
                    "is_triggering_entry": m.is_triggering_entry,
                    "is_error_confirmation": m.is_error_confirmation,
                }
                for pr in result.process_results
                for m in pr.movements
            ],
            "trace_report": trace,
            "mermaid": mermaid,
            "ir": nr,
            "result": result,
        }
