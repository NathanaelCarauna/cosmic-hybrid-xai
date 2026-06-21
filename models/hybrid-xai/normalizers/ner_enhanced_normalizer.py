"""NER-Enhanced SLM Normalizer — E33f.

Injects NER-extracted entities as structured hints into the SLM prompt,
reducing missed entities on complex UCs. The NER module identifies
functional users, data objects, persistence stores, and UI elements
before the SLM processes the requirement.

This is a wrapper around SLMNormalizer/OpenAINormalizer.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Optional

import spacy

from ..schema import Metadata, NormalizedRequirement, SourceFormat
from .base import BaseNormalizer

BASE = Path(__file__).resolve().parent.parent.parent.parent


def _load_pipeline_module():
    """Load the NER ontology pipeline from hyphenated path."""
    mod_path = Path(__file__).parent.parent / "ner_srl" / "ner_ontology_pipeline.py"
    spec = importlib.util.spec_from_file_location("ner_ontology_pipeline", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


class NEREnhancedNormalizer(BaseNormalizer):
    """Wraps a base normalizer with NER entity pre-annotation hints."""

    def __init__(
        self,
        base_normalizer: BaseNormalizer,
        ner_model_path: Optional[str] = None,
    ):
        self._base = base_normalizer
        self._ner_path = ner_model_path or str(BASE / "models" / "cosmic-ner" / "best")

        try:
            self._ner_nlp = spacy.load(self._ner_path)
        except OSError:
            self._ner_nlp = None

        pipe_mod = _load_pipeline_module()
        self._pipeline = pipe_mod.NerOntologyPipeline(
            ner_model_path=ner_model_path,
        )

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        hints = self._extract_hints(text)
        enhanced_text = self._inject_hints(text, hints)

        return self._base.normalize(
            enhanced_text,
            source_format,
            fp_id=fp_id,
            metadata=metadata,
        )

    def _extract_hints(self, text: str) -> dict:
        """Extract NER entities and ontology predictions."""
        ner_entities = {}
        if self._ner_nlp:
            doc = self._ner_nlp(text)
            for ent in doc.ents:
                ner_entities.setdefault(ent.label_, set()).add(ent.text)

        result = self._pipeline.measure(text)

        return {
            "functional_users": sorted(ner_entities.get("FUNC_USER", set())),
            "data_objects": sorted(ner_entities.get("DATA_OBJ", set())),
            "persist_stores": sorted(ner_entities.get("PERSIST_STORE", set())),
            "ui_elements": sorted(ner_entities.get("UI_ELEMENT", set())),
            "ontology_cfp": result.cfp,
            "ontology_breakdown": result.movement_breakdown,
        }

    def _inject_hints(self, text: str, hints: dict) -> str:
        """Inject NER hints as structured annotation before the requirement text."""
        hint_lines = [
            "\n--- NER Pre-Analysis (use as reference, not as final answer) ---",
        ]

        if hints["functional_users"]:
            hint_lines.append(f"Functional Users detected: {', '.join(hints['functional_users'])}")
        if hints["data_objects"]:
            hint_lines.append(f"Data Objects detected: {', '.join(hints['data_objects'])}")
        if hints["persist_stores"]:
            hint_lines.append(f"Persistent Storage references: {', '.join(hints['persist_stores'])}")
        if hints["ui_elements"]:
            hint_lines.append(f"UI Elements (typically 0 CFP): {', '.join(hints['ui_elements'])}")

        bd = hints["ontology_breakdown"]
        hint_lines.append(
            f"Estimated breakdown: ~{bd['E']}E {bd['R']}R {bd['W']}W {bd['X']}X = ~{hints['ontology_cfp']} CFP"
        )
        hint_lines.append("--- End NER Pre-Analysis ---\n")

        return text + "\n".join(hint_lines)
