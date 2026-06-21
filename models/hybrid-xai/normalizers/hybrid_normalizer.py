"""Hybrid Normalizer — spaCy NLP pre-processing + SLM refinement (E5).

Uses spaCy to extract candidate actors, objects, and directions from UC
steps, then passes a condensed representation to the SLM for final
classification. Reduces SLM input tokens by ~30% compared to raw text.

Cost: $0 (local inference).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..schema import (
    BoundaryContext,
    DataMovement,
    FunctionalProcess,
    Metadata,
    MovementType,
    NormalizedRequirement,
    SourceFormat,
    SubProcess,
)
from .base import BaseNormalizer
from .nlp_normalizer import NLPNormalizer, _detect_direction, _is_error_or_confirm, _get_nlp

_CONDENSED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "data_movements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string"},
                    "type": {"type": "string", "enum": ["E", "X", "R", "W", "NONE"]},
                    "data_group": {"type": "string"},
                    "object_of_interest": {"type": "string"},
                    "is_triggering_entry": {"type": "boolean"},
                    "is_error_confirmation": {"type": "boolean"},
                },
                "required": ["step", "type", "data_group", "object_of_interest"],
            },
        },
    },
    "required": ["data_movements"],
}

_HYBRID_SYSTEM = """You are a COSMIC ISO 19761 measurement expert.

I will give you a Use Case with NLP-analyzed steps. For each step, classify the COSMIC data movement:
- E (Entry): user sends data TO the system
- X (Exit): system sends/displays data TO the user  
- R (Read): system retrieves data FROM persistent storage
- W (Write): system saves data TO persistent storage
- NONE: UI navigation, showing empty forms, clicking buttons (no data moved)

Rules:
- "clicks button/link/menu" → NONE
- "shows empty form" → NONE  
- "browses results" (passive) → NONE
- "resumes at step N" → NONE
- First Entry = is_triggering_entry: true
- Error/confirmation messages = is_error_confirmation: true

Example:
Input steps: 1. [User] selects option "Rooms" | 2. [System] shows form to search | 3. [User] fills in room code and searches | 4. [System] searches for rooms matching criteria | 5. [System] displays rooms list | 5.a1 [System] shows error message
Output: {"data_movements": [
  {"step": "1", "type": "NONE", "data_group": "", "object_of_interest": ""},
  {"step": "2", "type": "NONE", "data_group": "", "object_of_interest": ""},
  {"step": "3", "type": "E", "data_group": "Room code filter", "object_of_interest": "Room", "is_triggering_entry": true},
  {"step": "4", "type": "R", "data_group": "Room records", "object_of_interest": "Room"},
  {"step": "5", "type": "X", "data_group": "Rooms list", "object_of_interest": "Room"},
  {"step": "5.a1", "type": "X", "data_group": "Error message", "object_of_interest": "Room", "is_error_confirmation": true}
]}"""


class HybridNormalizer(BaseNormalizer):
    """Hybrid spaCy + SLM normalizer."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        from llama_cpp import Llama

        self._nlp_engine = _get_nlp()
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        if source_format == SourceFormat.USE_CASE:
            return self._normalize_uc(text, fp_id, metadata)
        if source_format == SourceFormat.UC_NAME:
            nlp = NLPNormalizer()
            return nlp.normalize(text, source_format, fp_id=fp_id, metadata=metadata)
        return self._normalize_sentence(text, fp_id, metadata)

    def _normalize_uc(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        fields = self.parse_uc_fields(text)
        primary_actor = fields["primary_actor"] or "User"
        all_steps = fields["steps"] + fields["alt_flows"] + fields["exceptions"]

        candidates: list[dict] = []
        for step_id, actor, action in all_steps:
            if not action:
                continue
            if "resumes at step" in action.lower() or "flow of events ends" in action.lower():
                continue

            doc = self._nlp_engine(action)
            effective_actor = actor or "System"
            direction = _detect_direction(action, effective_actor, doc)

            obj = ""
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("dobj", "pobj", "attr"):
                    obj = chunk.text
                    break

            is_err = _is_error_or_confirm(action)

            candidates.append({
                "step": step_id,
                "actor": effective_actor,
                "action": action[:100],
                "nlp_direction": direction.value if direction else "none",
                "nlp_object": obj,
                "is_error_hint": is_err,
            })

        condensed = f"UC: {fields['id']} — {fields['name']}\nPrimary Actor: {primary_actor}\n\nSteps:\n"
        for c in candidates:
            condensed += f"  {c['step']}. [{c['actor']}] {c['action']}"
            if c["is_error_hint"]:
                condensed += " [ERROR/CONFIRM]"
            condensed += "\n"
        condensed += "\nClassify each step as E/X/R/W/NONE. Return JSON {\"data_movements\": [...]}."

        messages = [
            {"role": "system", "content": _HYBRID_SYSTEM},
            {"role": "user", "content": condensed},
        ]

        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object", "schema": _CONDENSED_SCHEMA},
            )
            raw_text = resp["choices"][0]["message"]["content"].strip()
        except Exception:
            raw_text = json.dumps({"data_movements": []})

        data = self._parse_response(raw_text)
        movements, sub_processes = self._build_movements(data, candidates, primary_actor)

        fp = FunctionalProcess(
            name=fields["name"] or fp_id,
            triggering_event=fields["name"] or "Unknown",
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=SourceFormat.USE_CASE,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=(metadata.project if metadata and metadata.project else "Unknown"),
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )

    def _normalize_sentence(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        from .slm_normalizer import SLMNormalizer
        slm = SLMNormalizer.__new__(SLMNormalizer)
        slm._llm = self._llm
        slm.temperature = self.temperature
        slm.max_tokens = self.max_tokens
        slm.model_path = ""
        return slm.normalize(text, SourceFormat.SENTENCE, fp_id=fp_id, metadata=metadata)

    def _parse_response(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"data_movements": []}

    def _build_movements(
        self, data: dict, candidates: list[dict], primary_actor: str
    ) -> tuple[list[DataMovement], list[SubProcess]]:
        movements: list[DataMovement] = []
        sub_processes: list[SubProcess] = []

        actor_map = {c["step"]: c["actor"] for c in candidates}

        for dm_data in data.get("data_movements", []):
            mt_str = dm_data.get("type", "NONE")
            if mt_str == "NONE":
                continue
            try:
                mt = MovementType(mt_str)
            except ValueError:
                continue

            step_id = dm_data.get("step", "?")
            actor = actor_map.get(step_id, primary_actor)
            fu = actor if mt in (MovementType.E, MovementType.X) else None

            dm = DataMovement(
                type=mt,
                data_group=dm_data.get("data_group", "unknown"),
                object_of_interest=dm_data.get("object_of_interest", "unknown"),
                functional_user=fu,
                is_triggering_entry=dm_data.get("is_triggering_entry", False),
                is_error_confirmation=dm_data.get("is_error_confirmation", False),
            )
            movements.append(dm)
            sub_processes.append(SubProcess(
                step_id=step_id,
                actor=actor,
                action_text=dm_data.get("data_group", ""),
                data_movement=dm,
            ))

        return movements, sub_processes
