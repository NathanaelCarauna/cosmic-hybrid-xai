"""E7: Step-wise SLM Normalizer — classifies each UC step individually.

Instead of asking the SLM to extract ALL data movements at once (which
causes under-counting on large UCs), this normalizer feeds each step
separately. This mirrors what COSMET does with GPT-4-turbo.

Addresses the core weakness: large UCs (9+ CFP) where the SLM "summarizes".
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..schema import (
    BoundaryContext,
    DataMovement,
    Direction,
    FunctionalProcess,
    Metadata,
    MovementType,
    NormalizedRequirement,
    SourceFormat,
    SubProcess,
)
from .base import BaseNormalizer

_STEP_SYSTEM = """You are a COSMIC ISO 19761 measurement expert.
I will give you ONE step from a Use Case. Classify it as a COSMIC data movement.

Types:
- E (Entry): user sends data TO the system (fills form, enters data, submits with domain content)
- X (Exit): system sends data TO the user (displays list, shows details, shows message)
- R (Read): system reads FROM persistent storage (searches database, retrieves records, checks file)
- W (Write): system writes TO persistent storage (saves, updates, deletes, stores)
- NONE: UI navigation, clicking buttons without data, showing empty forms, "resumes at step N"

IMPORTANT: If a step mentions MULTIPLE DISTINCT data stores (e.g., "reads rooms AND checks commission file AND reads booked dates"), output MULTIPLE movements, one per data store.

Output JSON: {"movements": [{"type": "E|X|R|W|NONE", "data_group": "...", "object": "..."}]}
For NONE: {"movements": [{"type": "NONE", "data_group": "", "object": ""}]}"""

_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "movements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["E", "X", "R", "W", "NONE"]},
                    "data_group": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["type", "data_group", "object"],
            },
        }
    },
    "required": ["movements"],
}

_STEP_EXAMPLES = """Examples:
Step: "User fills in room code filter and searches" → {"movements": [{"type": "E", "data_group": "Room code filter", "object": "Room"}]}
Step: "System searches for rooms" → {"movements": [{"type": "R", "data_group": "Room records", "object": "Room"}]}
Step: "System displays room list" → {"movements": [{"type": "X", "data_group": "Room list", "object": "Room"}]}
Step: "System saves reservation header and stores booked rooms" → {"movements": [{"type": "W", "data_group": "Reservation header", "object": "Reservation"}, {"type": "W", "data_group": "Booked rooms", "object": "Booked room"}]}
Step: "User clicks Search button" → {"movements": [{"type": "NONE", "data_group": "", "object": ""}]}
Step: "System shows error message" → {"movements": [{"type": "X", "data_group": "Error message", "object": "Error"}]}"""


class StepwiseNormalizer(BaseNormalizer):
    """Classifies each UC step individually via SLM."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
    ):
        from llama_cpp import Llama

        self.model_path = str(model_path)
        self.temperature = temperature
        self._llm = Llama(
            model_path=self.model_path,
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
        return self._normalize_sentence(text, fp_id, metadata, source_format)

    def _classify_step(self, step_text: str, actor: str, uc_context: str) -> list[dict]:
        prompt = f"UC context: {uc_context}\nStep: [{actor}] {step_text}\n\nClassify this step:"
        messages = [
            {"role": "system", "content": _STEP_SYSTEM},
            {"role": "user", "content": _STEP_EXAMPLES},
            {"role": "assistant", "content": "Understood. Send me a step to classify."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=self.temperature,
                response_format={"type": "json_object", "schema": _STEP_SCHEMA},
            )
            raw = resp["choices"][0]["message"]["content"].strip()
            data = json.loads(raw)
            return data.get("movements", [])
        except Exception:
            return [{"type": "NONE", "data_group": "", "object": ""}]

    def _normalize_uc(self, text: str, fp_id: str, metadata) -> NormalizedRequirement:
        source_format = SourceFormat.USE_CASE
        fields = self.parse_uc_fields(text)
        uc_context = f"{fields['id']} {fields['name']}"
        primary_actor = fields.get("primary_actor") or "User"

        all_steps = fields["steps"] + fields["alt_flows"] + fields["exceptions"]

        movements: list[DataMovement] = []
        sub_processes: list[SubProcess] = []
        has_trigger = False
        has_error_exit = False

        for step_id, actor, action in all_steps:
            classified = self._classify_step(action, actor or primary_actor, uc_context)
            for mv in classified:
                mt_str = mv.get("type", "NONE")
                if mt_str == "NONE":
                    continue
                try:
                    mt = MovementType(mt_str)
                except ValueError:
                    continue

                is_trigger = (mt == MovementType.E and not has_trigger)
                is_error = (mt == MovementType.X and
                            any(kw in mv.get("data_group", "").lower()
                                for kw in ("error", "confirm", "warning")))

                if is_error and has_error_exit:
                    continue
                if is_trigger:
                    has_trigger = True
                if is_error:
                    has_error_exit = True

                dm = DataMovement(
                    type=mt,
                    data_group=mv.get("data_group", "unknown"),
                    object_of_interest=mv.get("object", "unknown"),
                    functional_user=actor if mt in (MovementType.E, MovementType.X) else None,
                    is_triggering_entry=is_trigger,
                    is_error_confirmation=is_error,
                )
                movements.append(dm)
                sub_processes.append(SubProcess(
                    step_id=step_id,
                    actor=actor or primary_actor,
                    action_text=action,
                    data_movement=dm,
                ))

        fp = FunctionalProcess(
            name=fields["name"] or fp_id,
            triggering_event=fields["name"] or fp_id,
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=(metadata.project if metadata and metadata.project else "Unknown"),
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )

    def _normalize_sentence(self, text, fp_id, metadata, source_format):
        classified = self._classify_step(text, "User", text[:60])
        movements = []
        for mv in classified:
            mt_str = mv.get("type", "NONE")
            if mt_str == "NONE":
                continue
            try:
                mt = MovementType(mt_str)
            except ValueError:
                continue
            movements.append(DataMovement(
                type=mt,
                data_group=mv.get("data_group", "unknown"),
                object_of_interest=mv.get("object", "unknown"),
                functional_user="User" if mt in (MovementType.E, MovementType.X) else None,
                is_triggering_entry=(mt == MovementType.E and not movements),
            ))

        fp = FunctionalProcess(
            name=fp_id or "Unknown",
            triggering_event=fp_id or "Unknown",
            primary_functional_user="User",
            data_movements=movements,
        )
        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=text,
            boundary=BoundaryContext(software_name="Unknown", functional_users=["User"]),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )
