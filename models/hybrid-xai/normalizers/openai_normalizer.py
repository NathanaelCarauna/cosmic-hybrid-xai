"""OpenAI API Normalizer — uses GPT models with the same prompts/schema as SLMNormalizer.

Drop-in replacement for SLMNormalizer that sends requests to the OpenAI Chat
Completions API instead of local llama.cpp inference.  Shares the exact same
system prompt, few-shot examples, JSON schema, prompt builder, and response
parser so that results are directly comparable.

Usage:
    from .openai_normalizer import OpenAINormalizer
    norm = OpenAINormalizer("gpt-4o-mini")
"""
from __future__ import annotations

import json
import os
import time
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
from .slm_normalizer import (
    _FP_SCHEMA,
    _SYSTEM_PROMPT,
    _load_fewshots,
)


_STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "functional_processes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "data_movements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["E", "X", "R", "W"]},
                                "data_group": {"type": "string"},
                                "object_of_interest": {"type": "string"},
                                "functional_user": {"type": ["string", "null"]},
                                "is_triggering_entry": {"type": "boolean"},
                                "is_error_confirmation": {"type": "boolean"},
                            },
                            "required": [
                                "type", "data_group", "object_of_interest",
                                "functional_user", "is_triggering_entry", "is_error_confirmation",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "data_movements"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["functional_processes"],
    "additionalProperties": False,
}


class OpenAINormalizer(BaseNormalizer):
    """OpenAI API-backed normalizer producing IR-Full output.

    Reuses the prompt templates and JSON schema from SLMNormalizer so
    that the only variable is the LLM backend (cloud API vs local GGUF).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key: str | None = None,
        use_strict_schema: bool = False,
    ):
        import openai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_strict_schema = use_strict_schema

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = openai.OpenAI(api_key=key)
        self._fewshot_text = _load_fewshots()

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        user_prompt = self._build_prompt(text, source_format)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._fewshot_text},
            {"role": "assistant", "content": "I understand the COSMIC measurement rules and examples. Send me the requirement to analyze."},
            {"role": "user", "content": user_prompt},
        ]

        data = self._call_api(messages)
        functional_processes = self._build_processes(data, text, source_format)

        primary_actor = "User"
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            primary_actor = fields.get("primary_actor") or "User"

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=(metadata.project if metadata and metadata.project else "Unknown"),
                functional_users=[primary_actor],
            ),
            functional_processes=functional_processes,
            metadata=metadata or Metadata(source_file=fp_id),
        )

    def _call_api(self, messages: list[dict]) -> dict:
        try:
            rf = self._response_format()
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=rf,
            )
            raw = resp.choices[0].message.content.strip()

            if resp.usage:
                self._total_input_tokens += resp.usage.prompt_tokens
                self._total_output_tokens += resp.usage.completion_tokens
                self._update_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        except Exception as exc:
            import sys
            print(f"[OpenAINormalizer] API error: {exc}", file=sys.stderr)
            raw = json.dumps({"functional_processes": []})

        return self._parse_response(raw)

    def _response_format(self) -> dict:
        if not self.use_strict_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "cosmic_movements",
                "strict": True,
                "schema": _STRICT_SCHEMA,
            },
        }

    def _update_cost(self, inp: int, out: int) -> None:
        pricing = {
            "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
            "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
            "gpt-4-turbo": (10.0 / 1_000_000, 30.0 / 1_000_000),
        }
        rate = pricing.get(self.model, (0.15 / 1_000_000, 0.60 / 1_000_000))
        self._total_cost_usd += inp * rate[0] + out * rate[1]

    def get_usage_summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
        }

    def _build_prompt(self, text: str, source_format: SourceFormat) -> str:
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            steps_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["steps"]
            )
            alt_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["alt_flows"]
            )
            exc_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["exceptions"]
            )
            prompt = "Analyze this Use Case and extract all COSMIC data movements.\n\n"
            prompt += f"UC: {fields['id']} {fields['name']}\n"
            prompt += f"Description: {fields['description']}\n"
            prompt += f"Primary actor: {fields['primary_actor']}\n"
            prompt += f"\nMain Scenario:\n{steps_text}\n"
            if alt_text:
                prompt += f"\nAlternative Flows:\n{alt_text}\n"
            if exc_text:
                prompt += f"\nExceptions:\n{exc_text}\n"
            prompt += "\nOutput the COSMIC data movements as JSON:"
        elif source_format == SourceFormat.UC_NAME:
            prompt = f"Given only this Use Case name, estimate the COSMIC data movements:\nUC name: {text}\n\nOutput the COSMIC data movements as JSON:"
        else:
            prompt = f"Analyze this requirement sentence and extract COSMIC data movements:\nSentence: {text}\n\nOutput the COSMIC data movements as JSON:"
        return prompt

    @staticmethod
    def _parse_response(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"functional_processes": []}

    @staticmethod
    def _build_processes(
        data: dict, text: str, source_format: SourceFormat
    ) -> list[FunctionalProcess]:
        processes: list[FunctionalProcess] = []

        for fp_data in data.get("functional_processes", []):
            movements: list[DataMovement] = []
            sub_processes: list[SubProcess] = []

            for i, dm_data in enumerate(fp_data.get("data_movements", [])):
                try:
                    mt = MovementType(dm_data["type"])
                except (KeyError, ValueError):
                    continue

                dm = DataMovement(
                    type=mt,
                    data_group=dm_data.get("data_group", "unknown"),
                    object_of_interest=dm_data.get("object_of_interest", "unknown"),
                    functional_user=dm_data.get("functional_user"),
                    is_triggering_entry=dm_data.get("is_triggering_entry", False),
                    is_error_confirmation=dm_data.get("is_error_confirmation", False),
                )
                movements.append(dm)

                sp = SubProcess(
                    step_id=str(i + 1),
                    actor=dm_data.get("functional_user") or "System",
                    action_text=dm_data.get("data_group", ""),
                    data_movement=dm,
                )
                sub_processes.append(sp)

            fp = FunctionalProcess(
                name=fp_data.get("name", "Unknown"),
                triggering_event=fp_data.get("name", "Unknown"),
                primary_functional_user="User",
                sub_processes=sub_processes,
                data_movements=movements,
            )
            processes.append(fp)

        if not processes:
            processes.append(
                FunctionalProcess(
                    name="Unknown",
                    triggering_event="Unknown",
                    primary_functional_user="User",
                )
            )

        return processes
