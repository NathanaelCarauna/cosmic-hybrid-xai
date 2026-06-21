"""SLM Normalizer — local Small Language Model via llama-cpp-python.

Experiments E2/E3/E4: Uses Qwen2.5-7B, Qwen2.5-3B, or Phi-4-mini
to extract COSMIC data movements from requirement text.

Produces IR-Full output (M1 classifies E/X/R/W directly).
Uses JSON grammar-constrained decoding for structured output.

Cost: $0 (local CPU/GPU inference).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

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

logger = logging.getLogger(__name__)

_FP_SCHEMA: dict[str, Any] = {
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
                            "required": ["type", "data_group", "object_of_interest"],
                        },
                    },
                },
                "required": ["name", "data_movements"],
            },
        },
    },
    "required": ["functional_processes"],
}

_SYSTEM_PROMPT = """You are a COSMIC ISO 19761 functional size measurement expert.

Given a software requirement (use case, user story, or sentence), extract ALL data movements as structured JSON.

COSMIC movement types:
- E (Entry): Data moving FROM a functional user INTO the software boundary (user inputs, selections with domain data)
- X (Exit): Data moving FROM the software boundary TO a functional user (displays, lists, messages, reports)
- R (Read): Data moving FROM persistent storage INTO the software (database queries, file reads)
- W (Write): Data moving FROM the software TO persistent storage (saves, updates, deletes)

Rules:
1. Every process needs at least 1 triggering Entry (is_triggering_entry=true)
2. ALL user-facing messages (confirmation, error, success, warning) to the SAME functional user = 1 single Exit with is_error_confirmation=true. Never create separate Exits for confirmation and error messages.
3. UI navigation (clicking buttons/links without domain data) = NO movement
4. "Resumes at step N" or "flow ends" = NO movement
5. Each DISTINCT data group (file/table/entity) = SEPARATE movement. Do NOT merge reads from different stores.
6. "searches rooms" + "checks commission file" + "reads booked dates" = 3 separate R movements
7. "saves header" + "saves booked rooms" + "saves services" = 3 separate W movements
8. "displays rooms" + "displays services" = 2 separate X movements (different data groups)
9. "removes old records" + "stores new records" for different entities = separate W movements each
10. Functional_user is required for E and X movements (the human/system actor)
11. Count EVERY distinct persistent storage access — do not summarize multiple reads/writes into one
12. If you Read N distinct data groups to display them, you need N separate Exits (one per data group displayed). Data display Exits are NOT messages.
13. User selecting/choosing items from DIFFERENT data groups = separate Entry per data group
14. Internal validation/verification is NOT a Read unless it accesses a DISTINCT persistent data store
15. Delete operations require a Write movement (the deletion itself is a Write to persistent storage)"""

# Directory containing versioned few-shot examples (v2-clean — synthetic, no dataset overlap)
_FEWSHOTS_DIR = Path(__file__).parent / "fewshots" / "v2-clean"


def _load_fewshots() -> str:
    """Load all example_*.txt files from the fewshots directory, sorted by name."""
    examples = sorted(_FEWSHOTS_DIR.glob("example_*.txt"))
    if not examples:
        raise FileNotFoundError(
            f"No few-shot examples found in {_FEWSHOTS_DIR}. "
            "Expected files matching example_*.txt"
        )
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in examples)


class SLMNormalizer(BaseNormalizer):
    """SLM-based normalizer producing IR-Full output."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        chain_of_thought: bool = False,
        seed: int = 42,
    ):
        from llama_cpp import Llama

        self.model_path = str(model_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chain_of_thought = chain_of_thought
        self.seed = seed

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Load once at init to fail fast if files are missing
        self._fewshot_text = _load_fewshots()

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

        t0 = time.time()

        if self.chain_of_thought:
            data = self._normalize_cot(messages)
        else:
            data = self._normalize_direct(messages)

        elapsed = time.time() - t0

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
            metadata=metadata or Metadata(
                source_file=fp_id,
                domain=None,
                project=None,
            ),
        )

    def _call_llm_json(self, messages: list[dict]) -> str:
        """Call the LLM with retry on json parse failure. Returns raw text."""
        max_retries = 2
        backoff = [1, 4]

        for attempt in range(max_retries + 1):
            try:
                resp = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=self.seed,
                    response_format={"type": "json_object", "schema": _FP_SCHEMA},
                )
                raw_text = resp["choices"][0]["message"]["content"].strip()
                # Validate parseable before returning
                json.loads(raw_text)
                return raw_text
            except json.JSONDecodeError as exc:
                snippet = raw_text[:200] if "raw_text" in dir() else "<no output>"
                logger.warning(
                    "LLM JSON parse failure (attempt %d/%d): %s — output snippet: %r",
                    attempt + 1, max_retries + 1, exc, snippet,
                )
                if attempt < max_retries:
                    time.sleep(backoff[attempt])
            except Exception as exc:
                logger.warning("LLM call error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
                if attempt < max_retries:
                    time.sleep(backoff[attempt])

        return json.dumps({"functional_processes": []})

    def _normalize_direct(self, messages: list[dict]) -> dict:
        raw_text = self._call_llm_json(messages)
        return self._parse_response(raw_text)

    def _normalize_cot(self, messages: list[dict]) -> dict:
        """Two-pass chain-of-thought: reasoning first, then structured JSON."""
        cot_messages = messages.copy()
        cot_messages[-1] = {
            "role": "user",
            "content": messages[-1]["content"] + (
                "\n\nThink step by step:\n"
                "1. Identify the triggering event and primary actor\n"
                "2. For each step, determine if data crosses the boundary\n"
                "3. Classify each data movement as E/X/R/W\n"
                "4. Identify distinct data groups and objects of interest\n"
                "5. Mark error/confirmation exits\n"
                "6. Count the total CFP\n\n"
                "Reasoning:"
            ),
        }
        try:
            reasoning_resp = self._llm.create_chat_completion(
                messages=cot_messages,
                max_tokens=512,
                temperature=self.temperature,
                seed=self.seed,
            )
            reasoning = reasoning_resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("CoT reasoning call failed: %s", exc)
            reasoning = ""

        json_messages = cot_messages + [
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "Now output the data movements as JSON:"},
        ]
        raw_text = self._call_llm_json(json_messages)
        return self._parse_response(raw_text)

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
            prompt = f"Analyze this Use Case and extract all COSMIC data movements.\n\n"
            prompt += f"UC: {fields['id']} {fields['name']}\n"
            prompt += f"Description: {fields['description']}\n"
            prompt += f"Primary actor: {fields['primary_actor']}\n"
            prompt += f"\nMain Scenario:\n{steps_text}\n"
            if alt_text:
                prompt += f"\nAlternative Flows:\n{alt_text}\n"
            if exc_text:
                prompt += f"\nExceptions:\n{exc_text}\n"
            prompt += f"\nOutput the COSMIC data movements as JSON:"
        elif source_format == SourceFormat.UC_NAME:
            prompt = f"Given only this Use Case name, estimate the COSMIC data movements:\n"
            prompt += f"UC name: {text}\n"
            prompt += f"\nOutput the COSMIC data movements as JSON:"
        else:
            prompt = f"Analyze this requirement sentence and extract COSMIC data movements:\n"
            prompt += f"Sentence: {text}\n"
            prompt += f"\nOutput the COSMIC data movements as JSON:"
        return prompt

    def _parse_response(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"functional_processes": []}

    def _build_processes(
        self, data: dict, text: str, source_format: SourceFormat
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
