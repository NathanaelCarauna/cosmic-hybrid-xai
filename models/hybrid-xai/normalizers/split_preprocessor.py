"""M0 Atomic Split Preprocessor — COSMET-inspired sentence decomposition.

Rewrites compound UC steps into atomic, single-action numbered sentences
before passing to M1. Addresses paragraph-collapse failures where the SLM
maps dense multi-action steps (e.g. "updates header, removes rooms, stores
services") into 1-2 movements instead of the 6-7 distinct ones.

Works with both local SLM (llama-cpp) and OpenAI API backends.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .base import BaseNormalizer
from ..schema import Metadata, NormalizedRequirement, SourceFormat

_SPLIT_SYSTEM_PROMPT = """\
You are a sentence-splitting expert for COSMIC functional size measurement.

Your ONLY task: split compound sentences so that each output sentence describes exactly ONE action on ONE data entity. Keep all section headers (MAIN SCENARIO, Alternative flow, Exceptions) and renumber the sentences sequentially.

Rules:
- "The system stores A, saves B, and updates C" → 3 separate sentences (one per persistent action)
- "The system searches X and lists Y" → 2 sentences (one search/read, one display)
- "The system validates A, B, and C" → keep as ONE sentence (internal validation, not separate stores)
- "The system reads room details and searches for requestable services" → 2 sentences
- "The system removes old X, stores new Y, and updates Z" → 3 sentences (each a distinct persistent write)
- Keep alternative flows and exceptions as-is (just renumber if needed)
- Do NOT add new actions. Only split existing compound sentences.
- Do NOT remove or change the meaning of any sentence.
- Output ONLY the split text. No explanations."""

_SPLIT_EXAMPLE_1 = """Example 1:

Input:
MAIN SCENARIO:
1. The system searches for the season configuration in the given interval. The system stores the reservation header. In addition, the system stores the Booked Rooms, for each room saves the Requested Services, and stores the Children's Ages.
2. The system searches for and lists the data.

Split:
MAIN SCENARIO:
1. The system searches for the season configuration in the given interval.
2. The system stores the reservation header.
3. The system stores the Booked Rooms.
4. For each room, the system saves the Requested Services.
5. The system stores the Children's Ages.
6. The system searches for the data.
7. The system lists the data."""

_SPLIT_EXAMPLE_2 = """Example 2:

Input:
MAIN SCENARIO:
1. The system updates the customer's details if needed. The system removes the previously Requested Services, the previously Booked Rooms, the previous Children's Ages, and the Reservation header for the previous reservation.
2. The system reads room details and searches for the requestable room services for the selected rooms.
3. The system lists the selected rooms. For each room, the system lists the requestable services.

Split:
MAIN SCENARIO:
1. The system updates the customer's details if needed.
2. The system removes the previously Requested Services.
3. The system removes the previously Booked Rooms.
4. The system removes the previous Children's Ages.
5. The system removes the Reservation header for the previous reservation.
6. The system reads the room details for the selected rooms.
7. The system searches for the requestable room services for the selected rooms.
8. The system lists the selected rooms.
9. For each room, the system lists the requestable services."""

_SPLIT_EXAMPLE_3 = """Example 3:

Input:
MAIN SCENARIO:
1. C-Reg validates the data to ensure the proper formats and checks whether a Professor of that ID already exists. If the entered data are valid, C-Reg creates a record for the new Professor.
2. The system checks that the surname and name fields contain at least two characters, the fiscal code field is valid, the date of birth is correct, the email address is valid and records the new customer in persistent storage.

Split:
MAIN SCENARIO:
1. C-Reg validates the data to ensure proper formats.
2. C-Reg checks whether a Professor of that ID already exists.
3. If the entered data are valid, C-Reg creates a record for the new Professor.
4. The system checks that the surname and name fields contain at least two characters.
5. The system checks that the fiscal code field is valid.
6. The system checks that the date of birth is correct.
7. The system checks that the email address is valid.
8. The system records the new customer in persistent storage."""


class SplitPreprocessor:
    """Rewrites compound UC steps into atomic sentences via an LLM call.

    Supports two backends:
    - ``backend="local"``: uses an existing ``llama_cpp.Llama`` instance
    - ``backend="openai"``: uses the OpenAI Chat Completions API
    """

    def __init__(
        self,
        *,
        backend: str = "local",
        llm=None,
        openai_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens

        if backend == "local":
            if llm is None:
                raise ValueError("llm (Llama instance) required for local backend")
            self._llm = llm
        elif backend == "openai":
            import openai
            import os
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = openai.OpenAI(api_key=key)
            self._openai_model = openai_model
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def split(self, text: str, source_format: SourceFormat) -> str:
        """Return the atomized UC text, or the original if not a use case."""
        if source_format != SourceFormat.USE_CASE:
            return text

        system_steps = self._extract_system_steps(text)
        if not self._has_compound_steps(system_steps):
            return text

        messages = [
            {"role": "system", "content": _SPLIT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    _SPLIT_EXAMPLE_1 + "\n\n###\n\n"
                    + _SPLIT_EXAMPLE_2 + "\n\n###\n\n"
                    + _SPLIT_EXAMPLE_3
                ),
            },
            {
                "role": "assistant",
                "content": "I understand. I will split compound sentences into atomic single-action lines. Send me the text to split.",
            },
            {
                "role": "user",
                "content": f"Input:\n{text}\n\nSplit:",
            },
        ]

        split_text = self._call_llm(messages)
        if not split_text or len(split_text.strip()) < 20:
            return text
        return self._inject_split_steps(text, split_text)

    def _call_llm(self, messages: list[dict]) -> str:
        if self.backend == "local":
            return self._call_local(messages)
        return self._call_openai(messages)

    def _call_local(self, messages: list[dict]) -> str:
        try:
            msgs = [m.copy() for m in messages]
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += " /no_think"
            resp = self._llm.create_chat_completion(
                messages=msgs,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw = resp["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
            return raw
        except Exception:
            return ""

    def _call_openai(self, messages: list[dict]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._openai_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            import sys
            print(f"[SplitPreprocessor] API error: {exc}", file=sys.stderr)
            return ""

    @staticmethod
    def _extract_system_steps(text: str) -> list[str]:
        """Pull lines that belong to system actors from the UC."""
        steps = []
        for line in text.splitlines():
            stripped = line.strip()
            parts = stripped.split("\t")
            if len(parts) >= 3 and parts[1].strip().lower() == "system":
                steps.append(parts[2].strip())
        return steps

    @staticmethod
    def _has_compound_steps(steps: list[str]) -> bool:
        """Heuristic: does any system step contain compound actions?"""
        compound_markers = [
            ". The system ",
            ". In addition,",
            ", the system ",
            " and stores ",
            " and saves ",
            " and lists ",
            " and searches ",
            " removes the previously",
            " stores the ",
        ]
        for step in steps:
            if len(step) > 120:
                return True
            for marker in compound_markers:
                if marker.lower() in step.lower():
                    return True
        return False

    @staticmethod
    def _inject_split_steps(original: str, split_text: str) -> str:
        """Replace the body of the original UC with the split version,
        preserving the header (ID, description, primary actor, etc.)."""
        header_lines = []
        body_started = False
        for line in original.splitlines():
            if line.strip().upper().startswith("MAIN SCENARIO"):
                body_started = True
                break
            header_lines.append(line)

        if not body_started:
            return original

        clean_split = split_text.strip()
        if not clean_split.upper().startswith("MAIN SCENARIO"):
            clean_split = "MAIN SCENARIO\n" + clean_split

        return "\n".join(header_lines) + "\n" + clean_split
