"""Base normalizer interface for Module 1.

All normalizers accept raw requirement text and produce a
``NormalizedRequirement`` instance consumable by Module 2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..schema import (
    BoundaryContext,
    Metadata,
    NormalizedRequirement,
    SourceFormat,
)


class BaseNormalizer(ABC):
    """Abstract base for all normalizers."""

    @abstractmethod
    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        ...

    # ------------------------------------------------------------------
    # UC text parser (shared across normalizers)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_uc_fields(text: str) -> dict:
        """Extract structured fields from a COSMET-format UC text file.

        Returns dict with keys: id, name, description, primary_actor,
        supporting_actors, steps (list of (step_id, actor, action)),
        exceptions (list of (step_id, actor, action)).
        """
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        result: dict = {
            "id": "",
            "name": "",
            "description": "",
            "primary_actor": "",
            "supporting_actors": "",
            "steps": [],
            "exceptions": [],
            "alt_flows": [],
        }

        section = "header"
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("ID") and i == 0:
                i += 1
                continue

            if "\t" in line and section == "header":
                key, _, val = line.partition("\t")
                key = key.strip().lower()
                val = val.strip()
                if "description" in key:
                    result["description"] = val
                elif "primary actor" in key:
                    result["primary_actor"] = val
                elif "supporting" in key:
                    result["supporting_actors"] = val
                elif key.startswith("uc-") or (not result["name"] and key not in (
                    "entry condition", "exit condition", "on success",
                    "on failure", "priority", "extension points",
                    "generalization of",
                )):
                    if not any(skip in key for skip in ("extension", "priority", "entry cond", "exit cond", "on success", "on failure", "generalization")):
                        result["id"] = key
                        result["name"] = val

            if line.upper().startswith("MAIN SCENARIO"):
                section = "main"
                i += 1
                continue

            if line.lower().startswith("alternative flow"):
                section = "alt"
                i += 1
                continue

            if line.lower().startswith("exception"):
                section = "exception"
                i += 1
                continue

            if section in ("main", "alt", "exception") and "\t" in line:
                parts = line.split("\t", 2)
                if len(parts) >= 3:
                    step_id = parts[0].strip()
                    actor = parts[1].strip()
                    action = parts[2].strip()
                elif len(parts) == 2:
                    step_id = parts[0].strip()
                    actor = ""
                    action = parts[1].strip()
                else:
                    i += 1
                    continue

                entry = (step_id, actor, action)
                if section == "main":
                    result["steps"].append(entry)
                elif section == "alt":
                    result["alt_flows"].append(entry)
                elif section == "exception":
                    result["exceptions"].append(entry)

            i += 1

        if not result["name"] and len(lines) > 1:
            second = lines[1]
            if "\t" in second:
                parts = second.split("\t", 1)
                result["id"] = parts[0].strip()
                result["name"] = parts[1].strip() if len(parts) > 1 else ""

        return result

    @staticmethod
    def detect_source_format(text: str) -> SourceFormat:
        """Heuristic detection of the source format from raw text."""
        text_lower = text.lower()
        if "main scenario" in text_lower or "primary actor" in text_lower:
            return SourceFormat.USE_CASE
        if text_lower.startswith("as a ") or "i want to" in text_lower:
            return SourceFormat.USER_STORY
        if len(text.split()) <= 5:
            return SourceFormat.UC_NAME
        return SourceFormat.SENTENCE

    @staticmethod
    def load_fp_text(path: Path) -> str:
        """Read an FP text file."""
        return path.read_text(encoding="utf-8", errors="replace")
