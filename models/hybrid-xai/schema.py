"""COSMIC Normalized Requirement Schema.

Defines the Intermediate Representation (IR) contract between
Module 1 (Normalization) and Module 2 (Rule Engine).
Based on: docs/estimativa-cosmic-tese/modelo-normalizacao-semantica.md
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceFormat(str, Enum):
    USE_CASE = "use_case"
    USER_STORY = "user_story"
    SRS = "srs"
    SENTENCE = "sentence"
    UC_NAME = "uc_name"


class MovementType(str, Enum):
    E = "E"
    X = "X"
    R = "R"
    W = "W"


class Direction(str, Enum):
    USER_TO_SYSTEM = "user_to_system"
    SYSTEM_TO_USER = "system_to_user"
    SYSTEM_TO_STORAGE = "system_to_storage"
    STORAGE_TO_SYSTEM = "storage_to_system"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DataMovement:
    type: MovementType
    data_group: str
    object_of_interest: str
    functional_user: Optional[str] = None
    is_triggering_entry: bool = False
    is_error_confirmation: bool = False


@dataclass
class SubProcess:
    step_id: str
    actor: str
    action_text: str
    direction: Optional[Direction] = None
    data_movement: Optional[DataMovement] = None


@dataclass
class FunctionalProcess:
    name: str
    triggering_event: str
    primary_functional_user: str
    sub_processes: list[SubProcess] = field(default_factory=list)
    data_movements: list[DataMovement] = field(default_factory=list)

    @property
    def cfp(self) -> int:
        return len(self.data_movements)

    def validate(
        self, source_format: SourceFormat = SourceFormat.USE_CASE
    ) -> list[str]:
        """Return constraint violations (empty list = valid)."""
        errors: list[str] = []

        allow_single = source_format in (
            SourceFormat.SENTENCE,
            SourceFormat.UC_NAME,
        )
        if not allow_single and self.cfp < 2:
            errors.append(
                f"Min 2 movements for {source_format.value}, got {self.cfp}"
            )

        triggers = [m for m in self.data_movements if m.is_triggering_entry]
        if len(triggers) != 1:
            errors.append(
                f"Expected 1 triggering entry, found {len(triggers)}"
            )

        err_exits = [m for m in self.data_movements if m.is_error_confirmation]
        if len(err_exits) > 1:
            errors.append(
                f"Max 1 error/confirmation Exit, found {len(err_exits)}"
            )

        for m in self.data_movements:
            if (
                m.type in (MovementType.E, MovementType.X)
                and not m.functional_user
            ):
                errors.append(
                    f"E/X movement missing functional_user: {m.data_group}"
                )

        return errors


@dataclass
class BoundaryContext:
    software_name: str
    functional_users: list[str] = field(default_factory=list)
    persistent_stores: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    source_file: str = ""
    domain: Optional[str] = None
    project: Optional[str] = None


@dataclass
class NormalizedRequirement:
    id: str
    source_format: SourceFormat
    raw_text: str
    boundary: BoundaryContext
    functional_processes: list[FunctionalProcess] = field(default_factory=list)
    metadata: Metadata = field(default_factory=Metadata)

    @property
    def total_cfp(self) -> int:
        return sum(fp.cfp for fp in self.functional_processes)

    def validate_all(self) -> dict[str, list[str]]:
        """Validate every process; returns {name: [errors]}."""
        result: dict[str, list[str]] = {}
        for fp in self.functional_processes:
            errs = fp.validate(self.source_format)
            if errs:
                result[fp.name] = errs
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (JSON-ready)."""
        return _serialise(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedRequirement:
        return cls(
            id=d["id"],
            source_format=SourceFormat(d["source_format"]),
            raw_text=d["raw_text"],
            boundary=BoundaryContext(**d.get("boundary", {})),
            functional_processes=[
                _fp_from_dict(fp) for fp in d.get("functional_processes", [])
            ],
            metadata=Metadata(**d.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> NormalizedRequirement:
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# JSON Schema export (for grammar-constrained SLM decoding)
# ---------------------------------------------------------------------------

def json_schema() -> dict[str, Any]:
    """Return a JSON Schema describing NormalizedRequirement."""
    movement_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": [e.value for e in MovementType]},
            "data_group": {"type": "string"},
            "object_of_interest": {"type": "string"},
            "functional_user": {"type": ["string", "null"]},
            "is_triggering_entry": {"type": "boolean"},
            "is_error_confirmation": {"type": "boolean"},
        },
        "required": ["type", "data_group", "object_of_interest"],
    }
    subprocess_schema = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string"},
            "actor": {"type": "string"},
            "action_text": {"type": "string"},
            "direction": {
                "type": ["string", "null"],
                "enum": [e.value for e in Direction] + [None],
            },
            "data_movement": {
                "oneOf": [movement_schema, {"type": "null"}],
            },
        },
        "required": ["step_id", "actor", "action_text"],
    }
    fp_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "triggering_event": {"type": "string"},
            "primary_functional_user": {"type": "string"},
            "sub_processes": {"type": "array", "items": subprocess_schema},
            "data_movements": {"type": "array", "items": movement_schema},
            "cfp": {"type": "integer"},
        },
        "required": [
            "name",
            "triggering_event",
            "primary_functional_user",
            "sub_processes",
            "data_movements",
            "cfp",
        ],
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "NormalizedRequirement",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "source_format": {
                "type": "string",
                "enum": [e.value for e in SourceFormat],
            },
            "raw_text": {"type": "string"},
            "boundary": {
                "type": "object",
                "properties": {
                    "software_name": {"type": "string"},
                    "functional_users": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "persistent_stores": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["software_name"],
            },
            "functional_processes": {"type": "array", "items": fp_schema},
            "metadata": {
                "type": "object",
                "properties": {
                    "source_file": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "project": {"type": ["string", "null"]},
                },
            },
        },
        "required": [
            "id",
            "source_format",
            "raw_text",
            "boundary",
            "functional_processes",
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert Enum values to their .value for JSON."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


def _dm_from_dict(d: dict[str, Any] | None) -> DataMovement | None:
    if d is None:
        return None
    return DataMovement(
        type=MovementType(d["type"]),
        data_group=d["data_group"],
        object_of_interest=d["object_of_interest"],
        functional_user=d.get("functional_user"),
        is_triggering_entry=d.get("is_triggering_entry", False),
        is_error_confirmation=d.get("is_error_confirmation", False),
    )


def _sp_from_dict(d: dict[str, Any]) -> SubProcess:
    return SubProcess(
        step_id=d["step_id"],
        actor=d["actor"],
        action_text=d["action_text"],
        direction=Direction(d["direction"]) if d.get("direction") else None,
        data_movement=_dm_from_dict(d.get("data_movement")),
    )


def _fp_from_dict(d: dict[str, Any]) -> FunctionalProcess:
    return FunctionalProcess(
        name=d["name"],
        triggering_event=d["triggering_event"],
        primary_functional_user=d["primary_functional_user"],
        sub_processes=[_sp_from_dict(sp) for sp in d.get("sub_processes", [])],
        data_movements=[
            _dm_from_dict(m) for m in d.get("data_movements", [])  # type: ignore[arg-type]
        ],
    )
