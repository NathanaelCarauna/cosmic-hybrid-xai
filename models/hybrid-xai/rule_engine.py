"""Module 2 — COSMIC Deterministic Rule Engine.

Consumes a ``NormalizedRequirement`` produced by Module 1 and applies
COSMIC ISO 19761 rules to validate, deduplicate, and count CFP.

Supports two IR modes:
  * **IR-Full** — M1 already classified data movements (E/X/R/W).
    M2 validates constraints, deduplicates, and counts.
  * **IR-Semantic** — M1 only provides ``direction`` + ``action_text``.
    M2 classifies movements using verb taxonomy, then proceeds as IR-Full.

Reference: docs/estimativa-cosmic-tese/modelo-normalizacao-semantica.md §3.2, §6
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import (
    DataMovement,
    Direction,
    FunctionalProcess,
    MovementType,
    NormalizedRequirement,
    SourceFormat,
    SubProcess,
)

# ---------------------------------------------------------------------------
# Verb taxonomy for IR-Semantic classification (§4.1 of the model)
# ---------------------------------------------------------------------------

_ENTRY_VERBS = re.compile(
    r"\b(enter|fill|input|submit|select|choose|provide|inform|type|specify|"
    r"press|insert|pick|check|mark|tick|set|upload|attach)\w*\b",
    re.IGNORECASE,
)
_EXIT_VERBS = re.compile(
    r"\b(display|show|list|present|output|render|print|return|exhibit|"
    r"generate|produce|report|view|visualize|summarize|highlight)\w*\b",
    re.IGNORECASE,
)
_READ_VERBS = re.compile(
    r"\b(search|query|fetch|retrieve|read|look\s?up|find|get|load|obtain|"
    r"verify|validate|check|seek|browse|scan|inspect|locate|filter)\w*\b",
    re.IGNORECASE,
)
_WRITE_VERBS = re.compile(
    r"\b(save|store|write|record|persist|update|modify|delete|remove|"
    r"create|insert|register|add|log|archive|overwrite|append|erase)\w*\b",
    re.IGNORECASE,
)
_CONTROL_VERBS = re.compile(
    r"\b(click|navigate|close|cancel|abort|resume|return to|go back|"
    r"open|launch|exit|log\s?out|log\s?in)\w*\b",
    re.IGNORECASE,
)

_ERROR_PATTERNS = re.compile(
    r"\b(error|fail|invalid|exception|warning|not found|unable|denied|"
    r"reject|unsuccessful|alert|could not|cannot)\b",
    re.IGNORECASE,
)
_CONFIRM_PATTERNS = re.compile(
    r"\b(confirm|success|complete|done|acknowledge|notification|message)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class RuleEngineResult:
    """Output of the rule engine for one NormalizedRequirement."""
    requirement_id: str
    total_cfp: int
    process_results: list[ProcessResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    """Output for a single FunctionalProcess."""
    process_name: str
    cfp: int
    movements: list[DataMovement] = field(default_factory=list)
    removed: list[tuple[str, DataMovement]] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------


class CosmicRuleEngine:
    """Deterministic COSMIC measurement engine."""

    def measure(self, nr: NormalizedRequirement) -> RuleEngineResult:
        """Apply full COSMIC pipeline to a NormalizedRequirement."""
        result = RuleEngineResult(requirement_id=nr.id, total_cfp=0)

        for fp in nr.functional_processes:
            pr = self._process_fp(fp, nr.source_format)
            result.process_results.append(pr)
            result.total_cfp += pr.cfp

        return result

    # ------------------------------------------------------------------
    # Per-process pipeline
    # ------------------------------------------------------------------

    def _process_fp(
        self, fp: FunctionalProcess, source_format: SourceFormat
    ) -> ProcessResult:
        pr = ProcessResult(process_name=fp.name, cfp=0)

        movements = self._collect_movements(fp, pr)

        movements = self._classify_semantic(fp, movements, pr)

        movements = self._filter_control(movements, pr)
        movements = self._collapse_error_exits(movements, pr)
        movements = self._deduplicate(movements, pr)
        movements = self._ensure_triggering_entry(movements, pr, source_format)
        movements = self._enforce_minimum(movements, pr, source_format)

        pr.movements = movements
        pr.cfp = len(movements)
        return pr

    # ------------------------------------------------------------------
    # Step 1: Collect movements from sub_processes or data_movements
    # ------------------------------------------------------------------

    def _collect_movements(
        self, fp: FunctionalProcess, pr: ProcessResult
    ) -> list[DataMovement]:
        if fp.data_movements:
            return [copy.deepcopy(m) for m in fp.data_movements]

        movements: list[DataMovement] = []
        for sp in fp.sub_processes:
            if sp.data_movement:
                movements.append(copy.deepcopy(sp.data_movement))
        return movements

    # ------------------------------------------------------------------
    # Step 2: Classify movements from direction+verb (IR-Semantic mode)
    # ------------------------------------------------------------------

    def _classify_semantic(
        self,
        fp: FunctionalProcess,
        movements: list[DataMovement],
        pr: ProcessResult,
    ) -> list[DataMovement]:
        if movements:
            return movements

        classified: list[DataMovement] = []
        for sp in fp.sub_processes:
            if sp.data_movement:
                classified.append(copy.deepcopy(sp.data_movement))
                continue

            mt = self._infer_movement_type(sp)
            if mt is None:
                continue

            is_err = bool(_ERROR_PATTERNS.search(sp.action_text))
            is_conf = bool(
                _CONFIRM_PATTERNS.search(sp.action_text) and mt == MovementType.X
            )

            dm = DataMovement(
                type=mt,
                data_group=sp.action_text[:80],
                object_of_interest=sp.action_text[:40],
                functional_user=sp.actor if mt in (MovementType.E, MovementType.X) else None,
                is_triggering_entry=False,
                is_error_confirmation=is_err or is_conf,
            )
            classified.append(dm)
            pr.corrections.append(
                f"IR-Semantic: classified step {sp.step_id} as {mt.value}"
            )

        return classified

    def _infer_movement_type(self, sp: SubProcess) -> Optional[MovementType]:
        """Infer movement type from direction and verb patterns."""
        text = sp.action_text

        if _CONTROL_VERBS.search(text) and not _ENTRY_VERBS.search(text):
            return None

        if sp.direction:
            direction_map = {
                Direction.USER_TO_SYSTEM: MovementType.E,
                Direction.SYSTEM_TO_USER: MovementType.X,
                Direction.SYSTEM_TO_STORAGE: MovementType.W,
                Direction.STORAGE_TO_SYSTEM: MovementType.R,
            }
            if sp.direction in direction_map:
                return direction_map[sp.direction]

        scores = {
            MovementType.E: len(_ENTRY_VERBS.findall(text)),
            MovementType.X: len(_EXIT_VERBS.findall(text)),
            MovementType.R: len(_READ_VERBS.findall(text)),
            MovementType.W: len(_WRITE_VERBS.findall(text)),
        }
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best

        return None

    # ------------------------------------------------------------------
    # Step 3: Filter control commands (§3.5.10)
    # ------------------------------------------------------------------

    def _filter_control(
        self, movements: list[DataMovement], pr: ProcessResult
    ) -> list[DataMovement]:
        kept: list[DataMovement] = []
        for m in movements:
            if _CONTROL_VERBS.search(m.data_group) and not any(
                p.search(m.data_group) for p in [_ENTRY_VERBS, _READ_VERBS, _WRITE_VERBS, _EXIT_VERBS]
            ):
                pr.removed.append(("control_command", m))
            else:
                kept.append(m)
        return kept

    # ------------------------------------------------------------------
    # Step 4: Collapse error/confirmation Exits into at most 1 (§3.5.11)
    # ------------------------------------------------------------------

    _MESSAGE_EXIT_RE = re.compile(
        r"\b(confirm|success|error|fail|warning|alert|message|notification)\b",
        re.IGNORECASE,
    )

    def _collapse_error_exits(
        self, movements: list[DataMovement], pr: ProcessResult
    ) -> list[DataMovement]:
        err_exits = [m for m in movements if m.is_error_confirmation]
        if len(err_exits) > 1:
            kept: list[DataMovement] = []
            first_err_seen = False
            for m in movements:
                if m.is_error_confirmation:
                    if not first_err_seen:
                        first_err_seen = True
                        kept.append(m)
                    else:
                        pr.removed.append(("collapse_error_exit", m))
                else:
                    kept.append(m)
            pr.corrections.append(
                f"Collapsed {len(err_exits)} flagged error/confirmation Exits to 1"
            )
            movements = kept

        msg_exits = [
            m for m in movements
            if m.type == MovementType.X and self._MESSAGE_EXIT_RE.search(m.data_group)
        ]
        if len(msg_exits) >= 2:
            kept = []
            first_msg_seen = False
            for m in movements:
                if m.type == MovementType.X and self._MESSAGE_EXIT_RE.search(m.data_group):
                    if not first_msg_seen:
                        first_msg_seen = True
                        m.is_error_confirmation = True
                        kept.append(m)
                    else:
                        pr.removed.append(("collapse_message_exit", m))
                else:
                    kept.append(m)
            pr.corrections.append(
                f"Collapsed {len(msg_exits)} message-type Exits to 1 (§3.5.11)"
            )
            movements = kept

        return movements

    # ------------------------------------------------------------------
    # Step 5: Deduplicate identical movements (§4.1.7)
    # ------------------------------------------------------------------

    def _deduplicate(
        self, movements: list[DataMovement], pr: ProcessResult
    ) -> list[DataMovement]:
        seen: set[tuple[str, str, str]] = set()
        kept: list[DataMovement] = []
        for m in movements:
            key = (m.type.value, m.data_group.lower().strip(), m.object_of_interest.lower().strip())
            if key in seen:
                pr.removed.append(("duplicate", m))
            else:
                seen.add(key)
                kept.append(m)
        return kept

    # ------------------------------------------------------------------
    # Step 6: Ensure exactly 1 triggering Entry
    # ------------------------------------------------------------------

    def _ensure_triggering_entry(
        self,
        movements: list[DataMovement],
        pr: ProcessResult,
        source_format: SourceFormat,
    ) -> list[DataMovement]:
        triggers = [m for m in movements if m.is_triggering_entry]
        if len(triggers) == 1:
            return movements

        entries = [m for m in movements if m.type == MovementType.E]
        if not entries and not triggers:
            if source_format in (SourceFormat.SENTENCE, SourceFormat.UC_NAME):
                return movements
            pr.corrections.append("No Entry found; cannot assign triggering entry")
            return movements

        if len(triggers) == 0 and entries:
            entries[0].is_triggering_entry = True
            pr.corrections.append(
                f"Auto-assigned triggering entry to '{entries[0].data_group}'"
            )
        elif len(triggers) > 1:
            for t in triggers[1:]:
                t.is_triggering_entry = False
            pr.corrections.append(
                f"Multiple triggering entries; kept only the first"
            )

        return movements

    # ------------------------------------------------------------------
    # Step 7: Enforce minimum 2 movements per process
    # ------------------------------------------------------------------

    def _enforce_minimum(
        self,
        movements: list[DataMovement],
        pr: ProcessResult,
        source_format: SourceFormat,
    ) -> list[DataMovement]:
        if source_format in (SourceFormat.SENTENCE, SourceFormat.UC_NAME):
            return movements

        if len(movements) < 2:
            pr.corrections.append(
                f"Process has {len(movements)} movement(s), below COSMIC minimum of 2"
            )

        return movements
