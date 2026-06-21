"""E10: Keyword Counting Normalizer — pure regex pattern matching.

Zero-model approach: counts COSMIC movement keywords directly in the text.
Extremely fast, zero cost, fully deterministic.

Uses curated verb/pattern dictionaries mapped to E/X/R/W movement types.
Each distinct (verb_pattern, object) pair = 1 movement.
"""
from __future__ import annotations

import re
from typing import Optional

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

_ENTRY_PATTERNS = [
    r'\b(?:user|actor|operator|manager|admin|customer|clerk)\b.*?\b(?:fill\w*|enter\w*|input\w*|submit\w*|type\w*|select\w*|specif\w*|set\w*|provid\w*|upload\w*|insert\w*|choos\w*)\b',
    r'\b(?:fill\w*|enter\w*|input\w*|submit\w*|type\w*|select\w*|specif\w*|provid\w*|upload\w*|insert\w*|choos\w*)\b.*?\b(?:data|form|field|filter|criteria|detail|information|code|name|password|value)\b',
]

_READ_PATTERNS = [
    r'\b(?:system|application|software)\b.*?\b(?:search\w*|retriev\w*|read\w*|look\w*\s*up|fetch\w*|load\w*|check\w*|verif\w*|validat\w*|find\w*|query|queries|queri\w*)\b',
    r'\b(?:search\w*|retriev\w*|read\w*|fetch\w*|load\w*|check\w*|verif\w*|validat\w*|find\w*)\b.*?\b(?:database|storage|file|record|table|persistent|from)\b',
    r'\bsearch\w*\s+for\b',
]

_WRITE_PATTERNS = [
    r'\b(?:system|application|software)\b.*?\b(?:sav\w*|stor\w*|updat\w*|delet\w*|remov\w*|insert\w*|creat\w*|modif\w*|persist\w*|add\w*\s+to|record\w*|log\w*)\b',
    r'\b(?:sav\w*|stor\w*|updat\w*|delet\w*|remov\w*|persist\w*)\b.*?\b(?:database|storage|file|record|table|persistent)\b',
]

_EXIT_PATTERNS = [
    r'\b(?:system|application|software)\b.*?\b(?:display\w*|show\w*|present\w*|list\w*|print\w*|render\w*|exhibit\w*|output\w*|generat\w*\s+report|return\w*)\b',
    r'\b(?:display\w*|show\w*|present\w*|list\w*)\b.*?\b(?:result|data|information|message|report|list|detail|form|page|screen|confirmation|error)\b',
]

_ERROR_PATTERNS = [
    r'\b(?:error|warning|invalid|fail|not found|no result|duplicate|already exist|exception)\b',
    r'\bshow\w*\s+(?:error|warning|confirmation|system error)\b',
]

_SKIP_PATTERNS = [
    r'\b(?:click\w*|press\w*|tap\w*|navigate\w*|open\w*|close\w*|go\w*\s+to)\b.*?\b(?:button|link|menu|tab|page|screen|option)\b',
    r'\bresume\w*\s+at\s+step\b',
    r'\bflow\s+(?:ends|returns|resumes)\b',
    r'\bshow\w*\s+(?:empty\s+)?form\b',
]

_RE_FLAGS = re.IGNORECASE


class KeywordNormalizer(BaseNormalizer):
    """Counts COSMIC movements via regex keyword patterns."""

    def __init__(self):
        self._entry_re = [re.compile(p, _RE_FLAGS) for p in _ENTRY_PATTERNS]
        self._read_re = [re.compile(p, _RE_FLAGS) for p in _READ_PATTERNS]
        self._write_re = [re.compile(p, _RE_FLAGS) for p in _WRITE_PATTERNS]
        self._exit_re = [re.compile(p, _RE_FLAGS) for p in _EXIT_PATTERNS]
        self._error_re = [re.compile(p, _RE_FLAGS) for p in _ERROR_PATTERNS]
        self._skip_re = [re.compile(p, _RE_FLAGS) for p in _SKIP_PATTERNS]

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
        return self._normalize_text(text, fp_id, metadata, source_format)

    def _is_skip(self, line: str) -> bool:
        return any(p.search(line) for p in self._skip_re)

    def _classify_line(self, line: str) -> list[tuple[MovementType, str]]:
        if self._is_skip(line):
            return []

        results = []
        for p in self._entry_re:
            if p.search(line):
                obj = self._extract_object(line)
                results.append((MovementType.E, obj))
                break

        for p in self._read_re:
            m = p.search(line)
            if m:
                obj = self._extract_object(line)
                results.append((MovementType.R, obj))
                break

        for p in self._write_re:
            m = p.search(line)
            if m:
                obj = self._extract_object(line)
                results.append((MovementType.W, obj))
                break

        for p in self._exit_re:
            if p.search(line):
                obj = self._extract_object(line)
                results.append((MovementType.X, obj))
                break

        conjunctions = re.findall(r'\b(?:and|,)\s+(?:the\s+system\s+)?(?:also\s+)?(?:saves?|stores?|reads?|searches?|retrieves?|displays?|shows?|updates?|deletes?|removes?)\b', line, _RE_FLAGS)
        for conj in conjunctions:
            conj_lower = conj.lower()
            if any(w in conj_lower for w in ('save', 'store', 'update', 'delete', 'remove')):
                results.append((MovementType.W, self._extract_object(conj)))
            elif any(w in conj_lower for w in ('read', 'search', 'retriev')):
                results.append((MovementType.R, self._extract_object(conj)))
            elif any(w in conj_lower for w in ('display', 'show')):
                results.append((MovementType.X, self._extract_object(conj)))

        return results

    def _extract_object(self, text: str) -> str:
        m = re.search(r'\b(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:record|file|data|list|detail|form|message|table|header|code|filter)', text, _RE_FLAGS)
        if m:
            return m.group(1).strip()
        nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return nouns[0] if nouns else "data"

    def _normalize_uc(self, text: str, fp_id: str, metadata) -> NormalizedRequirement:
        fields = self.parse_uc_fields(text)
        primary_actor = fields.get("primary_actor") or "User"

        all_steps = fields["steps"] + fields["alt_flows"] + fields["exceptions"]
        movements: list[DataMovement] = []
        sub_processes: list[SubProcess] = []
        has_trigger = False
        has_error_exit = False

        for step_id, actor, action in all_steps:
            classified = self._classify_line(action)
            for mt, obj in classified:
                is_trigger = (mt == MovementType.E and not has_trigger)
                is_error = (mt == MovementType.X and
                            any(p.search(action) for p in self._error_re))
                if is_error and has_error_exit:
                    continue

                if is_trigger:
                    has_trigger = True
                if is_error:
                    has_error_exit = True

                dm = DataMovement(
                    type=mt,
                    data_group=f"{obj} {mt.value}",
                    object_of_interest=obj,
                    functional_user=actor if mt in (MovementType.E, MovementType.X) else None,
                    is_triggering_entry=is_trigger,
                    is_error_confirmation=is_error,
                )
                movements.append(dm)
                sub_processes.append(SubProcess(
                    step_id=step_id, actor=actor or primary_actor,
                    action_text=action, data_movement=dm,
                ))

            if not classified:
                multi_verbs = re.findall(
                    r'\b(searches?|reads?|checks?|retrieves?|saves?|stores?|updates?|deletes?|removes?|displays?|shows?|lists?)\b',
                    action, _RE_FLAGS
                )
                for verb in multi_verbs:
                    vl = verb.lower()
                    if vl.startswith(('search', 'read', 'check', 'retriev')):
                        mt = MovementType.R
                    elif vl.startswith(('save', 'store', 'update', 'delet', 'remov')):
                        mt = MovementType.W
                    elif vl.startswith(('display', 'show', 'list')):
                        mt = MovementType.X
                    else:
                        continue

                    dm = DataMovement(
                        type=mt, data_group=f"{verb} data",
                        object_of_interest="data",
                        functional_user=actor if mt == MovementType.X else None,
                    )
                    movements.append(dm)
                    sub_processes.append(SubProcess(
                        step_id=step_id, actor=actor or primary_actor,
                        action_text=action, data_movement=dm,
                    ))

        fp = FunctionalProcess(
            name=fields["name"] or fp_id,
            triggering_event=fields["name"] or fp_id,
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )
        return NormalizedRequirement(
            id=fp_id, source_format=SourceFormat.USE_CASE, raw_text=text,
            boundary=BoundaryContext(
                software_name=(metadata.project if metadata and metadata.project else "Unknown"),
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )

    def _normalize_text(self, text, fp_id, metadata, source_format):
        classified = self._classify_line(text)
        movements = []
        for mt, obj in classified:
            movements.append(DataMovement(
                type=mt, data_group=f"{obj} {mt.value}",
                object_of_interest=obj,
                functional_user="User" if mt in (MovementType.E, MovementType.X) else None,
                is_triggering_entry=(mt == MovementType.E and not movements),
            ))

        if not movements:
            for sent in re.split(r'[.;]', text):
                sent = sent.strip()
                if not sent:
                    continue
                verbs = re.findall(r'\b(enters?|fills?|submits?|provides?|inputs?)\b', sent, _RE_FLAGS)
                if verbs:
                    movements.append(DataMovement(
                        type=MovementType.E, data_group="input data",
                        object_of_interest="data", functional_user="User",
                        is_triggering_entry=not movements,
                    ))
                verbs = re.findall(r'\b(displays?|shows?|presents?|lists?|prints?)\b', sent, _RE_FLAGS)
                if verbs:
                    movements.append(DataMovement(
                        type=MovementType.X, data_group="output data",
                        object_of_interest="data", functional_user="System",
                    ))

        fp = FunctionalProcess(
            name=fp_id or "Unknown",
            triggering_event=fp_id or "Unknown",
            primary_functional_user="User",
            data_movements=movements,
        )
        return NormalizedRequirement(
            id=fp_id, source_format=source_format, raw_text=text,
            boundary=BoundaryContext(software_name="Unknown", functional_users=["User"]),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )
