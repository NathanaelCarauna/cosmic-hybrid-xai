"""NLP Normalizer — spaCy-based zero-cost floor (Experiment E1).

Uses spaCy dependency parsing to extract subject-verb-object triples
from requirement text and maps them to COSMIC data movements via
verb taxonomy heuristics. Produces IR-Semantic output (directions only),
leaving final E/X/R/W classification to Module 2.

Cost: $0 (local CPU only).
"""
from __future__ import annotations

import re
from typing import Optional

import spacy
from spacy.tokens import Doc, Span, Token

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

_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Verb → direction mapping
# ---------------------------------------------------------------------------

# \w* after stems captures inflected forms: fills, selects, shows, etc.
_USER_TO_SYSTEM = re.compile(
    r"\b(enter|fill|input|submit|select|choose|provide|inform|type|"
    r"specify|insert|pick|mark|tick|set|upload|attach|press)\w*\b",
    re.IGNORECASE,
)
_SYSTEM_TO_USER = re.compile(
    r"\b(display|show|list|present|output|render|print|exhibit|"
    r"generate|report|view|visualize|summarize|highlight)\w*\b",
    re.IGNORECASE,
)
_STORAGE_TO_SYSTEM = re.compile(
    r"\b(search|query|fetch|retrieve|read|look\s?up|find|get|load|"
    r"obtain|verify|validate|check|seek|browse|scan|inspect|locate|filter)\w*\b",
    re.IGNORECASE,
)
_SYSTEM_TO_STORAGE = re.compile(
    r"\b(save|store|write|record|persist|update|modify|delete|remove|"
    r"create|register|add|log|archive|overwrite|append|erase|empty|clear)\w*\b",
    re.IGNORECASE,
)
_CONTROL = re.compile(
    r"\b(click|navigate|close|cancel|abort|resume|return to|go back|"
    r"open|launch|exit|log\s?out|undo)\w*\b",
    re.IGNORECASE,
)
_ERROR_PATTERN = re.compile(
    r"\b(error|fail|invalid|exception|warning|not found|unable|denied|"
    r"reject|unsuccessful|alert|could not|cannot)\b",
    re.IGNORECASE,
)
_CONFIRM_PATTERN = re.compile(
    r"\b(confirm|success|complete|done|acknowledge|notification)\w*\b",
    re.IGNORECASE,
)

_SYSTEM_ACTORS = {"system", "system timer"}


def _classify_root_verb(lemma: str) -> Optional[str]:
    """Classify a single verb lemma into a direction category."""
    lemma = lemma.lower()
    u2s = {"enter", "fill", "input", "submit", "select", "choose", "provide",
           "inform", "type", "specify", "insert", "pick", "mark", "tick",
           "set", "upload", "attach", "press"}
    s2u = {"display", "show", "list", "present", "output", "render", "print",
           "exhibit", "generate", "report", "view", "visualize", "summarize",
           "highlight", "ask"}
    sto2s = {"search", "query", "fetch", "retrieve", "read", "find", "get",
             "load", "obtain", "verify", "validate", "check", "seek",
             "browse", "scan", "inspect", "locate", "filter", "look"}
    s2sto = {"save", "store", "write", "record", "persist", "update", "modify",
             "delete", "remove", "create", "register", "add", "log", "archive",
             "overwrite", "append", "erase", "empty", "clear"}
    ctrl = {"click", "navigate", "close", "cancel", "abort", "resume",
            "open", "launch", "exit", "undo", "refresh", "browse"}

    if lemma in ctrl:
        return "control"
    if lemma in s2u:
        return "system_to_user"
    if lemma in s2sto:
        return "system_to_storage"
    if lemma in sto2s:
        return "storage_to_system"
    if lemma in u2s:
        return "user_to_system"
    return None


def _detect_direction(action: str, actor: str, doc: Optional[Doc] = None) -> Optional[Direction]:
    """Map action text + actor to a data flow direction.

    Uses the spaCy ROOT verb for primary classification (more accurate),
    then falls back to whole-text regex scanning.
    """
    is_system = actor.lower() in _SYSTEM_ACTORS

    # --- Phase 1: ROOT verb classification (spaCy) ---
    root_cat: Optional[str] = None
    if doc is not None:
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                root_cat = _classify_root_verb(tok.lemma_)
                break
        if root_cat is None:
            for tok in doc:
                if tok.pos_ == "VERB":
                    root_cat = _classify_root_verb(tok.lemma_)
                    if root_cat:
                        break

    # Error/confirmation messages always flow system → user
    if _ERROR_PATTERN.search(action) or _CONFIRM_PATTERN.search(action):
        if is_system:
            return Direction.SYSTEM_TO_USER

    if root_cat == "control":
        has_data = (
            _USER_TO_SYSTEM.search(action)
            or _SYSTEM_TO_USER.search(action)
            or _STORAGE_TO_SYSTEM.search(action)
            or _SYSTEM_TO_STORAGE.search(action)
        )
        if not has_data:
            return None

    if root_cat and root_cat != "control":
        direction_map = {
            "user_to_system": Direction.USER_TO_SYSTEM,
            "system_to_user": Direction.SYSTEM_TO_USER,
            "system_to_storage": Direction.SYSTEM_TO_STORAGE,
            "storage_to_system": Direction.STORAGE_TO_SYSTEM,
        }
        return direction_map.get(root_cat)

    # --- Phase 2: Regex fallback (whole text) ---
    if is_system:
        if _SYSTEM_TO_USER.search(action):
            return Direction.SYSTEM_TO_USER
        if _SYSTEM_TO_STORAGE.search(action):
            return Direction.SYSTEM_TO_STORAGE
        if _STORAGE_TO_SYSTEM.search(action):
            return Direction.STORAGE_TO_SYSTEM
        return None
    else:
        if _CONTROL.search(action) and not _USER_TO_SYSTEM.search(action):
            return None
        if _USER_TO_SYSTEM.search(action):
            return Direction.USER_TO_SYSTEM
        return None


_UI_CONTROL_PATTERN = re.compile(
    r"\b(sub-option|link|button|menu|tab|page|screen|option)\b",
    re.IGNORECASE,
)
_EMPTY_FORM_PATTERN = re.compile(
    r"\bshow\w*\s+(a\s+)?(blank\s+)?(form|formatted\s+screen)\b",
    re.IGNORECASE,
)


def _is_ui_control_step(action: str, actor: str, direction: Optional[Direction]) -> bool:
    """Detect UI control steps that should not generate movements."""
    if direction == Direction.USER_TO_SYSTEM:
        action_lower = action.lower()
        if _UI_CONTROL_PATTERN.search(action) and "data" not in action_lower and "detail" not in action_lower:
            if not any(w in action_lower for w in ("fill", "enter", "submit", "input", "provide")):
                return True
    if direction == Direction.SYSTEM_TO_USER:
        if _EMPTY_FORM_PATTERN.search(action):
            return True
    return False


def _is_error_or_confirm(action: str) -> bool:
    return bool(_ERROR_PATTERN.search(action) or _CONFIRM_PATTERN.search(action))


# ---------------------------------------------------------------------------
# NLP extraction helpers
# ---------------------------------------------------------------------------

def _extract_object(doc: Doc) -> str:
    """Extract the main object noun phrase from a spaCy Doc."""
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in ("dobj", "pobj", "attr", "nsubjpass"):
            return chunk.text
    for tok in doc:
        if tok.dep_ in ("dobj", "pobj"):
            return tok.text
    return ""


def extract_all_objects(doc: Doc) -> list[str]:
    """Extract ALL object noun phrases from a spaCy Doc, excluding subjects.

    Returns distinct noun chunks whose root is dobj/pobj/attr/conj, plus
    any coordinated objects linked by ``conj`` dependency.  Used by E22
    token-level annotation for multi-object data group extraction.
    """
    objects: list[str] = []
    seen: set[int] = set()

    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == "nsubj":
            continue
        if chunk.root.dep_ in ("dobj", "pobj", "attr", "nsubjpass", "conj"):
            if chunk.root.i not in seen:
                objects.append(chunk.text)
                seen.add(chunk.root.i)

    if not objects:
        for tok in doc:
            if tok.dep_ in ("dobj", "pobj") and tok.i not in seen:
                objects.append(tok.text)
                seen.add(tok.i)

    return objects if objects else [""]


def _extract_actor_from_step(step_actor: str, text: str, doc: Doc) -> str:
    """Determine the actor from step metadata or NLP."""
    if step_actor:
        return step_actor
    for tok in doc:
        if tok.dep_ == "nsubj":
            return tok.text.capitalize()
    return "System"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class NLPNormalizer(BaseNormalizer):
    """Zero-cost spaCy-based normalizer producing IR-Semantic output."""

    def __init__(self):
        self._nlp = _get_nlp()

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
            return self._normalize_uc_name(text, fp_id, metadata)
        return self._normalize_sentence(text, fp_id, metadata)

    # ------------------------------------------------------------------
    # Use Case normalization
    # ------------------------------------------------------------------

    def _normalize_uc(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        fields = self.parse_uc_fields(text)
        primary_actor = fields["primary_actor"] or "User"
        all_steps = fields["steps"] + fields["alt_flows"] + fields["exceptions"]

        sub_processes: list[SubProcess] = []
        for step_id, actor, action in all_steps:
            if not action:
                continue
            if "resumes at step" in action.lower() or "flow of events ends" in action.lower():
                continue

            doc = self._nlp(action)
            effective_actor = actor or "System"
            direction = _detect_direction(action, effective_actor, doc)
            if direction is None:
                continue
            if _is_ui_control_step(action, effective_actor, direction):
                continue

            sp = SubProcess(
                step_id=step_id,
                actor=actor or "System",
                action_text=action,
                direction=direction,
                data_movement=None,
            )
            sub_processes.append(sp)

        fp = FunctionalProcess(
            name=fields["name"] or fp_id,
            triggering_event=fields["name"] or "User initiates action",
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=SourceFormat.USE_CASE,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=metadata.project if metadata and metadata.project else "Unknown",
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )

    # ------------------------------------------------------------------
    # Sentence normalization
    # ------------------------------------------------------------------

    def _normalize_sentence(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        doc = self._nlp(text)

        actor = "User"
        for tok in doc:
            if tok.dep_ == "nsubj":
                actor = tok.text.capitalize()
                break

        direction = _detect_direction(text, actor, doc)
        obj = _extract_object(doc)

        sp = SubProcess(
            step_id="1",
            actor=actor,
            action_text=text,
            direction=direction or Direction.USER_TO_SYSTEM,
        )

        fp = FunctionalProcess(
            name=fp_id or text[:60],
            triggering_event=text[:80],
            primary_functional_user=actor,
            sub_processes=[sp],
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=SourceFormat.SENTENCE,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=metadata.project if metadata and metadata.project else "Unknown",
                functional_users=[actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )

    # ------------------------------------------------------------------
    # UC-name-only normalization (Ochodek-437)
    # ------------------------------------------------------------------

    def _normalize_uc_name(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        doc = self._nlp(text)

        verb = ""
        obj_text = ""
        for tok in doc:
            if tok.pos_ == "VERB" and not verb:
                verb = tok.lemma_.lower()
            if tok.dep_ in ("dobj", "pobj", "attr") and not obj_text:
                obj_text = tok.text

        if not obj_text:
            for chunk in doc.noun_chunks:
                obj_text = chunk.text
                break

        if not obj_text:
            obj_text = text

        sub_processes: list[SubProcess] = []
        movements: list[DataMovement] = []

        movements.append(DataMovement(
            type=MovementType.E,
            data_group=f"{text} request",
            object_of_interest=obj_text,
            functional_user="User",
            is_triggering_entry=True,
        ))
        sub_processes.append(SubProcess(
            step_id="1", actor="User", action_text=f"initiates {text}",
            direction=Direction.USER_TO_SYSTEM,
        ))

        persist_verbs = {"save", "store", "register", "create", "insert", "add", "book", "submit", "record", "update", "modify", "edit", "delete", "remove"}
        read_verbs = {"view", "list", "search", "browse", "find", "get", "show", "display", "check", "verify", "report", "generate", "print", "export"}

        if verb in persist_verbs:
            movements.append(DataMovement(
                type=MovementType.W,
                data_group=f"{obj_text} record",
                object_of_interest=obj_text,
            ))
            sub_processes.append(SubProcess(
                step_id="2", actor="System", action_text=f"saves {obj_text}",
                direction=Direction.SYSTEM_TO_STORAGE,
            ))
            movements.append(DataMovement(
                type=MovementType.X,
                data_group=f"Confirmation",
                object_of_interest=obj_text,
                functional_user="User",
                is_error_confirmation=True,
            ))
            sub_processes.append(SubProcess(
                step_id="3", actor="System", action_text="shows confirmation",
                direction=Direction.SYSTEM_TO_USER,
            ))
        elif verb in read_verbs:
            movements.append(DataMovement(
                type=MovementType.R,
                data_group=f"{obj_text} records",
                object_of_interest=obj_text,
            ))
            sub_processes.append(SubProcess(
                step_id="2", actor="System", action_text=f"reads {obj_text}",
                direction=Direction.STORAGE_TO_SYSTEM,
            ))
            movements.append(DataMovement(
                type=MovementType.X,
                data_group=f"{obj_text} details",
                object_of_interest=obj_text,
                functional_user="User",
            ))
            sub_processes.append(SubProcess(
                step_id="3", actor="System", action_text=f"displays {obj_text}",
                direction=Direction.SYSTEM_TO_USER,
            ))
        else:
            movements.append(DataMovement(
                type=MovementType.R,
                data_group=f"{obj_text} data",
                object_of_interest=obj_text,
            ))
            sub_processes.append(SubProcess(
                step_id="2", actor="System", action_text=f"processes {obj_text}",
                direction=Direction.STORAGE_TO_SYSTEM,
            ))

        fp = FunctionalProcess(
            name=text,
            triggering_event=text,
            primary_functional_user="User",
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=SourceFormat.UC_NAME,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=metadata.project if metadata and metadata.project else "Unknown",
                functional_users=["User"],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )


# ---------------------------------------------------------------------------
# E22: Enhanced NLP Normalizer with token-level annotation
# ---------------------------------------------------------------------------

_DIR_TO_TYPE = {
    Direction.USER_TO_SYSTEM: MovementType.E,
    Direction.SYSTEM_TO_USER: MovementType.X,
    Direction.SYSTEM_TO_STORAGE: MovementType.W,
    Direction.STORAGE_TO_SYSTEM: MovementType.R,
}


class EnhancedNLPNormalizer(NLPNormalizer):
    """E22: NLP normalizer producing IR-Full output via token-level annotation.

    Key improvement over base NLPNormalizer: produces ``DataMovement`` objects
    with proper ``data_group`` names extracted from noun phrases, and extracts
    multiple objects per step when compound references are present.
    """

    def _normalize_uc(
        self, text: str, fp_id: str, metadata: Optional[Metadata]
    ) -> NormalizedRequirement:
        fields = self.parse_uc_fields(text)
        primary_actor = fields["primary_actor"] or "User"
        all_steps = fields["steps"] + fields["alt_flows"] + fields["exceptions"]

        sub_processes: list[SubProcess] = []
        movements: list[DataMovement] = []

        for step_id, actor, action in all_steps:
            if not action:
                continue
            if "resumes at step" in action.lower() or "flow of events ends" in action.lower():
                continue

            doc = self._nlp(action)
            effective_actor = actor or "System"
            direction = _detect_direction(action, effective_actor, doc)

            if direction is None:
                continue
            if _is_ui_control_step(action, effective_actor, direction):
                continue

            mt = _DIR_TO_TYPE.get(direction)
            if mt is None:
                continue

            objects = extract_all_objects(doc)
            is_msg = _is_error_or_confirm(action)

            for obj in objects:
                dg = obj if obj else action[:60]
                dm = DataMovement(
                    type=mt,
                    data_group=dg,
                    object_of_interest=obj or action[:40],
                    functional_user=effective_actor if mt in (MovementType.E, MovementType.X) else None,
                    is_triggering_entry=False,
                    is_error_confirmation=is_msg and mt == MovementType.X,
                )
                movements.append(dm)

            sp = SubProcess(
                step_id=step_id,
                actor=effective_actor,
                action_text=action,
                direction=direction,
            )
            sub_processes.append(sp)

        fp = FunctionalProcess(
            name=fields["name"] or fp_id,
            triggering_event=fields["name"] or "User initiates action",
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=SourceFormat.USE_CASE,
            raw_text=text,
            boundary=BoundaryContext(
                software_name=metadata.project if metadata and metadata.project else "Unknown",
                functional_users=[primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )
