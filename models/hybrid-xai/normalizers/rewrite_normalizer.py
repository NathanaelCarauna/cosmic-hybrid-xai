"""Rewrite Normalizer — SLM as text rewriter + NLP annotation pipeline.

Experiment E21 (RewriteNormalizer):
  Instead of asking the SLM to classify COSMIC movements directly, ask it
  to *rewrite* requirements as canonical atomic steps.  Each step is then
  classified via deterministic NLP direction detection + rule engine.

Experiment E23 (RewriteV2Normalizer):
  COSMIC-aware rewrite prompt + validation suppression + compound-step
  decomposition + gamma-cap correction.  Addresses three failure modes of
  E21: form-field over-decomposition, missing implicit Entries, and dense
  step under-parsing.

Experiment E21+E22 (RewriteTokenNormalizer):
  Combines SLM rewriting (E21) with enhanced token-level NLP annotation
  (E22) for multi-object extraction and better data group identification.

All normalizers produce IR-Full output (DataMovement objects with
explicit data_group names), letting the rule engine validate/deduplicate.
"""
from __future__ import annotations

import re
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
from .nlp_normalizer import (
    _detect_direction,
    _extract_object,
    _get_nlp,
    _is_error_or_confirm,
    _is_ui_control_step,
    _SYSTEM_ACTORS,
    extract_all_objects,
)
from .pipeline_normalizer import decompose_step

# ---------------------------------------------------------------------------
# Rewrite prompt — ~70 tokens system + ~100 tokens example ≈ 170 total
# (vs ~800+ for the full COSMIC classification prompt)
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = """\
You are a requirements analyst. Rewrite the requirement as a numbered list of atomic steps.

Format: N. [Actor] [verb] [data-object]

Rules:
- Actor = role name (e.g. "Hotel manager") or "System"
- One verb per step — split "reads X and saves Y" into two steps
- Name data objects explicitly — never use "it" or pronouns
- Navigation without data: "Actor navigates to screen-name"
- Messages: "System displays message-type to Actor"
- Skip "resumes at step N", "flow ends", "use case ends"
- Include error and exception handling steps"""

_REWRITE_EXAMPLE = """\
Input:
UC: Search rooms. Hotel manager selects Rooms menu, fills room code, \
system searches rooms and displays results. Exception: error if no results.

Output:
1. Hotel manager selects Rooms menu option
2. Hotel manager enters room code filter
3. System searches room records
4. System displays room list to Hotel manager
5. System displays error message to Hotel manager"""

# ---------------------------------------------------------------------------
# E23 COSMIC-aware rewrite prompt
# ---------------------------------------------------------------------------

_REWRITE_V2_SYSTEM_PROMPT = """\
You are a requirements analyst. Rewrite the requirement as a numbered list of data-movement steps.

Format: N. [Actor] [verb] [data-object]

Rules:
- Actor = role name (e.g. "Hotel manager") or "System"
- When a user fills in MULTIPLE fields of the SAME form, write ONE step: \
"Actor enters form-name data (field1, field2, ...)" — never split individual fields
- Internal validation (checking formats, ranges, business rules on already-entered data) \
is NOT a step — omit it entirely
- When the system accesses DIFFERENT persistent stores (files, tables, entities), \
write SEPARATE steps: "System reads store-A" and "System reads store-B"
- When the system removes/stores MULTIPLE distinct records, write a SEPARATE step per store
- "User clicks/selects [specific item]" = ONE data-selection step — include it
- Messages: "System displays message-type to Actor"
- Skip "resumes at step N", "flow ends", pure UI navigation without data"""

_REWRITE_V2_EXAMPLE_1 = """\
Input:
UC: Search rooms. Hotel manager selects Rooms menu, fills room code, \
system searches rooms and displays results. Exception: error if no results.

Output:
1. Hotel manager selects Rooms menu option
2. Hotel manager enters room code filter
3. System searches room records
4. System displays room list to Hotel manager
5. System displays error message to Hotel manager"""

_REWRITE_V2_EXAMPLE_2 = """\
Input:
UC: Insert room. Hotel manager clicks New, fills code/description/type/view/beds/notes, \
submits. System validates and saves room. Confirmation or error.

Output:
1. Hotel manager selects New room option
2. Hotel manager enters room details (code, description, type, view, beds, notes)
3. System saves room record
4. System displays confirmation message to Hotel manager
5. System displays error message to Hotel manager"""

# ---------------------------------------------------------------------------
# Validation / compound patterns for E23 post-processing
# ---------------------------------------------------------------------------

_VALIDATION_PATTERN = re.compile(
    r"\b(checks?\s+that|validates?\s+that|verifies?\s+that|ensures?\s+that"
    r"|checks?\s+if|validates?\s+the\s+given|validates?\s+the\s+provided)\b",
    re.IGNORECASE,
)
_NAMED_STORE_PATTERN = re.compile(
    r"\b(file|record|storage|database|table|store|persistent)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Direction → MovementType mapping
# ---------------------------------------------------------------------------

_DIR_TO_TYPE = {
    Direction.USER_TO_SYSTEM: MovementType.E,
    Direction.SYSTEM_TO_USER: MovementType.X,
    Direction.SYSTEM_TO_STORAGE: MovementType.W,
    Direction.STORAGE_TO_SYSTEM: MovementType.R,
}

_STEP_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# E21: RewriteNormalizer
# ---------------------------------------------------------------------------

class RewriteNormalizer(BaseNormalizer):
    """SLM rewrites requirements into canonical atomic steps; NLP classifies.

    Stage 1 — SLM produces a numbered list of ``[Actor] [verb] [object]``
    Stage 2 — spaCy extracts actor/verb/object, direction detection maps to E/X/R/W
    Stage 3 — Rule engine validates, deduplicates, collapses, and counts CFP
    """

    def __init__(
        self,
        slm_normalizer,
        *,
        max_tokens: int = 1024,
        fallback_to_slm: bool = True,
    ):
        self._llm = slm_normalizer._llm
        self._slm = slm_normalizer
        self._temperature = slm_normalizer.temperature
        self._max_tokens = max_tokens
        self._fallback = fallback_to_slm
        self._nlp = _get_nlp()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        prompt = self._build_prompt(text, source_format)
        rewritten = self._call_rewrite(prompt)
        steps = self._parse_steps(rewritten)

        if len(steps) < 2 and self._fallback:
            return self._slm.normalize(
                text, source_format, fp_id=fp_id, metadata=metadata,
            )

        return self._build_ir(steps, text, source_format, fp_id, metadata)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str, source_format: SourceFormat) -> str:
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(text)
            steps_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["steps"]
            )
            prompt = f"Rewrite as atomic steps:\n\nUC: {fields['name']}\n"
            if fields["description"]:
                prompt += f"Description: {fields['description']}\n"
            prompt += f"Actor: {fields['primary_actor'] or 'User'}\n"
            prompt += f"\nSteps:\n{steps_text}\n"

            alt_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["alt_flows"]
            )
            exc_text = "\n".join(
                f"  {sid}. [{actor}] {action}"
                for sid, actor, action in fields["exceptions"]
            )
            if alt_text:
                prompt += f"\nAlternatives:\n{alt_text}\n"
            if exc_text:
                prompt += f"\nExceptions:\n{exc_text}\n"
            return prompt

        if source_format == SourceFormat.UC_NAME:
            return f"Write the likely atomic steps for this use case:\n{text}"

        return f"Rewrite as atomic steps:\n{text}"

    # ------------------------------------------------------------------
    # SLM call (plain text, no JSON constraint)
    # ------------------------------------------------------------------

    def _call_rewrite(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": _REWRITE_EXAMPLE},
            {"role": "assistant", "content": "Understood. Send the requirement."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_steps(self, rewritten: str) -> list[tuple[str, str]]:
        """Parse numbered list → [(step_id, line_text), ...]."""
        return [
            (m.group(1), m.group(2).strip())
            for m in _STEP_RE.finditer(rewritten)
        ]

    # ------------------------------------------------------------------
    # IR construction
    # ------------------------------------------------------------------

    def _extract_step_objects(self, doc) -> list[str]:
        """Return data objects from a parsed step.  Override for E22."""
        obj = _extract_object(doc)
        return [obj] if obj else [""]

    def _build_ir(
        self,
        steps: list[tuple[str, str]],
        raw_text: str,
        source_format: SourceFormat,
        fp_id: str,
        metadata: Optional[Metadata],
    ) -> NormalizedRequirement:
        sub_processes: list[SubProcess] = []
        movements: list[DataMovement] = []
        actors_seen: set[str] = set()

        for step_id, line in steps:
            doc = self._nlp(line)

            actor = self._extract_step_actor(doc)
            action = self._strip_actor(line, actor)

            direction = _detect_direction(line, actor, doc)
            if direction is None:
                continue
            if _is_ui_control_step(action, actor, direction):
                continue

            mt = _DIR_TO_TYPE.get(direction)
            if mt is None:
                continue

            objects = self._extract_step_objects(doc)
            is_msg = _is_error_or_confirm(action)

            for obj in objects:
                dg = obj if obj else action[:60]
                dm = DataMovement(
                    type=mt,
                    data_group=dg,
                    object_of_interest=obj or action[:40],
                    functional_user=actor if mt in (MovementType.E, MovementType.X) else None,
                    is_triggering_entry=False,
                    is_error_confirmation=is_msg and mt == MovementType.X,
                )
                movements.append(dm)

            sp = SubProcess(
                step_id=step_id,
                actor=actor,
                action_text=action,
                direction=direction,
            )
            sub_processes.append(sp)

            if actor.lower() not in _SYSTEM_ACTORS:
                actors_seen.add(actor)

        fp_name = fp_id
        primary_actor = "User"
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(raw_text)
            fp_name = fields.get("name") or fp_id
            primary_actor = fields.get("primary_actor") or (
                next(iter(actors_seen), "User")
            )
        elif actors_seen:
            primary_actor = next(iter(actors_seen))

        fp = FunctionalProcess(
            name=fp_name or fp_id or "Unknown",
            triggering_event=fp_name or "User initiates action",
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=raw_text,
            boundary=BoundaryContext(
                software_name=(
                    metadata.project if metadata and metadata.project else "Unknown"
                ),
                functional_users=list(actors_seen) or [primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )

    # ------------------------------------------------------------------
    # NLP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_step_actor(doc) -> str:
        """Extract actor from the grammatical subject of a step."""
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ("nsubj", "nsubjpass"):
                return chunk.text
        for tok in doc:
            if tok.dep_ in ("nsubj", "nsubjpass"):
                return tok.text.capitalize()
        return "System"

    @staticmethod
    def _strip_actor(line: str, actor: str) -> str:
        """Remove leading actor name to isolate the action text."""
        if line.lower().startswith(actor.lower()):
            return line[len(actor):].strip()
        return line


# ---------------------------------------------------------------------------
# E21+E22: RewriteTokenNormalizer — combines SLM rewriting + enhanced NLP
# ---------------------------------------------------------------------------

class RewriteTokenNormalizer(RewriteNormalizer):
    """3-stage pipeline: SLM rewrite → token-level NLP annotation → rule engine.

    Overrides object extraction to use ``extract_all_objects`` (E22) which
    captures multiple data groups from compound noun phrases within a single
    canonical step.
    """

    def _extract_step_objects(self, doc) -> list[str]:
        return extract_all_objects(doc)


# ---------------------------------------------------------------------------
# E23: RewriteV2Normalizer — COSMIC-aware rewrite + smarter NLP + post-proc
# ---------------------------------------------------------------------------

class RewriteV2Normalizer(RewriteNormalizer):
    """Improved rewrite pipeline addressing three E21 failure modes.

    Fix 1 — COSMIC-aware prompt: groups form fields, omits validation steps,
             separates distinct store accesses.
    Fix 2 — Smarter direction detection: control verb + domain object = Entry;
             internal validation without named store = suppress Read.
    Fix 3 — Compound-step decomposition via ``decompose_step`` before NLP
             classification + gamma-cap correction for complex UCs.
    """

    _GAMMA_ALPHA = 0.35
    _GAMMA_BETA = 12
    _GAMMA_CAP = 0.30
    _STEP_THRESHOLD = 14

    def _call_rewrite(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": _REWRITE_V2_SYSTEM_PROMPT},
            {"role": "user", "content": _REWRITE_V2_EXAMPLE_1},
            {"role": "assistant", "content": (
                "1. Hotel manager selects Rooms menu option\n"
                "2. Hotel manager enters room code filter\n"
                "3. System searches room records\n"
                "4. System displays room list to Hotel manager\n"
                "5. System displays error message to Hotel manager"
            )},
            {"role": "user", "content": _REWRITE_V2_EXAMPLE_2},
            {"role": "assistant", "content": (
                "1. Hotel manager selects New room option\n"
                "2. Hotel manager enters room details "
                "(code, description, type, view, beds, notes)\n"
                "3. System saves room record\n"
                "4. System displays confirmation message to Hotel manager\n"
                "5. System displays error message to Hotel manager"
            )},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _build_ir(
        self,
        steps: list[tuple[str, str]],
        raw_text: str,
        source_format: SourceFormat,
        fp_id: str,
        metadata: Optional[Metadata],
    ) -> NormalizedRequirement:
        expanded_steps: list[tuple[str, str]] = []
        for step_id, line in steps:
            parts = decompose_step(line)
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    expanded_steps.append((f"{step_id}.{i+1}", part))
            else:
                expanded_steps.append((step_id, line))

        sub_processes: list[SubProcess] = []
        movements: list[DataMovement] = []
        actors_seen: set[str] = set()

        for step_id, line in expanded_steps:
            doc = self._nlp(line)

            actor = self._extract_step_actor(doc)
            action = self._strip_actor(line, actor)

            direction = _detect_direction(line, actor, doc)

            if direction is None:
                if self._is_control_with_domain_object(doc, action, actor):
                    direction = Direction.USER_TO_SYSTEM
                else:
                    continue

            if self._is_internal_validation(action, direction):
                continue

            if _is_ui_control_step(action, actor, direction):
                if self._has_domain_object(doc):
                    direction = Direction.USER_TO_SYSTEM
                else:
                    continue

            mt = _DIR_TO_TYPE.get(direction)
            if mt is None:
                continue

            objects = self._extract_step_objects(doc)
            is_msg = _is_error_or_confirm(action)

            for obj in objects:
                dg = obj if obj else action[:60]
                dm = DataMovement(
                    type=mt,
                    data_group=dg,
                    object_of_interest=obj or action[:40],
                    functional_user=actor if mt in (MovementType.E, MovementType.X) else None,
                    is_triggering_entry=False,
                    is_error_confirmation=is_msg and mt == MovementType.X,
                )
                movements.append(dm)

            sp = SubProcess(
                step_id=step_id,
                actor=actor,
                action_text=action,
                direction=direction,
            )
            sub_processes.append(sp)

            if actor.lower() not in _SYSTEM_ACTORS:
                actors_seen.add(actor)

        n_total = len(movements)
        if len(expanded_steps) >= self._STEP_THRESHOLD:
            correction = min(
                self._GAMMA_ALPHA * max(0, len(expanded_steps) - self._GAMMA_BETA),
                n_total * self._GAMMA_CAP,
            )
            n_to_remove = int(round(correction))
            if n_to_remove > 0:
                movements = movements[:n_total - n_to_remove]

        fp_name = fp_id
        primary_actor = "User"
        if source_format == SourceFormat.USE_CASE:
            fields = self.parse_uc_fields(raw_text)
            fp_name = fields.get("name") or fp_id
            primary_actor = fields.get("primary_actor") or (
                next(iter(actors_seen), "User")
            )
        elif actors_seen:
            primary_actor = next(iter(actors_seen))

        fp = FunctionalProcess(
            name=fp_name or fp_id or "Unknown",
            triggering_event=fp_name or "User initiates action",
            primary_functional_user=primary_actor,
            sub_processes=sub_processes,
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=raw_text,
            boundary=BoundaryContext(
                software_name=(
                    metadata.project if metadata and metadata.project else "Unknown"
                ),
                functional_users=list(actors_seen) or [primary_actor],
            ),
            functional_processes=[fp],
            metadata=metadata or Metadata(),
        )

    # ------------------------------------------------------------------
    # E23 Fix 2: Control verb + domain object → Entry
    # ------------------------------------------------------------------

    @staticmethod
    def _is_control_with_domain_object(doc, action: str, actor: str) -> bool:
        """Detect cases where a 'control' verb acts on a domain data object."""
        if actor.lower() in _SYSTEM_ACTORS:
            return False
        _CONTROL_VERBS = {"click", "select", "press", "choose", "pick", "tap"}
        root_verb = None
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                root_verb = tok.lemma_.lower()
                break
        if root_verb not in _CONTROL_VERBS:
            return False

        _UI_ONLY = {"button", "menu", "tab", "link", "page", "screen", "option",
                     "sub-option", "icon", "sidebar", "navbar", "toolbar"}
        for tok in doc:
            if tok.dep_ in ("dobj", "pobj", "attr") and tok.pos_ in ("NOUN", "PROPN"):
                if tok.lemma_.lower() not in _UI_ONLY:
                    return True
        return False

    @staticmethod
    def _has_domain_object(doc) -> bool:
        """Check if the step references a non-UI domain data object."""
        _UI_ONLY = {"button", "menu", "tab", "link", "page", "screen", "option",
                     "sub-option", "icon", "sidebar", "navbar", "toolbar"}
        for tok in doc:
            if tok.dep_ in ("dobj", "pobj", "attr") and tok.pos_ in ("NOUN", "PROPN"):
                if tok.lemma_.lower() not in _UI_ONLY:
                    return True
        return False

    # ------------------------------------------------------------------
    # E23 Fix 2b: Internal validation suppression
    # ------------------------------------------------------------------

    @staticmethod
    def _is_internal_validation(action: str, direction: Direction) -> bool:
        """Suppress Read movements that are really format/range validation."""
        if direction != Direction.STORAGE_TO_SYSTEM:
            return False
        if _VALIDATION_PATTERN.search(action) and not _NAMED_STORE_PATTERN.search(action):
            return True
        return False
