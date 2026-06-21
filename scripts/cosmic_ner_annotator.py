#!/usr/bin/env python3
"""COSMIC NER Annotation Pipeline — Expert-quality span + movement annotations.

Reads all 119 De Vito FPs, applies COSMIC ISO 19761 expert rules to produce:
  1. Entity span annotations (FUNC_USER, DATA_OBJ, ACTION_VERB, UI_ELEMENT, CONDITION)
  2. Movement labels per step (E/X/R/W/none) with data_group and functional_user
  3. Validates total movements against ground truth CFP

Output: JSONL files in datasets/ner-annotations/ (one per system).

NOT using SLM IRs as silver labels — uses NLP analysis + COSMIC ontology rules.
COSMET measurement CSVs used only as validation reference.

Usage:
  python3 scripts/cosmic_ner_annotator.py [--validate-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import spacy

# ── Config ────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "replications" / "cosmet-devito" / "datasets"
RESULTS_COSMET = BASE / "results" / "cosmet"
OUTPUT_DIR = BASE / "datasets" / "ner-annotations"

SYSTEMS = {
    "ALBERGATE": {
        "prompts": DATASETS / "MIS" / "ALBERGATE" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements",
    },
    "K01719": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01719" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01719-gpt-4-turbo-20260323T005355" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "K01719-gpt-4-turbo-20260323T005355" / "measurements",
    },
    "K01720": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01720" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01720-gpt-4-turbo-20260323T005553" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "K01720-gpt-4-turbo-20260323T005553" / "measurements",
    },
    "K01726": {
        "prompts": DATASETS / "IoT_Telemedicine" / "K01726" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "K01726-gpt-4-turbo-20260323T005703" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "K01726-gpt-4-turbo-20260323T005703" / "measurements",
    },
    "U-CURE": {
        "prompts": DATASETS / "ML" / "U-CURE" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "U-CURE-gpt-4-turbo-20260323T010152" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "U-CURE-gpt-4-turbo-20260323T010152" / "measurements",
    },
    "AutomaticLineSwitching": {
        "prompts": DATASETS / "Real-time" / "AutomaticLineSwitching" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "AutomaticLineSwitching-gpt-4-turbo-20260323T010407" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "AutomaticLineSwitching-gpt-4-turbo-20260323T010407" / "measurements",
    },
    "RiceCooker": {
        "prompts": DATASETS / "Real-time" / "RiceCooker" / "FP_prompts",
        "gt_csv": RESULTS_COSMET / "RiceCooker-gpt-4-turbo-20260323T010514" / "measurements" / "comparison_results.csv",
        "cosmet_measurements": RESULTS_COSMET / "RiceCooker-gpt-4-turbo-20260323T010514" / "measurements",
    },
}

# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class EntitySpan:
    start: int
    end: int
    text: str
    label: str  # FUNC_USER, DATA_OBJ, ACTION_VERB, UI_ELEMENT, CONDITION, PERSIST_STORE


@dataclass
class StepAnnotation:
    step_id: str
    actor: str
    action_text: str
    full_text: str
    offset_in_fp: int
    entities: list[EntitySpan] = field(default_factory=list)
    movement_type: Optional[str] = None  # E, X, R, W, or None
    data_group: str = ""
    functional_user: str = ""
    is_error_confirmation: bool = False
    reasoning: str = ""


@dataclass
class FPAnnotation:
    system: str
    fp_id: str
    fp_text: str
    cfp_gt: int
    entry_gt: Optional[int] = None
    read_gt: Optional[int] = None
    write_gt: Optional[int] = None
    exit_gt: Optional[int] = None
    steps: list[StepAnnotation] = field(default_factory=list)
    cfp_annotated: int = 0
    validation_status: str = ""


# ── Verb taxonomies (COSMIC ISO 19761) ────────────────────────────────

_ENTRY_LEMMAS = {
    "enter", "fill", "input", "submit", "select", "choose", "provide",
    "inform", "type", "specify", "insert", "pick", "mark", "tick", "set",
    "upload", "attach", "press", "give", "pass", "indicate",
    "confirm", "reply", "respond",
}

_EXIT_LEMMAS = {
    "display", "show", "list", "present", "output", "render", "print",
    "exhibit", "generate", "produce", "report", "view", "visualize",
    "summarize", "highlight", "return", "respond", "notify",
    "ask", "request", "send",
}

_READ_LEMMAS = {
    "search", "query", "fetch", "retrieve", "read", "find", "get", "load",
    "obtain", "look", "seek", "browse", "scan", "inspect", "locate",
    "filter", "access", "check",  # contextual: check against storage
}

_WRITE_LEMMAS = {
    "save", "store", "write", "record", "persist", "update", "modify",
    "delete", "remove", "create", "register", "add", "log", "archive",
    "overwrite", "append", "erase", "empty", "clear", "insert",
}

_CONTROL_LEMMAS = {
    "click", "navigate", "close", "cancel", "abort", "resume", "open",
    "launch", "exit", "refresh", "undo", "go",
}

_ERROR_PATTERN = re.compile(
    r"\b(error|fail|invalid|exception|warning|not found|unable|denied|"
    r"reject|unsuccessful|alert|could not|cannot|not valid)\b",
    re.IGNORECASE,
)
_CONFIRM_PATTERN = re.compile(
    r"\b(confirm|success|complete|done|acknowledge|notification)\w*\b",
    re.IGNORECASE,
)

_UI_ELEMENTS = re.compile(
    r"\b(button|menu|tab|screen|page|option|sub-option|link|form|icon|field|"
    r"checkbox|dropdown|dialog|modal|window|panel|bar|interface|section)\b",
    re.IGNORECASE,
)

_PERSIST_MARKERS = re.compile(
    r"\b(persistent storage|database|db|file|table|record|storage|repository)\b",
    re.IGNORECASE,
)

_SYSTEM_ACTORS = {"system", "the system"}

_RESUME_PATTERN = re.compile(
    r"resumes?\s+(at\s+)?step|flow\s+of\s+events\s+ends|include\s+the\s+use\s+case|"
    r"if\s+at\s+step\s+\d+",
    re.IGNORECASE,
)

_BLANK_FORM_PATTERN = re.compile(
    r"\bshow\w*\s+(a\s+)?(blank\s+)?(form|formatted\s+screen)\b",
    re.IGNORECASE,
)


# ── UC Parser ─────────────────────────────────────────────────────────

def parse_uc_text(text: str) -> dict:
    """Parse a UC prompt into structured fields."""
    lines = text.strip().split("\n")
    name = ""
    primary_actor = ""
    steps = []
    alt_flows = []
    exceptions = []

    section = "header"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith("main scenario"):
            section = "main"
            continue
        elif lower.startswith("alternative flow"):
            section = "alt"
            continue
        elif lower.startswith("exception"):
            section = "exception"
            continue
        elif any(lower.startswith(h) for h in [
            "reference table", "reference diagram", "gui:", "precondition",
        ]):
            section = "skip"
            continue

        if section == "header":
            parts = stripped.split("\t", 1)
            if len(parts) == 2:
                key, val = parts[0].strip().lower(), parts[1].strip()
                if key in ("id", ""):
                    if "–" in val:
                        name = val.split("–", 1)[1].strip()
                    elif "\t" in val:
                        sub = val.split("\t", 1)
                        name = sub[1].strip() if len(sub) > 1 else sub[0].strip()
                    else:
                        name = val
                elif key == "primary actor":
                    primary_actor = val
                elif key.startswith("actor"):
                    primary_actor = val.split(",")[0].strip()
            elif "–" in stripped and not name:
                name = stripped.split("–", 1)[1].strip()
            elif ":" in stripped and "description" in lower:
                pass
            continue

        if section == "skip":
            continue

        step = _parse_step_line(stripped)
        if step is None:
            continue

        if section == "main":
            steps.append(step)
        elif section == "alt":
            alt_flows.append(step)
        elif section == "exception":
            exceptions.append(step)

    if not name and lines:
        for line in lines[:5]:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].strip():
                name = parts[1].strip()
                break

    return {
        "name": name,
        "primary_actor": primary_actor,
        "steps": steps,
        "alt_flows": alt_flows,
        "exceptions": exceptions,
    }


def _parse_step_line(line: str) -> Optional[tuple[str, str, str]]:
    """Parse a step line into (step_id, actor, action)."""
    parts = line.split("\t")

    if len(parts) >= 3:
        step_id = parts[0].strip().rstrip(".")
        actor = parts[1].strip()
        action = parts[2].strip()
        if action:
            return (step_id, actor, action)

    if len(parts) == 2:
        step_id_raw = parts[0].strip()
        action_raw = parts[1].strip()
        step_match = re.match(r"^(\d+[\.\w]*)\s*(.*)", step_id_raw)
        if step_match:
            step_id = step_match.group(1).rstrip(".")
            leftover = step_match.group(2).strip()
            if leftover:
                return (step_id, leftover, action_raw)
            return (step_id, "", action_raw)
        return (step_id_raw.rstrip("."), "", action_raw)

    stripped = line.strip()
    match = re.match(r"^(\d+[\.\w]*)\.\s+(.+)", stripped)
    if match:
        return (match.group(1), "", match.group(2).strip())

    match = re.match(r"^(\d+[\.\w]*)\s+(.+)", stripped)
    if match:
        return (match.group(1).rstrip("."), "", match.group(2))

    return None


# ── COSMIC Expert Annotator ──────────────────────────────────────────

class CosmicAnnotator:
    """Applies COSMIC ISO 19761 expert rules to annotate entities and movements."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def annotate_fp(self, system: str, fp_id: str, text: str, gt: dict) -> FPAnnotation:
        parsed = parse_uc_text(text)
        ann = FPAnnotation(
            system=system,
            fp_id=fp_id,
            fp_text=text,
            cfp_gt=gt["y_true"],
            entry_gt=gt.get("entry_gt"),
            read_gt=gt.get("read_gt"),
            write_gt=gt.get("write_gt"),
            exit_gt=gt.get("exit_gt"),
        )

        all_steps = (
            [(s, "main") for s in parsed["steps"]]
            + [(s, "alt") for s in parsed["alt_flows"]]
            + [(s, "exception") for s in parsed["exceptions"]]
        )

        error_exit_seen = False

        for (step_id, actor, action), section in all_steps:
            if not action:
                continue

            if _RESUME_PATTERN.search(action):
                continue

            offset = text.find(action)

            effective_actor = actor or parsed.get("primary_actor", "User") or "User"

            if not actor:
                effective_actor = self._infer_actor_from_text(action, effective_actor)

            is_system = effective_actor.lower() in _SYSTEM_ACTORS or effective_actor.lower() == "the system"

            doc = self.nlp(action)
            entities = self._extract_entities(action, doc, effective_actor)

            sub_actions = self._split_compound_action(action, doc, effective_actor, is_system)

            for i, (sub_mv, sub_dg, sub_reason) in enumerate(sub_actions):
                step_ann = StepAnnotation(
                    step_id=f"{step_id}" if i == 0 else f"{step_id}.{i+1}",
                    actor=actor,
                    action_text=action,
                    full_text=action,
                    offset_in_fp=offset if offset >= 0 else 0,
                    entities=entities if i == 0 else [],
                )

                if sub_mv == "X" and (
                    _ERROR_PATTERN.search(action) or _CONFIRM_PATTERN.search(action)
                ):
                    step_ann.is_error_confirmation = True
                    if error_exit_seen and _ERROR_PATTERN.search(action):
                        step_ann.movement_type = None
                        step_ann.reasoning = "Error/confirm exit already counted (COSMIC: 1 Exit for all messages)"
                        ann.steps.append(step_ann)
                        continue
                    error_exit_seen = True

                step_ann.movement_type = sub_mv
                step_ann.data_group = sub_dg
                step_ann.functional_user = effective_actor if sub_mv in ("E", "X") else ""
                step_ann.reasoning = sub_reason
                ann.steps.append(step_ann)

        ann.cfp_annotated = sum(1 for s in ann.steps if s.movement_type is not None)
        ann.validation_status = self._validate(ann)
        return ann

    def _infer_actor_from_text(self, action: str, default_actor: str) -> str:
        """Infer the actor from the sentence subject."""
        lower = action.lower()
        if lower.startswith("the system ") or lower.startswith("system "):
            return "System"
        known = {
            "the actor": default_actor,
            "the user": default_actor,
        }
        for prefix, actor in known.items():
            if lower.startswith(prefix + " "):
                return actor
        return default_actor

    def _split_compound_action(
        self, action: str, doc, actor: str, is_system: bool,
    ) -> list[tuple[Optional[str], str, str]]:
        """Split compound actions (e.g., 'searches and displays') into multiple movements."""
        verbs = []
        for tok in doc:
            if tok.pos_ == "VERB" and tok.dep_ in ("ROOT", "conj", "advcl", "xcomp", "ccomp"):
                verbs.append(tok)

        if len(verbs) <= 1:
            entities = self._extract_entities(action, doc, actor)
            mv, dg, reason = self._classify_movement(
                action, doc, actor, is_system, "main", entities,
            )
            return [(mv, dg, reason)]

        results = []
        for verb in verbs:
            lemma = verb.lemma_.lower()

            objs = []
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    objs.append(child.text)
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objs.append(grandchild.text)
            dg = objs[0] if objs else ""

            mv_type = None
            reason = ""

            if is_system:
                if lemma in _WRITE_LEMMAS:
                    mv_type, reason = "W", f"System '{lemma}' → Write (compound)"
                elif lemma in _READ_LEMMAS:
                    if lemma in ("validate", "check", "verify") and not _PERSIST_MARKERS.search(action):
                        reason = f"System '{lemma}' → internal validation (compound)"
                    else:
                        mv_type, reason = "R", f"System '{lemma}' → Read (compound)"
                elif lemma in _EXIT_LEMMAS:
                    mv_type, reason = "X", f"System '{lemma}' → Exit (compound)"
                elif lemma in _ENTRY_LEMMAS:
                    mv_type, reason = "E", f"System receives '{lemma}' → Entry (compound)"
                elif lemma in _CONTROL_LEMMAS:
                    reason = f"System control '{lemma}' (compound)"
            else:
                if lemma in _ENTRY_LEMMAS or lemma == "select" or lemma == "choose" or lemma == "click":
                    if lemma == "click" and not objs:
                        reason = f"User click without domain data (compound)"
                    else:
                        mv_type, reason = "E", f"User '{lemma}' → Entry (compound)"
                elif lemma in _EXIT_LEMMAS:
                    mv_type, reason = "X", f"User '{lemma}' → Exit from system (compound)"
                elif lemma in _CONTROL_LEMMAS:
                    reason = f"User control '{lemma}' (compound)"

            if mv_type is not None:
                results.append((mv_type, dg, reason))

        if not results:
            entities = self._extract_entities(action, doc, actor)
            mv, dg, reason = self._classify_movement(
                action, doc, actor, is_system, "main", entities,
            )
            return [(mv, dg, reason)]

        return results

    def _extract_entities(self, text: str, doc, actor: str) -> list[EntitySpan]:
        """Extract NER-style entity spans from step text."""
        entities: list[EntitySpan] = []

        if actor and actor.lower() not in ("", "system"):
            idx = text.lower().find(actor.lower())
            if idx >= 0:
                entities.append(EntitySpan(idx, idx + len(actor), text[idx:idx + len(actor)], "FUNC_USER"))
        for match in re.finditer(r"\b(?:the\s+)?system\b", text, re.IGNORECASE):
            entities.append(EntitySpan(match.start(), match.end(), match.group(), "FUNC_USER"))

        for match in _UI_ELEMENTS.finditer(text):
            entities.append(EntitySpan(match.start(), match.end(), match.group(), "UI_ELEMENT"))

        for match in _PERSIST_MARKERS.finditer(text):
            entities.append(EntitySpan(match.start(), match.end(), match.group(), "PERSIST_STORE"))

        root_verb = None
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                root_verb = tok
                break
        if root_verb is None:
            for tok in doc:
                if tok.pos_ == "VERB":
                    root_verb = tok
                    break
        if root_verb:
            start = root_verb.idx
            end = start + len(root_verb.text)
            entities.append(EntitySpan(start, end, root_verb.text, "ACTION_VERB"))

        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "nsubj":
                continue
            if chunk.root.dep_ in ("dobj", "pobj", "attr", "nsubjpass", "conj", "appos"):
                chunk_text = chunk.text
                if _UI_ELEMENTS.search(chunk_text) and not any(
                    w in chunk_text.lower() for w in ("data", "detail", "list", "result", "code", "name", "field")
                ):
                    continue
                start = chunk.start_char
                end = chunk.end_char
                entities.append(EntitySpan(start, end, chunk_text, "DATA_OBJ"))

        for tok in doc:
            if tok.dep_ in ("dobj", "pobj") and not any(
                e.start <= tok.idx < e.end and e.label == "DATA_OBJ" for e in entities
            ):
                entities.append(EntitySpan(tok.idx, tok.idx + len(tok.text), tok.text, "DATA_OBJ"))

        for match in re.finditer(
            r"\b(if\s+(?:at\s+)?(?:no|the)?[^,\.]+?(?:,|$))",
            text, re.IGNORECASE,
        ):
            entities.append(EntitySpan(match.start(), match.end(), match.group(), "CONDITION"))

        return entities

    def _classify_movement(
        self, action: str, doc, actor: str, is_system: bool, section: str,
        entities: list[EntitySpan],
    ) -> tuple[Optional[str], str, str]:
        """Classify the COSMIC movement type for a step."""
        action_lower = action.lower()

        root_lemma = ""
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                root_lemma = tok.lemma_.lower()
                break
        if not root_lemma:
            for tok in doc:
                if tok.pos_ == "VERB":
                    root_lemma = tok.lemma_.lower()
                    break

        data_objs = [e for e in entities if e.label == "DATA_OBJ"]
        dg = data_objs[0].text if data_objs else ""

        if root_lemma in _CONTROL_LEMMAS and root_lemma not in ("select", "choose"):
            has_data_verb = any(
                tok.lemma_.lower() in (_ENTRY_LEMMAS | _EXIT_LEMMAS | _READ_LEMMAS | _WRITE_LEMMAS)
                for tok in doc if tok.pos_ == "VERB" and tok.dep_ != "ROOT"
            )
            if not has_data_verb:
                has_domain_obj = data_objs and not all(
                    _UI_ELEMENTS.search(o.text) for o in data_objs
                )
                if root_lemma == "click" and has_domain_obj:
                    if not is_system:
                        return "E", dg, f"User clicks on domain data '{dg}' → Entry"
                if any(e.label == "UI_ELEMENT" for e in entities) and not has_domain_obj:
                    return None, "", f"Control action '{root_lemma}' on UI element, no domain data"
                return None, "", f"Control action '{root_lemma}', no data movement"

        if is_system and _BLANK_FORM_PATTERN.search(action):
            return None, "", "System shows blank form — no data movement (COSMIC Guide §3.5)"

        if is_system and (_ERROR_PATTERN.search(action) or _CONFIRM_PATTERN.search(action)):
            return "X", "Messages", "Error/confirmation message → Exit (COSMIC: 1 Exit for all msgs)"

        if root_lemma == "browse" or (root_lemma == "view" and not is_system):
            return None, "", f"User browsing/viewing results — no data movement"

        if is_system:
            if root_lemma in _WRITE_LEMMAS:
                conj_objs = self._get_coordinated_objects(doc)
                if len(conj_objs) > 1:
                    return "W", ", ".join(conj_objs), f"System '{root_lemma}' multiple objects → Write (compound)"
                return "W", dg, f"System '{root_lemma}' → Write to persistent storage"

            if root_lemma in _READ_LEMMAS:
                if root_lemma in ("validate", "check", "verify"):
                    if _PERSIST_MARKERS.search(action) or any(
                        w in action_lower for w in ("against", "in the", "from the", "database")
                    ):
                        return "R", dg, f"System '{root_lemma}' with storage reference → Read"
                    return None, "", f"System '{root_lemma}' without storage reference → internal validation"
                return "R", dg, f"System '{root_lemma}' → Read from persistent storage"

            if root_lemma in _EXIT_LEMMAS:
                conj_objs = self._get_coordinated_objects(doc)
                if len(conj_objs) > 1:
                    return "X", ", ".join(conj_objs), f"System '{root_lemma}' multiple objects → Exit (compound)"
                return "X", dg, f"System '{root_lemma}' → Exit to functional user"

            if root_lemma in _ENTRY_LEMMAS:
                return "E", dg, f"System receives data '{root_lemma}' → Entry (unusual for system actor)"

            all_verbs = [tok.lemma_.lower() for tok in doc if tok.pos_ == "VERB"]
            for v in all_verbs:
                if v in _WRITE_LEMMAS:
                    return "W", dg, f"System secondary verb '{v}' → Write"
                if v in _READ_LEMMAS and v not in ("validate", "check", "verify"):
                    return "R", dg, f"System secondary verb '{v}' → Read"
                if v in _EXIT_LEMMAS:
                    return "X", dg, f"System secondary verb '{v}' → Exit"

            if any(w in action_lower for w in ("send", "respond", "signal")):
                return "X", dg, f"System sends/responds → Exit"

            return None, "", f"System action with unrecognized verb '{root_lemma}'"

        else:
            if root_lemma == "select" or root_lemma == "choose":
                if any(e.label == "UI_ELEMENT" for e in entities):
                    if data_objs and not all(
                        _UI_ELEMENTS.search(o.text) for o in data_objs
                    ):
                        return "E", dg, f"User selects domain data → Entry"
                    return None, "", f"User selects UI element → no data movement"
                return "E", dg, f"User '{root_lemma}' → Entry"

            if root_lemma in _ENTRY_LEMMAS:
                if root_lemma == "press" and any(e.label == "UI_ELEMENT" for e in entities):
                    return None, "", f"User presses UI element → no data movement (control)"
                return "E", dg, f"User '{root_lemma}' → Entry"

            if root_lemma in _EXIT_LEMMAS:
                return "X", dg, f"Non-system actor '{actor}' receives/displays data → Exit from system"

            if root_lemma in _CONTROL_LEMMAS:
                if any(tok.lemma_.lower() in _ENTRY_LEMMAS for tok in doc if tok.pos_ == "VERB"):
                    return "E", dg, f"User control+entry verb → Entry"
                return None, "", f"User control action '{root_lemma}'"

            all_verbs = [tok.lemma_.lower() for tok in doc if tok.pos_ == "VERB"]
            for v in all_verbs:
                if v in _ENTRY_LEMMAS and v != "press":
                    return "E", dg, f"User secondary verb '{v}' → Entry"

            if any(w in action_lower for w in ("send", "signal", "reply", "replie", "respond")):
                return "E", dg, f"External actor sends/signals/replies → Entry"

            return None, "", f"User action with unrecognized verb '{root_lemma}'"

    def _get_coordinated_objects(self, doc) -> list[str]:
        """Extract coordinated objects (X, Y, and Z) from compound sentences."""
        objects = []
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ("dobj", "pobj", "attr", "conj", "nsubjpass"):
                if chunk.root.dep_ != "nsubj":
                    objects.append(chunk.text)
        if len(objects) <= 1:
            for tok in doc:
                if tok.dep_ == "conj" and tok.head.dep_ in ("dobj", "pobj"):
                    if tok.text not in objects:
                        objects.append(tok.text)
        return objects

    def _validate(self, ann: FPAnnotation) -> str:
        """Validate annotation against ground truth."""
        if ann.cfp_annotated == ann.cfp_gt:
            return "EXACT_MATCH"

        delta = ann.cfp_annotated - ann.cfp_gt
        sign = "+" if delta > 0 else ""
        msg = f"MISMATCH: annotated={ann.cfp_annotated}, gt={ann.cfp_gt} ({sign}{delta})"

        if ann.entry_gt is not None:
            e = sum(1 for s in ann.steps if s.movement_type == "E")
            r = sum(1 for s in ann.steps if s.movement_type == "R")
            w = sum(1 for s in ann.steps if s.movement_type == "W")
            x = sum(1 for s in ann.steps if s.movement_type == "X")
            msg += f" [E:{e}vs{ann.entry_gt}, R:{r}vs{ann.read_gt}, W:{w}vs{ann.write_gt}, X:{x}vs{ann.exit_gt}]"

        return msg


# ── I/O ──────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Load GT from COSMET comparison_results.csv."""
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fp_id = row["fp"]
            entry = {"y_true": int(row["y_true"])}
            if "entry_gt" in row:
                entry["entry_gt"] = int(row["entry_gt"])
                entry["read_gt"] = int(row["read_gt"])
                entry["write_gt"] = int(row["write_gt"])
                entry["exit_gt"] = int(row["exit_gt"])
            gt[fp_id] = entry
    return gt


def save_annotations(annotations: list[FPAnnotation], output_path: Path):
    """Save annotations as JSONL."""
    with open(output_path, "w") as f:
        for ann in annotations:
            record = {
                "system": ann.system,
                "fp_id": ann.fp_id,
                "cfp_gt": ann.cfp_gt,
                "cfp_annotated": ann.cfp_annotated,
                "validation_status": ann.validation_status,
                "entry_gt": ann.entry_gt,
                "read_gt": ann.read_gt,
                "write_gt": ann.write_gt,
                "exit_gt": ann.exit_gt,
                "steps": [],
            }
            for step in ann.steps:
                step_rec = {
                    "step_id": step.step_id,
                    "actor": step.actor,
                    "action_text": step.action_text,
                    "movement_type": step.movement_type,
                    "data_group": step.data_group,
                    "functional_user": step.functional_user,
                    "is_error_confirmation": step.is_error_confirmation,
                    "reasoning": step.reasoning,
                    "entities": [
                        {"start": e.start, "end": e.end, "text": e.text, "label": e.label}
                        for e in step.entities
                    ],
                }
                record["steps"].append(step_rec)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COSMIC NER Annotator")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing annotations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotator = CosmicAnnotator()

    total_fps = 0
    total_exact = 0
    total_over = 0
    total_under = 0
    all_errors: list[tuple[str, str, int, int]] = []

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")

        gt = load_ground_truth(cfg["gt_csv"])
        prompt_dir = cfg["prompts"]

        annotations: list[FPAnnotation] = []

        fp_files = sorted(prompt_dir.glob("FP*.txt"), key=lambda p: _fp_sort_key(p.stem))

        for fp_file in fp_files:
            fp_id = fp_file.stem
            if fp_id not in gt:
                continue

            text = fp_file.read_text(encoding="utf-8")
            ann = annotator.annotate_fp(sys_name, fp_id, text, gt[fp_id])
            annotations.append(ann)

            status_icon = "✅" if ann.validation_status == "EXACT_MATCH" else "⚠️"
            print(f"  {status_icon} {fp_id}: annotated={ann.cfp_annotated} gt={ann.cfp_gt} → {ann.validation_status}")

            total_fps += 1
            if ann.validation_status == "EXACT_MATCH":
                total_exact += 1
            elif ann.cfp_annotated > ann.cfp_gt:
                total_over += 1
                all_errors.append((f"{sys_name}/{fp_id}", ann.validation_status, ann.cfp_annotated, ann.cfp_gt))
            else:
                total_under += 1
                all_errors.append((f"{sys_name}/{fp_id}", ann.validation_status, ann.cfp_annotated, ann.cfp_gt))

        output_path = OUTPUT_DIR / f"{sys_name}.jsonl"
        save_annotations(annotations, output_path)
        print(f"  → Saved {len(annotations)} annotations to {output_path.name}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total FPs: {total_fps}")
    print(f"  Exact match: {total_exact} ({100*total_exact/total_fps:.1f}%)")
    print(f"  Over-count: {total_over}")
    print(f"  Under-count: {total_under}")
    print(f"  MAE: {sum(abs(e[2]-e[3]) for e in all_errors)/total_fps:.3f}")

    if all_errors:
        print(f"\n  Mismatches:")
        for fp, status, ann, gt_val in sorted(all_errors, key=lambda x: abs(x[2]-x[3]), reverse=True)[:20]:
            print(f"    {fp}: {status}")


def _fp_sort_key(stem: str) -> int:
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main()
