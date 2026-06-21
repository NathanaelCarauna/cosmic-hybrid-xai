"""Smart Pipeline Normalizer — Router + Step Decomposer + SLM + Correction.

Addresses the three main weaknesses of E11:
1. Dense steps with multiple implicit operations (decomposer)
2. Different input types need different strategies (router)
3. Systematic under-counting on complex UCs (correction)

Pipeline:
  Input → Router → [Classify complexity] →
    Simple sentence → NLP heuristic
    UC name only   → NLP heuristic
    Simple UC      → SLM direct (v2 prompt)
    Complex UC     → Decompose steps → SLM (v2 prompt) → Correction
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


# ---------------------------------------------------------------------------
# Step Decomposer — splits dense steps into atomic sub-steps
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(
    r'\.\s+(?:The\s+system\s+|The\s+user\s+|For\s+each\s+)',
)

_AND_VERB = re.compile(
    r'\s+and\s+(?:the\s+system\s+)?(?=(?:searches?|reads?|retriev|checks?|loads?|fetch|'
    r'stores?|saves?|updates?|deletes?|removes?|creates?))',
    re.IGNORECASE,
)


def decompose_step(action: str) -> list[str]:
    """Split a dense UC step into atomic sub-steps.

    Three decomposition strategies applied in order:
    1. Sentence-boundary split: "stores X. The system reads Y" → 2 parts
    2. And-verb split: "reads X and stores Y" → 2 parts
    3. Noun enumeration: "searches for X, Y, and Z" → 3 parts
    """
    sentence_parts = _SENTENCE_BOUNDARY.split(action)
    if len(sentence_parts) > 1:
        fragments = _SENTENCE_BOUNDARY.findall(action)
        result = [sentence_parts[0].strip().rstrip('.')]
        for frag, part in zip(fragments, sentence_parts[1:]):
            result.append((frag.strip() + ' ' + part.strip()).rstrip('.'))
        result = [r for r in result if r and len(r) > 10]
        if len(result) >= 2:
            return result

    parts = _AND_VERB.split(action)
    if len(parts) >= 2:
        result = []
        for p in parts:
            p = p.strip().rstrip('.')
            if p and len(p) > 10:
                result.append(p)
        if len(result) >= 2:
            return result

    verb_match = re.match(
        r'(.*?(?:searches?\s+for|reads?\s+(?:from\s+\w+\s+storage\s+)?)'
        r'(?:the\s+)?)',
        action, re.IGNORECASE,
    )
    if verb_match:
        prefix = verb_match.group(1).strip()
        remainder = action[len(prefix):].strip()
        nouns = re.split(r',\s*(?:the\s+)?(?:and\s+(?:the\s+)?)?', remainder)
        nouns = [n.strip().rstrip('.') for n in nouns if n.strip() and len(n.strip()) > 2]
        if len(nouns) >= 2:
            return [f"{prefix} {n}" for n in nouns]

    return [action]


def decompose_uc_text(text: str) -> str:
    """Rewrite a UC text by decomposing dense steps into multiple steps.

    Modifies the tab-separated step lines in-place, expanding dense steps
    into multiple lines with sub-step IDs (e.g., 2 → 2a, 2b, 2c).
    """
    lines = text.split('\n')
    new_lines = []
    in_steps = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith('MAIN SCENARIO') or stripped.lower().startswith('alternative flow') or stripped.lower().startswith('exception'):
            in_steps = True
            new_lines.append(line)
            continue

        if in_steps and '\t' in stripped:
            parts = stripped.split('\t', 2)
            if len(parts) >= 3:
                step_id, actor, action = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                step_id, actor, action = parts[0], '', parts[1]
            else:
                new_lines.append(line)
                continue

            sub_actions = decompose_step(action)
            if len(sub_actions) > 1:
                for i, sub in enumerate(sub_actions):
                    sub_id = f"{step_id}{'abcdefghij'[i]}" if i < 10 else f"{step_id}_{i}"
                    new_lines.append(f"{sub_id}\t{actor}\t{sub}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


# ---------------------------------------------------------------------------
# Smart Router
# ---------------------------------------------------------------------------

class InputComplexity:
    SIMPLE_SENTENCE = "simple_sentence"
    UC_NAME = "uc_name"
    SIMPLE_UC = "simple_uc"
    COMPLEX_UC = "complex_uc"


def classify_input(text: str, source_format: SourceFormat) -> str:
    """Classify input complexity for routing."""
    if source_format == SourceFormat.SENTENCE:
        return InputComplexity.SIMPLE_SENTENCE
    if source_format == SourceFormat.UC_NAME:
        return InputComplexity.UC_NAME
    if source_format == SourceFormat.USER_STORY:
        return InputComplexity.SIMPLE_SENTENCE

    fields = BaseNormalizer.parse_uc_fields(text)
    n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])
    word_count = len(text.split())

    has_dense_steps = False
    for _, _, action in fields["steps"]:
        sub = decompose_step(action)
        if len(sub) >= 2:
            has_dense_steps = True
            break

    if n_steps >= 10 or word_count >= 300 or has_dense_steps:
        return InputComplexity.COMPLEX_UC

    return InputComplexity.SIMPLE_UC


# ---------------------------------------------------------------------------
# Pipeline Normalizer
# ---------------------------------------------------------------------------

class PipelineNormalizer(BaseNormalizer):
    """Smart pipeline: Router → Decomposer → SLM → Correction.

    Routes to the optimal normalizer based on input characteristics:
    - Simple sentence / UC name → NLP heuristic (fast, good for simple)
    - Simple UC → SLM direct (best for straightforward UCs)
    - Complex UC → Decompose dense steps + SLM + step-correction
    """

    def __init__(
        self,
        slm_normalizer: BaseNormalizer,
        nlp_normalizer: BaseNormalizer,
        *,
        correction_alpha: float = 0.35,
        correction_beta: float = 8.0,
        correction_gamma: float = 0.6,
        decompose: bool = True,
    ):
        self.slm = slm_normalizer
        self.nlp = nlp_normalizer
        self.alpha = correction_alpha
        self.beta = correction_beta
        self.gamma = correction_gamma
        self.decompose = decompose
        self._route_log: list[tuple[str, str]] = []

    @property
    def route_log(self):
        return list(self._route_log)

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        complexity = classify_input(text, source_format)
        self._route_log.append((fp_id, complexity))

        if complexity in (InputComplexity.SIMPLE_SENTENCE, InputComplexity.UC_NAME):
            return self.nlp.normalize(text, source_format, fp_id=fp_id, metadata=metadata)

        if complexity == InputComplexity.COMPLEX_UC:
            return self._handle_complex_uc(text, source_format, fp_id, metadata)

        return self.slm.normalize(text, source_format, fp_id=fp_id, metadata=metadata)

    def _handle_complex_uc(self, text, source_format, fp_id, metadata):
        if self.decompose:
            text = decompose_uc_text(text)

        nr = self.slm.normalize(text, source_format, fp_id=fp_id, metadata=metadata)

        fields = self.parse_uc_fields(text)
        n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])
        raw_correction = self.alpha * max(0, n_steps - self.beta)

        slm_cfp = sum(len(fp.data_movements) for fp in nr.functional_processes)
        max_correction = slm_cfp * self.gamma
        correction = min(raw_correction, max_correction)

        if correction > 0.5:
            n_extra = round(correction)
            for fp in nr.functional_processes:
                for i in range(n_extra):
                    fp.data_movements.append(DataMovement(
                        type=MovementType.R,
                        data_group=f"correction_read_{i}",
                        object_of_interest=f"correction_{i}",
                    ))

        return nr
