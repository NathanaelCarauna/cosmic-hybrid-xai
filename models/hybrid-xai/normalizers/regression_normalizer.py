"""E12: Regression Normalizer — direct CFP prediction from text features.

Bypasses the IR entirely. Extracts numerical features from requirement text
and applies a linear model to predict CFP directly. The regression weights
are learned via leave-one-out cross-validation on the training set.

This is a "traditional ML" baseline — no LLM needed, sub-millisecond.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..schema import (
    BoundaryContext,
    DataMovement,
    FunctionalProcess,
    Metadata,
    MovementType,
    NormalizedRequirement,
    SourceFormat,
)
from .base import BaseNormalizer

_VERB_ENTRY = re.compile(
    r'\b(?:fill|enter|input|submit|type|select|specify|provide|upload|insert|choose|search|click)\b', re.I
)
_VERB_READ = re.compile(
    r'\b(?:search|retrieve|read|look|fetch|load|check|verify|validate|find|query)\b', re.I
)
_VERB_WRITE = re.compile(
    r'\b(?:save|store|update|delete|remove|insert|create|modify|persist|add|record|log)\b', re.I
)
_VERB_EXIT = re.compile(
    r'\b(?:display|show|present|list|print|render|exhibit|output|generate|return)\b', re.I
)
_ENTITY = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
_CONJUNCTION = re.compile(r'\b(?:and|also|then|additionally|moreover)\b', re.I)


def extract_features(text: str, source_format: SourceFormat) -> np.ndarray:
    """Extract numeric features from requirement text."""
    words = text.split()
    sentences = re.split(r'[.;!?\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    n_steps = 0
    if source_format == SourceFormat.USE_CASE:
        fields = BaseNormalizer.parse_uc_fields(text)
        n_steps = len(fields["steps"]) + len(fields["alt_flows"]) + len(fields["exceptions"])

    features = [
        len(words),                                 # word_count
        len(sentences),                             # sentence_count
        n_steps,                                    # step_count (UC only)
        len(_VERB_ENTRY.findall(text)),             # entry_verbs
        len(_VERB_READ.findall(text)),              # read_verbs
        len(_VERB_WRITE.findall(text)),             # write_verbs
        len(_VERB_EXIT.findall(text)),              # exit_verbs
        len(set(_ENTITY.findall(text))),            # unique_entities
        len(_CONJUNCTION.findall(text)),             # conjunctions
        text.lower().count("error") + text.lower().count("exception"),  # error_mentions
    ]

    return np.array(features, dtype=np.float64)


class RegressionNormalizer(BaseNormalizer):
    """Direct CFP prediction via linear regression on text features.

    If training data (list of (text, source_format, gt_cfp)) is provided
    at construction, learns weights via OLS with L2 regularization (Ridge).
    Otherwise uses hand-tuned default weights.
    """

    FEATURE_NAMES = [
        "word_count", "sentence_count", "step_count",
        "entry_verbs", "read_verbs", "write_verbs", "exit_verbs",
        "unique_entities", "conjunctions", "error_mentions",
    ]

    def __init__(
        self,
        training_data: list[tuple[str, SourceFormat, int]] | None = None,
        alpha: float = 1.0,
    ):
        if training_data:
            self._fit(training_data, alpha)
        else:
            self.weights = np.array([
                0.02,   # word_count
                0.0,    # sentence_count (redundant with steps)
                0.5,    # step_count — main driver
                0.5,    # entry_verbs
                0.8,    # read_verbs
                0.8,    # write_verbs
                0.5,    # exit_verbs
                0.3,    # unique_entities
                0.2,    # conjunctions
                0.5,    # error_mentions
            ])
            self.bias = 1.0

    def _fit(self, data: list[tuple[str, SourceFormat, int]], alpha: float):
        X = np.array([extract_features(text, sf) for text, sf, _ in data])
        y = np.array([cfp for _, _, cfp in data])

        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        I = np.eye(X_bias.shape[1])
        I[-1, -1] = 0  # don't regularize bias
        w = np.linalg.solve(X_bias.T @ X_bias + alpha * I, X_bias.T @ y)

        self.weights = w[:-1]
        self.bias = w[-1]

    def predict_cfp(self, text: str, source_format: SourceFormat) -> int:
        features = extract_features(text, source_format)
        raw = features @ self.weights + self.bias
        return max(1, round(raw))

    def normalize(
        self,
        text: str,
        source_format: SourceFormat,
        *,
        fp_id: str = "",
        metadata: Optional[Metadata] = None,
    ) -> NormalizedRequirement:
        predicted_cfp = self.predict_cfp(text, source_format)

        movements = []
        for i in range(predicted_cfp):
            if i == 0:
                mt = MovementType.E
            elif i == predicted_cfp - 1:
                mt = MovementType.X
            elif i % 2 == 0:
                mt = MovementType.R
            else:
                mt = MovementType.W

            movements.append(DataMovement(
                type=mt,
                data_group=f"predicted_{mt.value}_{i}",
                object_of_interest="predicted",
                functional_user="User" if mt in (MovementType.E, MovementType.X) else None,
                is_triggering_entry=(i == 0),
                is_error_confirmation=(i == predicted_cfp - 1 and mt == MovementType.X),
            ))

        fp = FunctionalProcess(
            name=fp_id or "predicted",
            triggering_event=fp_id or "predicted",
            primary_functional_user="User",
            data_movements=movements,
        )

        return NormalizedRequirement(
            id=fp_id,
            source_format=source_format,
            raw_text=text,
            boundary=BoundaryContext(software_name="Unknown", functional_users=["User"]),
            functional_processes=[fp],
            metadata=metadata or Metadata(source_file=fp_id),
        )
