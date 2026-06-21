"""HKO linguistic feature extraction for R3 replication.

Implements the 13-feature vector from Hussain, Kosseim & Ormandjieva (2013):
  10 syntactic features (noun phrases, parentheses, active verbs, etc.)
  3 keyword features (discriminating NN/VB/JJ keywords per class)

Uses spaCy (en_core_web_sm) as a modern substitute for the original
Stanford Parser + Brill's POS tagger.
"""
from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Sequence

import numpy as np
import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


SYNTACTIC_FEATURE_NAMES = [
    "n_noun_phrases",
    "n_parentheses",
    "n_active_verbs",
    "n_tokens_in_parens",
    "n_conjunctions",
    "n_pronouns",
    "n_verb_phrases",
    "n_words",
    "n_sentences",
    "n_hapax",
]

KEYWORD_FEATURE_NAMES = [
    "nn_keywords",
    "vb_keywords",
    "jj_keywords",
]

ALL_FEATURE_NAMES = SYNTACTIC_FEATURE_NAMES + KEYWORD_FEATURE_NAMES


def _count_tokens_in_parens(text: str) -> int:
    """Count whitespace-delimited tokens inside parenthesized spans."""
    total = 0
    for m in re.finditer(r"\(([^)]*)\)", text):
        content = m.group(1).strip()
        if content:
            total += len(content.split())
    return total


def _count_verb_phrases(doc) -> int:
    """Count verb phrases as verbs that head at least one dependent."""
    count = 0
    for token in doc:
        if token.pos_ == "VERB":
            deps = [c for c in token.children if c.dep_ not in ("punct",)]
            if deps:
                count += 1
    return count


def _is_active_verb(token) -> bool:
    """A verb is active if it's not in a passive construction."""
    if token.pos_ != "VERB":
        return False
    for child in token.children:
        if child.dep_ in ("nsubjpass", "auxpass"):
            return False
    return True


def extract_syntactic_features(text: str) -> dict[str, int]:
    """Extract the 10 syntactic features from a single text."""
    nlp = _get_nlp()
    doc = nlp(text)

    words = [t for t in doc if not t.is_punct and not t.is_space]
    word_forms = [t.lower_ for t in words]
    freq = Counter(word_forms)

    return {
        "n_noun_phrases": len(list(doc.noun_chunks)),
        "n_parentheses": text.count("("),
        "n_active_verbs": sum(1 for t in doc if _is_active_verb(t)),
        "n_tokens_in_parens": _count_tokens_in_parens(text),
        "n_conjunctions": sum(1 for t in doc if t.pos_ in ("CCONJ", "SCONJ")),
        "n_pronouns": sum(1 for t in doc if t.pos_ == "PRON"),
        "n_verb_phrases": _count_verb_phrases(doc),
        "n_words": len(words),
        "n_sentences": max(len(list(doc.sents)), 1),
        "n_hapax": sum(1 for w, c in freq.items() if c == 1),
    }


# ---------------------------------------------------------------------------
# Keyword features (Section 5.3 of Hussain 2013)
# ---------------------------------------------------------------------------

_POS_GROUPS = {
    "nn": {"NOUN", "PROPN"},
    "vb": {"VERB"},
    "jj": {"ADJ"},
}


def _tokenize_by_pos(text: str) -> dict[str, list[str]]:
    """Return lists of lemmas grouped by NN/VB/JJ POS categories."""
    nlp = _get_nlp()
    doc = nlp(text)
    result: dict[str, list[str]] = {k: [] for k in _POS_GROUPS}
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        for group_name, pos_set in _POS_GROUPS.items():
            if token.pos_ in pos_set:
                result[group_name].append(token.lemma_.lower())
    return result


def build_keyword_lists(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    top_k: int = 20,
) -> dict[str, list[str]]:
    """Build discriminating keyword lists per POS group from training data.

    Uses log-likelihood ratio between each class and the rest, keeping
    the top_k most discriminating keywords per POS group (union across
    all class pairs). Faithful to HKO Section 5.3.
    """
    unique_labels = sorted(set(labels))
    per_class_pos: dict[str, dict[str, Counter]] = {
        lbl: {g: Counter() for g in _POS_GROUPS} for lbl in unique_labels
    }

    for text, lbl in zip(texts, labels):
        pos_tokens = _tokenize_by_pos(text)
        for group_name, tokens in pos_tokens.items():
            per_class_pos[lbl][group_name].update(tokens)

    keyword_lists: dict[str, list[str]] = {}
    for group_name in _POS_GROUPS:
        total_all = Counter()
        for lbl in unique_labels:
            total_all += per_class_pos[lbl][group_name]

        scored: dict[str, float] = {}
        vocab = set(total_all.keys())

        total_count = sum(total_all.values())
        for word in vocab:
            max_ratio = 0.0
            for lbl in unique_labels:
                c_in = per_class_pos[lbl][group_name].get(word, 0) + 1
                c_out = (total_all[word] - per_class_pos[lbl][group_name].get(word, 0)) + 1
                n_in = sum(per_class_pos[lbl][group_name].values()) + len(vocab)
                n_out = max(total_count - n_in + len(vocab), 1)
                ratio = log(c_in / n_in) - log(c_out / n_out)
                max_ratio = max(max_ratio, abs(ratio))
            scored[word] = max_ratio

        top_words = sorted(scored, key=scored.get, reverse=True)[:top_k]
        keyword_lists[group_name] = top_words

    return keyword_lists


def count_keyword_hits(
    text: str,
    keyword_lists: dict[str, list[str]],
) -> dict[str, int]:
    """Count occurrences of discriminating keywords in text per POS group."""
    pos_tokens = _tokenize_by_pos(text)
    return {
        f"{group_name}_keywords": sum(
            1 for t in pos_tokens.get(group_name, [])
            if t in kw_set
        )
        for group_name, kw_list in keyword_lists.items()
        for kw_set in [set(kw_list)]
    }


def extract_all_features(
    text: str,
    keyword_lists: dict[str, list[str]],
) -> np.ndarray:
    """Return full 13-feature vector for a single text."""
    syn = extract_syntactic_features(text)
    kw = count_keyword_hits(text, keyword_lists)
    return np.array([syn[f] for f in SYNTACTIC_FEATURE_NAMES]
                    + [kw[f] for f in KEYWORD_FEATURE_NAMES], dtype=float)


def extract_feature_matrix(
    texts: Sequence[str],
    keyword_lists: dict[str, list[str]],
) -> np.ndarray:
    """Return (n_samples, 13) feature matrix for a list of texts."""
    rows = [extract_all_features(t, keyword_lists) for t in texts]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Cached / split extraction (avoids redundant NLP calls across CV folds)
# ---------------------------------------------------------------------------

def extract_syntactic_matrix(texts: Sequence[str]) -> np.ndarray:
    """Extract syntactic features for all texts (10-dim). Expensive — call once."""
    rows = []
    for t in texts:
        d = extract_syntactic_features(t)
        rows.append([d[f] for f in SYNTACTIC_FEATURE_NAMES])
    return np.array(rows, dtype=float)


def extract_pos_tokens_batch(texts: Sequence[str]) -> list[dict[str, list[str]]]:
    """Tokenize all texts by POS group. Call once, reuse across folds."""
    return [_tokenize_by_pos(t) for t in texts]


def keyword_features_from_cache(
    pos_tokens_list: list[dict[str, list[str]]],
    keyword_lists: dict[str, list[str]],
) -> np.ndarray:
    """Compute keyword hit counts from cached POS tokens (fast)."""
    rows = []
    for pos_tokens in pos_tokens_list:
        row = []
        for group_name in ("nn", "vb", "jj"):
            kw_set = set(keyword_lists.get(group_name, []))
            hits = sum(1 for t in pos_tokens.get(group_name, []) if t in kw_set)
            row.append(hits)
        rows.append(row)
    return np.array(rows, dtype=float)


def assemble_features(
    syntactic_matrix: np.ndarray,
    keyword_matrix: np.ndarray,
) -> np.ndarray:
    """Combine pre-computed syntactic (10-col) and keyword (3-col) matrices."""
    return np.hstack([syntactic_matrix, keyword_matrix])
