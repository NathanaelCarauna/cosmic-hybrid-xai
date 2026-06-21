"""Compound Sentence Decomposer for COSMIC measurement.

Decomposes compound UC steps into atomic actions by:
  1. Detecting coordinated verbs (e.g., "searches and displays")
  2. Detecting coordinated objects (e.g., "stores X, Y, and Z")
  3. Splitting comma-separated enumerations in themes

Each atomic action maps to at most one COSMIC data movement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import spacy


@dataclass
class AtomicAction:
    subject: str
    verb: str
    verb_lemma: str
    object_text: str
    full_text: str
    action_index: int


class CompoundDecomposer:
    """Split compound UC steps into atomic actions."""

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def decompose(self, text: str) -> list[AtomicAction]:
        """Decompose a UC step into atomic actions."""
        doc = self.nlp(text)
        actions = []
        idx = 0

        for sent in doc.sents:
            root_verbs = self._find_coordinated_verbs(sent)

            for verb in root_verbs:
                subject = self._get_subject(verb, sent)
                objects = self._get_objects_with_coordination(verb)

                if not objects:
                    actions.append(AtomicAction(
                        subject=subject,
                        verb=verb.text,
                        verb_lemma=verb.lemma_.lower(),
                        object_text="",
                        full_text=sent.text,
                        action_index=idx,
                    ))
                    idx += 1
                elif len(objects) == 1:
                    actions.append(AtomicAction(
                        subject=subject,
                        verb=verb.text,
                        verb_lemma=verb.lemma_.lower(),
                        object_text=objects[0],
                        full_text=sent.text,
                        action_index=idx,
                    ))
                    idx += 1
                else:
                    for obj in objects:
                        actions.append(AtomicAction(
                            subject=subject,
                            verb=verb.text,
                            verb_lemma=verb.lemma_.lower(),
                            object_text=obj,
                            full_text=sent.text,
                            action_index=idx,
                        ))
                        idx += 1

        if not actions:
            actions.append(AtomicAction(
                subject="",
                verb="",
                verb_lemma="",
                object_text="",
                full_text=text,
                action_index=0,
            ))

        return actions

    def _find_coordinated_verbs(self, sent) -> list:
        """Find the root verb and any coordinated verbs."""
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break

        if root is None or root.pos_ not in ("VERB", "AUX"):
            for tok in sent:
                if tok.pos_ == "VERB":
                    root = tok
                    break

        if root is None:
            return []

        verbs = [root]
        for tok in sent:
            if tok.dep_ == "conj" and tok.head == root and tok.pos_ == "VERB":
                verbs.append(tok)

        return verbs

    def _get_subject(self, verb, sent) -> str:
        """Get the subject of a verb, inheriting from head if coordinated."""
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return self._expand_noun(child, sent)

        if verb.dep_ == "conj":
            for child in verb.head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return self._expand_noun(child, sent)

        return ""

    def _get_objects_with_coordination(self, verb) -> list[str]:
        """Get objects of verb, expanding coordinated nouns."""
        objects = []

        for child in verb.children:
            if child.dep_ in ("dobj", "attr"):
                objects.append(self._expand_noun_text(child))
                for conj in child.conjuncts:
                    objects.append(self._expand_noun_text(conj))

            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        objects.append(self._expand_noun_text(grandchild))
                        for conj in grandchild.conjuncts:
                            objects.append(self._expand_noun_text(conj))

        return objects

    def _expand_noun(self, token, sent) -> str:
        """Expand noun to full noun chunk."""
        doc = sent.as_doc()
        for chunk in doc.noun_chunks:
            if chunk.root.i == token.i:
                return chunk.text
        return token.text

    def _expand_noun_text(self, token) -> str:
        """Expand a noun token with its direct modifiers."""
        parts = []
        for child in token.children:
            if child.dep_ in ("compound", "amod", "det", "nummod", "poss", "punct"):
                if child.i < token.i:
                    parts.append((child.i, child.text))

        parts.append((token.i, token.text))

        for child in token.children:
            if child.dep_ in ("compound", "amod", "punct") and child.i > token.i:
                parts.append((child.i, child.text))

        parts.sort(key=lambda x: x[0])
        result = " ".join(p[1] for p in parts)
        result = re.sub(r"\s+(['\"\-])\s+", r"\1", result)
        return result
