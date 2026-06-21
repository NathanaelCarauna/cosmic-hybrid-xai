"""Semantic Role Labeling extractor using spaCy dependency parsing.

Extracts Agent-Verb-Theme triples from sentences and decomposes compound
sentences (coordinated objects/verbs) into atomic actions.

This is a lightweight alternative to AllenNLP SRL, optimized for COSMIC
functional size measurement domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy


@dataclass
class SemanticRole:
    agent: str = ""
    agent_span: tuple[int, int] = (0, 0)
    verb: str = ""
    verb_lemma: str = ""
    verb_span: tuple[int, int] = (0, 0)
    themes: list[str] = field(default_factory=list)
    theme_spans: list[tuple[int, int]] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    source_text: str = ""
    is_negated: bool = False


@dataclass
class DecomposedAction:
    agent: str
    verb: str
    verb_lemma: str
    theme: str
    theme_span: tuple[int, int]
    source_text: str
    is_negated: bool = False
    has_persist_marker: bool = False
    has_condition: bool = False
    prep_context: str = ""


_PERSIST_MARKERS = re.compile(
    r"\b(persistent storage|database|db|file|table|record|storage|repository)\b",
    re.IGNORECASE,
)


class SRLExtractor:
    """Extract semantic roles and decompose compound sentences."""

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def extract_roles(self, text: str) -> list[SemanticRole]:
        """Extract Agent-Verb-Theme triples from text."""
        doc = self.nlp(text)
        roles = []

        for sent in doc.sents:
            sent_roles = self._extract_from_sent(sent)
            roles.extend(sent_roles)

        return roles

    def decompose(self, text: str) -> list[DecomposedAction]:
        """Decompose text into atomic Agent-Verb-Theme actions."""
        roles = self.extract_roles(text)
        actions = []

        for role in roles:
            has_persist = bool(_PERSIST_MARKERS.search(text))

            if not role.themes:
                actions.append(DecomposedAction(
                    agent=role.agent,
                    verb=role.verb,
                    verb_lemma=role.verb_lemma,
                    theme="",
                    theme_span=(0, 0),
                    source_text=role.source_text,
                    is_negated=role.is_negated,
                    has_persist_marker=has_persist,
                ))
                continue

            for theme, span in zip(role.themes, role.theme_spans):
                prep = ""
                doc_theme = self.nlp(text)
                for tok in doc_theme:
                    if tok.text.lower() == theme.split()[-1].lower() and tok.head.dep_ == "prep":
                        prep = tok.head.text
                        break

                actions.append(DecomposedAction(
                    agent=role.agent,
                    verb=role.verb,
                    verb_lemma=role.verb_lemma,
                    theme=theme,
                    theme_span=span,
                    source_text=role.source_text,
                    is_negated=role.is_negated,
                    has_persist_marker=has_persist,
                    prep_context=prep,
                ))

        return actions

    def _extract_from_sent(self, sent) -> list[SemanticRole]:
        """Extract roles from a single sentence span."""
        roles = []
        verbs = []

        for tok in sent:
            if tok.pos_ == "VERB" and tok.dep_ in ("ROOT", "conj", "advcl", "xcomp", "ccomp", "relcl"):
                verbs.append(tok)

        for verb in verbs:
            role = SemanticRole(
                verb=verb.text,
                verb_lemma=verb.lemma_.lower(),
                verb_span=(verb.idx, verb.idx + len(verb.text)),
                source_text=sent.text,
            )

            role.is_negated = any(
                child.dep_ == "neg" for child in verb.children
            )

            agent = self._find_agent(verb, sent)
            if agent:
                role.agent = agent.text if hasattr(agent, "text") else str(agent)
                if hasattr(agent, "start_char"):
                    role.agent_span = (agent.start_char, agent.end_char)

            themes, spans = self._find_themes(verb, sent)
            role.themes = themes
            role.theme_spans = spans

            role.modifiers = self._find_modifiers(verb)

            roles.append(role)

        return roles

    def _find_agent(self, verb, sent):
        """Find the agent (subject) of a verb."""
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return self._expand_compound(child, sent)

        if verb.dep_ == "conj":
            head = verb.head
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return self._expand_compound(child, sent)

        return None

    def _find_themes(self, verb, sent) -> tuple[list[str], list[tuple[int, int]]]:
        """Find themes (objects) of a verb, expanding coordinated objects."""
        themes = []
        spans = []

        for child in verb.children:
            if child.dep_ in ("dobj", "attr"):
                base_theme = self._expand_compound(child, sent)
                themes.append(base_theme.text if hasattr(base_theme, "text") else str(base_theme))
                if hasattr(base_theme, "start_char"):
                    spans.append((base_theme.start_char, base_theme.end_char))
                else:
                    spans.append((child.idx, child.idx + len(child.text)))

                for conj in child.conjuncts:
                    conj_theme = self._expand_compound(conj, sent)
                    themes.append(conj_theme.text if hasattr(conj_theme, "text") else str(conj_theme))
                    if hasattr(conj_theme, "start_char"):
                        spans.append((conj_theme.start_char, conj_theme.end_char))
                    else:
                        spans.append((conj.idx, conj.idx + len(conj.text)))

            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        obj = self._expand_compound(grandchild, sent)
                        themes.append(obj.text if hasattr(obj, "text") else str(obj))
                        if hasattr(obj, "start_char"):
                            spans.append((obj.start_char, obj.end_char))
                        else:
                            spans.append((grandchild.idx, grandchild.idx + len(grandchild.text)))

                        for conj in grandchild.conjuncts:
                            conj_obj = self._expand_compound(conj, sent)
                            themes.append(conj_obj.text if hasattr(conj_obj, "text") else str(conj_obj))
                            if hasattr(conj_obj, "start_char"):
                                spans.append((conj_obj.start_char, conj_obj.end_char))
                            else:
                                spans.append((conj.idx, conj.idx + len(conj.text)))

        return themes, spans

    def _expand_compound(self, token, sent):
        """Expand a token to include its compound/amod modifiers (noun chunk)."""
        for chunk in sent.as_doc().noun_chunks:
            if chunk.root.i == token.i:
                return chunk
        return token

    def _find_modifiers(self, verb) -> list[str]:
        """Find adverbial/prepositional modifiers of a verb."""
        mods = []
        for child in verb.children:
            if child.dep_ in ("advmod", "npadvmod"):
                mods.append(child.text)
        return mods
