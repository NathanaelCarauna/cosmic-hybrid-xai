"""Semantic Role Labeling extractor using spaCy dependency parsing.

Extracts Agent-Verb-Theme triples from sentences (coordinated objects are
expanded in-place via `_find_themes`). Compound-sentence decomposition into
atomic actions is CompoundDecomposer's job (compound_decomposer.py); this
module used to have its own `decompose()` too, but it was dead code (no
caller — the pipeline only ever calls CompoundDecomposer.decompose) and was
removed in Task 6-bis along with its duplicate persist-marker regex.

This is a lightweight alternative to AllenNLP SRL, optimized for COSMIC
functional size measurement domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import spacy

from .language_profile import ENGLISH, LanguageProfile


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
    has_condition: bool = False


class SRLExtractor:
    """Extract semantic roles and decompose compound sentences."""

    def __init__(self, nlp=None, profile: LanguageProfile = ENGLISH):
        self.profile = profile
        self.nlp = nlp or spacy.load(profile.spacy_model)

    def extract_roles(self, text: str) -> list[SemanticRole]:
        """Extract Agent-Verb-Theme triples from text."""
        doc = self.nlp(text)
        roles = []

        for sent in doc.sents:
            sent_roles = self._extract_from_sent(sent)
            roles.extend(sent_roles)

        return roles

    def _extract_from_sent(self, sent) -> list[SemanticRole]:
        """Extract roles from a single sentence span."""
        roles = []
        verbs = []

        for tok in sent:
            if tok.pos_ == "VERB" and tok.dep_ in self.profile.verb_clause_deps:
                verbs.append(tok)

        for verb in verbs:
            role = SemanticRole(
                verb=verb.text,
                verb_lemma=verb.lemma_.lower(),
                verb_span=(verb.idx, verb.idx + len(verb.text)),
                source_text=sent.text,
            )

            role.is_negated = any(
                self.profile.is_negation(child) for child in verb.children
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
            if self.profile.is_subject(child):
                return self._expand_compound(child, sent)

        if verb.dep_ == "conj":
            head = verb.head
            for child in head.children:
                if self.profile.is_subject(child):
                    return self._expand_compound(child, sent)

        return None

    def _find_themes(self, verb, sent) -> tuple[list[str], list[tuple[int, int]]]:
        """Find themes (objects) of a verb, expanding coordinated objects."""
        themes = []
        spans = []

        for child in verb.children:
            if self.profile.is_direct_object(child):
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

            # Cadeia verbo->prep->pobj do esquema inglês do spaCy. Universal
            # Dependencies NÃO tem equivalente direto para nenhum dos dois
            # rótulos: a preposição vira "case" presa ao substantivo, e o
            # substantivo pende do VERBO (não da preposição) como "obl" ou
            # "nmod" — a estrutura se inverte, não é troca de rótulo. Os
            # predicados abaixo já existem para isolar o seam, mas o caminho
            # português de tema-via-preposição (via obl/nmod) ainda não foi
            # escrito aqui: é responsabilidade da construção do perfil
            # português na Task 9, com parses reais em mãos — não desta task.
            elif self.profile.is_preposition(child):
                for grandchild in child.children:
                    if self.profile.is_prepositional_object(grandchild):
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
        """Expand a token to include its compound/amod modifiers (noun chunk).

        noun_chunks depende do syntax iterator + vetores do modelo; parsers
        alternativos (ex.: spacy-stanza) não o expõem e as_doc() pode estourar.
        Nesse caso, degrada para o token nu — os léxicos de verbo (o que
        importa para o CFP) não dependem do noun chunk.
        """
        try:
            for chunk in sent.as_doc().noun_chunks:
                if chunk.root.i == token.i:
                    return chunk
        except (ValueError, NotImplementedError):
            pass
        return token

    def _find_modifiers(self, verb) -> list[str]:
        """Find adverbial/prepositional modifiers of a verb."""
        mods = []
        for child in verb.children:
            if child.dep_ in ("advmod", "npadvmod"):
                mods.append(child.text)
        return mods
