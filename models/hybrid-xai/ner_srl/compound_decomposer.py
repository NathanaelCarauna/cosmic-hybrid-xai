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

from .language_profile import ENGLISH, LanguageProfile


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

    def __init__(self, nlp=None, profile: LanguageProfile = ENGLISH):
        self.profile = profile
        self.nlp = nlp or spacy.load(profile.spacy_model)

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

        # PT: um modal ("o sistema DEVE receber, consultar e exibir") ancora a
        # coordenação nos INFINITIVOS via xcomp, não em conj do modal. Descemos
        # ao infinitivo. Gated no perfil: profile.modal_verbs é vazio no inglês.
        descended = self._descend_modal(root)

        if descended is root:
            # Caminho original (inglês, e PT sem modal): conj DIRETO do root.
            # Inalterado byte-a-byte → regressão do inglês preservada.
            verbs = [root]
            for tok in sent:
                if tok.dep_ == "conj" and tok.head == root and tok.pos_ == "VERB":
                    verbs.append(tok)
            return verbs

        # Caminho PT-modal: o parser rotula os infinitivos coordenados como
        # advcl ENCADEADO (consultar=advcl→receber; exibir=conj→consultar), não
        # conj flat. Segue a cadeia transitiva conj + advcl (sem marcador
        # subordinativo — "grava SE confirma" não é ação coordenada).
        #
        # Recalibração (Stanza): a travessia parte do ROOT (não só do verbo
        # descido), atravessando verbos modais/leves SEM contá-los, e segue
        # também coord_frontier_extra_deps (xcomp/parataxis) — o verbo de ação
        # real pende de um leve ("consegue MARCAR consulta … e pode DESMARCAR";
        # "recebe lembretes" em parataxis). Vazio no inglês → este branch nem é
        # alcançado (descended is root) e o comportamento fica preservado.
        extra = self.profile.coord_frontier_extra_deps
        verbs = []
        frontier = [root]
        seen = {root.i}
        while frontier:
            v = frontier.pop()
            if v.lemma_ not in self.profile.modal_verbs:
                verbs.append(v)  # modal é atravessado, não vira ação
            for tok in sent:
                if tok.i in seen or tok.pos_ != "VERB" or tok.head != v:
                    continue
                is_coord_advcl = (tok.dep_ == "advcl"
                                  and not self._has_subordinating_mark(tok)
                                  and not self._is_gerund(tok))
                if (tok.dep_ == "conj" or is_coord_advcl
                        or tok.dep_ in extra):
                    seen.add(tok.i)
                    frontier.append(tok)
        return verbs or [descended]

    def _has_subordinating_mark(self, verb) -> bool:
        """advcl com marcador (se/quando/porque/…) é cláusula subordinada — não
        é ação coordenada, não deve virar movimento. EXCEÇÃO: marcador de
        propósito (para/pra) é ação-fim, não subordinação — promovido a ação
        quando o perfil o declara (recalibração de parser)."""
        marks = [c.text.lower() for c in verb.children if c.dep_ == "mark"]
        if not marks:
            return False
        if self.profile.purpose_marks and all(m in self.profile.purpose_marks for m in marks):
            return False  # só marca(s) de propósito → é ação-fim, não subordina
        return True

    def _is_gerund(self, verb) -> bool:
        """Gerúndio ("cadastrar INFORMANDO e-mail") é cláusula de modo/meio, não
        um movimento coordenado — não deve virar ação atômica."""
        return "Ger" in verb.morph.get("VerbForm")

    def _descend_modal(self, root):
        """Desce por uma CADEIA de modais/invólucros (profile.modal_verbs) que
        governam um infinitivo via `xcomp`, até o verbo de ação real — cujas
        conjunções são os movimentos. Ex.: "deve permitir cadastrar" →
        deve→permitir→cadastrar. Sem modal (ou sem xcomp verbal), devolve o
        próprio nó. Inglês: modal_verbs vazio → devolve `root` inalterado."""
        node = root
        for _ in range(3):  # limite da cadeia (deve → permitir → ação)
            if node.lemma_.lower() not in self.profile.modal_verbs:
                break
            inf = next((c for c in node.children
                        if c.dep_ == "xcomp" and c.pos_ == "VERB"), None)
            if inf is None:
                break
            node = inf
        return node

    def _get_subject(self, verb, sent) -> str:
        """Get the subject of a verb, inheriting from the head through conj/xcomp
        chains. Verbos coordenados (conj) e infinitivos de modal-descent (xcomp)
        não carregam sujeito próprio — ele vive no verbo-cabeça / no modal.
        Ex.: em "o sistema deve receber, consultar e exibir", o sujeito
        "o sistema" pende de "deve"; sobe-se consultar→receber(conj)→deve(xcomp)
        até encontrá-lo. Inglês: verbos são ROOT/conj — o sujeito é achado no
        nível 1-2 como antes; o ramo xcomp não é exercitado."""
        node = verb
        for _ in range(4):  # limite de profundidade da cadeia
            for child in node.children:
                if self.profile.is_subject(child):
                    return self._expand_noun(child, sent)
            if node.dep_ in ("conj", "xcomp") and node.head is not node:
                node = node.head
            else:
                break
        return ""

    def _get_objects_with_coordination(self, verb) -> list[str]:
        """Get objects of verb, expanding coordinated nouns."""
        objects = []

        for child in verb.children:
            if self.profile.is_direct_object(child):
                objects.append(self._expand_noun_text(child))
                for conj in child.conjuncts:
                    objects.append(self._expand_noun_text(conj))

            # DÍVIDA (idêntica à documentada em srl_extractor._find_themes): o
            # esquema inglês do spaCy usa prep->pobj para tema via preposição.
            # Em Universal Dependencies (português) a preposição é `case` presa
            # ao substantivo, e o substantivo pende do verbo como `obl`/`nmod` —
            # inversão estrutural, não troca de rótulo. Este ramo é MORTO em PT
            # (UD nunca emite prep/pobj); o caminho `obl` não foi escrito porque
            # exige parses PT à frente (escrevê-lo cego criaria falsa completude).
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        objects.append(self._expand_noun_text(grandchild))
                        for conj in grandchild.conjuncts:
                            objects.append(self._expand_noun_text(conj))

        return objects

    def _expand_noun(self, token, sent) -> str:
        """Expand noun to full noun chunk.

        noun_chunks/as_doc() dependem do syntax iterator e vetores do modelo;
        parsers alternativos (spacy-stanza) podem não os expor. Degrada para o
        token nu — a expansão de nome não afeta o CFP (que vem do léxico de
        verbo). Com pt_core_news_sm o caminho normal é preservado.
        """
        try:
            doc = sent.as_doc()
            for chunk in doc.noun_chunks:
                if chunk.root.i == token.i:
                    return chunk.text
        except (ValueError, NotImplementedError):
            pass
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
