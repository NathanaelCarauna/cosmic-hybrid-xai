"""Full NER + SRL + Ontology Pipeline for COSMIC measurement.

End-to-end pipeline: Text → Parse UC → SRL Decompose → NER Tag → Ontology Classify → CFP
Each CFP is traceable to a specific text span in the original requirement.

Usage:
  pipeline = NerOntologyPipeline()
  result = pipeline.measure("UC text here...")
  print(result.cfp)
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import spacy

BASE = Path(__file__).resolve().parent.parent.parent.parent


def _load_module(name: str, path: str):
    """Load module from hyphenated path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_srl_mod = _load_module(
    "srl_extractor",
    str(Path(__file__).parent / "srl_extractor.py"),
)
_decomp_mod = _load_module(
    "compound_decomposer",
    str(Path(__file__).parent / "compound_decomposer.py"),
)
_onto_mod = _load_module(
    "cosmic_ontology",
    str(Path(__file__).parent / "cosmic_ontology.py"),
)
_ucname_mod = _load_module(
    "ucname_estimator",
    str(Path(__file__).parent / "ucname_estimator.py"),
)

SRLExtractor = _srl_mod.SRLExtractor
CompoundDecomposer = _decomp_mod.CompoundDecomposer
CosmicOntology = _onto_mod.CosmicOntology
ActionTuple = _onto_mod.ActionTuple
CosmicEntity = _onto_mod.CosmicEntity
MovementResult = _onto_mod.MovementResult
ucname_estimate = _ucname_mod.estimate

_ALL_KNOWN_VERB_LEMMAS = (
    _onto_mod._ENTRY_VERBS | _onto_mod._EXIT_VERBS | _onto_mod._READ_VERBS
    | _onto_mod._WRITE_VERBS | _onto_mod._CONTROL_VERBS | _onto_mod._AMBIGUOUS_VERBS
)


@dataclass
class PipelineResult:
    fp_id: str
    raw_text: str
    cfp: int
    movements: list[MovementResult]
    n_entries: int = 0
    n_reads: int = 0
    n_writes: int = 0
    n_exits: int = 0

    @property
    def movement_breakdown(self) -> dict[str, int]:
        return {"E": self.n_entries, "R": self.n_reads, "W": self.n_writes, "X": self.n_exits}


_SYSTEM_ACTORS = {"system", "the system"}

_RESUME_RE = re.compile(
    r"resumes?\s+(at\s+)?step|include\s+the\s+use\s+case|"
    r"(?:if\s+)?(?:at|in)\s+step\s+\d+\s+of\s+the\s+main|"
    r"if\s+(?:at\s+|in\s+)?step\s+\d+|"
    r"the\s+use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)|"
    r"use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)",
    re.IGNORECASE,
)

# Padrão "valida/checa ... e grava/salva X" (conserto de cancelamento, alta precisão).
# Verbo de validação seguido (na MESMA sentença, sem ponto) de verbo de escrita.
_VALIDATE_WRITE_RE = re.compile(
    r"\b(?:check|checks|checked|validate|validates|validated|verif\w+|ensure|ensures)\b"
    r"[^.]*\b(?:record|records|recorded|save|saves|saved|store|stores|stored|"
    r"persist\w*|registers?|registered)\b",
    re.IGNORECASE,
)
_WRITE_OBJ_RE = re.compile(
    r"\b(?:record|records|recorded|save|saves|saved|store|stores|stored|"
    r"persist\w*|registers?|registered)\b\s+"
    r"(?:the\s+|a\s+|an\s+|new\s+)*([a-z][\w'\- ]*?)"
    r"(?:\s+(?:in|to|into|on)\b|[.,;:]|$)",
    re.IGNORECASE,
)

# Read implícito na exibição (escopo: formato de SENTENÇA ÚNICA, sentence_mode).
# Princípio COSMIC: exibir um grupo de dados ARMAZENADO exige um Read antes do
# Exit. Numa sentença isolada (cada uma é um processo funcional próprio, sem
# Entry/Read anterior no mesmo processo) esse Read precisa ser contado; num UC
# multi-passo o dado pode já ter sido lido/digitado antes — por isso a regra é
# gateada por sentence_mode (não dispara em De Vito). Validada held-out (Zenodo
# 70/30, seed 42): MAE 0,374→0,345, SA +0,091→+0,161. Ver
# resultados-otimizacao-zenodo-sentencas.md.
_DISPLAY_READ_RE = re.compile(
    r"\b(?:show|shows|display|displays|displayed|list|lists|listed|"
    r"present|presents)\b",
    re.IGNORECASE,
)
# Objetos que NÃO são grupo de dados armazenado (mensagem/controle/artefato
# recém-gerado) — exibi-los é só Exit, sem Read. Excluem falsos positivos.
_DISPLAY_MSG_RE = re.compile(
    r"\b(?:message|notification|alert|confirmation|error|warning|popup|pop-up|"
    r"page|button|interface|tab|menu|screen|dialog|email|e-mail|homepage|"
    r"home\s+page|code|proof)\b",
    re.IGNORECASE,
)

_SHORT_INPUT_MAX_TOKENS = 10

_SHORT_KEYWORD_CFP = {
    "crud": 17, "manage": 12,
    "view": 4, "display": 4, "list": 4, "show": 4, "browse": 4, "print": 5,
    "add": 5, "create": 5, "register": 5, "insert": 5, "enter": 5, "import": 5,
    "delete": 4, "remove": 4,
    "edit": 6, "update": 6, "modify": 6, "change": 6,
    "login": 3, "log": 3, "authenticate": 3, "sign": 3,
    "search": 4, "find": 4, "filter": 4, "query": 4, "look": 4,
    "assign": 5, "approve": 4, "reject": 4, "accept": 4,
    "upload": 5, "download": 4, "export": 4,
    "send": 4, "receive": 4, "notify": 4, "inform": 4,
    "check": 4, "verify": 4, "validate": 4,
    "generate": 5, "report": 4, "calculate": 5, "compute": 5,
    "order": 6, "purchase": 6, "pay": 5,
    "run": 5, "execute": 5, "process": 5,
    "define": 6, "configure": 5, "select": 4, "provide": 5,
    "access": 4, "mark": 4, "unblock": 4, "block": 4,
    "transfer": 5, "set": 5,
}


class NerOntologyPipeline:
    """NER + SRL + Ontology pipeline for COSMIC functional size measurement."""

    def __init__(self, ner_model_path: Optional[str] = None,
                 use_ner: bool = False, use_short_input_estimate: bool = True,
                 inject_implicit_reads: bool = False, improved_dedup: bool = False,
                 dedup_by_object: bool = False, fix_validate_write: bool = True):
        # use_ner default False: a ablação (achado-ner-decorativo.md) mostrou que
        # o NER treinado NÃO altera o CFP (a ontologia, com regex de persistência/
        # UI + spaCy, faz o trabalho); desligá-lo é idêntico ou marginalmente
        # melhor e mais simples/rápido. use_ner=True ainda disponível p/ reprodução.
        self.nlp = spacy.load("en_core_web_sm")
        self.use_short_input_estimate = use_short_input_estimate
        # dedup_by_object: dentro de um passo, deduplica acoes por (verbo, objeto)
        # em vez de so verbo. Corrige a subcontagem de Writes em passos com
        # multiplas persistencias de objetos distintos ("stores X ... stores Y").
        # A dedup COSMIC correta (por grupo de dados) segue na ontologia.
        self.dedup_by_object = dedup_by_object
        # fix_validate_write: conserto de alta precisão dos "cancelamentos" — passo
        # de sistema "valida/checa ... e grava/salva X" produz 1 Write de X (a
        # validação de dado digitado é interna). Corrige a decomposição (recall de
        # Write, F1 por movimento) sem alterar o total. Ver resultados-fidelidade-*.
        self.fix_validate_write = fix_validate_write

        # Correção (2026-06-16): use_ner=False realmente desliga o NER. Antes,
        # `ner_path = ner_model_path or default` fazia None recair no default,
        # então a flag --no-ner do eval nunca desligava o NER.
        if use_ner:
            ner_path = ner_model_path or str(BASE / "models" / "cosmic-ner" / "best")
            try:
                self.ner_nlp = spacy.load(ner_path)
            except OSError:
                self.ner_nlp = None
        else:
            self.ner_nlp = None

        self.srl = SRLExtractor(nlp=self.nlp)
        self.decomposer = CompoundDecomposer(nlp=self.nlp)
        self.ontology = CosmicOntology(
            inject_implicit_reads=inject_implicit_reads,
            improved_dedup=improved_dedup)

    def measure(self, text: str, fp_id: str = "",
                uc_name: bool = False,
                sentence_mode: bool = False) -> PipelineResult:
        """Measure COSMIC CFP for a requirement text.

        uc_name=True roteia para o estimador per-formato de NOMES DE CASO DE USO
        (ucname_estimator). O roteamento por formato é responsabilidade do M1
        (que conhece o formato do item) — não se adivinha pelo texto, porque
        nome de UC e sentença curta são ambíguos na superfície.
        """
        if uc_name and self.use_short_input_estimate:
            return self._ucname_result(text, fp_id)

        parsed = self._parse_uc(text)
        primary_actor = parsed["primary_actor"]
        n_main_steps = len(parsed["steps"])
        all_steps = (
            [(s, "main") for s in parsed["steps"]]
            + [(s, "alt") for s in parsed["alt_flows"]]
            + [(s, "exception") for s in parsed["exceptions"]]
        )

        action_tuples: list = []

        for (step_id, actor, action_text), section in all_steps:
            if not action_text:
                continue
            if _RESUME_RE.search(action_text):
                continue

            effective_actor = actor or primary_actor or "User"
            effective_actor = self._infer_actor(action_text, effective_actor)

            if self.fix_validate_write:
                ov = self._validate_write_override(action_text, effective_actor, step_id, section)
                if ov is not None:
                    action_tuples.append(ov)
                    continue

            atoms = self.decomposer.decompose(action_text)
            atoms = self._apply_lemma_fallback(atoms, action_text)

            srl_roles = self.srl.extract_roles(action_text)
            ner_entities = self._run_ner(action_text) if self.ner_nlp else []

            negated_lemmas = {r.verb_lemma for r in srl_roles if r.is_negated}

            seen_verbs = set()
            for atom in atoms:
                vl = atom.verb_lemma
                key = ((vl, (atom.object_text or "").strip().lower())
                       if self.dedup_by_object else vl)
                if key in seen_verbs and vl not in ("", None):
                    continue
                seen_verbs.add(key)

                at = ActionTuple(
                    step_id=step_id,
                    agent=effective_actor,
                    verb=atom.verb,
                    verb_lemma=atom.verb_lemma,
                    theme=atom.object_text,
                    source_text=action_text,
                    entities=ner_entities,
                    section=section,
                    is_negated=vl in negated_lemmas,
                )
                action_tuples.append(at)

        movements = self.ontology.classify_actions(
            action_tuples, primary_actor, n_steps=n_main_steps,
        )

        active = [m for m in movements if m.movement_type is not None]
        cfp = len(active)

        if cfp == 0 and n_main_steps == 1 and action_tuples:
            src = action_tuples[0].source_text if action_tuples else ""
            movements.insert(0, MovementResult(
                movement_type=_onto_mod.CMovement.E,
                data_group="triggering_event",
                functional_user="User",
                rule_id="R_MIN_SINGLE_STEP",
                reasoning="Single-step FP with no classified movements → minimum 1 Entry",
                source_step_id=action_tuples[0].step_id if action_tuples else "",
                source_text=src,
                is_triggering=True,
            ))
            cfp = 1
            active = [movements[0]]

        # Read implícito na exibição (formato de sentença única — ver regex acima).
        # Se há Exit, não há Read, e a sentença exibe um grupo de dados armazenado
        # (verbo de exibição + objeto que não é mensagem/controle), conta-se o Read
        # que a busca em storage exige antes do Exit.
        if (sentence_mode
                and any(m.movement_type.value == "X" for m in active)
                and not any(m.movement_type.value == "R" for m in active)
                and _DISPLAY_READ_RE.search(text)
                and not _DISPLAY_MSG_RE.search(text)):
            exit_dg = next((m.data_group for m in active
                            if m.movement_type.value == "X"), "displayed_data")
            read_mov = MovementResult(
                movement_type=_onto_mod.CMovement.R,
                data_group=exit_dg,
                functional_user="System",
                rule_id="R_DISPLAY_IMPLICIT_READ",
                reasoning=("Sentence displays stored data → the functional process "
                           "must Read it from storage before the Exit (single-sentence "
                           "scope: no prior Entry/Read in this process)."),
                source_step_id=active[0].source_step_id if active else "",
                source_text=text,
                is_triggering=False,
            )
            movements.append(read_mov)
            active.append(read_mov)
            cfp += 1

        n_e = sum(1 for m in active if m.movement_type.value == "E")
        n_r = sum(1 for m in active if m.movement_type.value == "R")
        n_w = sum(1 for m in active if m.movement_type.value == "W")
        n_x = sum(1 for m in active if m.movement_type.value == "X")

        return PipelineResult(
            fp_id=fp_id,
            raw_text=text,
            cfp=cfp,
            movements=movements,
            n_entries=n_e,
            n_reads=n_r,
            n_writes=n_w,
            n_exits=n_x,
        )

    def _validate_write_override(self, action_text, actor, step_id, section):
        """Conserto de alta precisão dos cancelamentos: passo de sistema que
        'valida/checa ... e grava/salva X' produz 1 Write de X (a validação do
        dado digitado é interna, sem movimento). Mantém o total e corrige a
        decomposição. Retorna o ActionTuple de Write, ou None se não casa."""
        if str(actor).strip().lower() not in ("system", "the system"):
            return None
        if not _VALIDATE_WRITE_RE.search(action_text):
            return None
        m = _WRITE_OBJ_RE.search(action_text)
        obj = ((m.group(1).strip() if m else "") or "data")[:40]
        return ActionTuple(
            step_id=step_id, agent="system", verb="record", verb_lemma="record",
            theme=obj, source_text=action_text, entities=[], section=section,
            is_negated=False)

    def _apply_lemma_fallback(self, atoms: list, text: str) -> list:
        """When the decomposer extracts a verb lemma not in any known set,
        scan the spaCy doc for a token whose lemma IS recognised.

        Also handles cases where spaCy mistagged a verb as ADJ/NOUN
        (e.g. "System open Module" → "open" tagged ADJ but lemma is
        in _EXIT_VERBS).
        """
        needs_fix = any(
            a.verb_lemma and a.verb_lemma not in _ALL_KNOWN_VERB_LEMMAS
            for a in atoms
        )
        if not needs_fix:
            return atoms

        doc = self.nlp(text)
        candidates = []
        for tok in doc:
            lemma = tok.lemma_.lower()
            if lemma not in _ALL_KNOWN_VERB_LEMMAS:
                continue
            is_verb_pos = tok.pos_ in ("VERB", "AUX")
            is_adj_but_known = tok.pos_ == "ADJ" and lemma in _ALL_KNOWN_VERB_LEMMAS
            if not (is_verb_pos or is_adj_but_known):
                continue
            theme_parts = [
                c.text for c in tok.subtree
                if c != tok and c.dep_ not in ("nsubj", "nsubjpass", "punct")
            ]
            theme = " ".join(theme_parts).strip()
            candidates.append((lemma, tok.text, theme))

        if not candidates:
            return atoms

        fixed = []
        replaced = False
        for atom in atoms:
            if (
                not replaced
                and atom.verb_lemma
                and atom.verb_lemma not in _ALL_KNOWN_VERB_LEMMAS
                and candidates
            ):
                lemma, verb_text, theme = candidates[0]
                from compound_decomposer import AtomicAction
                fixed.append(AtomicAction(
                    subject=atom.subject,
                    verb=verb_text,
                    verb_lemma=lemma,
                    object_text=theme or atom.object_text,
                    full_text=atom.full_text,
                    action_index=atom.action_index,
                ))
                replaced = True
            else:
                fixed.append(atom)
        return fixed

    def _run_ner(self, text: str) -> list:
        """Run custom NER model on text."""
        if self.ner_nlp is None:
            return []
        doc = self.ner_nlp(text)
        return [
            CosmicEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

    def _infer_actor(self, text: str, default: str) -> str:
        lower = text.lower()
        if lower.startswith("the system ") or lower.startswith("system "):
            return "System"
        if lower.startswith("the actor ") or lower.startswith("the user "):
            return default
        return default

    def _ucname_result(self, text: str, fp_id: str) -> PipelineResult:
        """Estimativa per-formato para um NOME DE CASO DE USO (formato Ochodek),
        via ucname_estimator (modelo linear validado dev/teste held-out).

        Nomes de UC não carregam os movimentos de dados, então não há trace
        regra-a-trecho nem diagrama de sequência; a explicação (em
        MovementResult.reasoning) é a contribuição EXATA de cada feature (nível
        operação), fiel por construção."""
        name = text.strip()
        est = ucname_estimate(name)
        mv = MovementResult(
            movement_type=None,
            data_group=name,
            functional_user="User",
            rule_id="R_UCNAME_LINEAR",
            reasoning=est["explanation"],
            source_step_id="1",
            source_text=name,
        )
        return PipelineResult(
            fp_id=fp_id, raw_text=text, cfp=est["cfp"], movements=[mv],
            n_entries=0, n_reads=0, n_writes=0, n_exits=0,
        )

    def _try_short_input_estimate(self, parsed: dict, fp_id: str) -> PipelineResult | None:
        """[SUPERSEDED — não é mais chamado] heurístico antigo por tabela de
        palavras-chave. Roteamento de nomes de UC agora é por formato (uc_name=True
        em measure()), via _ucname_result. Mantido só por referência histórica.

        For ultra-short UC names (≤ N tokens), use keyword-based CFP estimation.

        Uses spaCy POS tagging to distinguish keyword-list UC names
        (e.g. "manage faculties crud") from real sentences
        (e.g. "System displays a page").  If spaCy finds a conjugated
        verb or determiners/auxiliaries the text is treated as a sentence.
        """
        steps = parsed["steps"]
        if len(steps) != 1 or parsed["alt_flows"] or parsed["exceptions"]:
            return None
        text = steps[0][2].strip() if steps else ""
        tokens = text.split()
        if len(tokens) > _SHORT_INPUT_MAX_TOKENS or len(tokens) == 0:
            return None
        if any(c in text for c in ".!?;:"):
            return None

        doc = self.nlp(text)
        if not doc:
            return None
        # Gate de nome de UC: aceita sintagmas curtos começando por verbo/substantivo
        # ("manage exams crud", "view room details"); rejeita sentenças que começam
        # por artigo/pronome/preposição/auxiliar ("the system displays...", "it
        # returns..."), que pertencem ao pipeline de movimentos. Substitui o antigo
        # gate por tabela de palavras-chave (acoplado ao heurístico in-sample).
        if doc[0].pos_ in ("DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART"):
            return None

        # Estimador per-formato validado (linear, dev/teste held-out) que
        # SUBSTITUI a antiga tabela palavra-chave->CFP in-sample. O gate acima
        # (first_lemma in _SHORT_KEYWORD_CFP) é mantido só para decidir QUAIS
        # inputs são nomes de UC; o CFP e a explicação vêm de ucname_estimator.
        est = ucname_estimate(text)
        cfp = est["cfp"]

        mv = MovementResult(
            movement_type=None,
            data_group=text,
            functional_user="User",
            rule_id="R_UCNAME_LINEAR",
            reasoning=est["explanation"],
            source_step_id="1",
            source_text=text,
        )
        return PipelineResult(
            fp_id=fp_id,
            raw_text=text,
            cfp=cfp,
            movements=[mv],
            n_entries=0,
            n_reads=0,
            n_writes=0,
            n_exits=0,
        )

    @staticmethod
    def _estimate_from_lemmas(doc) -> int | None:
        """Estimate CFP using spaCy lemmas matched against the keyword table."""
        lemmas = [tok.lemma_.lower() for tok in doc]
        raw_tokens = [tok.text.lower() for tok in doc]

        matched_cfps: list[int] = []
        matched_indices: set[int] = set()
        for i, lemma in enumerate(lemmas):
            if lemma in _SHORT_KEYWORD_CFP:
                matched_cfps.append(_SHORT_KEYWORD_CFP[lemma])
                matched_indices.add(i)
            elif raw_tokens[i] in _SHORT_KEYWORD_CFP:
                matched_cfps.append(_SHORT_KEYWORD_CFP[raw_tokens[i]])
                matched_indices.add(i)

        if not matched_cfps:
            default_cfp = max(3, len(raw_tokens) * 2)
            return default_cfp

        n_nouns = max(1, len(raw_tokens) - len(matched_indices))
        base = max(matched_cfps)
        if "crud" in raw_tokens:
            base = 17
        noun_multiplier = 1.0 + 0.3 * max(0, n_nouns - 1)
        return max(1, round(base * noun_multiplier))

    def _parse_uc(self, text: str) -> dict:
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

            bare = lower.strip().rstrip(":")
            if bare in ("main scenario", "main flow"):
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
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    if key in ("id", ""):
                        if "\t" in val:
                            sub = val.split("\t", 1)
                            name = sub[1].strip() if len(sub) > 1 else sub[0].strip()
                        else:
                            name = val
                    elif key == "primary actor":
                        primary_actor = val
                    elif key.startswith("actor"):
                        primary_actor = val.split(",")[0].strip()
                    elif key.startswith("secondary actor"):
                        pass
                elif "–" in stripped and not name:
                    name = stripped.split("–", 1)[1].strip()
                continue

            if section == "skip":
                continue

            step = self._parse_step(stripped)
            if step:
                target = steps if section == "main" else alt_flows if section == "alt" else exceptions
                target.append(step)
            elif section in ("main", "alt", "exception"):
                target = steps if section == "main" else alt_flows if section == "alt" else exceptions
                if target and stripped:
                    prev = target[-1]
                    target[-1] = (prev[0], prev[1], prev[2] + " " + stripped)

        return {
            "name": name,
            "primary_actor": primary_actor,
            "steps": steps,
            "alt_flows": alt_flows,
            "exceptions": exceptions,
        }

    def _parse_step(self, line: str) -> Optional[tuple[str, str, str]]:
        parts = line.split("\t")
        if len(parts) >= 3:
            return (parts[0].strip().rstrip("."), parts[1].strip(), parts[2].strip())
        if len(parts) == 2:
            step_match = re.match(r"^(\d+[\.\w]*)\s*(.*)", parts[0].strip())
            if step_match:
                sid = step_match.group(1).rstrip(".")
                leftover = step_match.group(2).strip()
                return (sid, leftover, parts[1].strip())
            return (parts[0].strip().rstrip("."), "", parts[1].strip())
        m = re.match(r"^(\d+[\.\w]*)\.\s+(.+)", line.strip())
        if m:
            return (m.group(1), "", m.group(2).strip())
        m = re.match(r"^(\d+[\.\w]*)\s+(.+)", line.strip())
        if m:
            return (m.group(1).rstrip("."), "", m.group(2))
        return None
