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
import types
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


def _load_ner_srl_submodule(mod_name: str, register_bare_alias: bool = False):
    """Load a submodule of ner_srl/ that uses a relative import.

    `models/hybrid-xai/` has a hyphen, so it can't be a dotted Python
    package: this file (and every script in the repo) loads its siblings
    ad hoc via `_load_module`, which executes each file as a rootless
    module with no parent package. That breaks `from .language_profile
    import ...` in srl_extractor.py/compound_decomposer.py, because a
    relative import needs a real package registered in sys.modules.

    Fix: register a lightweight synthetic package object (its `__path__`
    pointed at this same directory) so the standard import machinery can
    resolve the relative import on its own. No other behavior changes.

    The synthetic package is deliberately registered under `_ner_srl_shim`
    and NOT under `ner_srl` — `ner_srl` is the name of the *real* package
    (models/hybrid-xai/ner_srl/__init__.py, which does
    `from .srl_extractor import SRLExtractor, ...`). If the shim used that
    same name, it would permanently occupy `sys.modules["ner_srl"]` with an
    empty stub that never runs `__init__.py`; a later `import ner_srl` or
    `from ner_srl import SRLExtractor` in the same process would silently
    hit the incomplete stub from cache and fail with an obscure
    `AttributeError` instead of importing the real package. Using a
    distinct name means the shim can never shadow it.

    register_bare_alias additionally registers `mod_name` itself (e.g.
    "compound_decomposer") in sys.modules, for backward-compatible bare
    `from compound_decomposer import ...` style imports elsewhere in this
    file. Only pass True when there's a confirmed bare-import consumer —
    otherwise this would resurrect the same shadowing risk for that name.
    """
    pkg_name = "_ner_srl_shim"
    if pkg_name not in sys.modules:
        pkg_dir = Path(__file__).resolve().parent
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [str(pkg_dir)]
        pkg_mod.__package__ = pkg_name
        sys.modules[pkg_name] = pkg_mod

    mod = importlib.import_module(f"{pkg_name}.{mod_name}")
    if register_bare_alias:
        sys.modules.setdefault(mod_name, mod)
    return mod


_srl_mod = _load_ner_srl_submodule("srl_extractor")
_decomp_mod = _load_ner_srl_submodule("compound_decomposer", register_bare_alias=True)
_onto_mod = _load_ner_srl_submodule("cosmic_ontology")
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
LanguageProfile = _onto_mod.LanguageProfile
ENGLISH = _onto_mod.ENGLISH


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
    # Sinal de COMPLETUDE para a confiança (só no caminho do produto,
    # infer_implicit=True): True se o verbo de ação casou com uma classe
    # conhecida; False se havia verbo mas nenhum casou (motor "chutando" o
    # processo); None se não houve verbo ou o bloco não rodou. None => neutro.
    action_recognized: bool | None = None

    @property
    def movement_breakdown(self) -> dict[str, int]:
        return {"E": self.n_entries, "R": self.n_reads, "W": self.n_writes, "X": self.n_exits}


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
                 dedup_by_object: bool = False, fix_validate_write: bool = True,
                 profile: LanguageProfile = ENGLISH):
        # use_ner default False: a ablação (achado-ner-decorativo.md) mostrou que
        # o NER treinado NÃO altera o CFP (a ontologia, com regex de persistência/
        # UI + spaCy, faz o trabalho); desligá-lo é idêntico ou marginalmente
        # melhor e mais simples/rápido. use_ner=True ainda disponível p/ reprodução.
        self.profile = profile
        self.nlp = spacy.load(profile.spacy_model)
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

        self.srl = SRLExtractor(nlp=self.nlp, profile=profile)
        self.decomposer = CompoundDecomposer(nlp=self.nlp, profile=profile)
        self.ontology = CosmicOntology(
            inject_implicit_reads=inject_implicit_reads,
            improved_dedup=improved_dedup,
            profile=profile)

    def measure(self, text: str, fp_id: str = "",
                uc_name: bool = False,
                sentence_mode: bool = False,
                infer_implicit: bool = False) -> PipelineResult:
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
            if self.profile.resume_re.search(action_text):
                continue

            effective_actor = actor or primary_actor or "User"
            effective_actor = self._infer_actor(action_text, effective_actor)

            if self.fix_validate_write:
                ov = self._validate_write_override(action_text, effective_actor, step_id, section)
                if ov is not None:
                    action_tuples.append(ov)
                    continue

            atoms = self.decomposer.decompose(action_text)
            atoms = self._apply_lemma_fallback(atoms, action_text, infer_implicit)

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
                and self.profile.display_read_re.search(text)
                and not self.profile.display_msg_re.search(text)):
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

        # Inferência de movimentos IMPLÍCITOS (caminho de insumo pobre do produto:
        # descrição solta, processo de 1 passo). O COSMIC conta os movimentos que a
        # semântica do processo implica: uma escrita implica a ENTRADA do dado; uma
        # leitura implica a SAÍDA (o dado lido é devolvido); e um Exit de dado
        # armazenado implica um Read prévio. Só completa o PAR quando um lado existe
        # e o complemento falta (não infla processo já rico). Gated em infer_implicit
        # (a fachada do produto liga; as avaliações da tese NÃO → intocadas) e restrito
        # a processo de 1 passo (não mexe em UC multi-passo explícito → De Vito seguro).
        action_recognized = None  # sinal de completude p/ a confiança (ver PipelineResult)
        if infer_implicit and n_main_steps == 1 and active:
            def _has(t):
                return any(m.movement_type.value == t for m in active)

            def _add_implicit(mt, rule, why):
                mv = MovementResult(
                    movement_type=getattr(_onto_mod.CMovement, mt),
                    data_group=(active[0].data_group or "data"),
                    functional_user="System", rule_id=rule, reasoning=why,
                    source_step_id=active[0].source_step_id, source_text=text,
                    is_triggering=False)
                movements.append(mv)
                active.append(mv)

            # Classe do processo pelos verbos das ações (corrige a base pobre:
            # muitos verbos de criação/consulta caem como Entry-só).
            verbs = {at.verb_lemma for at in action_tuples}
            is_write = bool(verbs & self.profile.implicit_write_verbs)
            is_query = (not is_write) and bool(verbs & self.profile.implicit_query_verbs)
            is_send = bool(verbs & self.profile.implicit_send_verbs)
            # Cálculo (calcular/computar…): lê insumo e emite resultado → Read+Exit,
            # mesmo par que consulta. Só quando não é já escrita/consulta explícita.
            is_compute = (not is_write) and (not is_query) and \
                bool(verbs & self.profile.implicit_compute_verbs)
            # Completude: houve verbo de ação e ele casou com alguma classe?
            action_recognized = (is_write or is_query or is_send or is_compute) \
                if verbs else None

            # Envio a outro usuário funcional (email/notificação) → Exit.
            if is_send and not _has("X"):
                _add_implicit("X", "R_IMPLICIT_SEND_EXIT",
                              "Verbo de envio (enviar/notificar): saída ao usuário funcional → Exit.")

            if is_write:
                # criar/registrar/excluir… → o processo persiste o dado: Entry + Write.
                if not _has("W"):
                    _add_implicit("W", "R_IMPLICIT_WRITE",
                                  "Verbo de escrita: o processo persiste o dado → Write.")
                if not _has("E"):
                    _add_implicit("E", "R_IMPLICIT_ENTRY",
                                  "Verbo de escrita: o dado persistido entrou no processo → Entry.")
            elif is_query:
                # consultar/exibir/listar… → busca e devolve: Read + Exit.
                if not _has("R"):
                    _add_implicit("R", "R_IMPLICIT_READ",
                                  "Verbo de consulta: busca no armazenamento → Read.")
                if not _has("X"):
                    _add_implicit("X", "R_IMPLICIT_EXIT",
                                  "Verbo de consulta: o dado buscado é devolvido ao usuário → Exit.")
            elif is_compute:
                # calcular/computar… → lê o insumo e emite o resultado: Read + Exit.
                if not _has("R"):
                    _add_implicit("R", "R_IMPLICIT_COMPUTE_READ",
                                  "Verbo de cálculo: lê o insumo no armazenamento → Read.")
                if not _has("X"):
                    _add_implicit("X", "R_IMPLICIT_COMPUTE_EXIT",
                                  "Verbo de cálculo: o resultado é devolvido ao usuário → Exit.")

            # Completa o PAR faltante (fallback para os demais processos).
            if _has("W") and not _has("E"):
                _add_implicit("E", "R_IMPLICIT_ENTRY",
                              "Write implica Entry: o dado gravado entrou no processo (COSMIC).")
            if _has("R") and not _has("X"):
                _add_implicit("X", "R_IMPLICIT_EXIT",
                              "Read implica Exit: o dado lido é devolvido ao usuário funcional.")
            if _has("X") and not _has("R") and not is_send:
                # Envio (email/notificação) é Entry+Exit: a saída NÃO implica Read
                # prévio de armazenamento — só os Exit de dado consultado implicam.
                _add_implicit("R", "R_IMPLICIT_READ",
                              "Exit de dado armazenado implica Read prévio do armazenamento.")
            cfp = len([m for m in movements if m.movement_type is not None])

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
            action_recognized=action_recognized,
        )

    def _validate_write_override(self, action_text, actor, step_id, section):
        """Conserto de alta precisão dos cancelamentos: passo de sistema que
        'valida/checa ... e grava/salva X' produz 1 Write de X (a validação do
        dado digitado é interna, sem movimento). Mantém o total e corrige a
        decomposição. Retorna o ActionTuple de Write, ou None se não casa."""
        if not self.profile.is_system_actor(str(actor)):
            return None
        if not self.profile.validate_write_re.search(action_text):
            return None
        m = self.profile.write_obj_re.search(action_text)
        obj = ((m.group(1).strip() if m else "") or self.profile.default_data_label)[:40]
        return ActionTuple(
            step_id=step_id, agent=self.profile.canonical_system_actor,
            verb=self.profile.canonical_write_verb,
            verb_lemma=self.profile.canonical_write_verb,
            theme=obj, source_text=action_text, entities=[], section=section,
            is_negated=False)

    def _apply_lemma_fallback(self, atoms: list, text: str,
                              infer_implicit: bool = False) -> list:
        """When the decomposer extracts a verb lemma not in any known set,
        scan the spaCy doc for a token whose lemma IS recognised.

        Also handles cases where spaCy mistagged a verb as ADJ/NOUN
        (e.g. "System open Module" → "open" tagged ADJ but lemma is
        in profile.exit_verbs).
        """
        # Hoisted out of the token loop below: profile.all_verb_lemmas is a
        # @property that recomputes the union of six frozensets on every
        # access. It's accessed once per atom above and then again per-token
        # in the spaCy doc loop; caching it in a local here turns that into a
        # single computation for this call instead of one per token.
        known_lemmas = self.profile.all_verb_lemmas
        # Caminho de produto (infer_implicit): os verbos das classes de
        # inferência (write/query/send/compute) TAMBÉM são reconhecidos — pelo
        # bloco de implícitos, não pelos seis léxicos explícitos. Sem isto, um
        # verbo como "calcular" (só em implicit_compute_verbs) é tratado como
        # desconhecido e SUBSTITUÍDO por outro verbo da sentença, matando o
        # sinal (bug: "calcular … e avisar" virava "avisar … e avisar"). A tese
        # chama sem infer_implicit → known_lemmas inalterado, path intocado.
        if infer_implicit:
            known_lemmas = (known_lemmas | self.profile.implicit_write_verbs
                            | self.profile.implicit_query_verbs
                            | self.profile.implicit_send_verbs
                            | self.profile.implicit_compute_verbs)

        needs_fix = any(
            a.verb_lemma and a.verb_lemma not in known_lemmas
            for a in atoms
        )
        if not needs_fix:
            return atoms

        doc = self.nlp(text)
        candidates = []
        for tok in doc:
            lemma = tok.lemma_.lower()
            if lemma not in known_lemmas:
                continue
            is_verb_pos = tok.pos_ in ("VERB", "AUX")
            is_adj_but_known = tok.pos_ == "ADJ" and lemma in known_lemmas
            if not (is_verb_pos or is_adj_but_known):
                continue
            theme_parts = [
                c.text for c in tok.subtree
                if c != tok and not self.profile.is_subject(c) and c.dep_ != "punct"
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
                and atom.verb_lemma not in known_lemmas
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
        if lower.startswith(self.profile.actor_system_prefixes):
            return self.profile.canonical_system_actor
        if lower.startswith(self.profile.actor_user_prefixes):
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
            if bare in self.profile.uc_main_labels:
                section = "main"
                continue
            elif lower.startswith(self.profile.uc_alt_prefix):
                section = "alt"
                continue
            elif lower.startswith(self.profile.uc_exception_prefix):
                section = "exception"
                continue
            elif lower.startswith(self.profile.uc_skip_prefix):
                section = "skip"
                continue

            if section == "header":
                parts = stripped.split("\t", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    if key in self.profile.uc_id_labels or key == "":
                        if "\t" in val:
                            sub = val.split("\t", 1)
                            name = sub[1].strip() if len(sub) > 1 else sub[0].strip()
                        else:
                            name = val
                    elif key in self.profile.uc_primary_actor_labels:
                        primary_actor = val
                    # Ator secundário ANTES de ator: em português "atores
                    # secundários"/"atores de apoio" começam com o prefixo de
                    # ator ("ator"), então checar ator primeiro os capturaria e
                    # sobrescreveria o primary_actor. Em inglês não há colisão
                    # ("secondary actor" não começa com "actor"), então a ordem
                    # é indiferente lá — a regressão de 517 confirma.
                    elif key.startswith(self.profile.uc_secondary_actor_prefix):
                        pass
                    elif key.startswith(self.profile.uc_actor_prefix):
                        primary_actor = val.split(",")[0].strip()
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
