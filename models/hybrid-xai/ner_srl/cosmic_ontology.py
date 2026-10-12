"""COSMIC ISO 19761 Ontology — Rule-based movement classification engine.

Encodes ~40 inference rules derived from the COSMIC Measurement Manual and
the De Vito COSMET ground-truth measurements. Takes NER entities + SRL
decomposed actions and produces classified COSMIC data movements.

Key capabilities:
  - Verb-to-movement classification with contextual disambiguation
  - Agent role resolution (human user, external system, device)
  - Persistent storage detection (explicit vs implicit)
  - Error/confirmation message collapsing
  - UI control action filtering
  - Compound action handling
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .language_profile import ENGLISH, LanguageProfile


# ── Movement types ────────────────────────────────────────────────────

class CMovement(str, Enum):
    E = "E"
    X = "X"
    R = "R"
    W = "W"


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class CosmicEntity:
    text: str
    label: str  # FUNC_USER, DATA_OBJ, ACTION_VERB, UI_ELEMENT, PERSIST_STORE, CONDITION
    start: int = 0
    end: int = 0


@dataclass
class ActionTuple:
    """An atomic Agent-Verb-Theme action extracted from UC step."""
    step_id: str
    agent: str
    verb: str
    verb_lemma: str
    theme: str
    source_text: str
    entities: list[CosmicEntity] = field(default_factory=list)
    section: str = "main"  # main, alt, exception
    is_negated: bool = False


@dataclass
class MovementResult:
    """A classified COSMIC data movement with traceability."""
    movement_type: Optional[CMovement]
    data_group: str
    functional_user: str
    rule_id: str
    reasoning: str
    source_step_id: str
    source_text: str
    is_triggering: bool = False
    is_error_confirmation: bool = False


# Léxicos de verbos, regexes e nomes de sistema agora vêm do LanguageProfile
# injetado em CosmicOntology (ver __init__ e self.profile.* nos métodos abaixo).
# O perfil ENGLISH em language_profile.py é a cópia verbatim destas constantes.

# ── Ontology engine ──────────────────────────────────────────────────

class CosmicOntology:
    """COSMIC rule-based movement classifier.

    Flags de correção (auditoria 2026-06-16). Defaults = comportamento canônico,
    validado empiricamente como o melhor (ver resultados-correcoes-2026-06-16.md):
      inject_implicit_reads (default False): liga _inject_implicit_reads (que era
        código morto). TESTADO: ligar PIORA a acurácia (De Vito MAE 0,933→1,042,
        sobreconta Reads), então é mantido DESLIGADO. Disponível só por reprodução.
      improved_dedup (default False): normalização de grupo de dados menos rasa
        (whitespace, pontuação, plural guardado). TESTADO: neutro na acurácia
        (totais idênticos); útil apenas para limpar rótulos do trace.
    """

    def __init__(self, inject_implicit_reads: bool = False,
                 improved_dedup: bool = False,
                 profile: LanguageProfile = ENGLISH):
        self.inject_implicit_reads = inject_implicit_reads
        self.improved_dedup = improved_dedup
        self.profile = profile

    def classify_actions(
        self,
        actions: list[ActionTuple],
        primary_actor: str = "",
        n_steps: int = 0,
    ) -> list[MovementResult]:
        """Classify a list of atomic actions into COSMIC movements.

        Args:
            actions: decomposed atomic actions from SRL
            primary_actor: the declared primary actor of the UC
            n_steps: number of main-scenario steps (controls triggering entry injection)
        """
        results = []
        error_exit_counted = False
        first_entry_found = False

        for action in actions:
            if self.profile.include_uc_re.search(action.source_text):
                continue

            actor = action.agent or primary_actor or "User"
            is_system = self.profile.is_system_actor(actor)

            result = self._apply_rules(action, actor, is_system)

            if result.movement_type == CMovement.X and result.is_error_confirmation:
                if error_exit_counted:
                    result = MovementResult(
                        movement_type=None,
                        data_group=result.data_group,
                        functional_user=result.functional_user,
                        rule_id="R_ERR_COLLAPSE",
                        reasoning="Error/confirm exit already counted (COSMIC: 1 Exit for all msgs to same FU)",
                        source_step_id=action.step_id,
                        source_text=action.source_text,
                    )
                else:
                    error_exit_counted = True

            if result.movement_type == CMovement.E and not first_entry_found:
                result.is_triggering = True
                first_entry_found = True

            results.append(result)

        inject_trigger = n_steps >= 2 or n_steps == 0
        if not first_entry_found and inject_trigger:
            for r in results:
                if r.movement_type is not None:
                    synthetic = MovementResult(
                        movement_type=CMovement.E,
                        data_group="triggering_event",
                        functional_user=primary_actor or "User",
                        rule_id="R_TRIGGER_INJECT",
                        reasoning="No Entry found; injecting triggering Entry (COSMIC §3.2.1)",
                        source_step_id=results[0].source_step_id if results else "",
                        source_text="",
                        is_triggering=True,
                    )
                    results.insert(0, synthetic)
                    break

        if self.inject_implicit_reads:
            results = self._inject_implicit_reads(results)
        results = self._dedup_data_groups(results)

        return results

    def _normalize_dg_old(self, dg: str) -> str:
        """Normalização canônica (rasa) — pré-correção."""
        s = dg.lower().strip()
        s = self.profile.article_re.sub("", s)
        return s

    def _normalize_dg(self, dg: str) -> str:
        """Normaliza um grupo de dados para comparação de dedup.

        improved_dedup=False reproduz a versão rasa (_normalize_dg_old). A versão
        corrigida também colapsa whitespace/pontuação e singulariza um 's' final
        (guardado contra status/address/analysis)."""
        if not getattr(self, "improved_dedup", True):
            return self._normalize_dg_old(dg)
        import re as _re
        s = dg.lower().strip()
        s = self.profile.article_re.sub("", s)
        s = _re.sub(r"\s+", " ", s).strip(" \t\n.,;:\"'()[]")
        toks = []
        for w in s.split(" "):
            if (
                len(w) > 4
                and w.endswith(self.profile.plural_suffix)
                and not w.endswith(self.profile.plural_exceptions)
            ):
                w = w[: -len(self.profile.plural_suffix)]
            toks.append(w)
        return " ".join(toks)

    def _dedup_data_groups(self, results: list[MovementResult]) -> list[MovementResult]:
        """COSMIC ISO 19761: same-direction movements on the same data group count as one."""
        seen: set[tuple[str, str]] = set()
        deduped = []
        for r in results:
            if r.movement_type is None or r.is_error_confirmation:
                deduped.append(r)
                continue
            dg = self._normalize_dg(r.data_group)
            if not dg or dg in ("messages", "triggering_event"):
                deduped.append(r)
                continue
            key = (r.movement_type.value, dg)
            if key in seen:
                deduped.append(MovementResult(
                    movement_type=None,
                    data_group=r.data_group,
                    functional_user=r.functional_user,
                    rule_id="R_DG_DEDUP",
                    reasoning=f"Duplicate {r.movement_type.value} on '{dg}' — collapsed (ISO 19761)",
                    source_step_id=r.source_step_id,
                    source_text=r.source_text,
                ))
            else:
                seen.add(key)
                deduped.append(r)
        return deduped

    def _inject_implicit_reads(self, results: list[MovementResult]) -> list[MovementResult]:
        """If a system Exit has no prior Read for the same data group, inject an implicit Read.

        Conservative: only inject when (a) the FP already has at least one Read,
        (b) the Exit data group contains domain-data keywords, and (c) the data
        group was not provided by a prior Entry.
        """
        _SKIP_DGS = {"messages", "triggering_event", ""}
        read_dgs: set[str] = set()
        entry_dgs: set[str] = set()
        has_any_read = False
        for r in results:
            if r.movement_type is None:
                continue
            ndg = self._normalize_dg(r.data_group)
            if r.movement_type == CMovement.R:
                read_dgs.add(ndg)
                has_any_read = True
            elif r.movement_type == CMovement.E:
                entry_dgs.add(ndg)

        if not has_any_read:
            return results

        augmented: list[MovementResult] = []
        for r in results:
            if (
                r.movement_type == CMovement.X
                and not r.is_error_confirmation
                and r.rule_id in ("R3_SYS_EXIT",)
            ):
                ndg = self._normalize_dg(r.data_group)
                if (
                    ndg not in _SKIP_DGS
                    and ndg not in read_dgs
                    and ndg not in entry_dgs
                    and self.profile.domain_data_re.search(ndg)
                ):
                    implicit = MovementResult(
                        movement_type=CMovement.R,
                        data_group=r.data_group,
                        functional_user="",
                        rule_id="R_IMPLICIT_READ",
                        reasoning=f"Implicit Read for '{ndg}' — Exit has no prior Read/Entry",
                        source_step_id=r.source_step_id,
                        source_text=r.source_text,
                    )
                    augmented.append(implicit)
                    read_dgs.add(ndg)
            augmented.append(r)
        return augmented

    def _apply_rules(
        self, action: ActionTuple, actor: str, is_system: bool,
    ) -> MovementResult:
        """Apply COSMIC rules to a single atomic action."""
        lemma = action.verb_lemma
        text = action.source_text
        theme = action.theme
        entities = action.entities

        has_persist = self.profile.persist_re.search(text) or any(
            e.label == "PERSIST_STORE" for e in entities
        )
        has_ui = self.profile.ui_re.search(text) or self.profile.quoted_ui_re.search(text) or any(
            e.label == "UI_ELEMENT" for e in entities
        )
        has_data_obj = any(e.label == "DATA_OBJ" for e in entities) or bool(theme)
        is_error = self.profile.error_re.search(text)
        is_confirm = self.profile.confirm_re.search(text)

        def mk(mv, dg, rule, reason, err_conf=False):
            return MovementResult(
                movement_type=mv,
                data_group=dg or theme or "",
                functional_user=actor if mv in (CMovement.E, CMovement.X) else "",
                rule_id=rule,
                reasoning=reason,
                source_step_id=action.step_id,
                source_text=action.source_text,
                is_error_confirmation=err_conf,
            )

        def mk_none(rule, reason):
            return MovementResult(
                movement_type=None,
                data_group="",
                functional_user="",
                rule_id=rule,
                reasoning=reason,
                source_step_id=action.step_id,
                source_text=action.source_text,
            )

        # ── R0: Negated action → No movement ─────────────────────
        if action.is_negated:
            return mk_none("R0_NEGATED", f"Negated verb '{lemma}' — no data movement")

        # ── R1: System shows blank/empty form → No movement ────────────
        if is_system and self.profile.blank_form_re.search(text):
            return mk_none("R1_BLANK_FORM", "System shows blank form — no data (COSMIC Guide §3.5)")

        # ── R2: Error/confirmation message → single Exit ──────────
        if is_system and (is_error or is_confirm) and lemma in self.profile.exit_verbs:
            return mk(CMovement.X, "Messages", "R2_ERR_CONFIRM",
                       f"System error/confirm message → 1 Exit", err_conf=True)

        # ── R3: System display/show with domain data → Exit ──────
        if is_system and lemma in self.profile.exit_verbs:
            if lemma in self.profile.exit_external_request_verbs:
                return mk(CMovement.X, theme, "R3_SYS_REQUEST",
                           f"System '{lemma}' to external actor → Exit")
            if lemma in self.profile.exit_return_ui_verbs and has_ui and not any(
                e.label == "DATA_OBJ" for e in entities
            ):
                return mk_none("R3_RETURN_UI",
                                f"System returns to UI (screen) — no data movement")
            return mk(CMovement.X, theme, "R3_SYS_EXIT",
                       f"System '{lemma}' → Exit to functional user")

        # ── R4: System read/search from storage → Read ───────────
        if is_system and lemma in self.profile.read_verbs:
            return mk(CMovement.R, theme, "R4_SYS_READ",
                       f"System '{lemma}' → Read from persistent storage")

        # ── R5: System save/store/delete → Write ─────────────────
        if is_system and lemma in self.profile.write_verbs:
            return mk(CMovement.W, theme, "R5_SYS_WRITE",
                       f"System '{lemma}' → Write to persistent storage")

        # ── R6: System receives from external actor → Entry ──────
        if is_system and lemma in self.profile.entry_verbs:
            return mk(CMovement.E, theme, "R6_SYS_ENTRY",
                       f"System receives '{lemma}' → Entry (from external FU)")

        # ── R7: Ambiguous verbs — context resolution ─────────────
        if is_system and lemma in self.profile.ambiguous_verbs:
            if has_persist:
                if lemma in self.profile.ambiguous_read_verbs:
                    return mk(CMovement.R, theme, "R7_AMBIG_READ",
                               f"System '{lemma}' with storage reference → Read")
                return mk(CMovement.R, theme, "R7_AMBIG_READ",
                           f"System '{lemma}' with persistence marker → Read")
            if lemma in self.profile.ambiguous_read_verbs:
                return mk_none("R7_AMBIG_INTERNAL",
                                f"System '{lemma}' without storage → internal validation, no movement")
            if lemma in self.profile.ambiguous_process_verbs:
                return mk_none("R7_AMBIG_PROCESS",
                                f"System '{lemma}' → internal processing, no direct movement")

        # ── R8: User control action (click, navigate, press) → No movement ─
        if not is_system and lemma in self.profile.control_verbs:
            if lemma in self.profile.control_click_verbs and has_data_obj and not has_ui:
                return mk(CMovement.E, theme, "R8_CLICK_DATA",
                           f"User clicks on domain data → Entry")
            return mk_none("R8_CONTROL",
                            f"User control '{lemma}' on UI → no data movement")

        # ── R9: User select/choose ────────────────────────────────
        if not is_system and lemma in self.profile.entry_select_verbs:
            theme_is_ui = self.profile.ui_re.search(theme) if theme else False
            if has_ui and (not has_data_obj or theme_is_ui):
                return mk_none("R9_SELECT_UI",
                                f"User selects UI element → no data movement")
            if not theme or theme_is_ui:
                return mk_none("R9_SELECT_UI",
                                f"User selects without domain data → no data movement")
            return mk(CMovement.E, theme, "R9_SELECT_DATA",
                       f"User '{lemma}' domain data → Entry")

        # ── R10: User data entry (fill, submit, provide, etc.) → Entry
        if not is_system and lemma in self.profile.entry_verbs:
            if lemma in self.profile.control_click_verbs and has_ui:
                return mk_none("R10_PRESS_UI",
                                f"User presses button → no data movement")
            if lemma in self.profile.entry_confirm_verbs and not has_data_obj:
                return mk_none("R10_CONFIRM_NO_DATA",
                                f"User confirms without data → no movement")
            return mk(CMovement.E, theme, "R10_USER_ENTRY",
                       f"User '{lemma}' → Entry")

        # ── R11: External actor displays/receives → Exit from system
        if not is_system and lemma in self.profile.exit_verbs:
            return mk(CMovement.X, theme, "R11_EXT_EXIT",
                       f"Non-system actor '{actor}' receives/displays → Exit from system")

        # ── R12: User browsing results → No movement ─────────────
        if not is_system and lemma in self.profile.control_browse_verbs:
            return mk_none("R12_BROWSE",
                            f"User browsing → no data movement")

        # ── R13: External actor sends/signals → Entry ─────────────
        if not is_system and lemma in self.profile.entry_external_send_verbs:
            return mk(CMovement.E, theme, "R13_EXT_SEND",
                       f"External actor '{lemma}' → Entry to system")

        # ── R14: System secondary action fallback ─────────────────
        if is_system:
            if has_persist:
                return mk(CMovement.R, theme, "R14_SYS_FALLBACK_READ",
                           f"System unrecognized verb with storage → Read (fallback)")
            if has_data_obj:
                return mk(CMovement.X, theme, "R14_SYS_FALLBACK_EXIT",
                           f"System unrecognized verb with data → Exit (fallback)")
            return mk_none("R14_SYS_UNKNOWN",
                            f"System unrecognized verb '{lemma}' without data context")

        # ── R15: User fallback ────────────────────────────────────
        if has_data_obj:
            return mk(CMovement.E, theme, "R15_USER_FALLBACK",
                       f"User unrecognized verb with data → Entry (fallback)")

        return mk_none("R_DEFAULT", f"No applicable rule for verb '{lemma}'")
