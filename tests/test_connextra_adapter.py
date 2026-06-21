"""S4-T4 — Testes do adaptador Connextra (E34a)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from adapters.connextra_adapter import (  # noqa: E402
    connextra_to_ucs, _split_subactions, _verb_class,
)


class TestFrameParsing:
    def test_basic_frame(self):
        ucs = connextra_to_ucs(
            "As a user, I want to view my profile, so that I can check my data.",
            "FP001")
        assert ucs is not None and len(ucs) == 1
        assert "MAIN SCENARIO" in ucs[0]

    def test_non_connextra_returns_none(self):
        assert connextra_to_ucs("The system shall persist all orders.", "FP001") is None

    def test_benefit_clause_dropped(self):
        ucs = connextra_to_ucs(
            "As a user, I want to delete my account, so that nobody emails me.",
            "FP001")
        # cláusula 'so that' não deve aparecer nos passos (protocolo §3.6)
        steps = ucs[0].split("MAIN SCENARIO")[1]
        assert "emails me" not in steps

    def test_an_article_and_want_variants(self):
        assert connextra_to_ucs(
            "As an admin, I would like to view the audit log.", "X") is not None


class TestDecomposition:
    def test_crud_coordination_decomposes(self):
        # regra 4.6: CRUD explícito → 1 processo por verbo
        ucs = connextra_to_ucs(
            "As a manager, I want to create, update and delete any records in my repository.",
            "FP076")
        assert len(ucs) == 3

    def test_read_coordination_single_process(self):
        # regra 4.3: coordenação de consulta → 1 processo
        ucs = connextra_to_ucs(
            "As a user, I want to filter, sort and search items by criteria.",
            "FP015")
        assert len(ucs) == 1

    def test_or_coordination_single_process(self):
        # alternativas ('or') = mesmo processo funcional
        subs = _split_subactions("add useful metadata or add new records to the dataset")
        assert len(subs) == 1


class TestVerbClasses:
    def test_classes(self):
        assert _verb_class("view") == "read"
        assert _verb_class("create") == "create"
        assert _verb_class("update") == "update"
        assert _verb_class("delete") == "delete"
        assert _verb_class("notify") == "send"
        assert _verb_class("manage") == "manage"
        assert _verb_class("login") == "auth"
        assert _verb_class("perform") == "compute"
        assert _verb_class("xyzzy") == "generic"
