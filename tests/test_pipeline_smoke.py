"""S4-T4 — Smoke test end-to-end do pipeline E33d (sem NER custom, p/ rapidez)."""
import importlib.util
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent

UC_LOGIN = """ID
UC-T1\tUser login
Description\tThis function allows the user to log in.
Primary actor\tUser
MAIN SCENARIO
1\tUser\tThe user fills in the username and password and submits the form.
2\tSystem\tThe system reads the user credentials from the database.
3\tSystem\tThe system displays the main dashboard to the user.
"""

UC_CREATE = """ID
UC-T2\tRegister product
Description\tThis function allows the manager to register a new product.
Primary actor\tManager
MAIN SCENARIO
1\tManager\tThe manager enters the product details and submits the form.
2\tSystem\tThe system saves the new product to the database.
3\tSystem\tThe system shows a confirmation message.
"""


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod.NerOntologyPipeline()  # sem NER custom (rápido, só ontologia)


def test_login_uc(pipeline):
    res = pipeline.measure(UC_LOGIN, fp_id="UC-T1")
    assert res.cfp >= 2, "login deve ter pelo menos Entry + Exit"
    assert res.n_entries >= 1
    assert res.n_exits >= 1
    # todo movimento contado deve citar regra e ter tipo válido
    for m in res.movements:
        assert m.rule_id


def test_create_uc(pipeline):
    res = pipeline.measure(UC_CREATE, fp_id="UC-T2")
    assert res.n_entries >= 1, "entrada de dados do produto"
    assert res.n_writes >= 1, "persistência do produto"
    assert res.n_exits >= 1, "confirmação"
    assert res.cfp == res.n_entries + res.n_reads + res.n_writes + res.n_exits


def test_deterministic(pipeline):
    a = pipeline.measure(UC_LOGIN, fp_id="UC-T1")
    b = pipeline.measure(UC_LOGIN, fp_id="UC-T1")
    assert a.cfp == b.cfp
