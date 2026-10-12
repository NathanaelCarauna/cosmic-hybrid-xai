"""Predicados do LanguageProfile, testados isoladamente com frases mínimas."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import spacy

BASE = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(BASE / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lp = _load("language_profile", "models/hybrid-xai/ner_srl/language_profile.py")


@pytest.fixture(scope="module")
def nlp_en():
    return spacy.load("en_core_web_sm")


def _tok(doc, text):
    return next(t for t in doc if t.text == text)


def test_english_negation(nlp_en):
    doc = nlp_en("The system does not display the order.")
    assert lp.ENGLISH.is_negation(_tok(doc, "not"))
    assert not lp.ENGLISH.is_negation(_tok(doc, "display"))


def test_english_direct_object(nlp_en):
    doc = nlp_en("The system displays the order.")
    assert lp.ENGLISH.is_direct_object(_tok(doc, "order"))
    assert not lp.ENGLISH.is_direct_object(_tok(doc, "system"))


def test_english_subject(nlp_en):
    doc = nlp_en("The system displays the order.")
    assert lp.ENGLISH.is_subject(_tok(doc, "system"))


def test_english_system_actor():
    assert lp.ENGLISH.is_system_actor("System")
    assert lp.ENGLISH.is_system_actor("the system")
    assert not lp.ENGLISH.is_system_actor("Hotel manager")


def test_english_lexicons_are_verbatim():
    # Trava de regressão: se alguém editar o léxico inglês, este teste quebra.
    assert "display" in lp.ENGLISH.exit_verbs
    assert "enter" in lp.ENGLISH.entry_verbs
    assert "search" in lp.ENGLISH.read_verbs
    assert "save" in lp.ENGLISH.write_verbs
    assert "click" in lp.ENGLISH.control_verbs
    assert "check" in lp.ENGLISH.ambiguous_verbs
    # Contagens conferidas no cosmic_ontology.py antes de escrever o plano.
    assert len(lp.ENGLISH.entry_verbs) == 32
    assert len(lp.ENGLISH.exit_verbs) == 34
    assert len(lp.ENGLISH.read_verbs) == 20
    assert len(lp.ENGLISH.write_verbs) == 33
    assert len(lp.ENGLISH.control_verbs) == 15
    assert len(lp.ENGLISH.ambiguous_verbs) == 9


def test_english_direct_object_inclui_copular_attr(nlp_en):
    # "attr" é como o esquema inglês do spaCy rotula o predicativo de cópula.
    # O extrator o trata como objeto, então o predicado precisa aceitá-lo.
    doc = nlp_en("The result is the confirmation screen.")
    attr_tok = next(t for t in doc if t.dep_ == "attr")
    assert lp.ENGLISH.is_direct_object(attr_tok)


def test_english_passive_subject(nlp_en):
    doc = nlp_en("The order is displayed by the system.")
    assert lp.ENGLISH.is_passive_subject(_tok(doc, "order"))


def test_english_relative_clause(nlp_en):
    # spaCy só rotula "relcl" com o pronome relativo "which"; com "that" ele
    # produz "acl" (verificado imprimindo os deps — não é o predicado que
    # está errado, é a frase de teste que precisa do pronome certo).
    doc = nlp_en("The system displays the order which the customer created.")
    assert any(lp.ENGLISH.is_relative_clause(t) for t in doc)


def _load_onto_mod():
    """Carrega cosmic_ontology.py resolvendo seu import relativo interno.

    `cosmic_ontology.py` faz `from .language_profile import ...` — um import
    relativo que exige um pacote pai registrado em sys.modules. `_load` (acima)
    cria o módulo com um nome sem ponto (__package__ == ""), o que faz esse
    import relativo estourar `ImportError: attempted relative import with no
    known parent package` ANTES mesmo de chegar em `_canary_profile()`.
    `ner_ontology_pipeline.py` resolve o mesmo problema para si mesmo com um
    pacote sintético `_ner_srl_shim` (ver `_load_ner_srl_submodule` lá); aqui
    replicamos o mesmo mecanismo para carregar `cosmic_ontology` isoladamente.
    """
    import importlib
    import types

    pkg_name = "_ner_srl_shim"
    if pkg_name not in sys.modules:
        pkg_dir = BASE / "models" / "hybrid-xai" / "ner_srl"
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [str(pkg_dir)]
        pkg_mod.__package__ = pkg_name
        sys.modules[pkg_name] = pkg_mod
    return importlib.import_module(f"{pkg_name}.cosmic_ontology")


def _canary_profile():
    """Perfil sintético sem nenhuma palavra inglesa nos léxicos.

    Se alguma regra depender de literal cru em inglês, ela não dispara com
    este perfil — e o teste correspondente falha. É o que impede que a
    família de resíduos de idioma seja reintroduzida.

    "retornar"/"exit_return_ui_verbs" foi acrescentado na Task 6-bis (revisão,
    Importante 2) só para o novo caso R3_RETURN_UI — os oito casos originais
    não o usam.
    """
    import dataclasses
    return dataclasses.replace(
        lp.ENGLISH,
        code="canary",
        system_names=frozenset({"sistema"}),
        entry_verbs=frozenset({"informar", "selecionar", "confirmar"}),
        exit_verbs=frozenset({"exibir", "solicitar", "retornar"}),
        read_verbs=frozenset({"consultar"}),
        write_verbs=frozenset({"gravar"}),
        control_verbs=frozenset({"clicar"}),
        ambiguous_verbs=frozenset({"validar", "processar"}),
        exit_external_request_verbs=frozenset({"solicitar"}),
        exit_return_ui_verbs=frozenset({"retornar"}),
        ambiguous_read_verbs=frozenset({"validar"}),
        ambiguous_process_verbs=frozenset({"processar"}),
        control_click_verbs=frozenset({"clicar"}),
        entry_select_verbs=frozenset({"selecionar"}),
        entry_confirm_verbs=frozenset({"confirmar"}),
        canonical_system_actor="sistema",
        canonical_write_verb="gravar",
    )


@pytest.mark.parametrize("lemma,actor,rule_esperada", [
    ("solicitar",  "Sistema", "R3_SYS_REQUEST"),
    ("consultar",  "Sistema", "R4_SYS_READ"),
    ("gravar",     "Sistema", "R5_SYS_WRITE"),
    ("validar",    "Sistema", "R7_AMBIG_READ"),
    ("processar",  "Sistema", "R7_AMBIG_PROCESS"),
    ("clicar",     "Usuário", "R8_CLICK_DATA"),
    ("selecionar", "Usuário", "R9_SELECT_DATA"),
    ("informar",   "Usuário", "R10_USER_ENTRY"),
])
def test_canary_regras_disparam_sem_literal_ingles(lemma, actor, rule_esperada):
    onto_mod = _load_onto_mod()
    onto = onto_mod.CosmicOntology(profile=_canary_profile())

    # R7_AMBIG_READ só dispara com has_persist=True (senão cai em
    # R7_AMBIG_INTERNAL); has_persist vem de profile.persist_re no texto OU
    # de uma entidade PERSIST_STORE — usamos a entidade para não depender de
    # nenhuma palavra (nem em português) casar com o persist_re herdado do
    # ENGLISH. R7_AMBIG_PROCESS exige o oposto (has_persist=False), que é o
    # padrão sem entidades.
    entities = []
    if lemma == "validar":
        entities = [onto_mod.CosmicEntity(
            text="banco de dados", label="PERSIST_STORE")]

    acao = onto_mod.ActionTuple(
        step_id="1", agent=actor, verb=lemma, verb_lemma=lemma,
        theme="dados do cliente",
        source_text=f"{actor} {lemma} os dados do cliente",
        entities=entities,
        is_negated=False,
    )
    resultados = onto.classify_actions([acao], primary_actor=actor, n_steps=2)
    regras = {getattr(r, "rule_id", "") for r in resultados}
    assert rule_esperada in regras, (
        f"regra {rule_esperada} não disparou com léxico não-inglês; "
        f"disparou {regras} — provável literal cru em cosmic_ontology.py"
    )


@pytest.mark.parametrize("lemma,actor,theme,ui_marker,persist_marker,rule_esperada", [
    # R3_RETURN_UI: verbo de exit_return_ui_verbs, marcador de UI, sem
    # entidade DATA_OBJ. Sem o campo migrado, cai em R3_SYS_EXIT (a regra
    # SEMPRE dispara — outro Exit — então só o rule_id denuncia o literal
    # cru; por isso os 8 casos originais não pegavam isto).
    ("retornar", "Sistema", "", True, False, "R3_RETURN_UI"),
    # R7_AMBIG_INTERNAL: verbo de ambiguous_read_verbs, SEM persistência
    # (irmão de R7_AMBIG_READ, já coberto acima). Sem o campo migrado, a
    # regra não dispara nenhuma sub-regra e cai em R14_SYS_FALLBACK_EXIT.
    ("validar", "Sistema", "dados do cliente", False, False, "R7_AMBIG_INTERNAL"),
    # R9_SELECT_UI: verbo de entry_select_verbs, marcador de UI, sem objeto
    # de dado. Sem o campo migrado, o bloco R9 inteiro é pulado e o lema
    # (que também está em entry_verbs) cai em R10_USER_ENTRY.
    ("selecionar", "Usuário", "", True, False, "R9_SELECT_UI"),
    # R10_CONFIRM_NO_DATA: verbo de entry_confirm_verbs, sem objeto de
    # dado. Sem o campo migrado, cai no fallback final do bloco R10,
    # R10_USER_ENTRY (a regra sempre dispara alguma coisa — de novo só o
    # rule_id denuncia).
    ("confirmar", "Usuário", "", False, False, "R10_CONFIRM_NO_DATA"),
])
def test_canary_branches_irmaos_sem_literal_ingles(
    lemma, actor, theme, ui_marker, persist_marker, rule_esperada
):
    """Cobre os branches IRMÃOS que test_canary_regras_disparam_sem_literal_ingles
    (acima) não cobria (revisão da Task 6-bis, Importante 2).

    O canário original só prova que a regra DISPARA (movement_type is not
    None); os quatro casos acima migram literais em que a regra sempre
    dispara ALGUMA coisa, e o literal cru só decide qual sub-regra (rule_id)
    — ou seja, qual mensagem/rótulo de rastreio. Sem estes casos, reintroduzir
    o literal em `exit_return_ui_verbs`, no branch sem persistência de
    `ambiguous_read_verbs`, em `entry_select_verbs` ou em
    `entry_confirm_verbs` deixaria os 8 testes originais verdes.
    """
    onto_mod = _load_onto_mod()
    profile = _canary_profile()
    onto = onto_mod.CosmicOntology(profile=profile)

    entities = []
    if ui_marker:
        entities.append(onto_mod.CosmicEntity(
            text="elemento de interface", label="UI_ELEMENT"))
    if persist_marker:
        entities.append(onto_mod.CosmicEntity(
            text="banco de dados", label="PERSIST_STORE"))

    acao = onto_mod.ActionTuple(
        step_id="1", agent=actor, verb=lemma, verb_lemma=lemma,
        theme=theme,
        source_text=f"{actor} {lemma}",
        entities=entities,
        is_negated=False,
    )
    resultados = onto.classify_actions([acao], primary_actor=actor, n_steps=2)
    regras = {getattr(r, "rule_id", "") for r in resultados}
    assert rule_esperada in regras, (
        f"regra {rule_esperada} não disparou com léxico não-inglês; "
        f"disparou {regras} — provável literal cru em cosmic_ontology.py"
    )


def test_canary_validate_write_override_usa_valores_canonicos():
    """Cobre o Step 5 (`_validate_write_override`), que os 8 casos originais
    e os 4 acima não tocam — aquele caminho não passa por `classify_actions`
    a partir de um `ActionTuple` já pronto, ele CONSTRÓI a ActionTuple
    sintética dentro de `ner_ontology_pipeline.py`.

    `validate_write_re`/`write_obj_re` (Superfície 4, padrões) ainda não são
    migrados — são regex crus em inglês no perfil ENGLISH, e a Task 9 é quem
    decide seus equivalentes em português; usar texto em inglês aqui só para
    ACIONAR esses dois regexes é intencional e não enfraquece o teste. O que
    o teste trava é outra coisa: os literais que a Task 6-bis migrou nesta
    função — `agent`/`verb`/`verb_lemma` da ActionTuple sintética, que hoje
    vêm de `profile.canonical_system_actor`/`profile.canonical_write_verb`
    em vez de "system"/"record" cravados. Prova disso: usamos os valores
    canônicos do perfil-canário ("sistema"/"gravar", que não aparecem em
    nenhum léxico inglês) e confirmamos que `CosmicOntology` classifica o
    resultado como Write (R5_SYS_WRITE) — só possível se `is_system_actor`
    reconhecer o agente sintético e `write_verbs` reconhecer o verbo
    sintético, ambos contra o perfil-canário.
    """
    pipeline_mod = _load(
        "ner_ontology_pipeline_canary_test",
        "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py",
    )
    onto_mod = _load_onto_mod()
    profile = _canary_profile()
    pipeline = pipeline_mod.NerOntologyPipeline(profile=profile, use_ner=False)

    action_text = "The system checks the input and records the customer data."
    ov = pipeline._validate_write_override(action_text, "Sistema", "1", "main")

    assert ov is not None, (
        "validate_write_re/write_obj_re (ainda em inglês, fora do escopo "
        "desta task) não casaram — ajuste o texto de gatilho do teste, não "
        "o profile"
    )
    assert ov.agent == profile.canonical_system_actor == "sistema"
    assert ov.verb_lemma == profile.canonical_write_verb == "gravar"

    onto = onto_mod.CosmicOntology(profile=profile)
    resultados = onto.classify_actions([ov], primary_actor="sistema", n_steps=2)
    regras = {getattr(r, "rule_id", "") for r in resultados}
    assert "R5_SYS_WRITE" in regras, (
        f"ActionTuple sintética de _validate_write_override não foi "
        f"classificada como Write; disparou {regras} — provável literal cru "
        f"('system'/'record') reintroduzido em _validate_write_override"
    )


@pytest.fixture(scope="module")
def nlp_pt():
    return spacy.load("pt_core_news_sm")


def test_portuguese_negation(nlp_pt):
    doc = nlp_pt("O sistema não exibe o pedido do cliente.")
    assert lp.PORTUGUESE.is_negation(_tok(doc, "não"))
    assert not lp.PORTUGUESE.is_negation(_tok(doc, "exibe"))


def test_portuguese_negation_ignores_other_advmod(nlp_pt):
    # advmod que NÃO é negação não pode disparar R0_NEGATED
    doc = nlp_pt("O sistema sempre exibe o pedido.")
    assert not lp.PORTUGUESE.is_negation(_tok(doc, "sempre"))


def test_portuguese_direct_object(nlp_pt):
    doc = nlp_pt("O sistema exibe o pedido.")
    assert lp.PORTUGUESE.is_direct_object(_tok(doc, "pedido"))
    assert not lp.PORTUGUESE.is_direct_object(_tok(doc, "sistema"))


def test_portuguese_subject(nlp_pt):
    doc = nlp_pt("O sistema exibe o pedido.")
    assert lp.PORTUGUESE.is_subject(_tok(doc, "sistema"))


def test_portuguese_system_actor():
    assert lp.PORTUGUESE.is_system_actor("Sistema")
    assert lp.PORTUGUESE.is_system_actor("o sistema")
    assert lp.PORTUGUESE.is_system_actor("A aplicação")
    assert not lp.PORTUGUESE.is_system_actor("Gerente do hotel")


def test_parse_uc_pt_segmenta_corpus_real():
    """_parse_uc sob perfil PT extrai passos dos documentos traduzidos.

    Rótulo de seção mal configurado -> parsed["steps"] vazio -> 0 CFP silencioso.
    Cobre o De Vito (Cenário principal, na verdade em maiúsculas "CENÁRIO
    PRINCIPAL" em ALBERGATE_FP10.txt) e o Zenodo (MAIN SCENARIO em inglês,
    preservado da tradução em FP1000.txt).
    """
    pipe_mod = _load("ner_ontology_pipeline",
                     "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py")
    pipe = pipe_mod.NerOntologyPipeline(use_ner=False, profile=lp.PORTUGUESE)
    base = BASE / "datasets" / "pt-br"
    amostras = [
        base / "devito" / "ALBERGATE_FP10.txt",
        base / "zenodo300" / "FP1000.txt",
    ]
    for f in amostras:
        parsed = pipe._parse_uc(f.read_text(encoding="utf-8"))
        assert parsed["steps"], f"{f.name}: nenhum passo extraído (rótulo de seção?)"


def test_shim_nao_sombreia_pacote_ner_srl():
    """O shim de import não pode ocupar o nome do pacote real.

    models/hybrid-xai/ tem hífen e não é pacote Python, então o pipeline
    registra um pacote sintético para resolver imports relativos. Se esse
    shim usasse o nome 'ner_srl', um futuro 'import ner_srl' pegaria o
    stub incompleto do cache e falharia com AttributeError obscuro.
    """
    import sys
    _load("ner_ontology_pipeline",
          "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py")
    assert "ner_srl" not in sys.modules or hasattr(
        sys.modules["ner_srl"], "SRLExtractor"
    ), "shim vazio registrado sob o nome do pacote real"


def test_infer_actor_pt_reconhece_sistema():
    """Passo português "O sistema exibe X" tem que ser tratado como ação de
    sistema. Regressão: _infer_actor retornava o literal inglês "System",
    que is_system_actor não reconhece sob o perfil PT.
    """
    pipe_mod = _load("ner_ontology_pipeline",
                     "models/hybrid-xai/ner_srl/ner_ontology_pipeline.py")
    pipe = pipe_mod.NerOntologyPipeline(use_ner=False, profile=lp.PORTUGUESE)
    inferido = pipe._infer_actor("O sistema exibe os dados do cliente", "Sistema")
    assert lp.PORTUGUESE.is_system_actor(inferido), (
        f"_infer_actor devolveu {inferido!r}, que o perfil PT não reconhece como sistema"
    )
