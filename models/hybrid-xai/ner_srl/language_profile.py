"""Perfil de idioma: concentra tudo que depende de língua natural.

A cascata de 26 regras da ontologia COSMIC raciocina sobre PAPÉIS (ator,
ação, tema, negação), que são independentes de idioma. O que depende de
idioma são seis superfícies, todas reunidas aqui.

O perfil expõe PREDICADOS e não rótulos de dependência, porque o mapeamento
não é um rename: Universal Dependencies v2 eliminou o rótulo `neg`, de modo
que negação em português é `advmod` com lema "não", enquanto em inglês o
spaCy usa o esquema estilo ClearNLP com `neg` dedicado.

ATENÇÃO: o perfil ENGLISH é extração VERBATIM das constantes que estavam em
cosmic_ontology.py. Qualquer alteração nele muda os números canônicos da
dissertação. Ver tests/golden/README.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageProfile:
    """Tudo que depende de idioma, para um idioma."""

    code: str
    spacy_model: str

    # Superfície 2 — nomes de sistema (eixo primário da cascata)
    system_names: frozenset[str]

    # Superfície 3 — léxicos de verbos (casam por lema)
    entry_verbs: frozenset[str]
    exit_verbs: frozenset[str]
    read_verbs: frozenset[str]
    write_verbs: frozenset[str]
    control_verbs: frozenset[str]
    ambiguous_verbs: frozenset[str]

    # Superfície 4 — padrões
    error_re: re.Pattern
    confirm_re: re.Pattern
    ui_re: re.Pattern
    quoted_ui_re: re.Pattern
    persist_re: re.Pattern
    blank_form_re: re.Pattern
    include_uc_re: re.Pattern
    domain_data_re: re.Pattern

    # Superfície 5 — rótulos de seção de caso de uso (comparados em minúsculas)
    uc_main_labels: frozenset[str]
    uc_alt_prefix: tuple[str, ...]
    uc_exception_prefix: tuple[str, ...]
    uc_skip_prefix: tuple[str, ...]

    # Superfície 5 (cont.) — rótulos de CHAVE de cabeçalho de caso de uso, em
    # _parse_uc (comparados em minúsculas contra a chave antes do "\t"). A
    # chave "" (vazia) NÃO entra aqui: é estrutural — marca uma linha de
    # cabeçalho sem rótulo, não uma palavra de idioma nenhum.
    uc_id_labels: frozenset[str]
    uc_primary_actor_labels: frozenset[str]
    uc_actor_prefix: tuple[str, ...]
    uc_secondary_actor_prefix: tuple[str, ...]

    # Superfície 6 — prefixos de inferência de ator em _infer_actor (comparados
    # no início do texto em minúsculas via str.startswith, por isso tuple).
    actor_system_prefixes: tuple[str, ...]
    actor_user_prefixes: tuple[str, ...]

    # Superfície 6 (cont.) — padrões aplicados direto ao texto de entrada em
    # ner_ontology_pipeline.py (measure() e _validate_write_override), migrados
    # das antigas constantes de módulo _RESUME_RE/_VALIDATE_WRITE_RE/
    # _WRITE_OBJ_RE/_DISPLAY_READ_RE/_DISPLAY_MSG_RE.
    resume_re: re.Pattern
    validate_write_re: re.Pattern
    write_obj_re: re.Pattern
    display_read_re: re.Pattern
    display_msg_re: re.Pattern

    # Superfície 1 — rótulos de dependência, por predicado
    # "attr" é a construção copular do esquema inglês do spaCy ("X is the Y"),
    # tratada como objeto pelo extrator. Fica DENTRO do predicado para que o
    # perfil concentre toda a dependência de idioma; UD não tem equivalente.
    _dep_obj: frozenset[str] = field(default=frozenset({"dobj", "attr"}))
    _dep_subj: frozenset[str] = field(default=frozenset({"nsubj"}))
    _dep_subj_pass: frozenset[str] = field(default=frozenset({"nsubjpass"}))
    _dep_relcl: frozenset[str] = field(default=frozenset({"relcl"}))
    _dep_neg: frozenset[str] = field(default=frozenset({"neg"}))
    _neg_lemmas: frozenset[str] = field(default=frozenset())

    # Rótulos de dependência que identificam um token VERBO como ação
    # relevante em srl_extractor.py::_extract_from_sent. "relcl" é o rótulo
    # do esquema inglês do spaCy para oração relativa com pronome "which";
    # Universal Dependencies usa "acl:relcl" (e às vezes "acl" simples, como
    # em orações com "that"/"que"). Sem parametrizar isto, um verbo em
    # oração relativa em português ("o pedido que o cliente criou") nunca
    # seria reconhecido como ação — movimento perdido, em silêncio. Placeholder
    # verbatim do inglês aqui; a Task 9 preenche o valor UD no perfil PORTUGUESE.
    verb_clause_deps: frozenset[str] = field(
        default=frozenset({"ROOT", "conj", "advcl", "xcomp", "ccomp", "relcl"}))

    # Classes de verbo para a INFERÊNCIA DE MOVIMENTOS IMPLÍCITOS (só no caminho
    # de produto, gated em measure(infer_implicit=True); default vazio → inglês/tese
    # intocados). Um processo de escrita implica Entry+Write; um de consulta,
    # Read+Exit — o motor completa o padrão quando o insumo pobre expõe só um lado.
    implicit_write_verbs: frozenset[str] = frozenset()
    implicit_query_verbs: frozenset[str] = frozenset()
    implicit_send_verbs: frozenset[str] = frozenset()
    # Verbos de cálculo/derivação (calcular, computar…): o processo LÊ o insumo
    # armazenado e EMITE o resultado → Read+Exit (mesmo par que consulta). Campo
    # próprio (não misturado em query) para o sinal de completude ser explícito.
    # Default VAZIO → inglês intocado; preenchido só no PT.
    implicit_compute_verbs: frozenset[str] = frozenset()
    # Marcadores de PROPÓSITO ("para/pra <infinitivo>"): uma cláusula advcl com
    # essa marca é uma ação-fim ("informar o peso PARA calcular o frete" = 2
    # ações), NÃO uma subordinada condicional/temporal (se/quando/porque). Só
    # promovida quando o parser dá a estrutura correta (recalibração p/ Stanza).
    # Default VAZIO → inglês intocado.
    purpose_marks: frozenset[str] = frozenset()
    # Deps extra seguidas na travessia de coordenação (xcomp/parataxis) — o
    # verbo de ação pende de um verbo leve. Default VAZIO → inglês intocado.
    coord_frontier_extra_deps: frozenset[str] = frozenset()

    # Verbos modais que governam um infinitivo de ação via `xcomp`
    # ("o sistema DEVE receber, consultar e exibir"). O decompositor desce do
    # modal ao infinitivo para que a coordenação dos infinitivos vire ações
    # atômicas. Default VAZIO → no-op (inglês intocado); preenchido só no PT.
    modal_verbs: frozenset[str] = frozenset()

    # Fallback por forma de superfície (ligado quando a lematização é fraca)
    match_surface_form: bool = False

    # Sub-léxicos de desambiguação DENTRO das regras. Sem eles, cada porte
    # para um novo idioma reintroduz literais crus em R3, R7, R8, R9 e R10.
    exit_external_request_verbs: frozenset[str] = field(
        default=frozenset({"request", "send", "forward", "transmit"}))
    exit_return_ui_verbs: frozenset[str] = field(default=frozenset({"return"}))
    ambiguous_read_verbs: frozenset[str] = field(
        default=frozenset({"check", "validate", "verify"}))
    ambiguous_process_verbs: frozenset[str] = field(
        default=frozenset({"process", "handle", "manage", "execute", "perform", "run"}))
    control_click_verbs: frozenset[str] = field(default=frozenset({"click", "press"}))
    control_browse_verbs: frozenset[str] = field(default=frozenset({"browse"}))
    entry_select_verbs: frozenset[str] = field(default=frozenset({"select", "choose"}))
    entry_confirm_verbs: frozenset[str] = field(default=frozenset({"confirm"}))
    entry_external_send_verbs: frozenset[str] = field(
        default=frozenset({"send", "signal", "reply", "respond"}))

    # Normalização de grupo de dados para deduplicação
    article_re: re.Pattern = field(default_factory=lambda: re.compile(r"^(the|a|an)\s+"))
    plural_suffix: str = "s"
    plural_exceptions: tuple[str, ...] = ("ss", "us", "is")

    # Valores canônicos para ActionTuple sintética (_validate_write_override)
    canonical_system_actor: str = "system"
    canonical_write_verb: str = "record"
    default_data_label: str = "data"

    # Objeto de preposição: o esquema inglês usa prep->pobj; UD usa case+obl/nmod
    _dep_prep: frozenset[str] = field(default=frozenset({"prep"}))
    _dep_pobj: frozenset[str] = field(default=frozenset({"pobj"}))

    # ── Predicados ────────────────────────────────────────────────────
    def is_direct_object(self, token) -> bool:
        return token.dep_ in self._dep_obj

    def is_subject(self, token) -> bool:
        return token.dep_ in self._dep_subj or token.dep_ in self._dep_subj_pass

    def is_passive_subject(self, token) -> bool:
        return token.dep_ in self._dep_subj_pass

    def is_relative_clause(self, token) -> bool:
        return token.dep_ in self._dep_relcl

    def is_negation(self, token) -> bool:
        if token.dep_ not in self._dep_neg:
            return False
        if not self._neg_lemmas:
            return True
        return token.lemma_.lower() in self._neg_lemmas

    def is_system_actor(self, actor: str) -> bool:
        return actor.strip().lower() in self.system_names

    def is_prepositional_object(self, token) -> bool:
        """Objeto de preposição.

        Inglês (esquema ClearNLP do spaCy): a preposição é um token `prep`
        e o substantivo é seu filho `pobj`. Universal Dependencies não tem
        nenhum dos dois: a preposição vira `case` presa ao substantivo, e
        o substantivo pende do verbo como `obl` ou `nmod`.
        """
        return token.dep_ in self._dep_pobj

    def is_preposition(self, token) -> bool:
        return token.dep_ in self._dep_prep

    def verb_key(self, token) -> str:
        """Chave usada para casar contra os léxicos."""
        return token.lemma_.lower()

    def verb_keys(self, token) -> tuple[str, ...]:
        """Chaves candidatas, incluindo o fallback de superfície."""
        lemma = token.lemma_.lower()
        if self.match_surface_form:
            return (lemma, token.text.lower())
        return (lemma,)

    @property
    def all_verb_lemmas(self) -> frozenset[str]:
        """União dos seis léxicos de verbo — todos os lemas conhecidos deste idioma.

        Usada como fallback de reconhecimento (ex.: ner_ontology_pipeline decidir
        se um lema extraído pelo decompositor é "conhecido" antes de tentar
        corrigi-lo via nova varredura do doc spaCy)."""
        return (
            self.entry_verbs | self.exit_verbs | self.read_verbs
            | self.write_verbs | self.control_verbs | self.ambiguous_verbs
        )


# ── Perfil inglês — CÓPIA VERBATIM de cosmic_ontology.py:72-150 ───────

ENGLISH = LanguageProfile(
    code="en",
    spacy_model="en_core_web_sm",
    system_names=frozenset({"system", "the system"}),
    entry_verbs=frozenset({
        "enter", "fill", "input", "submit", "select", "choose", "provide",
        "inform", "type", "specify", "insert", "pick", "mark", "tick", "set",
        "upload", "attach", "give", "pass", "indicate", "send", "signal",
        "reply", "respond", "confirm", "import",
        "approve", "accept", "reject", "vote", "receive", "deny",
    }),
    exit_verbs=frozenset({
        "display", "show", "list", "present", "output", "render", "print",
        "exhibit", "generate", "produce", "report", "view", "visualize",
        "summarize", "highlight", "notify", "ask", "request", "send",
        "forward", "transmit", "return", "export", "emit", "publish",
        "broadcast", "expose", "prompt", "warn", "alert",
        "open", "redirect", "communicate", "transition",
    }),
    read_verbs=frozenset({
        "search", "query", "fetch", "retrieve", "read", "find", "get", "load",
        "obtain", "seek", "scan", "inspect", "locate", "filter", "access",
        "look", "lookup", "consult", "extract", "pull",
    }),
    write_verbs=frozenset({
        "save", "store", "write", "record", "persist", "update", "modify",
        "delete", "remove", "create", "register", "add", "log", "archive",
        "overwrite", "append", "erase", "empty", "clear", "change", "insert",
        "alter", "approve", "accept", "reject", "deny", "assign",
        "activate", "deactivate", "enable", "disable", "lock", "unlock",
    }),
    control_verbs=frozenset({
        "click", "navigate", "close", "cancel", "abort", "resume", "open",
        "launch", "exit", "refresh", "undo", "go", "press", "browse",
        "return",
    }),
    ambiguous_verbs=frozenset({
        "check", "validate", "verify", "process", "handle", "manage",
        "execute", "perform", "run",
    }),
    error_re=re.compile(
        r"\b(error|fail|invalid|exception|warning|not found|unable|denied|"
        r"reject|unsuccessful|alert|not valid|inconsistenc)\w*\b",
        re.IGNORECASE,
    ),
    confirm_re=re.compile(
        r"\b(confirm|success|complete|done|acknowledg|notification)\w*\b",
        re.IGNORECASE,
    ),
    ui_re=re.compile(
        r"\b(button|menu|tab|screen|page|option|sub-option|link|form|icon|"
        r"checkbox|dropdown|dialog|modal|window|panel|bar|interface|section)\b",
        re.IGNORECASE,
    ),
    quoted_ui_re=re.compile(
        r'["“”‘’\'](Cancel|Back|OK|Close|Confirm|Save|Reset|Submit)'
        r'["“”‘’\']',
        re.IGNORECASE,
    ),
    persist_re=re.compile(
        r"\b(persistent storage|database|db|file|table|records?|storage|repository)\b",
        re.IGNORECASE,
    ),
    blank_form_re=re.compile(
        r"\b(?:show|display|present|render)\w*\s+(?:a\s+)?(?:blank\s+)?"
        r"(?:formatted?\s+)?(?:form|screen)\b(?!\s+containing)",
        re.IGNORECASE,
    ),
    include_uc_re=re.compile(
        r"include\s+the\s+use\s+case|resume\w*\s+at\s+step|"
        r"(?:if\s+)?(?:at|in)\s+step\s+\d+\s+of\s+the\s+main|"
        r"if\s+(at\s+|in\s+)?step\s+\d+.*use\s+case|"
        r"the\s+use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)|"
        r"use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)",
        re.IGNORECASE,
    ),
    domain_data_re=re.compile(
        r"\b(details?|list|results?|data|records?|information|status|"
        r"configuration|settings?|profile|summary|history|report)\b",
        re.IGNORECASE,
    ),
    uc_main_labels=frozenset({"main scenario", "main flow"}),
    uc_alt_prefix=("alternative flow",),
    uc_exception_prefix=("exception",),
    uc_skip_prefix=("reference table", "reference diagram", "gui:", "precondition"),
    uc_id_labels=frozenset({"id"}),
    uc_primary_actor_labels=frozenset({"primary actor"}),
    uc_actor_prefix=("actor",),
    uc_secondary_actor_prefix=("secondary actor",),
    actor_system_prefixes=("the system ", "system "),
    actor_user_prefixes=("the actor ", "the user "),
    resume_re=re.compile(
        r"resumes?\s+(at\s+)?step|include\s+the\s+use\s+case|"
        r"(?:if\s+)?(?:at|in)\s+step\s+\d+\s+of\s+the\s+main|"
        r"if\s+(?:at\s+|in\s+)?step\s+\d+|"
        r"the\s+use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)|"
        r"use\s+case\s+\S+.*?\s+is\s+(initiated|started|included)",
        re.IGNORECASE,
    ),
    # Padrão "valida/checa ... e grava/salva X" (conserto de cancelamento, alta
    # precisão). Verbo de validação seguido (na MESMA sentença, sem ponto) de
    # verbo de escrita.
    validate_write_re=re.compile(
        r"\b(?:check|checks|checked|validate|validates|validated|verif\w+|ensure|ensures)\b"
        r"[^.]*\b(?:record|records|recorded|save|saves|saved|store|stores|stored|"
        r"persist\w*|registers?|registered)\b",
        re.IGNORECASE,
    ),
    write_obj_re=re.compile(
        r"\b(?:record|records|recorded|save|saves|saved|store|stores|stored|"
        r"persist\w*|registers?|registered)\b\s+"
        r"(?:the\s+|a\s+|an\s+|new\s+)*([a-z][\w'\- ]*?)"
        r"(?:\s+(?:in|to|into|on)\b|[.,;:]|$)",
        re.IGNORECASE,
    ),
    # Read implícito na exibição (escopo: formato de SENTENÇA ÚNICA,
    # sentence_mode). Princípio COSMIC: exibir um grupo de dados ARMAZENADO
    # exige um Read antes do Exit. Numa sentença isolada (cada uma é um
    # processo funcional próprio, sem Entry/Read anterior no mesmo processo)
    # esse Read precisa ser contado; num UC multi-passo o dado pode já ter
    # sido lido/digitado antes — por isso a regra é gateada por sentence_mode
    # (não dispara em De Vito). Validada held-out (Zenodo 70/30, seed 42): MAE
    # 0,374→0,345, SA +0,091→+0,161. Ver resultados-otimizacao-zenodo-sentencas.md.
    display_read_re=re.compile(
        r"\b(?:show|shows|display|displays|displayed|list|lists|listed|"
        r"present|presents)\b",
        re.IGNORECASE,
    ),
    # Objetos que NÃO são grupo de dados armazenado (mensagem/controle/artefato
    # recém-gerado) — exibi-los é só Exit, sem Read. Excluem falsos positivos.
    display_msg_re=re.compile(
        r"\b(?:message|notification|alert|confirmation|error|warning|popup|pop-up|"
        r"page|button|interface|tab|menu|screen|dialog|email|e-mail|homepage|"
        r"home\s+page|code|proof)\b",
        re.IGNORECASE,
    ),
)


# ── Perfil português ──────────────────────────────────────────────────
# Somente Superfícies 1 (rótulos de dependência UD) e 2 (nomes de sistema),
# mais os rótulos de seção/cabeçalho de caso de uso (Superfície 5) — sem eles
# _parse_uc não segmenta NENHUM documento do corpus e o pipeline mediria 0 CFP
# em silêncio, o que contradiz o próprio propósito desta task. Os léxicos de
# verbo (Superfície 3), os padrões regex (Superfície 4), os prefixos de
# inferência de ator e os regexes de texto cru (Superfície 6) e os
# sub-léxicos de desambiguação entram na Task 9, ajustados EXCLUSIVAMENTE
# sobre a partição de desenvolvimento (dev-20). Cada campo abaixo é
# preenchido EXPLICITAMENTE (mesmo quando o placeholder é "vazio") para que
# nenhum campo esquecido herde silenciosamente o default do perfil ENGLISH —
# um dataclass field(default=...) não avisa quando fica de fora de uma
# chamada de construtor.

_EMPTY_RE = re.compile(r"(?!x)x")  # nunca casa; placeholder explícito p/ Task 9
_NEVER_SUFFIX = "￿"  # sentinela que nenhuma palavra real termina; placeholder
                          # seguro p/ plural_suffix (uma string vazia faria
                          # str.endswith("") ser sempre True e truncaria toda
                          # palavra em cosmic_ontology.py, corrompendo a dedup)

# Task 9 — Léxicos de verbo (Superfície 3), ajustados EXCLUSIVAMENTE sobre o
# dev-20 (kappa_subset). Ponto de partida: tradução da AMPLITUDE INTEIRA dos
# seis léxicos ingleses (não só os poucos verbos do rascunho do brief) — ver
# scripts/tune_pt_profile_on_dev.py para o histórico de rodadas.
#
# ARMADILHA DE LEMATIZAÇÃO (pt_core_news_sm, modelo "_sm", pequeno): para
# vários verbos regulares de 1ª conjugação (-ar), a 3ª pessoa do singular do
# presente do indicativo NÃO é reduzida ao infinitivo — o lematizador ora
# devolve a própria forma de superfície, ora (para a MESMA palavra, em
# contexto ligeiramente diferente) confunde a marcação POS com substantivo/
# adjetivo homógrafo (ex.: "consulta", "grava"→"grar", "extrai", "removar").
# Como o perfil casa por LEMA (match_surface_form=False, Task 2: lematização
# adequada) e a extração em srl_extractor.py exige tok.pos_ == "VERB", incluir
# a forma alternativa no frozenset é seguro (só importa quando o spaCy already
# marcou o token como VERB) e é a única forma de não perder o movimento em
# silêncio. Cada forma extra abaixo foi confirmada rodando
# pt_core_news_sm 3.8.14 em frases-modelo ("Sistema <verbo> os dados [para
# X].") — ver notas por verbo abaixo. Verbos onde o modelo NUNCA marca VERB
# (ex.: "clicar"→sempre NOUN "clica"/"clico"; "armazenar", "votar",
# "submeter", "navegar", "limpar", "imprimir" → frequentemente ADJ) são uma
# limitação do tagger, não do léxico — mantidos aqui defensivamente (custo
# zero) e documentados no relatório da Task 9.
_PT_ENTRY_VERBS = frozenset({
    "informar", "preencher", "inserir", "digitar", "submeter", "submete",
    "enviar", "selecionar", "escolher", "escolhe", "fornecer", "especificar",
    "marcar", "assinalar", "definir", "configurar", "carregar", "anexar",
    "anexo", "indicar", "sinalizar", "responder", "confirmar", "importar",
    "aprovar", "aceitar", "rejeitar", "votar", "vota", "receber", "negar",
})
_PT_EXIT_VERBS = frozenset({
    "exibir", "mostrar", "apresentar", "listar", "imprimir", "imprime",
    "gerar", "produzir", "reportar", "relatar", "visualizar", "ver",
    "resumir", "destacar", "notificar", "solicitar", "pedir", "enviar",
    "encaminhar", "transmitir", "retornar", "retorno", "voltar", "volta",
    "devolver", "exportar", "emitir", "emitar", "publicar", "divulgar",
    "disponibilizar", "expor", "avisar", "alertar", "abrir", "redirecionar",
    "comunicar", "convidar", "esclarecer",
})
_PT_READ_VERBS = frozenset({
    "pesquisar", "buscar", "procurar", "consultar", "consulta", "ler", "lê",
    "obter", "recuperar", "encontrar", "localizar", "carregar",
    "inspecionar", "filtrar", "filtra", "acessar", "extrair", "extrai",
    "descobrir",
})
_PT_WRITE_VERBS = frozenset({
    "salvar", "armazenar", "armazena", "gravar", "grar", "escrever",
    "registrar", "persistir", "atualizar", "modificar", "alterar", "editar",
    "excluir", "remover", "removar", "apagar", "deletar", "deleta", "limpar",
    "limpa", "criar", "cadastrar", "adicionar", "arquivar", "sobrescrever",
    "sobrescreve", "anexar", "anexo", "inserir", "aprovar", "aceitar",
    "rejeitar", "negar", "atribuir", "ativar", "ativo", "desativar",
    "habilitar", "desabilitar", "bloquear", "desbloquear", "desbloqueia",
})
_PT_CONTROL_VERBS = frozenset({
    "clicar", "clica", "clico", "navegar", "navega", "fechar", "cancelar",
    "cancela", "abortar", "aborto", "retomar", "abrir", "iniciar", "inicia",
    "lançar", "sair", "atualizar", "desfazer", "ir", "pressionar", "aperta",
    "apertar", "voltar", "volta", "retornar", "retorno", "percorrer",
})
_PT_AMBIGUOUS_VERBS = frozenset({
    "verificar", "checar", "conferir", "validar", "processar", "tratar",
    "gerenciar", "administrar", "executar", "realizar", "rodar",
})

PORTUGUESE = LanguageProfile(
    code="pt",
    spacy_model="pt_core_news_sm",

    # Superfície 2 — nomes de sistema. Conjunto amplo observado no corpus
    # PT-BR (De Vito + Zenodo traduzido): "sistema"/"o sistema" dominam, mas
    # "aplicação"/"plataforma"/"software" também aparecem como o próprio ator
    # do UC. O ator composto "User/System" do Zenodo (linha "1 User/System
    # ...") NÃO entra aqui de propósito: não é um nome de sistema puro, então
    # is_system_actor deve retornar False para ele — mesmo comportamento do
    # inglês, que também não reconhece atores compostos.
    system_names=frozenset({
        "sistema", "o sistema", "a aplicação", "aplicação",
        "a plataforma", "plataforma", "o software", "software",
    }),

    # Superfície 3 — léxicos de verbo. Task 9 (amplitude traduzida dos seis
    # léxicos ingleses — ver _PT_*_VERBS acima e task-9-report.md).
    entry_verbs=_PT_ENTRY_VERBS,
    exit_verbs=_PT_EXIT_VERBS,
    read_verbs=_PT_READ_VERBS,
    write_verbs=_PT_WRITE_VERBS,
    control_verbs=_PT_CONTROL_VERBS,
    ambiguous_verbs=_PT_AMBIGUOUS_VERBS,

    # Modais PT que prendem o infinitivo de ação via xcomp. "O sistema DEVE
    # receber, consultar e exibir" ancorava tudo em "dever" (1 ação); descendo
    # ao infinitivo, os coordenados (receber/consultar/exibir) viram 3 ações.
    modal_verbs=frozenset({"dever", "poder", "precisar", "querer",
                           "necessitar", "conseguir", "ir",
                           # invólucros de permissão: "deve PERMITIR cadastrar"
                           # = o sistema cadastra; desce ao verbo de ação.
                           "permitir", "possibilitar", "habilitar"}),

    # Inferência de implícitos (produto): processo de escrita → Entry+Write;
    # de consulta → Read+Exit.
    implicit_write_verbs=frozenset({
        "criar", "cadastrar", "registrar", "gravar", "salvar", "persistir",
        "armazenar", "anexar", "adicionar", "inserir", "atualizar", "alterar",
        "editar", "modificar", "remover", "excluir", "deletar", "apagar",
        "marcar", "confirmar", "aprovar", "definir", "configurar", "agendar",
        # v7 (dev): operações de persistência que jogavam o motor no mínimo.
        "reservar", "descontar", "reembolsar", "prescrever", "documentar",
        "liberar", "candidatar", "reescrever"}),
    implicit_query_verbs=frozenset({
        "consultar", "buscar", "ver", "visualizar", "exibir", "mostrar",
        "apresentar", "listar", "exportar", "gerar", "obter", "pesquisar",
        "filtrar", "acessar", "relatar",
        "rastrear"}),  # v7 (dev): rastrear = ler estado e devolver → Read+Exit
    # Envio a outro usuário funcional (email/notificação) → Exit.
    implicit_send_verbs=frozenset({
        "enviar", "mandar", "notificar", "avisar", "comunicar", "encaminhar",
        "remeter", "disparar", "alertar"}),
    # Cálculo/derivação → Read (insumo) + Exit (resultado). v7 (dev).
    implicit_compute_verbs=frozenset({
        "calcular", "computar", "recalcular", "apurar", "estimar", "determinar"}),
    # Cláusula de propósito "para/pra <infinitivo>" = ação-fim (recalibração parser).
    purpose_marks=frozenset({"para", "pra"}),
    # Deps extra na travessia de coordenação (recalibração Stanza): o verbo de
    # ação real frequentemente pende de um verbo leve via xcomp ("consegue
    # MARCAR") ou parataxis ("recebe lembretes"). Segue-os, atravessando modais.
    coord_frontier_extra_deps=frozenset({"xcomp", "parataxis"}),

    # Superfície 4 — padrões regex sobre texto. Task 9.
    error_re=re.compile(
        r"\b(erro|falha|falhou|inválid|exceção|excecao|aviso|não encontrad|"
        r"nao encontrad|incapaz|negad|rejeit|malsucedid|inconsistên|inconsisten)\w*\b",
        re.IGNORECASE,
    ),
    confirm_re=re.compile(
        r"\b(confirma|sucesso|conclu|finaliz|realizad|notifica)\w*\b",
        re.IGNORECASE,
    ),
    ui_re=re.compile(
        r"\b(botão|botao|menu|aba|tela|página|pagina|opção|opcao|link|campo|"
        r"formulário|formulario|ícone|icone|caixa|seletor|diálogo|dialogo|"
        r"modal|janela|painel|barra|interface|seção|secao)\b",
        re.IGNORECASE,
    ),
    quoted_ui_re=re.compile(
        r'["“”‘’\']'
        r'(Cancelar|Voltar|OK|Fechar|Confirmar|Salvar|Limpar|Enviar|Excluir|'
        r'Remover|Sim|Não|Nao)'
        r'["“”‘’\']',
        re.IGNORECASE,
    ),
    persist_re=re.compile(
        r"\b(armazenamento persistente|banco de dados|base de dados|arquivo|"
        r"tabela|registros?|armazenamento|repositório|repositorio)\b",
        re.IGNORECASE,
    ),
    # Espelha blank_form_re do perfil ENGLISH acima (~lc 260-264):
    #   r"\b(?:show|display|present|render)\w*\s+(?:a\s+)?(?:blank\s+)?
    #     (?:formatted?\s+)?(?:form|screen)\b(?!\s+containing)"
    # Estrutura espelhada termo a termo:
    #   - "(?:show|display|present|render)\w*" -> "(?:exib|mostr|apresent)\w*"
    #     (verbo de exibição, \w* cobre conjugações);
    #   - "(?:a\s+)?" -> "(?:um\s+|o\s+|a\s+)?" (artigo opcional, já existia);
    #   - "(?:blank\s+)?" era OBRIGATÓRIO na versão anterior deste regex
    #     (exigia "em branco"/"vazio" literal) — corrigido para OPCIONAL,
    #     igual ao inglês: exibir um formulário/tela SEM qualificador nenhum
    #     já conta como "em branco" por padrão (COSMIC: superfície vazia sem
    #     conteúdo declarado = sem movimento). Em português o adjetivo vem
    #     DEPOIS do substantivo ("formulário vazio", "tela em branco"), ao
    #     contrário do inglês ("a blank form") — por isso o grupo
    #     "(?:em\s+branco|vazi\w+)" fica APÓS formulário/tela, não antes;
    #   - "(?!\s+containing)" -> "(?!\s+(?:contendo|com\s+(?:o|a|os|as)\b))"
    #     (lookahead negativo: um formulário/tela exibido CONTENDO dados tem
    #     conteúdo -> não é "sem movimento", é Exit normal. Confirmado contra
    #     o corpus real: datasets/pt-br/devito/ALBERGATE_FP38.txt:17 "O
    #     sistema exibe um formulário para buscar pedidos CONTENDO os
    #     seguintes campos: mesa, data inicial, data final e status." — essa
    #     frase NÃO deve casar como formulário em branco).
    blank_form_re=re.compile(
        r"\b(?:exib|mostr|apresent)\w*\s+(?:um\s+|uma\s+|o\s+|a\s+)?"
        r"(?:formulário|formulario|tela)\b"
        r"(?:\s+(?:em\s+branco|vazi\w+))?"
        r"(?!\s+(?:contendo|com\s+(?:o|a|os|as)\b))",
        re.IGNORECASE,
    ),
    include_uc_re=re.compile(
        r"inclui\s+o\s+caso\s+de\s+uso|retoma\w*\s+no\s+passo|"
        r"(?:se\s+)?no\s+passo\s+\d+\s+do\s+(?:cenário|cenario)\s+principal|"
        r"o\s+caso\s+de\s+uso\s+\S+.*?\s+é\s+(iniciado|incluído|incluido)",
        re.IGNORECASE,
    ),
    domain_data_re=re.compile(
        r"\b(detalhes?|lista|resultados?|dados|registros?|informaç\w+|"
        r"informac\w+|status|situação|situacao|configuraç\w+|configurac\w+|"
        r"perfil|resumo|histórico|historico|relatório|relatorio|critérios?|"
        r"criterios?)\b",
        re.IGNORECASE,
    ),

    # Superfície 5 — rótulos de seção de _parse_uc. Valores confirmados
    # contra o corpus real (ver task-8-report.md para o inventário):
    #   - De Vito traduziu "MAIN SCENARIO" -> "CENÁRIO PRINCIPAL" (bare é
    #     comparado em minúsculas, então a caixa alta do arquivo não importa);
    #   - o Zenodo (300 arquivos) manteve "MAIN SCENARIO" em inglês;
    #   - "fluxo alternativo" cobre as variantes numeradas via startswith
    #     ("fluxo alternativo n.º 01", etc.);
    #   - "exceç" (sem acento/plural fixo) cobre "Exceções" (dominante, 101x),
    #     "Exceção" e "Exceção n.º 01" de uma vez; "fluxo de exceção" é uma
    #     variante à parte que não compartilha esse prefixo.
    uc_main_labels=frozenset({"cenário principal", "fluxo principal", "main scenario"}),
    uc_alt_prefix=("fluxo alternativo", "alternative flow"),
    uc_exception_prefix=("exceç", "fluxo de exceção", "exception"),
    uc_skip_prefix=("tabela de referência", "diagrama de referência", "gui:",
                    "pré-condição", "pre-condicao", "reference table",
                    "reference diagram", "precondition"),

    # Superfície 5 (cont.) — rótulos de CHAVE de cabeçalho. "Description"
    # (Zenodo, em inglês) e "Descrição" (De Vito) apareceriam aqui se o
    # LanguageProfile tivesse um campo de rótulo dedicado a descrição — não
    # tem (_parse_uc não usa nenhuma chave "description"/"descrição"; o campo
    # "name" do UC vem de uc_id_labels ou de um cabeçalho sem chave), então
    # não há nada a preencher para esse rótulo específico.
    uc_id_labels=frozenset({"id"}),
    uc_primary_actor_labels=frozenset({"ator principal", "primary actor"}),
    uc_actor_prefix=("ator", "actor"),
    uc_secondary_actor_prefix=("ator secundário", "atores secundários", "secondary actor"),

    # Superfície 6 — prefixos de inferência de ator em texto cru. Task 9.
    # NOTA (corrigida nas correções da Task 9): _infer_actor em
    # ner_ontology_pipeline.py retornava o literal inglês "System" (não lido
    # do perfil) quando actor_system_prefixes casava — como is_system_actor
    # compara contra system_names (que não contém "system" bare), esse ramo
    # ficava com is_system=False mesmo casando o prefixo, classificando quase
    # todo passo de sistema como ação de usuário. Corrigido: _infer_actor
    # agora retorna self.profile.canonical_system_actor. Preenchido aqui
    # porque é a Superfície 6 corretamente pertencente a este perfil, e o
    # _parse_uc normal (actor explícito por passo, tab-separado) não passa
    # por este fallback.
    actor_system_prefixes=("o sistema ", "sistema "),
    actor_user_prefixes=("o usuário ", "o ator ", "a usuária "),

    # Superfície 6 (cont.) — regexes de texto cru usados fora do spaCy
    # (measure()/_validate_write_override). Task 9. São regexes de SUPERFÍCIE
    # (formas conjugadas via \w*), não de lema — por isso incluem os radicais
    # que cobrem as conjugações reais (grav\w*, salv\w*, etc.), incluindo a
    # forma fixa do adaptador Connextra PT (grava/consulta/exibe/informa).
    resume_re=re.compile(
        r"retoma\w*\s+no\s+passo|inclui\s+o\s+caso\s+de\s+uso|"
        r"(?:se\s+)?no\s+passo\s+\d+\s+do\s+(?:cenário|cenario)\s+principal|"
        r"o\s+caso\s+de\s+uso\s+\S+.*?\s+é\s+(iniciado|incluído|incluido)",
        re.IGNORECASE,
    ),
    # Padrão "valida/verifica ... e grava/salva X" (conserto de alta
    # precisão dos cancelamentos, espelhando validate_write_re do inglês).
    validate_write_re=re.compile(
        r"\b(?:verific\w+|valid\w+|check\w+|confir\w+|garant\w+|assegur\w+)\b"
        r"[^.]*\b(?:grav\w*|grar\w*|salv\w*|armazen\w*|persist\w*|registr\w*)\b",
        re.IGNORECASE,
    ),
    write_obj_re=re.compile(
        r"\b(?:grav\w*|salv\w*|armazen\w*|persist\w*|registr\w*)\b\s+"
        r"(?:o\s+|a\s+|os\s+|as\s+|um\s+|uma\s+|novo\s+|nova\s+)*"
        r"([a-zà-ÿ][\wà-ÿ'\- ]*?)"
        r"(?:\s+(?:em|no|na|nos|nas|para|ao|à)\b|[.,;:]|$)",
        re.IGNORECASE,
    ),
    # Read implícito na exibição (formato de sentença única — mesma lógica
    # do inglês, ver display_read_re/display_msg_re em ENGLISH acima).
    display_read_re=re.compile(
        r"\b(?:exib\w*|mostr\w*|apresent\w*|list\w*)\b",
        re.IGNORECASE,
    ),
    display_msg_re=re.compile(
        r"\b(mensage\w*|notificaç\w*|notificac\w*|alert\w*|confirmaç\w*|"
        r"confirmac\w*|erro\w*|aviso\w*|popup|pop-up|página|pagina|botão|"
        r"botao|interface|aba|menu|tela|diálogo|dialogo|e-?mail|"
        r"página\s+inicial|pagina\s+inicial|código|codigo|comprovante)\b",
        re.IGNORECASE,
    ),

    # Superfície 1 — rótulos de dependência, Universal Dependencies v2 (o
    # esquema que pt_core_news_sm usa, diferente do esquema estilo ClearNLP
    # do modelo inglês). "attr" (copular) não tem equivalente UD dedicado —
    # por isso NÃO entra em _dep_obj: UD trata o predicativo de cópula como
    # "root" com o verbo "ser"/"estar" como cop, não como filho do verbo
    # principal; sem ocorrência disso no corpus, deixamos _dep_obj só com
    # "obj" (não introduzimos um rótulo não verificado).
    _dep_obj=frozenset({"obj"}),
    _dep_subj=frozenset({"nsubj"}),
    _dep_subj_pass=frozenset({"nsubj:pass"}),
    _dep_relcl=frozenset({"acl:relcl"}),
    _dep_neg=frozenset({"advmod"}),
    # Não vazio de propósito: é o que impede que TODO advmod (ex.: "sempre",
    # "rapidamente") seja lido como negação — is_negation só checa lema
    # quando esta lista existe (ver docstring do predicado acima).
    _neg_lemmas=frozenset({"não", "nunca", "jamais", "nem"}),

    # Rótulos de oração relativa/subordinada para reconhecimento de VERBO em
    # srl_extractor.py. UD acrescenta "acl:relcl" (equivalente a "relcl" do
    # esquema inglês) e "acl" (oração adnominal sem pronome relativo — comum
    # no corpus PT em construções como "a solicitação de X", geradas pelo
    # adaptador Connextra no ramo genérico; sem "acl" aqui, esses verbos
    # nunca seriam vistos por extract_roles/is_negation). Valor prescrito
    # (Task 9): ver task-9-report.md para exemplos reais do corpus dev-20
    # onde "acl" é o único dep_ do verbo relevante.
    verb_clause_deps=frozenset(
        {"ROOT", "conj", "advcl", "xcomp", "ccomp", "acl:relcl", "acl"}),

    # Perfil PT casa por lema (Task 2 confirmou lematização adequada, 98,2%).
    match_surface_form=False,

    # Sub-léxicos de desambiguação (dentro das regras R3/R7/R8/R9/R10). Task 9.
    # Mesma armadilha de lematização documentada acima nas listas _PT_*_VERBS;
    # cada sub-léxico é um SUBCONJUNTO do léxico principal correspondente
    # (mesma relação do perfil ENGLISH).
    exit_external_request_verbs=frozenset({
        "solicitar", "pedir", "enviar", "encaminhar", "transmitir", "convidar",
    }),
    exit_return_ui_verbs=frozenset({"retornar", "retorno", "voltar", "volta"}),
    ambiguous_read_verbs=frozenset({"verificar", "checar", "conferir", "validar"}),
    ambiguous_process_verbs=frozenset({
        "processar", "tratar", "gerenciar", "administrar", "executar",
        "realizar", "rodar",
    }),
    control_click_verbs=frozenset({
        "clicar", "clica", "clico", "pressionar", "aperta", "apertar",
    }),
    control_browse_verbs=frozenset({"percorrer"}),
    entry_select_verbs=frozenset({"selecionar", "escolher", "escolhe"}),
    entry_confirm_verbs=frozenset({"confirmar"}),
    entry_external_send_verbs=frozenset({"enviar", "sinalizar", "responder"}),

    # Normalização de grupo de dados para deduplicação. Task 9.
    # article_re: artigos definidos/indefinidos portugueses (o/a/os/as/
    # um/uma/uns/umas), mesmo papel do "^(the|a|an)\s+" inglês.
    article_re=re.compile(r"^(o|a|os|as|um|uma|uns|umas)\s+", re.IGNORECASE),
    # plural_suffix/plural_exceptions: a morfologia de plural do português NÃO
    # é só "tirar s" (há -ão→-ões/-ães/-ãos, -l→-is, -m→-ns, -r/-z→+es) — uma
    # heurística de sufixo único não pode cobrir isso com segurança. Optamos
    # por um valor CONSERVADOR: reaproveitar a heurística "-s" do inglês
    # (mesma "-s" final é também o padrão dominante de plural em português
    # para os substantivos de domínio deste corpus: dado(s), registro(s),
    # usuário(s), resultado(s)), com as MESMAS exceções do inglês (ss/us/is),
    # que por coincidência fonética também protegem os invariantes em -us/-is
    # do português (status, vírus, ônibus, lápis, tênis). O QUE ESTA
    # HEURÍSTICA NÃO RESOLVE: plurais em -ão/-ões (ex.: "informações" vs
    # "informação" não colapsam para a mesma chave de dedup — "informações"
    # vira "informaçõe", que não bate com o singular "informação"). Isso é
    # SEGURO (não funde palavras diferentes por engano) mas incompleto
    # (não deduplica esse padrão específico) — aceitável porque, como o
    # próprio campo aponta, a dedup por grupo de dados é secundária face à
    # correção do léxico de verbos (Superfície 3, o alvo real desta task).
    plural_suffix="s",
    plural_exceptions=("ss", "us", "is"),

    # Valores canônicos para ActionTuple sintética (_validate_write_override).
    # canonical_write_verb usa o infinitivo "gravar" (não o lema "grar" do
    # spaCy) — ambos estão em write_verbs, então R5 dispara de qualquer
    # forma; "gravar" foi escolhido por legibilidade no rastro de regras.
    canonical_system_actor="sistema",
    canonical_write_verb="gravar",
    default_data_label="dados",

    # Objeto de preposição: UD usa case+obl/nmod, sem equivalente a prep/pobj.
    # Sem ocorrência verificada no corpus para mapear com segurança; vazio
    # (Task 9, se necessário).
    _dep_prep=frozenset(),
    _dep_pobj=frozenset(),
)
