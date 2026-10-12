"""Adaptadores de formato PT (M1) — imperativa, Given-When-Then e informal.

Seguem o padrão do connextra_adapter: extraem a(s) cláusula(s) de AÇÃO do
molde do formato e reusam a geração canônica de passos (`_split_subactions`,
`_verb_class`, `_build_uc`) — mesma regra 4.6 (escritas coordenadas = processos
separados; leituras coordenadas = um processo). Determinístico, grátis,
explicável. Só PT (lang="pt").

GWT foge do padrão dos outros dois porque o molde já separa gatilho de
resposta: o QUANDO é o evento de gatilho (Entradas) e o ENTÃO é a resposta do
sistema (uma movimentação por cláusula) — tudo dentro de UM processo
funcional, não vários. Ver `gwt_to_ucs`.
"""
from __future__ import annotations
import re

from .connextra_adapter import (_QUERY_VERBS_PT, _WRITE_VERBS_PT, _build_uc,
                                _clean_obj, _split_subactions, _verb_class)

# ── Imperativa: "O sistema deve/tem que/precisa [permitir] <ação>" ──────────
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:o\s+sistema\s+|a\s+aplica\w+\s+|a\s+plataforma\s+)?"
    r"(?:deve|dever[áa]|tem\s+que|precisa|necessita|ir[áa])\s+"
    r"(?:(?:poder|permitir|possibilitar|habilitar)\s+(?:que\s+\S+\s+)?)?"
    r"(?P<action>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Envio a outro usuário funcional (regra 4.2: notificação = Exit). O léxico do
# connextra_adapter não tem essa classe (Connextra raramente a expressa); GWT e
# informal expressam muito ("então envia SMS", "me avisa quando...").
_SEND_VERBS_PT = frozenset({
    "enviar", "mandar", "notificar", "avisar", "comunicar", "encaminhar",
    "disparar", "alertar", "transmitir", "remeter",
})
# Escrita não coberta pelo léxico CRUD do Connextra: GWT/informal dizem o que
# o sistema FAZ com o dado ("efetua a transferência", "bloqueia as datas"),
# não o verbo CRUD.
_WRITE_EXTRA_PT = frozenset({
    "gravar", "salvar", "armazenar", "persistir", "efetuar", "realizar",
    "processar", "bloquear", "liberar", "reservar", "debitar", "creditar",
    "reembolsar", "estornar", "lançar", "acumular", "emitir",
})
# Respostas do sistema que só precisam ser RECONHECIDAS como ação (a classe
# fica genérica = Exit): sem elas a cláusula é lida como coordenação nominal e
# a movimentação se perde ("o gerente recebe notificação", "a prescrição
# chega na farmácia"). Modais ficam de fora de propósito — "pode levantar"
# tem que ancorar em "levantar", não em "poder".
_RESPONSE_VERBS_PT = frozenset({
    "receber", "chegar", "confirmar", "calcular", "entrar", "sair", "aguardar",
    "validar", "sugerir", "manter", "oferecer", "apresentar", "responder",
    "retornar", "devolver", "levantar", "medir", "integrar", "colocar",
    "solicitar", "tentar", "escolher", "clicar", "acessar", "prescrever",
    "submeter", "candidatar", "atingir", "cair", "subir", "transferir",
})
_KNOWN_VERBS_PT = (_WRITE_VERBS_PT | _QUERY_VERBS_PT | _SEND_VERBS_PT
                   | _WRITE_EXTRA_PT | _RESPONSE_VERBS_PT)

_SYS_PREFIX_RE = re.compile(
    r"^\s*(?:ent[aã]o\s+|depois\s+|ap[oó]s\s+isso\s+|em\s+seguida\s+)?"
    r"(?:o\s+sistema|a\s+aplica\w+|a\s+plataforma|o\s+app)\s+",
    re.IGNORECASE)
_COORD_RE = re.compile(r",\s*(?:e\s+)?|\s+e\s+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ\-]+")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


# ── Lematização mínima de superfície ────────────────────────────────────────
# Os léxicos PT do connextra_adapter são infinitivos, mas GWT e informal vêm
# com verbo FLEXIONADO ("então envia", "o sistema calcula", "me avisa"). Aqui
# só desfazemos as flexões que esses dois formatos realmente usam (3ª pessoa
# do presente, gerúndio, pretérito) — não é um lematizador geral, e por isso
# só aceitamos o palpite quando ele cai num verbo conhecido.
_INFLEX = (
    (r"ando$", "ar"), (r"endo$", "er"), (r"indo$", "ir"),   # gerúndio
    (r"ou$", "ar"), (r"eu$", "er"), (r"iu$", "ir"),         # pretérito 3ª
    (r"ei$", "ar"), (r"aram$", "ar"), (r"eram$", "er"),
    (r"am$", "ar"), (r"em$", "er"),                          # 3ª plural
)


def _lemma_pt(word: str) -> str:
    """Infinitivo do verbo se o palpite cair num verbo conhecido; senão a
    própria palavra (o chamador trata 'não é verbo conhecido')."""
    w = word.lower().strip(".,;:!?'\"()")
    if not w or w in _KNOWN_VERBS_PT:
        return w
    cands = [w + "r"]                       # 3ª sing. -ar: "envia" → "enviar"
    if w.endswith("e"):                     # 3ª sing. -er/-ir: "exibe"/"recebe"
        cands += [w[:-1] + "ir", w + "r"]
    for suf, inf in _INFLEX:
        if re.search(suf, w):
            cands.append(re.sub(suf, inf, w))
    for c in cands:
        if c in _KNOWN_VERBS_PT:
            return c
    return w


def _find_verb(part: str) -> tuple[int, str] | None:
    """(índice do token, infinitivo) do primeiro verbo conhecido da cláusula."""
    toks = _TOKEN_RE.findall(part)
    for i, t in enumerate(toks):
        lem = _lemma_pt(t)
        if lem in _KNOWN_VERBS_PT:
            return i, lem
    return None


# O léxico de consulta do Connextra junta duas coisas que, DENTRO de uma
# resposta do sistema, são movimentações diferentes: exibir manda dado para
# fora (Exit) e consultar/verificar lê da base (Read). Na user story isso não
# aparece porque o processo inteiro é gerado do zero (E+R+X); no ENTÃO cada
# cláusula é UMA movimentação, então a distinção passa a valer.
_DISPLAY_VERBS_PT = frozenset({
    "exibir", "mostrar", "apresentar", "listar", "visualizar",
    "pré-visualizar", "informar",
})


def _movement_class(verb: str | None) -> str:
    """Classe de movimentação da cláusula de resposta do sistema."""
    if verb is None:
        return "generic"
    if verb in _DISPLAY_VERBS_PT or verb in _SEND_VERBS_PT:
        return "send"          # dado cruza a fronteira para fora = Exit
    if verb in _WRITE_EXTRA_PT:
        return "write"
    vc = _verb_class(verb, "pt")
    return "write" if vc in ("create", "update", "delete") else vc


# Sufixos que denunciam verbo sem depender de léxico (infinitivo, gerúndio,
# pretérito e 3ª do plural). "-a"/"-e" ficam de fora: colidem com o grosso dos
# substantivos PT ("hora", "etapa", "entrega") e transformariam coordenação
# nominal em movimentação nova.
_VERB_SUF_RE = re.compile(r"^\w+(?:ando|endo|indo|ar|er|ir|ou|iu|eu|am|em)$",
                          re.IGNORECASE)
_NAO_VERBO = frozenset({  # fecham em sufixo verbal mas são palavras de apoio
    "sistema", "problema", "item", "imagem", "ordem", "origem", "viagem",
    "mensagem", "margem", "porcentagem", "nuvem", "também", "além", "ninguém",
    "alguém", "bem", "vem", "tem", "sem", "em", "com", "por", "ter", "ser",
    "poder", "dever", "querer", "ir", "estar", "haver", "par",
})


def _tem_verbo(part: str) -> tuple[int, str | None] | None:
    """(índice do token, infinitivo conhecido|None) se a parte traz uma AÇÃO.

    Duas evidências, nesta ordem: verbo do léxico (dá também a classe) ou
    sufixo verbal inequívoco (só detecta — a classe fica genérica).
    """
    achado = _find_verb(part)
    if achado is not None:
        return achado
    for i, t in enumerate(_TOKEN_RE.findall(part)):
        if _VERB_SUF_RE.match(t) and t.lower() not in _NAO_VERBO:
            return i, None
    return None


def _obj_after_verb(part: str, idx: int | None) -> str:
    """Objeto = o que vem depois do verbo (ou a cláusula inteira)."""
    toks = _TOKEN_RE.findall(part)
    rest = toks[idx + 1:] if idx is not None else toks
    return _clean_obj(" ".join(rest), lang="pt") if rest else "a solicitação"


def _split_clause(clause: str) -> list[tuple[str, str]]:
    """Divide uma cláusula em movimentações: [(objeto, classe), ...].

    Só vira movimentação nova a parte que traz uma AÇÃO; parte coordenada sem
    verbo é continuação do mesmo grupo de dados. É o que separa "exibe
    localização, hora e etapa" (um Exit de um grupo de dados) de "grava o
    pedido e envia o comprovante" (dois).

    O trecho que vem ANTES da primeira ação é sujeito ou adjunto ("o gerente"
    em "o gerente recebe notificação"; "após aprovação" em "após aprovação, o
    sistema bloqueia as datas") — contexto, não grupo de dados, e por isso
    fica fora do objeto. Só quando a cláusula inteira não tem ação nenhuma ele
    vira a movimentação, para não perder a resposta do sistema.
    """
    partes: list[list] = []   # [idx_verbo, classe, [trechos]]
    antes: list[str] = []
    for bruto in _COORD_RE.split(_SYS_PREFIX_RE.sub("", clause.strip())):
        p = bruto.strip().rstrip(".;")
        if not p:
            continue
        achado = _tem_verbo(p)
        if achado is None:
            if partes:
                partes[-1][2].append(p)   # mesmo grupo de dados
            else:
                antes.append(p)
            continue
        idx, lem = achado
        partes.append([idx, _movement_class(lem), [p]])
    if not partes:
        return [(_clean_obj(" ".join(antes), lang="pt"), "generic")] if antes else []
    return [(_obj_after_verb(" ".join(trechos), idx), cls)
            for idx, cls, trechos in partes]


def _uc_from_steps(text: str, fp_id: str, steps: list[tuple[str, str]]) -> str:
    """Monta o pseudo-UC a partir de passos já resolvidos [(ator, corpo)]."""
    linhas = ["ID", f"{fp_id}\t{text[:60]}", f"Description\t{_flat(text)}",
              "MAIN SCENARIO"]
    for n, (ator, corpo) in enumerate(steps, 1):
        linhas.append(f"{n}\t{ator}\t{corpo}")
    return "\n".join(linhas) + "\n"


def _build_from_action(text: str, fp_id: str, action: str) -> list[str] | None:
    """Roteia a cláusula de ação pela geração canônica (regra 4.6)."""
    subs = _split_subactions(action, "pt")
    if not subs:
        return None
    classes = [_verb_class(v, "pt") for v, _ in subs]
    # leitura/análise coordenada = um processo (regra 4.3)
    if len(subs) > 1 and all(c in ("read", "compute", "generic") for c in classes):
        read_first = next(((v, o) for (v, o), c in zip(subs, classes) if c == "read"),
                          subs[0])
        subs = [read_first]
    if len(subs) > 1:  # escritas distintas = processos separados (regra 4.6)
        return [_build_uc(text, f"{fp_id}-p{i+1}", "Usuário", [sa], "pt")
                for i, sa in enumerate(subs)]
    return [_build_uc(text, fp_id, "Usuário", subs, "pt")]


def imperative_to_ucs(text: str, fp_id: str, lang: str = "pt") -> list[str] | None:
    """"O sistema deve <ação>" → pseudo-UC(s). None se não casar o molde."""
    if lang != "pt":
        return None
    m = _IMPERATIVE_RE.match(_flat(text))
    if not m:
        return None
    return _build_from_action(text, fp_id, m.group("action").strip())


# ── GWT ─────────────────────────────────────────────────────────────────────
_DADO_RE = re.compile(r"\bdado\s+(?:que\s+)?\b", re.IGNORECASE)
_QUANDO_RE = re.compile(r"\bquando\b", re.IGNORECASE)
_ENTAO_RE = re.compile(r"\bent[aã]o\b", re.IGNORECASE)
_DET_RE = re.compile(r"^(?:o|a|os|as|um|uma|uns|umas|seu|sua|meu|minha|"
                     r"esse|essa|este|esta|cada|que)$", re.IGNORECASE)


_ESTADO_RE = re.compile(
    r"^(?:foi|foram|[eé]|s[aã]o|est[aá]|est[aã]o|tem|t[eê]m|tinha|possui|"
    r"havia|h[aá]|com|sem|de|do|da|em|no|na)$", re.IGNORECASE)


_PARTICIPIO_RE = re.compile(r"^\w{3,}(?:ado|ada|ados|adas|ido|ida|idos|idas)$",
                            re.IGNORECASE)


def _np_head(clause: str, n: int = 3) -> str:
    """Sintagma nominal de abertura da cláusula (o sujeito da pré-condição).

    Para no primeiro verbo, cópula ou particípio: "uma encomenda foi
    despachada e possui código" → "encomenda". O particípio precisa parar
    junto porque o motor o lê como ação — deixar "paciente cadastrado" no
    grupo de dados fazia uma consulta virar Write.
    """
    saida: list[str] = []
    for t in _TOKEN_RE.findall(clause):
        if _DET_RE.match(t):
            continue
        if (_ESTADO_RE.match(t) or _PARTICIPIO_RE.match(t)
                or _tem_verbo(t) is not None):
            break
        saida.append(t)
        if len(saida) >= n:
            break
    return " ".join(saida) if saida else "os dados"


def _gwt_clauses(text: str) -> tuple[str, str, str] | None:
    """(dado, quando, entao). Aceita o molde por palavra-chave e, no texto
    cru de 3 linhas, a posição — os itens crus omitem "então" ("...faz
    colheita / atualiza estoque")."""
    flat = _flat(text)
    mq, me = _QUANDO_RE.search(flat), _ENTAO_RE.search(flat)
    if mq and me and me.start() > mq.end():
        md = _DADO_RE.search(flat[:mq.start()])
        dado = flat[md.end():mq.start()] if md else flat[:mq.start()]
        return dado.strip(" ;,."), flat[mq.end():me.start()].strip(" ;,."), \
            flat[me.end():].strip(" ;,.:")
    linhas = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(linhas) >= 3:  # cru sem palavra-chave: DADO / QUANDO / ENTÃO
        d = _DADO_RE.sub("", linhas[0]).strip()
        q = _QUANDO_RE.sub("", linhas[1]).strip()
        e = _ENTAO_RE.sub("", " ".join(linhas[2:])).strip()
        return d, q, e
    return None


def gwt_to_ucs(text: str, fp_id: str, lang: str = "pt") -> list[str] | None:
    """GWT → UM pseudo-UC. None se não houver estrutura QUANDO…ENTÃO.

    Ao contrário de Connextra/imperativa, o molde GWT já vem decomposto e
    descreve UM processo funcional: o QUANDO é o evento de gatilho (Entrada
    por cláusula de ação) e o ENTÃO é a resposta do sistema (uma movimentação
    por cláusula — Write para escrita, Exit para envio/confirmação, Read+Exit
    para consulta). Por isso a regra 4.6 NÃO se aplica aqui: cláusulas
    coordenadas do ENTÃO são movimentações do mesmo processo, não processos
    novos — o gatilho é único.
    """
    if lang != "pt":
        return None
    cl = _gwt_clauses(text)
    if cl is None:
        return None
    dado, quando, entao = cl
    if not (quando and entao):
        return None

    steps: list[tuple[str, str]] = []
    for obj, _cls in _split_clause(quando) or [("a solicitação", None)]:
        steps.append(("Usuário", f"informa os dados de {obj}"))
    # DADO = estado já persistido que o processo precisa ler para responder
    # (regra 4.1: o processo lê o que a pré-condição estabelece). Não é
    # cosmético: sem este Read o dev cai de MAE 0.708 para 1.125 e o viés vai
    # a -1.04 — a pré-condição é a leitura que faltava. O grupo de dados é o
    # SUJEITO da pré-condição ("dado que uma encomenda foi despachada e possui
    # código" → "encomenda"), não a cláusula inteira — senão o motor acha
    # vários candidatos a Read dentro do mesmo grupo.
    if dado:
        steps.append(("Sistema", f"consulta {_np_head(dado)} na base de dados"))
    leu = bool(dado)
    saiu = False
    for obj, cls in _split_clause(entao):
        if cls == "write":
            steps.append(("Sistema", f"grava {obj} na base de dados"))
        elif cls == "read":  # consulta/verificação interna: só lê
            steps.append(("Sistema", f"consulta {obj} na base de dados"))
            leu = True
        elif cls == "send":
            steps.append(("Sistema", f"envia {obj} para Usuário"))
            saiu = True
        else:  # confirmação/resposta genérica do sistema = Exit
            steps.append(("Sistema", f"exibe {obj} para Usuário"))
            saiu = True
    if saiu and not leu:
        # não se exibe o que não se leu: sem DADO nem consulta explícita, a
        # leitura do processo fica implícita no ENTÃO (regra 4.3).
        steps.insert(len(steps) - 1,
                     ("Sistema", f"consulta {_np_head(entao)} na base de dados"))
    if len(steps) < 2:
        return None
    return [_uc_from_steps(text, fp_id, steps)]


# ── Informal ────────────────────────────────────────────────────────────────
# Texto livre, sem molde: nenhuma decomposição confiável em processos. O que o
# diagnóstico mostra é subcontagem sistemática (viés -1.4) — o motor não acha
# passo nenhum na fala solta. O adaptador ancora a AÇÃO PRINCIPAL e emite UM
# processo funcional canônico; decompor em vários exigiria distinguir ação de
# adorno conversacional, o que este formato não sinaliza.
_MODAL_RE = re.compile(
    r"\b(?:quero|queria|gostaria|preciso|precisa\w*|pode|poder|podia|"
    r"consegue|conseguir|deve|deveria|d[aá]\s+pra|querendo|t[oô]\s+querendo)\b",
    re.IGNORECASE)
_INFINITIVO_RE = re.compile(r"^[a-zà-ÿ]+(?:ar|er|ir)$", re.IGNORECASE)
# Passo de confirmação do gerador canônico. O texto é contrato fixo do
# `_steps_for_pt` do connextra_adapter — test_informal_nao_conta_confirmacao
# quebra alto se ele mudar.
_CONFIRMACAO = "confirmação de"


def _sem_confirmacao(uc: str) -> str:
    """Tira o Exit de confirmação e renumera os passos.

    Numa user story ("quero cadastrar X") a confirmação faz parte do que o
    autor pediu; em texto livre não há evidência nenhuma de que ela exista, e
    o COSMIC só conta como Exit a mensagem que carrega dado do objeto de
    interesse — um "salvo com sucesso" seco não é movimentação. É a diferença
    que o painel independente marcou 4/4 nos itens de escrita ("marcar entrega
    como concluída" = E+W, sem X), e vale 0.083 de MAE lá.
    """
    linhas, n = [], 0
    for l in uc.split("\n"):
        if _CONFIRMACAO in l:
            continue
        campos = l.split("\t")
        if len(campos) == 3 and campos[0].isdigit():
            n += 1
            campos[0] = str(n)
            l = "\t".join(campos)
        linhas.append(l)
    return "\n".join(linhas)
_RUIDO_RE = re.compile(
    r"\b(?:tipo\s+assim|tipo|sabe|mesmo|memo|n[eé]|a\s+gente|s[oó]\s+isso|"
    r"de\s+verdade|mesma\s+coisa)\b", re.IGNORECASE)


def informal_to_ucs(text: str, fp_id: str, lang: str = "pt") -> list[str] | None:
    """Texto informal → UM pseudo-UC canônico. None se não achar ação."""
    if lang != "pt":
        return None
    flat = _RUIDO_RE.sub(" ", _flat(text))
    flat = re.sub(r"\s+", " ", flat).strip()
    toks = _TOKEN_RE.findall(flat)
    if not toks:
        return None

    # Duas evidências de ação, da mais forte para a mais fraca. Sufixo verbal
    # sozinho NÃO basta aqui (basta no GWT, onde o molde já garante que a
    # cláusula é resposta do sistema): em texto livre ele dispara em frase que
    # não descreve processo nenhum — "melhorar o desempenho" (NFR) e "tá dando
    # erro" (relato de bug) viravam processos de 3 CFP contra gabarito 0.
    alvo: tuple[str, int] | None = None
    achado = _find_verb(flat)
    if achado is not None:                      # 1) verbo do léxico (dá classe)
        alvo = (achado[1], achado[0])
    else:
        m = _MODAL_RE.search(flat)              # 2) infinitivo após modal
        if m is not None:
            pos = len(_TOKEN_RE.findall(flat[:m.end()]))
            i = next((i for i in range(pos, min(pos + 3, len(toks)))
                      if _INFINITIVO_RE.match(toks[i])), None)
            if i is not None:
                alvo = (toks[i], i)
    if alvo is None:
        return None
    verbo, idx = alvo
    obj = " ".join(toks[idx + 1:]).strip() or "a solicitação"
    return [_sem_confirmacao(_build_uc(text, fp_id, "Usuário", [(verbo, obj)], "pt"))]
