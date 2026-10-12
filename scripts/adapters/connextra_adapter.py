"""Connextra → pseudo-UC adapter (normalizador de formato para user stories).

Converte "As a <role>, I want to <action>, so that <benefit>" em um cenário
UC canônico que o parser do pipeline NER+Ontologia (E33d) consegue medir.

Fonte das regras: docs/estimativa-cosmic-tese/r4-annotation-protocol.md
(o padrão de medição usado pelos anotadores humanos), NÃO o ground truth:
  - Passo 2 (protocolo): todo processo funcional inicia com Entry de gatilho.
  - Passo 3: uma US típica tem 1 Exit (resultado/confirmação do processo).
  - Passo 4: Reads implícitas obrigatórias (ler antes de atualizar/deletar;
    ler para exibir/listar/validar).
  - Passo 5: criar/atualizar/deletar → 1 Write no objeto.
  - Passo 6: cláusula "so that" é contexto — descartada (limitação documentada:
    benefícios que revelam Exits extras não são capturados).
  - Regra 4.1: verbos de gestão genérica ("manage") → mínimo razoável E+X+R+W.
  - Regra 4.3: view/see/list/search → E (critérios) + R (ler) + X (exibir).
  - Regra 4.6 (adjudicação r2): CRUD explícito coordenado ("create, update and
    delete X") decompõe em um processo funcional por verbo.

Desenvolvimento calibrado APENAS no subconjunto dev (20 US kappa_subset);
avaliação única no restante (78) — ver three-annotator-analysis.md.
"""
from __future__ import annotations

import re

# Lexicons derivados das tabelas do protocolo (§3 passos 2-5, §4.1, §4.3).
CREATE_VERBS = {
    "add", "create", "insert", "register", "record", "upload", "submit",
    "post", "publish", "import", "attach", "assign", "set", "define",
    "specify", "apply", "schedule", "build", "generate", "make", "include",
}
READ_VERBS = {
    "view", "see", "search", "find", "list", "display", "show", "browse",
    "look", "locate", "access", "read", "check", "review", "monitor",
    "track", "explore", "discover", "watch", "know", "understand",
    "identify", "compare", "verify", "select", "filter", "sort", "download",
    "export", "retrieve", "query", "assess", "evaluate", "analyze",
    "visualize", "preview", "trace", "determine",
}
UPDATE_VERBS = {
    "update", "edit", "change", "modify", "rename", "move", "reorder",
    "configure", "adjust", "customize", "enable", "disable", "activate",
    "deactivate", "approve", "reject", "mark", "flag", "merge", "link",
    "associate", "organize", "restore", "reset", "rate", "tag", "annotate",
    "correct", "fix", "revise", "indicate",
}
DELETE_VERBS = {"delete", "remove", "deselect", "unlink", "cancel", "clear", "deaccession"}
SEND_VERBS = {
    "send", "notify", "email", "share", "invite", "alert", "broadcast",
    "communicate", "report", "forward", "distribute",
}
MANAGE_VERBS = {"manage", "administer", "maintain", "handle", "control"}
AUTH_VERBS = {"login", "log", "authenticate", "signin"}
# Computação sobre dados: protocolo passo 4 (ler dados de entrada é
# obrigatório) + passo 5 (resultado persistido) + passo 3 (resultado exibido).
COMPUTE_VERBS = {
    "perform", "compute", "calculate", "process", "convert", "transform",
    "aggregate", "summarize", "measure", "estimate", "predict", "detect",
}

ALL_ACTION_VERBS = (CREATE_VERBS | READ_VERBS | UPDATE_VERBS | DELETE_VERBS
                    | SEND_VERBS | MANAGE_VERBS | AUTH_VERBS | COMPUTE_VERBS)

CONNEXTRA_RE = re.compile(
    r"^\s*as\s+an?\s+(?P<role>[^,]+?)\s*,\s*"
    r"i\s+(?:want|would\s+like|need|wish|am\s+able|should\s+be\s+able)\s*"
    r"(?:to\s+)?(?:be\s+able\s+to\s+)?(?P<action>.+?)"
    r"(?:\s*,?\s*so\s+that\s+.+)?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# "create, update and delete X" / "add new X and edit existing X"
COORD_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+|\s+or\s+", re.IGNORECASE)

# Frame português: "Como (um/uma/o/a) <papel>, (eu) quero/gostaria de/preciso/
# desejo/devo poder <ação>[, para que/a fim de/de modo que <benefício>]."
CONNEXTRA_RE_PT = re.compile(
    r"^\s*como\s+(?:um[a]?\s+|o\s+|a\s+)?(?P<role>[^,]+?)\s*,\s*"
    r"(?:eu\s+)?(?:quero|gostaria\s+de|preciso|desejo|devo\s+poder)\s*"
    r"(?:poder\s+)?(?P<action>.+?)"
    r"(?:\s*,?\s*(?:para\s+que|a\s+fim\s+de|de\s+modo\s+que)\s+.+)?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# "criar, atualizar e excluir X" / "filtrar e ordenar X"
COORD_SPLIT_RE_PT = re.compile(r",\s*(?:e\s+)?|\s+e\s+|\s+ou\s+", re.IGNORECASE)

_FRAMES = {"en": (CONNEXTRA_RE, COORD_SPLIT_RE),
           "pt": (CONNEXTRA_RE_PT, COORD_SPLIT_RE_PT)}

# Classes de verbo em português (mesmas famílias das inglesas, usadas para
# selecionar o verbo/ator dos passos gerados quando lang == "pt").
#
# Correção de paridade de léxico: o inglês (CREATE_VERBS, UPDATE_VERBS,
# DELETE_VERBS, READ_VERBS acima, linhas 28-40) cobre ~97 verbos nessas
# quatro classes (as únicas que _verb_class/_steps_for_pt reconhecem para
# português — send/manage/auth/compute do inglês, linhas 41-60, não têm
# branch de geração de passos em português e por isso não entram aqui: um
# verbo dessas classes cai no branch "generic" de _steps_for_pt, o mesmo
# fallback que o próprio inglês usa para qualquer verbo fora de suas 8
# classes, ex. "connect" em "As a parent, I want to be able to connect with
# the staff in case of an emergency" -> datasets/pt-br/user-stories/FP034.txt
# "entrar em contato com a equipe"). Os conjuntos abaixo traduzem a amplitude
# INTEIRA de CREATE_VERBS/UPDATE_VERBS/DELETE_VERBS/READ_VERBS (cada verbo
# comentado cita o verbo inglês espelhado); antes só ~20 verbos cobriam uma
# fração pequena disso, e casos reais do corpus ficavam em "generic" por
# omissão de tradução (não por serem estruturalmente genéricos): "indicar"
# (mirror de "indicate", UPDATE_VERBS), "verificar" (mirror de "verify",
# READ_VERBS), "especificar" (mirror de "specify", CREATE_VERBS), "avançar"
# (mirror de "move", UPDATE_VERBS — fonte real: datasets/raw/
# user-stories-dalpiaz/g02-federalspending.txt "I want to move on to round 2
# of Homepage edits" == datasets/pt-br/user-stories/FP001.txt "eu quero
# avançar para a rodada 2"), "saber" (mirror de "know", READ_VERBS).
_CREATE_VERBS_PT = frozenset({
    "criar",       # create
    "cadastrar",   # register
    "adicionar",   # add
    "inserir",     # insert
    "registrar",   # record
    "submeter",    # submit
    "postar",      # post
    "publicar",    # publish
    "importar",    # import
    "anexar",      # attach
    "atribuir",    # assign
    "definir",     # set / define
    "especificar",  # specify
    "aplicar",     # apply
    "agendar",     # schedule
    "construir",   # build
    "gerar",       # generate
    "incluir",     # include
    "fazer",       # make
})
_UPDATE_VERBS_PT = frozenset({
    "atualizar",    # update
    "editar",       # edit
    "alterar",      # change
    "modificar",    # modify
    "renomear",     # rename
    "mover",        # move
    "avançar",      # move (on) — realização idiomática real do corpus
    "reordenar",    # reorder
    "configurar",   # configure
    "ajustar",      # adjust
    "customizar",   # customize
    "personalizar",  # customize (sinônimo)
    "habilitar",    # enable
    "desabilitar",  # disable
    "ativar",       # activate
    "desativar",    # deactivate
    "aprovar",      # approve
    "rejeitar",     # reject
    "marcar",       # mark
    "sinalizar",    # flag
    "mesclar",      # merge
    "vincular",     # link
    "associar",     # associate
    "organizar",    # organize
    "restaurar",    # restore
    "redefinir",    # reset
    "classificar",  # rate
    "etiquetar",    # tag
    "anotar",       # annotate
    "corrigir",     # correct
    "consertar",    # fix
    "revisar",      # revise
    "indicar",      # indicate
})
_DELETE_VERBS_PT = frozenset({
    "excluir",       # delete
    "remover",       # remove
    "apagar",        # delete (sinônimo)
    "deletar",       # delete (sinônimo)
    "desmarcar",     # deselect
    "desvincular",   # unlink
    "cancelar",      # cancel
    "limpar",        # clear
    "desincorporar",  # deaccession
})
_QUERY_VERBS_PT = frozenset({
    "consultar",       # query / consult / lookup
    "buscar",          # search
    "pesquisar",       # search (sinônimo)
    "procurar",        # search (sinônimo)
    "visualizar",      # view / visualize
    "ver",             # see
    "listar",          # list
    "filtrar",         # filter
    "ordenar",         # sort
    "exportar",        # export
    "acompanhar",      # track
    "encontrar",       # find
    "exibir",          # display
    "mostrar",         # show
    "navegar",         # browse
    "olhar",           # look
    "extrair",         # extract
    "puxar",           # pull
    "checar",          # check
    "monitorar",       # monitor
    "explorar",        # explore
    "descobrir",       # discover
    "observar",        # watch
    "saber",           # know
    "entender",        # understand
    "identificar",     # identify
    "comparar",        # compare
    "verificar",       # verify
    "selecionar",      # select
    "baixar",          # download
    "recuperar",       # retrieve
    "analisar",        # assess / analyze
    "avaliar",         # evaluate
    "pré-visualizar",  # preview
    "rastrear",        # trace
    "determinar",      # determine
    "conferir",        # review
})
# União derivada (mesmo padrão de ALL_ACTION_VERBS acima, linha 62-63: uma
# constante calculada a partir das classes finas, não mantida à mão em
# paralelo — evita as duas listas divergirem por esquecimento).
_WRITE_VERBS_PT = _CREATE_VERBS_PT | _UPDATE_VERBS_PT | _DELETE_VERBS_PT


def _verb_class(verb: str, lang: str = "en") -> str:
    v = verb.lower()
    if lang == "pt":
        if v in _DELETE_VERBS_PT:
            return "delete"
        if v in _UPDATE_VERBS_PT:
            return "update"
        if v in _QUERY_VERBS_PT:
            return "read"
        if v in _CREATE_VERBS_PT:
            return "create"
        return "generic"
    if v in MANAGE_VERBS:
        return "manage"
    if v in AUTH_VERBS:
        return "auth"
    if v in DELETE_VERBS:
        return "delete"
    if v in UPDATE_VERBS:
        return "update"
    if v in READ_VERBS:
        return "read"
    if v in SEND_VERBS:
        return "send"
    if v in CREATE_VERBS:
        return "create"
    if v in COMPUTE_VERBS:
        return "compute"
    return "generic"


def _split_subactions(action: str, lang: str = "en") -> list[tuple[str, str]]:
    """Divide a cláusula de ação em sub-ações (verbo, objeto).

    Trata coordenação de verbos com objeto compartilhado ("create, update and
    delete any records") e coordenação de VPs ("add new sources and edit
    existing sources"). Retorna [(verbo, objeto), ...].
    """
    action = action.strip().rstrip(".")
    words = action.split()
    if not words:
        return []

    coord_re = COORD_SPLIT_RE_PT if lang == "pt" else COORD_SPLIT_RE
    all_action_verbs = ((_WRITE_VERBS_PT | _QUERY_VERBS_PT) if lang == "pt"
                        else ALL_ACTION_VERBS)
    and_word = r"e" if lang == "pt" else r"and"
    or_word = r"ou" if lang == "pt" else r"or"

    # Coordenação exclusivamente por "or"/"ou" = caminhos alternativos do
    # MESMO processo funcional (um único evento de gatilho) — não decompõe.
    if re.search(rf"\b{or_word}\b", action, re.IGNORECASE) and not re.search(
            rf",|\b{and_word}\b", action, re.IGNORECASE):
        first = re.split(r"\s+", action, 1)
        verb = first[0]
        obj = first[1].strip() if len(first) > 1 else "the request"
        return [(verb, obj)]

    # Caso 1: verbos coordenados no início com objeto compartilhado
    # "create, update and delete <obj>"
    m = re.match(
        rf"^(\w+)\s*,\s*(\w+),?\s+{and_word}\s+(\w+)\s+(.+)$", action,
        re.IGNORECASE,
    )
    if m and all(_verb_class(v, lang) != "generic" or v.lower() in all_action_verbs
                 for v in (m.group(1), m.group(2), m.group(3))):
        verbs = [m.group(1), m.group(2), m.group(3)]
        if all(v.lower() in all_action_verbs for v in verbs):
            obj = m.group(4).strip()
            return [(v, obj) for v in verbs]

    # Caso 2: VPs coordenadas — "v1 obj1 and v2 obj2" (cada parte inicia com verbo)
    parts = coord_re.split(action)
    if len(parts) > 1:
        sub = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            first = re.split(r"\s+", p, 1)
            v = first[0].lower()
            if v in all_action_verbs and len(first) > 1:
                sub.append((first[0], first[1].strip()))
            elif sub:
                # fragmento sem verbo: anexa ao objeto anterior
                pv, po = sub[-1]
                sub[-1] = (pv, f"{po} {and_word} {p}")
            else:
                sub = []
                break
        if len(sub) >= 2:
            return sub

    # Caso 3: ação única — primeiro verbo conhecido + resto como objeto
    first = re.split(r"\s+", action, 1)
    verb = first[0]
    obj = first[1].strip() if len(first) > 1 else "the request"
    return [(verb, obj)]


def _clean_obj(obj: str, max_words: int = 8, lang: str = "en") -> str:
    """Objeto curto para data group (remove subordinadas longas)."""
    obj = re.sub(r"\s*\(.*?\)\s*", " ", obj)
    if lang == "pt":
        obj = re.split(
            r"\s+(?:para\s+que|a\s+fim\s+de|de\s+modo\s+que|quando|se|porque)\s+",
            obj, 1, re.IGNORECASE)[0]
    else:
        obj = re.split(r"\s+(?:so\s+that|when|if|because|in\s+order\s+to)\s+",
                       obj, 1, re.IGNORECASE)[0]
    words = obj.strip().rstrip(".").split()
    default = "os dados" if lang == "pt" else "the data"
    return " ".join(words[:max_words]) if words else default


def _steps_for_pt(role: str, verb: str, obj: str, vc: str, o: str, tag: str) -> list[str]:
    """Gera passos UC canônicos em português para uma sub-ação.

    Vocabulário fixo (contrato com a Task 9 / léxicos PT): `informar` para
    Entry (dado do usuário), `consultar` para Read, `exibir` para Exit,
    `gravar` para Write. Ator do sistema: `Sistema`.
    """
    steps: list[str] = []
    if vc == "read":
        # Regra 4.3: E (critérios) + R (ler) + X (exibir)
        steps.append(f"{role} informa os critérios de busca de {o}")
        steps.append(f"Sistema consulta {o} na base de dados")
        steps.append(f"Sistema exibe {o} para {role}")
    elif vc == "create":
        # Passo 5: criar → Write; Passo 2/3: Entry gatilho + Exit confirmação
        steps.append(f"{role} informa os dados de {o}")
        steps.append(f"Sistema grava o registro de {o} na base de dados")
        steps.append(f"Sistema exibe a confirmação de {tag}")
    elif vc == "update":
        # Passo 4: ler registro atual antes de modificar
        steps.append(f"{role} informa os dados atualizados de {o}")
        steps.append(f"Sistema consulta o {o} atual na base de dados")
        steps.append(f"Sistema grava o {o} atualizado na base de dados")
        steps.append(f"Sistema exibe a confirmação de {tag}")
    elif vc == "delete":
        steps.append(f"{role} informa os dados de {o} para exclusão")
        steps.append(f"Sistema consulta o registro de {o} na base de dados")
        steps.append(f"Sistema grava a exclusão do registro de {o} na base de dados")
        steps.append(f"Sistema exibe a confirmação de {tag}")
    else:
        # genérico: processo mínimo (Entry gatilho + ação crua + Exit resultado)
        steps.append(f"{role} informa a solicitação de {verb} {o}")
        steps.append(f"Sistema {verb} {o}")
        steps.append(f"Sistema exibe o resultado de {tag}")
    return steps


def _steps_for(role: str, verb: str, obj: str, idx: int, lang: str = "en") -> list[str]:
    """Gera passos UC canônicos para uma sub-ação, conforme protocolo §3-§4."""
    vc = _verb_class(verb, lang)
    o = _clean_obj(obj, lang=lang)
    tag = f"{verb.lower()} {o}"  # data groups distintos por processo (anti-dedup)
    role = role.strip() or ("Usuário" if lang == "pt" else "User")
    if lang == "pt":
        return _steps_for_pt(role, verb, obj, vc, o, tag)
    steps: list[str] = []
    if vc == "read":
        # Regra 4.3: E (critérios) + R (ler) + X (exibir)
        steps.append(f"{role} enters the search criteria for {o}")
        steps.append(f"System reads the {o} from the database")
        steps.append(f"System displays the {o} to the {role}")
    elif vc == "create":
        # Passo 5: criar → Write; Passo 2/3: Entry gatilho + Exit confirmação
        steps.append(f"{role} enters the {o} details")
        steps.append(f"System saves the {o} to the database")
        steps.append(f"System displays the confirmation of {tag}")
    elif vc == "update":
        # Passo 4: ler registro atual antes de modificar
        steps.append(f"{role} enters the updated {o} details")
        steps.append(f"System reads the current {o} from the database")
        steps.append(f"System saves the updated {o} to the database")
        steps.append(f"System displays the confirmation of {tag}")
    elif vc == "delete":
        steps.append(f"{role} selects the {o} for deletion")
        steps.append(f"System reads the {o} record from the database")
        steps.append(f"System deletes the {o} record from the database")
        steps.append(f"System displays the confirmation of {tag}")
    elif vc == "send":
        # Regra 4.2: notificação = Exit; Entry de gatilho com os dados a enviar
        steps.append(f"{role} enters the {o} message details")
        steps.append(f"System sends the {o} to the recipient")
        steps.append(f"System displays the confirmation of {tag}")
    elif vc == "auth":
        # Regra 4.4: login canônico — E credenciais + R dados de auth + X resultado
        steps.append(f"{role} enters the login credentials")
        steps.append(f"System reads the user authentication data from the database")
        steps.append(f"System displays the authentication result to the {role}")
    elif vc == "manage":
        # Regra 4.1: mínimo razoável E+X+R+W
        steps.append(f"{role} enters the {o} management request")
        steps.append(f"System reads the current {o} from the database")
        steps.append(f"System saves the modified {o} to the database")
        steps.append(f"System displays the confirmation of {tag}")
    elif vc == "compute":
        # Passo 4 (ler entrada obrigatório) + passo 5 (persistir resultado)
        steps.append(f"{role} enters the request to {verb} {o}")
        steps.append(f"System reads the {o} input data from the database")
        steps.append(f"System saves the {tag} result to the database")
        steps.append(f"System displays the {tag} result to the {role}")
    else:
        # genérico: processo mínimo (Entry gatilho + ação crua + Exit resultado)
        steps.append(f"{role} enters the request to {verb} {o}")
        steps.append(f"System {verb}s the {o}")
        steps.append(f"System displays the result of {tag}")
    return steps


def _build_uc(text: str, fp_id: str, role: str,
              subactions: list[tuple[str, str]], lang: str = "en") -> str:
    system_actor = "Sistema" if lang == "pt" else "System"
    lines = [
        "ID",
        f"{fp_id}\t{text[:60]}",
        f"Description\t{text}",
        "MAIN SCENARIO",
    ]
    n = 0
    for i, (verb, obj) in enumerate(subactions):
        for step in _steps_for(role, verb, obj, i, lang):
            n += 1
            actor = role if step.lower().startswith(role.lower()) else system_actor
            body = (step[len(role):].strip()
                    if step.lower().startswith(role.lower())
                    else step[len(system_actor):].strip())
            lines.append(f"{n}\t{actor}\t{body}")
    return "\n".join(lines) + "\n"


def connextra_to_ucs(text: str, fp_id: str, lang: str = "en") -> list[str] | None:
    """Converte uma user story Connextra em pseudo-UCs (um por processo
    funcional). None se o texto não casar com o frame Connextra.

    Regra 4.6: ações coordenadas com verbos de escrita distintos (CRUD
    explícito) são processos funcionais separados — cada um medido em
    isolamento (Entry de gatilho, confirmação e dedup próprios) e somados.
    Coordenação puramente de leitura (ex.: "filter, sort and search") é UM
    processo (regra 4.3: critérios variáveis ainda são 1 grupo de dados).

    `lang`: "en" (padrão, frame Connextra clássico) ou "pt" (frame
    "Como um <papel>, quero <ação>, para que <benefício>").
    """
    frame_re, _coord_re = _FRAMES[lang]
    m = frame_re.match(text.strip())
    if not m:
        return None
    role = m.group("role").strip()
    action = m.group("action").strip()
    subactions = _split_subactions(action, lang)
    if not subactions:
        return None

    classes = [_verb_class(v, lang) for v, _ in subactions]
    if len(subactions) > 1 and all(c in ("read", "compute") for c in classes):
        # coordenação de consulta/análise → um único processo (regra 4.3:
        # critérios variáveis ainda são 1 grupo de dados de busca)
        read_first = next(
            ((v, o) for (v, o), c in zip(subactions, classes) if c == "read"),
            subactions[0],
        )
        subactions = [read_first]

    if len(subactions) > 1:
        # processos funcionais separados (regra 4.6)
        return [
            _build_uc(text, f"{fp_id}-p{i+1}", role, [sa], lang)
            for i, sa in enumerate(subactions)
        ]
    return [_build_uc(text, fp_id, role, subactions, lang)]


def connextra_to_uc(text: str, fp_id: str, lang: str = "en") -> str | None:
    """Compat: retorna o primeiro UC (uso simples, sem decomposição)."""
    ucs = connextra_to_ucs(text, fp_id, lang)
    return ucs[0] if ucs else None
