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


def _verb_class(verb: str) -> str:
    v = verb.lower()
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


def _split_subactions(action: str) -> list[tuple[str, str]]:
    """Divide a cláusula de ação em sub-ações (verbo, objeto).

    Trata coordenação de verbos com objeto compartilhado ("create, update and
    delete any records") e coordenação de VPs ("add new sources and edit
    existing sources"). Retorna [(verbo, objeto), ...].
    """
    action = action.strip().rstrip(".")
    words = action.split()
    if not words:
        return []

    # Coordenação exclusivamente por "or" = caminhos alternativos do MESMO
    # processo funcional (um único evento de gatilho) — não decompõe.
    if re.search(r"\bor\b", action, re.IGNORECASE) and not re.search(
            r",|\band\b", action, re.IGNORECASE):
        first = re.split(r"\s+", action, 1)
        verb = first[0]
        obj = first[1].strip() if len(first) > 1 else "the request"
        return [(verb, obj)]

    # Caso 1: verbos coordenados no início com objeto compartilhado
    # "create, update and delete <obj>"
    m = re.match(
        r"^(\w+)\s*,\s*(\w+),?\s+and\s+(\w+)\s+(.+)$", action, re.IGNORECASE
    )
    if m and all(_verb_class(v) != "generic" or v.lower() in ALL_ACTION_VERBS
                 for v in (m.group(1), m.group(2), m.group(3))):
        verbs = [m.group(1), m.group(2), m.group(3)]
        if all(v.lower() in ALL_ACTION_VERBS for v in verbs):
            obj = m.group(4).strip()
            return [(v, obj) for v in verbs]

    # Caso 2: VPs coordenadas — "v1 obj1 and v2 obj2" (cada parte inicia com verbo)
    parts = COORD_SPLIT_RE.split(action)
    if len(parts) > 1:
        sub = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            first = re.split(r"\s+", p, 1)
            v = first[0].lower()
            if v in ALL_ACTION_VERBS and len(first) > 1:
                sub.append((first[0], first[1].strip()))
            elif sub:
                # fragmento sem verbo: anexa ao objeto anterior
                pv, po = sub[-1]
                sub[-1] = (pv, f"{po} and {p}")
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


def _clean_obj(obj: str, max_words: int = 8) -> str:
    """Objeto curto para data group (remove subordinadas longas)."""
    obj = re.sub(r"\s*\(.*?\)\s*", " ", obj)
    obj = re.split(r"\s+(?:so\s+that|when|if|because|in\s+order\s+to)\s+",
                   obj, 1, re.IGNORECASE)[0]
    words = obj.strip().rstrip(".").split()
    return " ".join(words[:max_words]) if words else "the data"


def _steps_for(role: str, verb: str, obj: str, idx: int) -> list[str]:
    """Gera passos UC canônicos para uma sub-ação, conforme protocolo §3-§4."""
    vc = _verb_class(verb)
    o = _clean_obj(obj)
    tag = f"{verb.lower()} {o}"  # data groups distintos por processo (anti-dedup)
    role = role.strip() or "User"
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
              subactions: list[tuple[str, str]]) -> str:
    lines = [
        "ID",
        f"{fp_id}\t{text[:60]}",
        f"Description\t{text}",
        "MAIN SCENARIO",
    ]
    n = 0
    for i, (verb, obj) in enumerate(subactions):
        for step in _steps_for(role, verb, obj, i):
            n += 1
            actor = role if step.lower().startswith(role.lower()) else "System"
            body = (step[len(role):].strip()
                    if step.lower().startswith(role.lower())
                    else step[len("System"):].strip())
            lines.append(f"{n}\t{actor}\t{body}")
    return "\n".join(lines) + "\n"


def connextra_to_ucs(text: str, fp_id: str) -> list[str] | None:
    """Converte uma user story Connextra em pseudo-UCs (um por processo
    funcional). None se o texto não casar com o frame Connextra.

    Regra 4.6: ações coordenadas com verbos de escrita distintos (CRUD
    explícito) são processos funcionais separados — cada um medido em
    isolamento (Entry de gatilho, confirmação e dedup próprios) e somados.
    Coordenação puramente de leitura (ex.: "filter, sort and search") é UM
    processo (regra 4.3: critérios variáveis ainda são 1 grupo de dados).
    """
    m = CONNEXTRA_RE.match(text.strip())
    if not m:
        return None
    role = m.group("role").strip()
    action = m.group("action").strip()
    subactions = _split_subactions(action)
    if not subactions:
        return None

    classes = [_verb_class(v) for v, _ in subactions]
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
            _build_uc(text, f"{fp_id}-p{i+1}", role, [sa])
            for i, sa in enumerate(subactions)
        ]
    return [_build_uc(text, fp_id, role, subactions)]


def connextra_to_uc(text: str, fp_id: str) -> str | None:
    """Compat: retorna o primeiro UC (uso simples, sem decomposição)."""
    ucs = connextra_to_ucs(text, fp_id)
    return ucs[0] if ucs else None
