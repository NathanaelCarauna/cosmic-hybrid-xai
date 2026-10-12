"""Adaptadores de formato PT (M1): imperativa (v10), GWT e informal (v11).

Testa as REGRAS de normalização (o que vira passo), não a contagem final —
essa é do motor e está coberta em test_estimator_facade.py.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from adapters import format_adapters as FA  # noqa: E402
from adapters.format_adapters import (gwt_to_ucs, imperative_to_ucs,  # noqa: E402
                                      informal_to_ucs)


def _passos(uc):
    """Corpo dos passos do MAIN SCENARIO do pseudo-UC."""
    linhas = uc.strip().split("\n")
    return [l.split("\t", 2)[2]
            for l in linhas[linhas.index("MAIN SCENARIO") + 1:]]


# ── imperativa (v10) ────────────────────────────────────────────────────────

def test_imperativa_casa_e_gera_uc():
    ucs = imperative_to_ucs("O sistema deve cadastrar um cliente.", "T1", lang="pt")
    assert ucs and "MAIN SCENARIO" in ucs[0]
    assert "Sistema" in ucs[0] or "Usuário" in ucs[0]


def test_imperativa_permitir_desce_a_acao():
    # "deve permitir cadastrar" → a ação é cadastrar, não permitir
    ucs = imperative_to_ucs("O sistema deve permitir cadastrar uma conta.", "T2", lang="pt")
    assert ucs and "permitir" not in ucs[0].lower().split("main scenario")[1]


def test_imperativa_nao_casa_texto_solto():
    assert imperative_to_ucs("Cliente quer ver o pedido", "T3", lang="pt") is None


def test_imperativa_continua_decompondo_crud_coordenado():
    # regra 4.6 na imperativa segue valendo (regressão do v10)
    ucs = imperative_to_ucs("O sistema deve permitir criar, editar e excluir "
                            "produtos.", "T", lang="pt")
    assert len(ucs) == 3


def test_lang_en_nao_suportado():
    assert imperative_to_ucs("The system must register X.", "T6", lang="en") is None
    assert gwt_to_ucs("GIVEN a user; WHEN he asks; THEN it shows.", "T", "en") is None
    assert informal_to_ucs("I want to see my data", "T", "en") is None


# ── lematização de superfície ───────────────────────────────────────────────

def test_lemma_reconhece_flexoes_que_gwt_e_informal_usam():
    assert FA._lemma_pt("envia") == "enviar"         # 3ª sing. -ar
    assert FA._lemma_pt("exibe") == "exibir"         # 3ª sing. -ir
    assert FA._lemma_pt("recebe") == "receber"       # 3ª sing. -er
    assert FA._lemma_pt("cadastrou") == "cadastrar"  # pretérito


def test_lemma_nao_inventa_verbo_para_substantivo():
    # só aceita o palpite se cair em verbo conhecido — senão devolve a palavra
    for nao_verbo in ("hora", "etapa", "encomenda", "sistema"):
        assert FA._lemma_pt(nao_verbo) == nao_verbo


# ── divisão de cláusulas (o coração do GWT) ─────────────────────────────────

def test_split_clause_separa_acoes_coordenadas():
    partes = FA._split_clause("o sistema grava o pedido e envia o comprovante")
    assert [c for _, c in partes] == ["write", "send"]


def test_split_clause_nao_quebra_coordenacao_nominal():
    # "exibe A, B e C" = um Exit de um grupo de dados, não três
    partes = FA._split_clause(
        "o sistema exibe a localização atual, a hora da última atualização "
        "e a data estimada de entrega")
    assert len(partes) == 1
    assert partes[0][1] == "send", "exibir manda dado para fora = Exit"


def test_split_clause_separa_exibir_de_consultar():
    # dentro da resposta do sistema as duas não são a mesma movimentação
    assert FA._split_clause("o sistema exibe o saldo")[0][1] == "send"
    assert FA._split_clause("o sistema verifica a disponibilidade")[0][1] == "read"


def test_split_clause_trecho_antes_da_acao_nao_vira_movimentacao():
    # adjunto/sujeito é contexto: não multiplica movimentação nem entra no objeto
    partes = FA._split_clause("após aprovação, o sistema grava o pedido")
    assert len(partes) == 1
    assert partes[0] == ("o pedido", "write")


def test_split_clause_sem_acao_nenhuma_ainda_rende_movimentacao():
    # resposta do sistema sem verbo reconhecível não pode sumir
    partes = FA._split_clause("comprovante de pagamento por e-mail")
    assert len(partes) == 1 and partes[0][1] == "generic"


def test_split_clause_reconhece_acao_fora_do_lexico_por_sufixo():
    # "levantar" não está em léxico nenhum; o sufixo de infinitivo basta
    assert len(FA._split_clause("o paciente pode levantar os medicamentos")) == 1


# ── GWT ─────────────────────────────────────────────────────────────────────

def test_gwt_sem_estrutura_retorna_none():
    assert gwt_to_ucs("O sistema deve exibir o saldo.", "T5", lang="pt") is None


def test_gwt_le_a_precondicao_do_dado():
    # regra 4.1: a pré-condição é o estado persistido que o processo lê.
    # (No v10 experimental o DADO era descartado; medir mostrou que isso
    # custava MAE 0.708 → 1.125 no dev de GWT.)
    uc = gwt_to_ucs("DADO que a encomenda foi despachada; QUANDO o cliente "
                    "consulta o código; ENTÃO exibe a localização.", "T", "pt")[0]
    consultas = [p for p in _passos(uc) if p.startswith("consulta")]
    assert consultas and "encomenda" in consultas[0]
    # o grupo de dados é o SUJEITO da pré-condição, não a cláusula inteira
    assert "despachada" not in consultas[0]


def test_gwt_gatilho_do_quando_vira_entry():
    uc = gwt_to_ucs("DADO que há saldo; QUANDO o cliente submete o pedido; "
                    "ENTÃO grava o pedido.", "T", "pt")[0]
    assert _passos(uc)[0].startswith("informa")


def test_gwt_entao_vira_resposta_do_sistema():
    uc = gwt_to_ucs("Aplicar cupom.\nCritério: DADO um carrinho; QUANDO informa "
                    "o código; ENTÃO o sistema valida o cupom e atualiza o "
                    "total", "T4", lang="pt")[0]
    passos = " ".join(_passos(uc)).lower()
    assert "cupom" in passos and "total" in passos


def test_gwt_cru_de_tres_linhas_sem_palavra_chave():
    # os itens crus omitem "então": a posição das linhas é o molde
    ucs = gwt_to_ucs("fazendeiro registra plantação de milho\n"
                     "quando faz colheita\n"
                     "atualiza estoque e registra quantidade", "T", "pt")
    assert ucs is not None
    gravacoes = [p for p in _passos(ucs[0]) if p.startswith("grava")]
    assert len(gravacoes) == 2, "as duas escritas coordenadas do ENTÃO"


def test_gwt_sempre_um_processo():
    # ao contrário de Connextra/imperativa, o gatilho é único → regra 4.6 não
    # se aplica: respostas coordenadas são movimentações do MESMO processo
    ucs = gwt_to_ucs("DADO que há saldo; QUANDO submete o pedido; ENTÃO o "
                     "sistema grava o pedido, envia a confirmação e "
                     "atualiza o calendário.", "T", "pt")
    assert len(ucs) == 1


# ── informal ────────────────────────────────────────────────────────────────

def test_informal_ancora_no_verbo_do_lexico():
    uc = informal_to_ucs("eu queria ver quantos gigas eu gastei", "T", "pt")[0]
    assert any(p.startswith("consulta") for p in _passos(uc)), "ver → Read"


def test_informal_aceita_infinitivo_apos_modal():
    # "transferir" não está no léxico CRUD; o modal ancora a ação
    ucs = informal_to_ucs("preciso transferir grana pra várias contas", "T", "pt")
    assert ucs is not None and len(ucs) == 1


def test_informal_ignora_ruido_conversacional():
    com = informal_to_ucs("tipo assim, salvar os negócio sabe, tipo uma lista",
                          "T", "pt")[0]
    sem = informal_to_ucs("salvar os negócio uma lista", "T", "pt")[0]
    assert _passos(com) == _passos(sem)


def test_informal_e_sempre_um_processo():
    # texto livre não sinaliza fronteira de processo: um só, por desenho
    ucs = informal_to_ucs("Paciente consegue marcar consulta, receber lembrete "
                          "e desmarcar depois.", "T", "pt")
    assert len(ucs) == 1


def test_informal_sem_acao_devolve_none():
    assert informal_to_ucs("o negócio ali do lado, aquilo mesmo", "T", "pt") is None


def test_informal_nao_conta_confirmacao_mas_user_story_conta():
    # texto livre não pede confirmação; a user story pede. Este teste prende
    # os DOIS lados do contrato: se o gerador canônico mudar a redação do
    # passo de confirmação, o filtro do informal para de casar e isto quebra.
    livre = informal_to_ucs("preciso cadastrar um cliente novo", "T", "pt")[0]
    assert FA._CONFIRMACAO not in livre
    story = imperative_to_ucs("O sistema deve cadastrar um cliente.", "T", "pt")[0]
    assert FA._CONFIRMACAO in story, "contrato do _steps_for_pt mudou"


def test_informal_passos_ficam_numerados_em_sequencia():
    uc = informal_to_ucs("preciso cadastrar um cliente novo", "T", "pt")[0]
    linhas = uc.strip().split("\n")
    nums = [l.split("\t")[0] for l in linhas[linhas.index("MAIN SCENARIO") + 1:]]
    assert nums == [str(i) for i in range(1, len(nums) + 1)]
