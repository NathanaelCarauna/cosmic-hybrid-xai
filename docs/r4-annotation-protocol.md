# Protocolo de Anotação COSMIC para Histórias de Usuário — R4

**Versão:** 1.0  
**Data:** 2026-05-25  
**Authors:** (omitted for double-anonymous review)
**Contexto:** Sprint 2, Track B — dataset R4 (100 User Stories do Dalpiaz Mendeley Data)

---

## 1. Introdução

### 1.1 Por que medir CFP em Histórias de Usuário?

O método COSMIC (ISO/IEC 19761) define o tamanho funcional de um software pela contagem de movimentos de dados entre processos funcionais e seus usuários funcionais ou armazenamentos persistentes. Cada movimento conta 1 CFP (Cosmic Function Point). Historicamente, o COSMIC foi aplicado a Casos de Uso e especificações SRS; no entanto, metodologias ágeis substituíram esses formatos por **Histórias de Usuário** (User Stories), que são mais curtas e orientadas ao valor para o usuário.

O objetivo deste track de pesquisa é criar um *ground truth* de 100 histórias anotadas em CFP para (a) alimentar a coluna 4 da matriz cross-dataset de heterogeneidade e (b) treinar e avaliar o framework Hybrid-XAI em formato de requisito ágil — o formato mais comum em projetos open-source contemporâneos.

A medição de CFP em User Stories é possível e recomendada pelo COSMIC Guide (seção 3.6), mas exige atenção especial porque cada história tende a mapear diretamente a **um único processo funcional** de nível granular, com poucos movimentos explícitos. A tendência à subestimação de movimentos implícitos (especialmente Reads e Writes) é o principal desafio deste formato.

### 1.2 Escopo e limites deste protocolo

Este protocolo aplica-se exclusivamente a histórias no formato canônico de Connextra:

> **"As a [role], I want [action] so that [benefit]."**

Não cobre:
- User Stories em outros idiomas (o dataset Dalpiaz é em inglês).
- Épicos ou temas que descrevem conjuntos de funcionalidades.
- Critérios de aceitação (ACs) — estes podem ajudar na interpretação, mas não alteram a US anotada.

### 1.3 Referências normativas

- COSMIC Measurement Manual v4.0.2 (fonte normativa em inglês, abreviado "MM")
- *A Guide to Software Size Measurement*, Symons v1.0 (2020) — especialmente seção 3.6 (User Stories)
- Manual Brasileiro v3.0.1 (tradução ISO/IEC 19761:2003)

---

## 2. Tipos de Movimentos de Dados

### 2.1 Entry (E) — Entrada

Uma **Entry** ocorre quando um grupo de dados move-se de um **usuário funcional externo** (tipicamente o ator humano da US) **para dentro do processo funcional** do software. Na linguagem das User Stories, Entries correspondem a dados que o usuário submete, informa, envia ou fornece ao sistema para que o processo seja executado.

**Regra-chave:** toda US que representa um processo funcional dispara ao menos **uma Entry** — a Entry disparadora (*triggering Entry*), que inicia o processo e carrega os dados do evento que o motivou. Sem Entry disparadora, não há processo funcional. Exemplos de Entries em User Stories: preenchimento de formulário, envio de solicitação, upload de arquivo, login com credenciais, seleção de opção que carrega dados de contexto para o sistema.

**Atenção:** um único formulário com múltiplos campos ainda é **uma única Entry** se mover um único grupo de dados sobre um único objeto de interesse (ex.: dados do pedido, dados do usuário). Só há duas Entries se o processo recebe grupos de dados de objetos de interesse **distintos** em momentos distintos do mesmo processo.

### 2.2 Exit (X) — Saída

Uma **Exit** ocorre quando um grupo de dados move-se **do processo funcional para um usuário funcional externo** — ou seja, o sistema envia, exibe, notifica ou retorna algo ao usuário ou a outro sistema externo. Na linguagem das User Stories, Exits correspondem ao resultado que o sistema apresenta ou envia ao ator ou a terceiros.

**Regra-chave:** o COSMIC consolida todas as mensagens de erro e confirmação em **uma única Exit por processo** (MM 3.5.11). Portanto, se uma US descreve "o sistema exibe confirmação ou mensagem de erro", isso conta como **uma** Exit, não duas. Notificações (e-mail, push, alertas) também são Exits — dados saem do processo em direção a um usuário funcional. Relatórios, dashboards e downloads de arquivo são Exits. Comandos de controle puro (navegação entre telas sem dados de negócio, pressionar "OK" para fechar diálogo) **não** contam como Exit (MM 3.5.10).

### 2.3 Read (R) — Leitura

Uma **Read** ocorre quando o processo funcional recupera um grupo de dados de um **armazenamento persistente** (banco de dados, arquivo, cache persistente, índice) **para uso interno no próprio processo**. Na linguagem das User Stories, Reads ocorrem quando o sistema precisa buscar dados existentes para realizar o que o usuário pediu — verificar permissão, recuperar perfil, consultar histórico, checar disponibilidade.

**Regra-chave:** Reads são invisíveis ao usuário final — ocorrem dentro do sistema. Se o usuário não sabe que o sistema está lendo algo, mas o sistema precisa ler para processar, **há uma Read**. Por exemplo, para exibir a página de perfil do usuário, o sistema precisa ler os dados do perfil — mesmo que a US não mencione explicitamente "buscar no banco".

**Atenção:** Reads são os movimentos mais frequentemente esquecidos em User Stories. Anote todos os grupos de dados que o sistema **precisa recuperar** do armazenamento para completar o processo, mesmo quando a US descreve apenas o resultado visível.

### 2.4 Write (W) — Escrita

Uma **Write** ocorre quando o processo funcional persiste, atualiza ou remove um grupo de dados em um **armazenamento persistente**. Na linguagem das User Stories, Writes acontecem sempre que o sistema deve guardar algo novo, alterar algo existente ou deletar um registro.

**Regra-chave:** criação, atualização e deleção de um mesmo objeto de interesse contam como **Writes distintas** apenas se ocorrem sobre objetos distintos no mesmo processo funcional. Um processo de "cadastro de usuário" que escreve os dados do usuário no banco é **uma Write**. Um processo de "fazer pedido" que grava o pedido e também atualiza o estoque representa **duas Writes** (dois objetos de interesse: Pedido e Estoque).

---

## 3. Algoritmo de Identificação de Movimentos

Siga este algoritmo passo a passo para cada User Story. Registre o raciocínio na coluna `movements_detail`.

### Passo 1 — Identificar o processo funcional

Leia a cláusula **"I want [action]"**. Esse é o candidato a processo funcional. Confirme que representa uma resposta completa e atômica a um único evento disparador (ex.: "register an account", "search for products", "submit a report"). Se a cláusula contiver mais de um processo funcional distinto (ver Passo 6), divida a análise.

**Perguntas a fazer:**
- O que o ator está tentando fazer?
- É uma ação única ou uma combinação de ações?
- Qual é o evento que dispara o processo?

### Passo 2 — Contar Entries

Identifique os dados que o **ator (role)** envia ao sistema para iniciar e executar o processo:

- Dados de formulário submetidos pelo usuário
- Credenciais de login
- Arquivo enviado
- Parâmetros de busca/filtro inseridos pelo usuário
- Seleção de item que passa dados ao sistema

Regra mínima: **sempre há pelo menos 1 Entry** (a Entry disparadora). Se a US descreve apenas "ver/visualizar" algo sem nenhum dado do usuário além de uma seleção ou clique, pode haver Entry=1 implícita (identificação do item a ver, ex.: ID do pedido clicado).

### Passo 3 — Contar Exits

Identifique o que o sistema **retorna, exibe ou envia** ao usuário ou sistema externo:

- Página de resultado, relatório, lista de itens
- Mensagem de confirmação (agrupa todas as confirmações e erros em 1 Exit)
- Notificação enviada (e-mail, push, SMS)
- Arquivo gerado para download
- Resposta de API enviada a sistema externo

Regra: uma US típica tem **1 Exit** (o resultado principal do processo). Se há também uma notificação para um ator diferente, isso é uma **segunda Exit**.

### Passo 4 — Contar Reads

Pergunte: **o que o sistema precisa buscar no armazenamento persistente para executar o processo?**

- Para autenticar: ler credenciais/perfil do usuário
- Para exibir: ler os dados do objeto a exibir
- Para atualizar: ler o registro atual antes de modificar
- Para validar: ler regras, permissões, configurações
- Para listar: ler coleção de registros

Escreva explicitamente os grupos de dados que precisam ser lidos, mesmo que a US não os mencione. **Reads implícitas são obrigatórias se o processo não pode ocorrer sem elas.**

### Passo 5 — Contar Writes

Pergunte: **o processo persiste, atualiza ou remove dados?**

- Criar novo registro: 1 Write para o objeto criado
- Atualizar registro existente: 1 Write para o objeto atualizado
- Deletar registro: 1 Write (deleção lógica ou física)
- Se atualiza dois objetos de interesse distintos: 2 Writes

"View/see/display" sem nenhuma persistência de estado (ex.: sem log de acesso): Write=0.

### Passo 6 — Interpretar a cláusula "so that [benefit]"

A cláusula de benefício normalmente fornece **contexto**, não movimentos adicionais. Porém, em alguns casos, revela processos implícitos:

- "so that I can track my orders" → pode indicar que há um log ou histórico sendo gravado, mas só adiciona Write se o rastreamento não estava implícito no processo principal.
- "so that I am notified" → se a notificação não estava clara na cláusula de ação, adiciona 1 Exit.
- "so that the admin can review" → pode indicar um Exit adicional para um ator diferente.

**Regra geral:** só adicione movimentos pela cláusula de benefício se o benefício descreve claramente um fluxo de dados adicional não capturado nos passos anteriores.

---

## 4. Regras de Desambiguação

### 4.1 CRUD implícito

Quando a US usa verbos de gestão genérica como **"manage", "administer", "maintain"**, assuma o padrão mínimo de CRUD:

| Operação implícita | Entry | Exit | Read | Write |
|--------------------|-------|------|------|-------|
| Criar              | 1     | 1    | 0    | 1     |
| Ler/Listar         | 1     | 1    | 1    | 0     |
| Atualizar          | 1     | 1    | 1    | 1     |
| Deletar            | 1     | 1    | 1    | 1     |

Para "manage X" como US única (sem subcláusulas que especifiquem a operação), anote o **mínimo razoável**: 1E+1X+1R+1W = **4 CFP** e marque `ambiguity_flag=true` com nota explicando a ambiguidade.

### 4.2 Notificações

Notificações (e-mail, push, SMS, alertas no sistema) são **Exits** — dados saem do processo em direção a um usuário funcional. Cada destinatário distinto pode ser um usuário funcional distinto, mas Exits para múltiplos destinatários do **mesmo tipo** (ex.: lista de administradores) contam como **uma única Exit** (um grupo de dados enviado).

### 4.3 View / See / Display

Processos que descrevem apenas visualização de dados:

- "I want to view my profile" → 1E (identificação do usuário/sessão) + 1X (dados do perfil exibidos) + 1R (ler perfil do banco) + 0W = **3 CFP**
- "I want to see a list of products" → 1E (parâmetros de filtro/navegação) + 1X (lista exibida) + 1R (ler produtos) + 0W = **3 CFP**
- Se a US menciona "search" com critérios variáveis, a Entry pode ser mais rica, mas ainda é 1 grupo de dados de busca = 1E.

### 4.4 Login / Autenticação

"As a user, I want to log in so that I can access my account" é o processo canônico de autenticação:

- **1 Entry** — credenciais (usuário + senha ou token)
- **1 Read** — ler dados de autenticação do usuário no banco
- **1 Exit** — resultado da autenticação (sessão iniciada / erro)
- **0 Write** — a menos que haja gravação de sessão/log de acesso (adicionar 1W se relevante)

Mínimo: **3 CFP**. Com log de acesso: **4 CFP**.

### 4.5 Upload de arquivo

"As a user, I want to upload a document" — o arquivo é o grupo de dados da Entry:

- **1 Entry** — arquivo + metadados (nome, tipo, etc.)
- **1 Write** — persistir o arquivo no armazenamento
- **1 Exit** — confirmação de sucesso/erro
- **0 Read** — a menos que valide contra dados existentes (ex.: verificar quota)

Mínimo: **3 CFP**.

### 4.6 Histórias compostas ("and", "also", "as well as")

User Stories que descrevem dois ou mais processos funcionais distintos no mesmo enunciado devem ser **decompostas em sub-stories** antes da anotação. Critério: se a remoção de uma cláusula deixa a US ainda completa e independente, ela representa um processo separado.

**Exemplo:**
> "I want to register an account **and** receive a welcome email."

Decompor em:
1. Register account: 1E + 1X + 0R + 1W = **3 CFP**
2. Send welcome email (Exit adicional do mesmo processo): pode ser 1X adicional no mesmo processo funcional = total **4 CFP** (se enviado como parte do mesmo evento de registro)

**Critério prático:** se os dois processos ocorrem em **resposta ao mesmo evento disparador**, mantenha como um único processo com múltiplos movimentos. Se ocorrem em resposta a **eventos distintos**, são dois processos. Marque `ambiguity_flag=true` e documente.

---

## 5. Exemplos com Gabarito

Cada exemplo apresenta a US, a análise dos movimentos com justificativa e o total CFP.

---

**Exemplo 1** — Visualização simples

> "As a user, I want to view my profile page so that I can see my personal information."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Identificador de sessão/usuário | Entry disparadora: usuário navega para a própria página |
| R1 | Dados do perfil (nome, e-mail, bio) | Sistema recupera o perfil do banco para exibir |
| X1 | Dados do perfil exibidos | Sistema renderiza a página com os dados |

**CFP: 3** (1E + 1X + 1R + 0W)

---

**Exemplo 2** — Criação de registro

> "As a new user, I want to create an account so that I can access the platform."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Dados de cadastro (nome, e-mail, senha) | Entry disparadora: formulário de registro |
| W1 | Novo registro de usuário | Sistema persiste o novo usuário no banco |
| X1 | Confirmação ou mensagem de erro | Sistema informa o resultado ao usuário |

**CFP: 3** (1E + 1X + 0R + 1W)

---

**Exemplo 3** — Autenticação

> "As a registered user, I want to log in with my email and password so that I can access my account."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Credenciais (e-mail, senha) | Entry disparadora: formulário de login |
| R1 | Dados de autenticação do usuário | Sistema consulta banco para verificar credenciais |
| W1 | Registro de sessão/log de acesso | Sistema persiste a sessão do usuário autenticado |
| X1 | Resultado (sessão iniciada / erro) | Sistema informa o resultado ao usuário |

**CFP: 4** (1E + 1X + 1R + 1W)

---

**Exemplo 4** — Busca com filtros

> "As a customer, I want to search for products by category and price range so that I can find what I need."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Parâmetros de busca (categoria, faixa de preço) | Entry disparadora: critérios de busca informados pelo usuário |
| R1 | Conjunto de produtos que atendem aos critérios | Sistema recupera produtos do banco conforme filtros |
| X1 | Lista de produtos filtrados | Sistema exibe os resultados ao usuário |

**CFP: 3** (1E + 1X + 1R + 0W)

---

**Exemplo 5** — Atualização de registro

> "As a user, I want to update my email address so that I receive notifications at the correct address."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Novo endereço de e-mail | Entry disparadora: usuário submete o novo e-mail |
| R1 | Dados atuais do perfil do usuário | Sistema lê o registro atual antes de modificar |
| W1 | Perfil do usuário atualizado | Sistema persiste o novo e-mail no banco |
| X1 | Confirmação de atualização | Sistema informa sucesso ou erro ao usuário |

**CFP: 4** (1E + 1X + 1R + 1W)

---

**Exemplo 6** — Upload de arquivo

> "As a project manager, I want to upload project documents so that the team can access them."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Arquivo + metadados (nome, tipo, projeto associado) | Entry disparadora: upload do documento |
| R1 | Dados do projeto para validação de contexto | Sistema verifica se o projeto existe e se usuário tem permissão |
| W1 | Documento persistido no armazenamento | Sistema salva o arquivo e seus metadados |
| X1 | Confirmação ou erro de upload | Sistema informa o resultado ao usuário |

**CFP: 4** (1E + 1X + 1R + 1W)

---

**Exemplo 7** — Notificação para ator externo

> "As an administrator, I want to send a notification to all users when the system is under maintenance so that they are informed in advance."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Dados da notificação (mensagem, horário) | Entry disparadora: administrador compõe a notificação |
| R1 | Lista de usuários a notificar | Sistema recupera lista de usuários ativos do banco |
| W1 | Registro de notificação enviada (log) | Sistema persiste o histórico da notificação |
| X1 | Confirmação ao administrador (notificação disparada) | Sistema informa ao admin que a notificação foi enviada |
| X2 | Notificações enviadas aos usuários | Dados saem do processo em direção a cada usuário (um grupo de dados para usuário como ator distinto) |

**CFP: 5** (1E + 2X + 1R + 1W)

> **Correção (2026-06-04):** versão anterior deste exemplo rotulava "CFP: 6" por erro de aritmética (1+2+1+1 = 5). O total correto é **5**. Atenção a este padrão ao anotar broadcasts/notificações.

---

**Exemplo 8** — Deleção de registro

> "As a user, I want to delete my account so that my personal data is removed from the system."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Confirmação de intenção de deleção (senha ou "confirmar") | Entry disparadora: usuário confirma a ação irreversível |
| R1 | Dados do perfil do usuário | Sistema lê o registro a ser deletado |
| W1 | Deleção do registro do usuário | Sistema remove ou marca como deletado no banco |
| X1 | Confirmação de encerramento da conta | Sistema informa o resultado |

**CFP: 4** (1E + 1X + 1R + 1W)

---

**Exemplo 9** — Processo com múltiplos objetos de interesse

> "As a customer, I want to place an order so that the selected items are reserved and payment is processed."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Dados do pedido (itens, quantidades, endereço de entrega) | Entry disparadora: cliente submete o pedido |
| R1 | Dados do estoque dos itens selecionados | Sistema verifica disponibilidade antes de confirmar |
| R2 | Dados de pagamento/perfil financeiro do cliente | Sistema valida o método de pagamento |
| W1 | Novo registro de pedido | Sistema cria o pedido no banco |
| W2 | Atualização do estoque (reserva dos itens) | Sistema decrementa o estoque dos itens reservados |
| X1 | Confirmação do pedido (número do pedido, resumo) | Sistema exibe/envia a confirmação ao cliente |

**CFP: 7** (1E + 1X + 2R + 2W)

---

**Exemplo 10** — Relatório / Listagem com exportação

> "As a manager, I want to export the monthly sales report as a CSV file so that I can analyze the data in a spreadsheet."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Parâmetros do relatório (mês, período, filtros) | Entry disparadora: manager configura e solicita o relatório |
| R1 | Dados de vendas do período selecionado | Sistema recupera os registros de vendas do banco |
| W1 | Log de geração do relatório | Sistema registra que o relatório foi gerado (auditoria) |
| X1 | Arquivo CSV gerado para download | Sistema envia o arquivo ao usuário |

**CFP: 4** (1E + 1X + 1R + 1W)

---

**Exemplo 11** — US ambígua de CRUD ("manage")

> "As an admin, I want to manage user accounts so that the system access is controlled."

Esta US é ambígua: "manage" pode significar criar, listar, editar ou desativar contas — possivelmente todos. Para fins de anotação, use o mínimo razoável para um único processo de gestão:

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Dados da ação de gestão (ação + dados da conta-alvo) | Entry disparadora |
| R1 | Dados da conta do usuário-alvo | Sistema recupera o registro a gerenciar |
| W1 | Registro da conta atualizado/criado/removido | Sistema persiste a mudança |
| X1 | Confirmação da operação | Sistema informa o resultado |

**CFP: 4** (1E + 1X + 1R + 1W) — `ambiguity_flag=true`, `notes="'manage' é ambíguo; anotado como ciclo de gestão único"`

---

**Exemplo 12** — US de visualização pura sem Read explícito

> "As a visitor, I want to see the homepage so that I know what the product offers."

| Movimento | Grupo de dados | Justificativa |
|-----------|----------------|---------------|
| E1 | Requisição de navegação (acesso à URL) | Entry disparadora: mesmo sem dados explícitos do usuário, a requisição HTTP é a Entry |
| R1 | Conteúdo da homepage (textos, imagens, configurações) | Sistema recupera o conteúdo do armazenamento |
| X1 | Página renderizada ao visitante | Sistema exibe o conteúdo |

**CFP: 3** (1E + 1X + 1R + 0W)

> **Nota:** para US de páginas completamente estáticas (sem leitura de BD), R1 pode ser 0 → CFP=2. Decida conforme o contexto da aplicação descrita na US.

---

## 6. Casos de Borda

### 6.1 US vazia ou incompleta

Se a US estiver ausente, corrompida ou tiver menos de 5 palavras na cláusula de ação, **não anotar**. Marcar `ambiguity_flag=true` e `notes="US inválida para anotação COSMIC"`. O CFP deve ser deixado em branco (não zero).

### 6.2 US ambígua sem resolução possível

Se após aplicar as regras de desambiguação não for possível determinar os movimentos com razoável confiança, anote o **intervalo mínimo-máximo plausível** na coluna `notes` (ex.: "CFP estimado entre 2 e 5") e marque `ambiguity_flag=true`. Registre o CFP total como o valor mínimo plausível.

### 6.3 "As a system" — ator não humano

Quando o ator é um sistema externo ou processo automatizado (ex.: "As a payment gateway", "As a scheduler"), o processo funcional ainda é válido. O ator sistema é um **usuário funcional externo não humano**. Aplique o algoritmo normalmente; o ator externo emite Entries e recebe Exits como qualquer usuário funcional.

### 6.4 US sem cláusula "so that"

A ausência de cláusula de benefício não invalida a anotação. Aplique os Passos 1–5 normalmente. A cláusula "so that" não é obrigatória para identificar os movimentos de dados — ela apenas fornece contexto adicional. Marque `notes="US sem cláusula de benefício"`, mas não adicione `ambiguity_flag=true` por esse motivo isolado.

### 6.5 US em inglês informal com contrações e erros

O dataset Dalpiaz contém US em inglês informal. Normalize mentalmente a US antes de anotar:

- Contrações: "I wanna" → "I want to"
- Abreviações: "mgmt" → "management"
- Erros de digitação que não alteram o significado: corrigir mentalmente e registrar na coluna `notes` se houver ambiguidade

Não deixar de anotar por causa de linguagem informal — anote e documente.

### 6.6 US que mencionam "real-time" ou "live"

Dados "em tempo real" (ex.: "I want to see live updates") implicam polling ou streaming — o que ainda são movimentos Read+Exit em cada ciclo. Para fins de anotação de CFP, trate como **um único processo funcional**: 1R (leitura periódica do estado) + 1X (exibição atualizada). CFP=3 com a Entry disparadora implícita.

---

## 7. Instruções para Anotadores

### 7.1 Template a preencher

O arquivo `datasets/user-stories-100-template.csv` contém as 100 US a anotar com as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `us_id` | string | Identificador único da US (ex.: `dalpiaz-001`) |
| `project_id` | string | Projeto de origem |
| `us_text` | string | Texto completo da US ("As a... I want... so that...") |
| `us_text_role` | string | Componente [role] extraído |
| `us_text_action` | string | Componente [action] extraído |
| `us_text_benefit` | string | Componente [benefit] extraído (vazio se ausente) |
| `count_entry` | int | Número de Entries (E) contadas |
| `count_exit` | int | Número de Exits (X) contadas |
| `count_read` | int | Número de Reads (R) contadas |
| `count_write` | int | Número de Writes (W) contadas |
| `cfp_total` | int | Total CFP = E + X + R + W |
| `movements_detail` | string | Descrição textual de cada movimento (formato livre) |
| `ambiguity_flag` | bool | `true` se a US é ambígua ou requer interpretação especial |
| `notes` | string | Observações adicionais, decisões de interpretação |
| `annotator_id` | string | Identificador do anotador (ex.: `a1`, `a2`) |
| `annotated_at` | string | Data e hora da anotação (ISO 8601, ex.: `2026-05-26T14:30:00`) |

### 7.2 Como preencher `movements_detail`

Use o formato compacto, uma linha por movimento:

```
E1: dados do cadastro (nome, e-mail, senha) — formulário de registro
R1: perfil do usuário — verificação de e-mail duplicado
W1: novo usuário criado no banco
X1: confirmação de cadastro ou mensagem de erro
```

Não é necessário escrever em inglês: use o idioma que for mais natural para o anotador (pt-BR ou en).

### 7.3 Como sinalizar ambiguidade

Quando uma US admite interpretações diferentes que levam a contagens distintas:

1. Defina os movimentos para a interpretação mais provável (mínimo razoável).
2. Marque `ambiguity_flag=true`.
3. Na coluna `notes`, explique brevemente: qual é a ambiguidade, quais interpretações foram consideradas e qual foi escolhida e por quê.

**Exemplo de nota:** `"'manage' pode ser create/read/update/delete; anotado como update genérico (1E+1X+1R+1W=4); se expandido para CRUD completo, seria 14 CFP"`

### 7.4 Subconjunto para Cohen's κ

As 20 US marcadas com `kappa_subset=true` no arquivo de template devem ser anotadas por **dois anotadores independentes** sem consulta mútua. Após ambas as anotações, o pesquisador principal calculará o κ e agendará uma sessão de consenso para as discordâncias.

**Regra:** durante a anotação do subconjunto κ, não consulte o outro anotador. Dúvidas devem ser documentadas em `notes` e resolvidas na sessão de consenso.

### 7.5 Ritmo e checkpoints

Recomenda-se anotar em sessões de 25 US, com pausa entre sessões. Ao final de cada bloco de 25, revise as últimas 5 US da sessão anterior para garantir consistência. A primeira sessão deve incluir a calibração: os Exemplos 1–12 da Seção 5 devem ser anotados de forma independente e comparados com o gabarito antes de iniciar as 100 US do dataset.

---

*Fim do protocolo. Dúvidas sobre interpretações específicas devem ser registradas como issues no repositório ou encaminhadas ao pesquisador principal antes de anotar.*
