# Análise de Divergências entre Anotadores — R4 User Stories

**Data:** 2026-06-04
**Anotador 1 (A1):** `user-stories-100-annotated-ai.csv` — pré-anotação Claude Opus 4.8
**Anotador 2 (A2):** `user-stories-100-annotated-cosmic-expert.csv` — anotador independente
**Itens comparados:** 98 (US028 e US062 deixados em branco por ≥1 anotador → tratados à parte)

---

## 1. Concordância global (CFP total)

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Concordância exata | 83/98 (84.7%) | — |
| Diferença média absoluta | 0.245 CFP | — |
| **κ de Cohen (exato)** | **0.7046** | **Substantial** |
| **κ de Cohen (ponderado linear)** | **0.6640** | **Substantial** |

> κ "Substantial" (Landis & Koch) é o patamar esperado e aceitável para anotação COSMIC de requisitos ágeis, formato reconhecidamente ambíguo (protocolo §1.1).

### 1.1 κ por tipo de movimento

| Movimento | κ (linear) | Concordância exata | Observação |
|-----------|-----------|--------------------|------------|
| Entry | 0.000 | 99% | Sem variância: todos = 1 Entry disparadora (esperado §2.1). κ indefinido por ausência de variância, não por discordância. |
| Exit | 0.393 | 98% | Divergência concentrada nos raros casos X=2 (notificação/broadcast). |
| Read | 0.704 | 89% | Principal fonte de divergência — Reads implícitas (§2.3). |
| Write | 0.875 | 94% | Alta concordância. |

**Diagnóstico:** as divergências concentram-se em **Reads e Exits**, exatamente os movimentos mais ambíguos do formato user story. Entries e Writes são quase unânimes.

---

## 2. Padrão sistemático identificado

Nas 11 divergências de |Δ|=1, **a IA (A1) super-contou** Reads/Writes especulativos em 9 casos; o anotador humano (A2) foi consistentemente mais conservador e mais aderente ao protocolo:

- A IA adicionou **Reads de validação** não exigidos (US039, US080, US085, US091) — registrar/criar nem sempre exige ler.
- A IA adicionou **Writes de sessão/log** não explícitos (US091 login — protocolo §4.4 dá mínimo 3 sem log).
- A IA contou **Write duplicado** para atributos do mesmo objeto (US093 branding = atributo da coleção, 1 Write).

> Isto **inverte** a limitação #1 antecipada na metodologia (que previa sub-contagem de Reads pela IA). Na prática, a IA **super-contou** movimentos implícitos. O anotador humano corrige na direção certa.

---

## 3. Divergências que exigem consenso (|Δ| ≥ 2)

| US | CFP IA | CFP A2 | Δ | Resolução recomendada |
|----|:-----:|:-----:|:-:|-----------------------|
| **US076** | 4 | 11 | 7 | ⚠️ **Decisão humana.** "create, update and delete any records" — CRUD explícito. §4.6 manda decompor → A2 (11) é protocolar, mas gera outlier (1 texto = 11 CFP) que distorce a avaliação dos modelos. **Opções:** (a) aceitar 11 e marcar US como composta excluída do MAE; (b) anotar processo primário e remover do conjunto de avaliação. |
| **US033** | 4 | 6→**5** | 2 | **Corrigir para 5.** Broadcast (Ex.7): E1+X2+R1+W1 = 5. A2 escreveu 6 por **erro herdado do Exemplo 7 do protocolo** (lista 5 movimentos, rotula "6"). A IA omitiu o W1 (log). → 5. |
| **US047** | 4 | 6 | 2 | **→ A2 (6).** Composta "and" (§4.6): enfileirar replicação + exibir status = E1+X2+R2+W1. Decomposição do A2 é protocolar. |
| **US099** | 4 | 6 | 2 | **→ A2 (6).** "based on analysis + annotations + comments" = 3 fontes de Read distintas + W(avaliação). A2 captura as 3 fontes. |

---

## 4. ⚠️ Erro no protocolo (Exemplo 7) a corrigir

O **Exemplo 7** do `r4-annotation-protocol.md` declara:

> "**CFP: 6** (1E + 2X + 1R + 1W)"

Mas 1+2+1+1 = **5**, não 6. O movimento listado é correto (5 movimentos); o rótulo "6" é um erro de aritmética. Esse erro se propagou para a anotação de US033 do A2 (cfp=6 com movimentos somando 5 — única inconsistência interna detectada). **Ação:** corrigir o Exemplo 7 para CFP 5.

---

## 5. Itens em branco (protocolo §6.1)

| US | A1 (IA) | A2 | Resolução |
|----|:------:|:---:|-----------|
| US028 | 2 | *(branco)* | **→ branco.** "see beta up on cloud.gov" = tarefa de deploy, sem movimentos de dados (§6.1). A2 correto. Excluir da avaliação. |
| US062 | 2 | *(branco)* | **→ branco.** "device adapted for visual troubles" = NFR de acessibilidade (§6.1). A2 correto. Excluir da avaliação. |

A IA não deveria ter atribuído 2; o anotador humano seguiu o §6.1 corretamente (deixar em branco, não estimar).

---

## 6. Protocolo de resolução aplicado

Conforme §7.4 do protocolo:
- **|Δ| ≥ 2** → reunião de consenso entre anotadores (4 itens: US076, US033, US047, US099).
- **|Δ| = 1** → anotador principal decide com justificativa (11 itens).
- **Branco vs valor** → alinhar ao §6.1 (2 itens → branco).

Recomendações pré-preenchidas em `datasets/user-stories-100-consensus-worksheet.csv` (coluna `cfp_recomendado`). O anotador principal confirma/ajusta na coluna `cfp_final_humano`.

---

## 7. Ground-truth consolidado (proposto)

`datasets/user-stories-100-ground-truth-PROPOSED.csv`:

| Origem do rótulo | n |
|------------------|---|
| Concordância plena A1=A2 | 83 |
| Resolvido por recomendação | 14 |
| **Pendente decisão humana** | 1 (US076) |
| Branco §6.1 (excluído) | 2 (US028, US062) |
| **Total com CFP final** | **97** (média 3.44, mediana 3, faixa 3–6) |

---

## 8. Nota metodológica crítica (independência do ground truth)

Para o κ reportado servir como evidência de **independência** entre o ground truth e os modelos avaliados (E30f, R1 — ambos LLM/ML), é **necessário que o anotador A2 seja um especialista humano independente**, não outra execução de LLM. Se A2 for humano, o fluxo é defensável:

> "Pré-anotação assistida por IA (Opus 4.8), validada contra anotador humano independente; concordância κ=0.70 (substancial); divergências resolvidas em sessão de consenso conduzida por especialista humano. Os rótulos finais resultam de adjudicação humana."

Se A2 também for IA, o κ mede apenas consistência entre LLMs e **não** estabelece independência — nesse caso ainda é obrigatório um anotador humano antes de usar o R4 na avaliação. **Confirmar a natureza de A2 antes de fechar o ground truth.**
