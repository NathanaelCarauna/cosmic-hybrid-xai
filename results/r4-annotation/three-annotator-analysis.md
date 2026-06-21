# R4 — Análise com Terceiro Anotador (A3, expert humano 2)

**Data:** 2026-06-10
**Arquivo:** `datasets/user-stories-100-annotated-a1.csv` (100/100 preenchidas, consistência interna perfeita: cfp = E+X+R+W em todas as linhas)

## Anotadores

| Código | Quem | Arquivo |
|--------|------|---------|
| A1 | IA (Claude Opus 4.8, pré-anotação) | `user-stories-100-annotated-ai.csv` |
| A2 | Expert humano 1 | `user-stories-100-annotated-cosmic-expert.csv` |
| A3 | Expert humano 2 (novo) | `user-stories-100-annotated-a1.csv` |
| GT | Adjudicação humana rodada 1 (A1×A2) | `user-stories-100-ground-truth-FINAL.csv` |

## Concordância (n=98, excluídas US028/US062 §6.1)

### Pareada (Cohen's κ)

| Par | κ exato | κ linear-ponderado | Concordância | MAE |
|-----|:------:|:------:|:------:|:---:|
| **A2 × A3 (humano × humano)** | **0.659** | **0.705** | 81.6% | 0.255 |
| A1(IA) × A2 | 0.705 | 0.664 | 84.7% | 0.245 |
| A1(IA) × A3 | 0.659 | 0.637 | 81.6% | 0.316 |
| GT × A3 (validação externa) | 0.657 | 0.659 | 81.6% | 0.316 |
| GT × A2 | 0.960 | 0.937 | 98.0% | 0.082 |
| GT × A1(IA) | 0.722 | 0.712 | 85.7% | 0.163 |

### Multi-anotador

- **Fleiss' κ (3 anotadores): 0.672** — Substancial (Landis & Koch)
- Concordância plena 3-way: 74/98 (75.5%)

### Leituras metodológicas

1. **κ humano×humano = 0.66/0.71 (Substancial)** — este é o número canônico de
   confiabilidade inter-anotador para a tese, livre da ressalva "IA participou
   do κ". Estabelece que a tarefa tem ambiguidade intrínseca nesse patamar.
2. **A IA concorda com cada humano (~0.66–0.70) tanto quanto os humanos
   concordam entre si (0.66).** A pré-anotação por IA opera no mesmo nível de
   confiabilidade de um anotador humano — defesa direta da metodologia
   AI-assisted (`r4-ai-assisted-annotation-methodology.md`).
3. GT×A2 = 0.96 é esperado (A2 participou da adjudicação); GT×A3 = 0.66 é a
   **validação externa independente** do ground truth.

### Por movimento (A2 × A3)

| Movimento | κ | Concordância |
|-----------|:--:|:----:|
| Entries | 0.191 | 91.8% |
| Exits | 0.482 | 93.9% |
| Reads | 0.699 | 88.8% |
| Writes | 0.850 | 91.8% |

(κ de Entries é baixo apesar de 91.8% de concordância: quase todos os itens têm
1 Entry — paradoxo do κ com classe majoritária dominante.)

## Divergências A3 × GT: 18/98

Deltas (A3 − GT): +1 em 10 itens, −1 em 5, +2 em 1, +7 em 2 (US076/US078, CRUD
decomposto). A3 tende a contar +1 movimento em itens ambíguos (mesma direção da
IA na rodada 1).

### Itens contestados para re-adjudicação (rodada 2): 10

Planilha: `datasets/user-stories-100-consensus-worksheet-r2.csv`

| us_id | A1 | A2 | A3 | GT r1 | Maioria | Situação |
|-------|:--:|:--:|:--:|:-----:|:-------:|----------|
| US008 | 4 | 5 | 4 | 5 | 4 | maioria contradiz GT |
| US013 | 4 | 3 | 4 | 3 | 4 | maioria contradiz GT |
| US039 | 4 | 3 | 4 | 3 | 4 | maioria contradiz GT |
| US045 | 4 | 3 | 4 | 3 | 4 | maioria contradiz GT |
| US076 | 4 | 11 | 11 | 4 | 11 | **ambos humanos = 11** (CRUD, regra 4.6) |
| US078 | 4 | 4 | 11 | 4 | 4 | mesmo padrão de US076 — exige decisão de regra consistente |
| US085 | 4 | 3 | 4 | 3 | 4 | maioria contradiz GT |
| US091 | 4 | 3 | 4 | 3 | 4 | maioria contradiz GT |
| US093 | 4 | 3 | 5 | 3 | — | sem maioria |
| US099 | 4 | 6 | 5 | 6 | — | sem maioria |

**Decisão de regra pendente (US076/US078):** user stories compostas com
processos CRUD explícitos ("create, update and delete") devem ser decompostas
em múltiplos processos funcionais (regra 4.6 → CFP ≈ 11) ou anotadas como um
único processo (→ CFP 4)? A decisão deve ser aplicada uniformemente aos dois
itens. Ambos os experts humanos decompuseram pelo menos um deles.

### Exclusões §6.1 (US028/US062)

A3 atribuiu CFP=3 a ambas, mas as notas marcam **AMBIGUA** ("tarefa de deploy,
não processo funcional típico" / "NFR de acessibilidade") — consistente com o
racional de exclusão da rodada 1. **Exclusão mantida.**

## Salvaguarda metodológica

A re-adjudicação (rodada 2) deve ser feita **cega aos resultados dos modelos**:
o adjudicador decide com base apenas no texto da US e nas notas dos três
anotadores. Caso o GT mude, todos os modelos (E33d, E30f, R3-HKO) serão
re-avaliados sobre o GT v2 e a matriz será reconstruída — os números da rodada 1
ficam preservados em `results/` para auditoria.

---

## Resolução — Rodada 2 (2026-06-10)

Adjudicação humana (cega aos modelos) registrada em
`user-stories-100-consensus-worksheet-r2.csv`, coluna `cfp_final_humano`:

| us_id | GT r1 | GT v2 | Decisão |
|-------|:-----:|:-----:|---------|
| US008 | 5 | 4 | maioria 2/3 |
| US013 | 3 | 4 | maioria 2/3 |
| US039 | 3 | 4 | maioria 2/3 |
| US045 | 3 | 4 | maioria 2/3 |
| US076 | 4 | 11 | **regra 4.6: CRUD explícito decomposto** (ambos experts) |
| US078 | 4 | 11 | regra 4.6 aplicada consistentemente com US076 |
| US085 | 3 | 4 | maioria 2/3 |
| US091 | 3 | 4 | maioria 2/3 |
| US093 | 3 | 4 | adjudicação (sem maioria) |
| US099 | 6 | 6 | mantido |

**GT v2:** `datasets/user-stories-100-ground-truth-FINAL-v2.csv` — 9 alterações.
Distribuição (n=98): {3: 55, 4: 37, 5: 2, 6: 2, 11: 2}, média 3.643.

### Validação do GT v2 contra os anotadores

| Par | κ exato | κ ponderado | Concordância | MAE |
|-----|:------:|:------:|:------:|:---:|
| GTv2 × A1(IA) | 0.825 | 0.755 | 90.8% | 0.235 |
| GTv2 × A2 | 0.827 | 0.842 | 90.8% | 0.153 |
| GTv2 × A3 | 0.815 | 0.863 | 89.8% | 0.102 |

O GT v2 é **equilibrado** entre os três anotadores (κ ≈ 0.82 com cada um),
enquanto o v1 era enviesado para A2 (κ=0.96 com A2 vs 0.66 com A3) — propriedade
desejável de um consenso adjudicado.

### Re-avaliação dos modelos (GT v2)

| Método | MAE v1 → v2 | SA v2 (definição consistente) |
|--------|:-----------:|:----------:|
| E33d | 2.408 → 2.602 | **−2.606** |
| E30f-clean | 1.357 → 1.490 | −1.065 |
| R1-COSMET | — → 0.888 | −0.230 |
| R3-HKO (home, melhor variante) | 0.429 → 0.602 | +0.166 |

> **ERRATA (2026-06-10):** a versão anterior desta seção reportava SA do E33d
> como +0.286 e o claim "único método com SA>0 nos 4 formatos". O script do
> E33d calculava `1 − MAE/média(GT)` (= 1−NMAE) em vez do SA padrão
> (`1 − MAE/MAE_baseline`). Sob a métrica consistente, o E33d colapsa em user
> stories (prediz ~1 CFP para quase tudo; Spearman −0.08) e o claim foi
> retirado. Ver `feedback-auditoria-2026-05-21.md` §19.11.

Runs GT-v1 removidas do `runs_registry.csv` (diretórios preservados em
`results/` para auditoria); matriz e figura reconstruídas.

---

## Adendo — E34a (adaptador Connextra, pós-errata SA)

Após a errata da métrica SA (auditoria §19.11), o colapso do E33d em user
stories foi diagnosticado (100% das US caíam em `R15_USER_FALLBACK` → 1 Entry)
e corrigido com o **adaptador de formato Connextra (E34a)** — regras derivadas
do protocolo de anotação, calibrado APENAS no dev-20 (`kappa_subset`),
avaliado uma única vez no test-78. Cobertura do frame Connextra: 98/98 (0
fallbacks).

| Método (célula US, GT v2) | MAE | SA | n |
|---------------------------|----:|---:|--:|
| **E34a (test, congelado)** | **0,500** | **+0,261** | 78 |
| R3-HKO (home, treina no GT) | 0,602 | +0,166 | 98 |
| R1-COSMET | 0,888 | −0,230 | 98 |
| E30f-clean | 1,490 | −1,065 | 98 |
| E33d sem adaptador | 2,602 | −2,606 | 98 |

Ver `feedback-auditoria-2026-05-21.md` §19.12 e
`results/ner-ontology/e34a_user_stories_{results.csv,summary.json}`.
