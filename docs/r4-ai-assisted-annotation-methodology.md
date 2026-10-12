# Metodologia de Anotação Assistida por IA — R4 (User Stories)

**Versão:** 2.0
**Contexto:** ground truth COSMIC para 100 histórias de usuário da coleção de Dalpiaz (Mendeley Data)

Este documento descreve o fluxo de anotação tal como executado e reportado na
dissertação (Capítulo 4, seção "Construção do ground truth para histórias de
usuário"). A versão 1.0 era a nota de planejamento anterior à anotação e
previa um fluxo diferente (um único humano revisando o rascunho da IA); ela
foi substituída por esta.

---

## 1. O risco a evitar: circularidade

O conjunto R4 é usado para **avaliar** métodos que, em parte, usam modelos de
linguagem. Se o ground truth fosse a saída de um LLM, um método poderia
parecer bom em histórias de usuário apenas por concordar com outro LLM. O
fluxo abaixo foi desenhado para que o ground truth não dependa de nenhum
modelo avaliado.

## 2. Fluxo executado

| Etapa | Artefato | Responsável |
|-------|----------|-------------|
| 1. Amostragem (100 histórias, semente fixa, estratificada por projeto) | `scripts/sample_user_stories.py` | script |
| 2. Protocolo de anotação, redigido antes da anotação | `docs/r4-annotation-protocol.md` | autor |
| 3. Pré-anotação com justificativa por movimento (anotador A1) | `datasets/user-stories-100-annotated-ai.csv` | Claude Opus 4.8 |
| 4. Anotação independente das 100 histórias (anotador A2) | `datasets/user-stories-100-annotated-cosmic-expert.csv` | humano |
| 5. Anotação independente das 100 histórias (anotador A3) | `datasets/user-stories-100-annotated-a1.csv` | humano |
| 6. Adjudicação das divergências, em duas rodadas | `datasets/user-stories-100-consensus-worksheet.csv`, `…-worksheet-r2.csv` | adjudicador humano |
| 7. Ground truth final (98 itens) | `datasets/user-stories-100-ground-truth-FINAL-v2.csv` | — |

## 3. Salvaguardas

1. **Pré-anotação ≠ ground truth.** A saída do modelo conta como um dos três
   anotadores (A1) e não é adotada como gabarito sem adjudicação humana.
2. **Humanos sem acesso à pré-anotação.** Os dois anotadores humanos (A2 e A3)
   aplicaram o protocolo às 100 histórias de forma independente, sem acesso às
   anotações alheias, incluída a do modelo. Isso evita ancoragem dos humanos
   no modelo.
3. **Adjudicação às cegas.** As divergências foram resolvidas por adjudicador
   humano sem acesso aos resultados de qualquer método avaliado. A segunda
   rodada, motivada pela entrada do terceiro anotador, reexaminou dez itens
   contestados e fixou uma decisão de regra (a decomposição de processos CRUD
   explícitos), aplicada uniformemente ao conjunto.
4. **Confiabilidade canônica é humano × humano.** O κ reportado como
   confiabilidade do ground truth é o calculado entre A2 e A3.

## 4. Concordância (n = 98)

Detalhes em `results/r4-annotation/three-annotator-analysis.md`.

| Par | κ exato | κ ponderado linear |
|-----|:-------:|:------------------:|
| A2 × A3 (humano × humano) | 0,659 | 0,705 |
| A1 (IA) × A2 | 0,705 | 0,664 |
| A1 (IA) × A3 | 0,659 | 0,637 |

κ de Fleiss para os três anotadores: 0,672 (concordância substancial na
interpretação de Landis e Koch). A pré-anotação concorda com cada humano em
nível equivalente ao da concordância entre os próprios humanos.

Duas histórias (US028 e US062) foram excluídas por critério documentado no
protocolo, o que leva o conjunto final a 98 itens.

## 5. Limitações

1. A pré-anotação tende a subestimar leituras implícitas, o erro mais comum
   em histórias de usuário (protocolo, Seção 2.3).
2. Em casos ambíguos, a pré-anotação adota o mínimo razoável de movimentos.
3. A distribuição de CFP é estreita (2 a 4 na pré-anotação), porque uma
   história de usuário corresponde tipicamente a um único processo funcional
   granular.
