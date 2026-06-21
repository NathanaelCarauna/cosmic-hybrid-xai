# Metodologia de Anotação Assistida por IA — R4 (User Stories)

**Versão:** 1.0
**Data:** 2026-06-04
**Contexto:** Sprint 2, Track B — ground truth COSMIC para 100 User Stories do Dalpiaz Mendeley Data

---

## 1. Justificativa metodológica

A anotação COSMIC de 100 user stories foi conduzida em um fluxo **AI-assisted com validação humana por especialista**, padrão crescentemente adotado em NLP (ACL/EMNLP 2023–2025) e defensável em contexto de mestrado, desde que respeitadas as salvaguardas abaixo.

### 1.1 O risco a evitar: circularidade

O dataset R4 é usado para **avaliar** os modelos da matriz de heterogeneidade (E30f, E33d, R3). Se o ground truth fosse produzido por um LLM e os modelos avaliados fossem também LLMs, haveria circularidade:

> "O modelo X é bom em user stories porque concorda com outro LLM que anotou as user stories."

### 1.2 As salvaguardas que tornam o ground truth independente

1. **Pré-anotação ≠ ground truth.** A saída do Claude Opus 4.8 é um *rascunho* (`cfp_annotator1` em `user-stories-100-annotated-ai.csv`), não o rótulo final.
2. **Revisão humana obrigatória.** O pesquisador principal (anotador A1) revisa todas as 100 anotações com o protocolo COSMIC em mãos, corrigindo `cfp_annotator1` onde discordar. Apenas após essa revisão o valor é considerado ground truth.
3. **Concordância inter-anotadores com humano (não com a IA).** O subconjunto de 20 (`kappa_subset=True`) é anotado de forma **independente** por um segundo anotador humano (A2), **sem acesso às anotações da IA**. O κ de Cohen reportado é entre A1 (humano revisado) e A2 (humano independente). Esse número prova que o ground truth não depende dos modelos avaliados.

### 1.3 Como reportar na dissertação

> "A pré-anotação COSMIC foi realizada com assistência do modelo Claude Opus 4.8 aplicando o protocolo formal da Seção [X]. Todos os rótulos foram revisados e corrigidos pelo autor (especialista COSMIC). A concordância inter-anotadores foi medida em um subconjunto de 20 itens anotados independentemente por um segundo especialista humano, resultando em κ = [valor]."

A pré-anotação por IA **não** deve ser apresentada como ground truth; é uma ferramenta de produtividade que reduz o tempo de anotação de ~10h para ~3–5h, mantendo o julgamento final humano.

---

## 2. Fluxo executado

| Etapa | Artefato | Responsável | Status |
|-------|----------|-------------|--------|
| 1. Amostragem (100 US, seed=42, estratificado por projeto) | `datasets/cross-sample/r4-100-us.csv` | script | ✅ |
| 2. Pré-anotação COSMIC | `datasets/user-stories-100-annotated-ai.csv` | Claude Opus 4.8 | ✅ |
| 3. Revisão e correção das 100 | (mesmo arquivo, coluna `cfp_annotator1`) | A1 (humano) | ⏳ pendente |
| 4. Anotação independente das 20 κ | `datasets/user-stories-100-kappa-blank.csv` | A2 (humano) | ⏳ pendente |
| 5. Cálculo κ A1×A2 + sessão de consenso | `scripts/compute_annotation_kappa.py` | A1 | ⏳ pendente |
| 6. Rodar modelos no ground truth final | matriz cross-dataset (coluna User Stories) | script | ⏳ pendente |

---

## 3. Resumo da pré-anotação da IA

- **Modelo:** Claude Opus 4.8 (`claude-opus-4-8`)
- **Protocolo aplicado:** `r4-annotation-protocol.md` v1.0 (algoritmo Passos 1–6, regras 4.1–4.6, casos de borda 6.1–6.6)
- **Distribuição CFP:** min=2, max=4, média=3.40, mediana=3.0 (2:3 itens, 3:54 itens, 4:43 itens)
- **Itens marcados `[AMBIGUO]`:** 22/100 — exigem atenção prioritária do revisor humano

### 3.1 Por que a distribuição é estreita (2–4 CFP)

Conforme antecipado na Seção 1.1 do protocolo, user stories mapeiam tipicamente a **um único processo funcional granular**, com poucos movimentos explícitos. Os padrões dominantes:

- **Visualização/busca** (3 CFP): E(parâmetros) + R(dados) + X(resultado) — 54 itens
- **Criação/atualização/deleção** (3–4 CFP): + W(persistência), eventualmente + R(registro atual)
- **NFR/workflow/deploy** (2 CFP, ambíguo): itens que não representam processo funcional COSMIC claro

### 3.2 Itens que mais demandam revisão humana (os 22 ambíguos)

| Categoria | IDs | Motivo |
|-----------|-----|--------|
| CRUD "manage/maintain" (regra 4.1) | US064, US089 | "manage/maintain" → ciclo genérico 4 CFP |
| Composto "and/also" (regra 4.6) | US047, US050, US056, US076, US078, US093 | múltiplos processos no mesmo enunciado |
| NFR / não-funcional | US013, US028, US061, US062, US095 | usabilidade, deploy, privacidade, acessibilidade, link-rot |
| Epic-level | US045, US065 | billing system, multiplayer — múltiplos processos |
| Vago / frase truncada | US001, US008, US009, US035, US057, US074, US099 | interpretação incerta dos movimentos |

**Recomendação ao revisor:** começar pelos 22 ambíguos, decidir caso a caso, e documentar a decisão na coluna `notes`. Para os 78 restantes, a revisão é mais rápida (padrões CRUD/view bem definidos).

---

## 4. Limitações declaradas

1. A pré-anotação da IA pode sistematicamente subestimar **Reads implícitas** (o erro mais comum em US, Seção 2.3 do protocolo). O revisor deve verificar especialmente se há leituras de armazenamento não capturadas.
2. A IA tende a usar o **mínimo razoável** em casos ambíguos; o especialista pode legitimamente expandir.
3. O κ deve ser calculado **A1 (humano) × A2 (humano)**, nunca incluindo a IA, sob pena de invalidar o argumento de independência.
