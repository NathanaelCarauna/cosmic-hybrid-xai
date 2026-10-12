# Avaliação Por-Movimento — De Vito 119 FPs (Sprint 4, T1)

**Data:** 2026-06-11 · **GT:** XLSX oficiais CFPS-grade (7 sistemas) · **Gerado por:** `scripts/movement_level_eval.py`

Motivação: o caso FP41 (auditoria em `m3-materials/README.md` §5) mostrou que **acerto de total ≠ acerto de decomposição**. Esta análise quantifica a qualidade da decomposição — a base de qualquer trace explicável. Métrica: micro precision/recall/F1 sobre contagens por tipo (match = min(pred, gt) por FP).

## Resultados (n=119)

| Método | Mov | Precision | Recall | F1 | MAE/FP | Viés |
|--------|:---:|:---------:|:------:|:--:|:------:|:----:|
| E33d | E | 0,826 | 0,944 | **0,881** | 0,34 | +0,19 |
| E33d | X | 0,914 | 0,901 | **0,908** | 0,33 | −0,03 |
| E33d | R | 0,927 | 0,724 | **0,813** | 0,29 | −0,19 |
| E33d | W | 0,989 | 0,798 | **0,883** | 0,20 | −0,19 |
<!-- atualizado 2026-06-16 com fix_validate_write ON (conserto de fidelidade) -->>
<!-- atualizado 2026-07-29: extract_movements_from_csv passou a somar a coluna CFP
     em vez de contar rótulos DM. ALBERGATE/FP55 contava linhas de fluxo alternativo
     com CFP=0 e somava 12 contra os 10 da linha TOTAL do próprio arquivo. Só o R1
     é afetado (o E33d vem de e33d_results.csv). -->
| R1-COSMET | E | 0,858 | 0,975 | 0,913 | 0,25 | +0,18 |
| R1-COSMET | X | 0,995 | 0,953 | 0,974 | 0,09 | −0,08 |
| R1-COSMET | R | 1,000 | 0,962 | 0,981 | 0,03 | −0,03 |
| R1-COSMET | W | 1,000 | 0,939 | 0,968 | 0,06 | −0,06 |

**Decomposição exata (4 tipos simultaneamente corretos):**

| Método | Decomp. exata | Total exato com decomp. ERRADA (cancelamento) |
|--------|:-------------:|:--------------------------------------------:|
| E33d | 54/119 (45,4%) | **7 FPs** |
| R1-COSMET | 83/119 (69,7%) | 4 FPs |

## Leituras (honestas)

1. **Após o conserto de fidelidade (fix_validate_write), o cancelamento caiu para 7/119 (6%)** (antes 14/119, 12%). Decomposição exata subiu de 46 para 54/119. Implicação para a escrita: reportar exact-match de total E de decomposição.
2. **O conserto elevou o recall de Writes de 0,675 para 0,798** (viés −0,30→−0,19); o ponto fraco remanescente passou a ser o recall de Reads (0,724, F1 0,813), por leituras implícitas não capturadas. Entries/Exits seguem bons (F1 0,88/0,91).
3. **R1-COSMET (gpt-4-turbo) decompõe muito melhor no home dataset** (F1 0,91–0,98; 70% de decomposição exata). O LLM grande entende movimentos implícitos que o pipeline simbólico não captura. O diferencial do Hybrid-XAI não é decompor melhor que o GPT-4 em UCs — é decompor de forma **auditável** (regra ISO + trecho), **local, a custo zero** e **transferível** (matriz 20/20; o R1 colapsa fora de UCs).
4. **⚠️ Ponto de honestidade para o M3/defesa:** o COSMET também produz uma tabela de movimentos (é daí que extraímos as contagens dele). A Condição B do survey ("apenas o CFP") representa a *saída típica de um estimador caixa-preta*, não a capacidade plena do COSMET. O diferencial real da Condição A é o **trace de regras** (qual regra ISO, aplicada a qual trecho) + diagrama — não a mera existência de movimentos. Registrar essa nuance no instrumento/dissertação para blindar contra a objeção "mas o COSMET também explica".

## Artefatos

- `movement_eval.csv` — métricas agregadas por método × tipo
- `per_fp.csv` — contagens GT vs pred por FP (119 × 2 métodos), com flags `decomp_exact` e `total_exact_only`
