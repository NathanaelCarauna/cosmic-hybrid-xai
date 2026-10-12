# M3 — Avaliacao functionally-grounded (sem sujeitos humanos)

Hold-out De Vito, 30 FPs. LIME: K=8 reexecucoes/FP, num_features=25, num_samples=400, top-k=8. IC95 bootstrap percentilico (10k, seed 42).

## Metrica 2 — Estabilidade da explicacao

| Explicacao | Jaccard top-k [IC95] | Spearman pesos [IC95] |
|---|---|---|
| Trace simbolico (deterministico) | 1,000 (variancia 0) | 1,000 (variancia 0) |
| LIME post-hoc | 0.505 [0.463; 0.549] | 0.684 [0.654; 0.713] |

## Metrica 3 — Ancoragem semantica (massa de atribuicao em trechos funcionais)

| Explicacao | Ancoragem [IC95] |
|---|---|
| Trace simbolico (por construcao) | 1,000 |
| LIME — todos os tokens | 0.570 [0.525; 0.612] |
| LIME — sem stopwords | 0.449 [0.403; 0.491] |

medianas: Jaccard 0.493; ancoragem todos 0.589; sem stopwords 0.482.

Metrica 1 (fidelidade/correcao do trace): ver results/movement-level/ (F1 por tipo de movimento contra gabarito oficial).
