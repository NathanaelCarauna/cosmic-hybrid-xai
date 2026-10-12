# Tabela Estatística Canônica (Sprint 4, T2)

Bootstrap percentílico 10k (seed 42); Wilcoxon signed-rank sobre |erro| por item; Cliff's δ (limiar Romano et al.).

## A — Células (MAE e SA com IC95)

`p(SA<=0)` é o p unilateral das mesmas reamostragens do IC: fração dos bootstraps com SA não-positiva. Qualifica as células cujo IC bilateral cruza zero.

| Método | Dataset | n | MAE [IC95] | SA [IC95] | SA | p(SA<=0) |
|--------|---------|--:|------------|-----------|----|---------:|
| E30f | devito | 30 | 1.133 [0.667, 1.733] | 0.285 [-0.064, 0.448] | indistinguível de 0 | 0.045 |
| E30f | ochodek | 200 | 7.4 [6.490, 8.330] | -0.475 [-0.623, -0.365] | negativo | 1.0 |
| E30f | user-stories | 98 | 1.49 [1.255, 1.745] | -1.065 [-1.770, -0.617] | negativo | 1.0 |
| E30f | zenodo | 200 | 1.185 [1.045, 1.330] | -1.756 [-2.419, -1.292] | negativo | 1.0 |
| E33d | devito | 119 | 0.933 [0.672, 1.235] | 0.555 [0.412, 0.668] | positivo | 0.0 |
| E33d | ochodek | 437 | 4.41 [4.034, 4.805] | 0.067 [0.018, 0.108] | positivo | 0.004 |
| E33d | zenodo | 2040 | 0.345 [0.321, 0.371] | 0.161 [0.096, 0.222] | positivo | 0.0 |
| E33d-sem-adaptador | user-stories | 98 | 2.602 [2.378, 2.878] | -2.606 [-3.629, -1.902] | negativo | 1.0 |
| E34a | user-stories | 98 | 0.469 [0.347, 0.602] | 0.349 [0.067, 0.569] | positivo | 0.0074 |
| E34a-test78 | user-stories | 78 | 0.5 [0.359, 0.654] | 0.261 [-0.026, 0.486] | indistinguível de 0 | 0.038 |
| R1-COSMET | devito | 119 | 0.37 [0.227, 0.546] | 0.824 [0.751, 0.883] | positivo | 0.0 |
| R1-COSMET | ochodek | 200 | 6.7 [5.840, 7.625] | -0.336 [-0.468, -0.233] | negativo | 1.0 |
| R1-COSMET | user-stories | 98 | 0.888 [0.683, 1.133] | -0.23 [-0.555, -0.009] | negativo | 0.9796 |
| R1-COSMET | zenodo | 200 | 0.775 [0.650, 0.910] | -0.802 [-1.294, -0.455] | negativo | 1.0 |
| R1-COSMET-test78 | user-stories | 78 | 0.859 [0.667, 1.064] | -0.27 [-0.646, -0.002] | negativo | 0.9761 |

## B — Comparações pareadas

Wilcoxon signed-rank sobre |erro| por item; *p* (Holm) = valor-p ajustado por Holm-Bonferroni sobre a família dos 5 testes pareados (α=0,05).

| Comparação | Dataset | MAE₁ | MAE₂ | p (Wilcoxon) | p (Holm) | Sig. (Holm) | Cliff's δ | Magnitude | Melhor |
|------------|---------|-----:|-----:|--------------|----------|-------------|----------:|-----------|--------|
| E33d vs R1-COSMET | devito (119) | 0.933 | 0.37 | 4.17e-05 | 8.34e-05 | sim | 0.253 | pequeno | R1-COSMET |
| E30f vs R1-COSMET | zenodo (200) | 1.185 | 0.775 | 9.76e-07 | 2.93e-06 | sim | 0.263 | pequeno | R1-COSMET |
| E34a vs R1-COSMET | user-stories (98) | 0.469 | 0.888 | 1.34e-03 | 1.34e-03 | sim | -0.224 | pequeno | E34a |
| E34a vs E30f | user-stories (98) | 0.469 | 1.49 | 5.55e-11 | 2.22e-10 | sim | -0.548 | grande | E34a |
| E34a vs E33d-sem-adaptador | user-stories (98) | 0.469 | 2.602 | 8.00e-20 | 4.00e-19 | sim | -0.94 | grande | E34a |

## C — Robustez da manchete à exclusão dos itens de desenvolvimento

O adaptador Connextra foi calibrado observando o gabarito de 20 histórias (`kappa_subset`). Esta comparação repete o par da Parte B apenas sobre os 78 itens restantes. **Fora da família de Holm**: é reanálise de um subconjunto do mesmo par, não hipótese independente.

| Comparação | Dataset | MAE₁ | MAE₂ | p (Wilcoxon) | Cliff's δ | Magnitude | Melhor |
|------------|---------|-----:|-----:|--------------|----------:|-----------|--------|
| E34a vs R1-COSMET (dev-20 excluído) | user-stories (78) | 0.5 | 0.859 | 3.80e-03 | -0.234 | pequeno | E34a |
