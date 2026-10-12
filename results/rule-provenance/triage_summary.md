# Triagem por proveniência de regra (sinal de confiança de custo zero)

Item marcado = ao menos um movimento classificado por regra de baixa
certeza (fallback/ambiguidade). Nenhuma chamada de API, nenhum treino.

| Formato | n | marcados | % itens | % da massa de erro | MAE marcados | MAE demais | discriminação |
|---|--:|--:|--:|--:|--:|--:|--:|
| DE VITO (casos de uso) | 119 | 40 | 33.6% | 46.0% | 1.275 | 0.759 | 1.7x |
| ZENODO (sentenças) | 2040 | 554 | 27.2% | 40.6% | 0.516 | 0.281 | 1.8x |
| USER STORIES (Connextra) | 98 | 29 | 29.6% | 41.3% | 0.655 | 0.391 | 1.7x |

Leitura: a proveniência da regra separa, sem custo, uma minoria de
itens que concentra desproporcionalmente o erro — nos três formatos, no
mesmo sentido. É subproduto do trace do M3 e estruturalmente indisponível
para métodos que não expõem a regra aplicada.
