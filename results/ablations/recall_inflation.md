# Inflação por recall — via simbólica, De Vito (119 processos)

Todas as variantes são opções do construtor do pipeline, desligadas por
padrão. Reexecutadas no mesmo commit e sobre o mesmo conjunto.

| Variante | MAE | Δ MAE | SA | exatos | viés/proc. |
|---|--:|--:|--:|--:|--:|
| canônico (E33d implantado) | 0.9328 | +0.0000 | +0.5550 | 61 | -0.2101 |
| + leitura implícita na exibição (global) | 0.9664 | +0.0336 | +0.5389 | 59 | -0.1429 |
| + dedup por verbo+objeto | 1.4706 | +0.5378 | +0.2983 | 42 | +1.0000 |
| + injeção de leituras implícitas | 1.0420 | +0.1092 | +0.5028 | 54 | +0.0000 |
| + dedup melhorada (semântica) | 0.9328 | +0.0000 | +0.5550 | 61 | -0.2101 |

Leitura: toda alternativa que aumenta o recall de movimentos sobre a
extração não calibrada piora o MAE e desloca o viés para cima. O
gargalo é a calibração da extração, não a cobertura de regras.
