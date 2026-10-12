# Golden baseline do motor inglês

CFP por item produzido pelo motor **antes** do porte para português
(`scripts/build_golden_baseline.py`).

Estes arquivos são a rede de segurança do refactor: `tests/test_english_regression.py`
exige **igualdade exata**. Se um valor mudar, o refactor alterou o comportamento
em inglês e os números canônicos da dissertação deixariam de valer.

**Nunca regenere estes arquivos para "consertar" um teste que falhou.**
Uma falha significa que o código mudou, não que o baseline está errado.
