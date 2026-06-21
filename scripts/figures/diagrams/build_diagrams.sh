#!/usr/bin/env bash
# Renderiza os diagramas da dissertação (D1-D5) para PDF + PNG e copia os PDF
# para imagens/ do LaTeX. Fontes: Graphviz (.dot) e Mermaid (.mmd).
# Uso: bash scripts/figures/diagrams/build_diagrams.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../../../../docs/estimativa-cosmic-tese/LATEX_DISSERTAÇÃO_MESTRADO_2024/imagens"

echo "== Graphviz (D1-D4) =="
for dot in d1_arquitetura_hybridxai d2_via_simbolica_e33d d3_via_neural_e30f d4_competidores \
           d6_modelo_cosmic d7_representacao_intermediaria d8_adaptador_connextra d9_construcao_gt; do
  dot -Tpdf "$HERE/$dot.dot" -o "$HERE/$dot.pdf"
  dot -Tpng -Gdpi=160 "$HERE/$dot.dot" -o "$HERE/$dot.png"
  cp "$HERE/$dot.pdf" "$DEST/$dot.pdf"
  echo "  [OK] $dot"
done

echo "== Mermaid (D5) =="
# mmdc gera PDF/PNG; fundo branco p/ legibilidade.
mmdc -i "$HERE/d5_isomorfismo_sequencia.mmd" -o "$HERE/d5_isomorfismo_sequencia.pdf" -b white --pdfFit >/dev/null 2>&1 || \
  mmdc -i "$HERE/d5_isomorfismo_sequencia.mmd" -o "$HERE/d5_isomorfismo_sequencia.pdf" -b white >/dev/null 2>&1
mmdc -i "$HERE/d5_isomorfismo_sequencia.mmd" -o "$HERE/d5_isomorfismo_sequencia.png" -b white -s 2 >/dev/null 2>&1
cp "$HERE/d5_isomorfismo_sequencia.pdf" "$DEST/d5_isomorfismo_sequencia.pdf"
echo "  [OK] d5_isomorfismo_sequencia"

echo "Diagramas em $DEST"
