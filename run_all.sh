#!/usr/bin/env bash
# S4-T5 — Replicação total do pipeline experimental da tese.
#
# Uso:
#   bash run_all.sh --dry-run     # valida insumos sem executar nada
#   bash run_all.sh               # roda etapas locais (GPU p/ E28a)
#   RUN_API=1 bash run_all.sh     # inclui etapas pagas (R1-COSMET via OpenAI)
#
# Pré-requisitos: python3.10 + requirements.lock.txt; GGUF Qwen3-8B em
# models/gguf/ (SHA no README); OPENAI_API_KEY apenas se RUN_API=1.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3.10
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

step() { echo; echo "══ $1"; }
need_file() {
  if [[ ! -e "$2" ]]; then echo "  [FALTA] $2 ($1)"; MISSING=1;
  else echo "  [ok] $2"; fi
}
run() {
  if [[ $DRY -eq 1 ]]; then echo "  [dry-run] $*"; else echo "  \$ $*"; "$@"; fi
}

MISSING=0
step "0. Validação de insumos"
need_file "modelo SLM"        models/gguf/Qwen3-8B-Q4_K_M.gguf
need_file "manifest"          datasets/manifest.csv
need_file "holdout De Vito"   datasets/holdout/devito_holdout.csv
need_file "GT user stories"   datasets/user-stories-100-ground-truth-FINAL-v2.csv
need_file "input US"          results/cross-input/user-stories-98/ground_truth.csv
need_file "few-shots limpos"  models/hybrid-xai/normalizers/fewshots/v2-clean/example_01.txt
need_file "NER treinado"      models/cosmic-ner/best/meta.json
need_file "train E28a (M3)"   results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245/comparison_results.csv
[[ $MISSING -eq 1 && $DRY -eq 0 ]] && { echo "Insumos faltando — abortando."; exit 1; }

step "1. Testes (gate de sanidade)"
run $PY -m pytest tests/ -q

step "2. E28a — Qwen3-8B local (GPU; horas) [pulado se resultados existem]"
if [[ -d results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245 ]]; then
  echo "  [skip] runs E28a canônicas presentes (re-rodar exige remover dirs)"
else
  run $PY scripts/hybrid_xai_runner.py --dataset devito --devito-scope all \
      --normalizer slm --model-path models/gguf/Qwen3-8B-Q4_K_M.gguf
fi

step "3. E30f frozen — cross-dataset"
for ds in zenodo ochodek user-stories; do
  echo "  (e30f $ds — consulte apply_e30f_cross_dataset.py p/ dirs E28a)"
done
run bash -c 'echo "  [info] e30f usa runs E28a existentes; ver docs sprint2"'

step "4. E33d / E34a — pipeline simbólico (rápido)"
for ds in devito zenodo ochodek; do
  run $PY scripts/eval_ner_ontology_pipeline.py --dataset "$ds"
done
run $PY scripts/eval_ner_ontology_pipeline.py --dataset user-stories
run $PY scripts/eval_ner_ontology_pipeline.py --dataset user-stories --adapter connextra

step "5. R3-HKO (CV local)"
for ds in devito zenodo ochodek user-stories; do
  run $PY scripts/r3_hko_runner.py --dataset "$ds" --classifier c45 --n-classes 3
done

step "6. R1-COSMET (API paga — opt-in via RUN_API=1)"
if [[ "${RUN_API:-0}" == "1" ]]; then
  run $PY scripts/cosmet_batch_runner.py results/cross-input/zenodo-200 --dataset zenodo-200-r1
  run $PY scripts/cosmet_batch_runner.py results/cross-input/ochodek-200 --dataset ochodek-200-r1
  run $PY scripts/cosmet_batch_runner.py results/cross-input/user-stories-98 --dataset user-stories-98-r1
else
  echo "  [skip] RUN_API!=1 (custo ~\$32; runs canônicas já em results/cosmet/)"
fi

step "7. Consolidação — matriz, análises, estatística, figuras"
run $PY scripts/build_cross_dataset_matrix.py
run $PY scripts/error_analysis/collect_per_item.py
run $PY scripts/movement_level_eval.py
run $PY scripts/build_cost_benchmark.py
run $PY scripts/build_final_statistical_table.py
run $PY scripts/exp_pred25_all_cells.py
run $PY scripts/build_consolidated_table.py
run $PY scripts/figures/generate_canonical_figures.py
run $PY scripts/m3_generate_materials.py

step "8. M3 quantitativo — explicabilidade functionally-grounded (CPU; ~2h) [pulado se existe]"
if [[ -f results/m3-quantitative/summary.md ]]; then
  echo "  [skip] M3 quantitativo presente (remover results/m3-quantitative/ p/ re-rodar)"
else
  run $PY scripts/m3_quantitative_eval.py
fi

step "Concluído"
[[ $DRY -eq 1 ]] && echo "(dry-run: nada foi executado)" || \
  echo "Artefatos: results/cross-dataset-matrix, movement-level, final_statistical_table.*, figures/, m3-materials/, m3-quantitative/"
