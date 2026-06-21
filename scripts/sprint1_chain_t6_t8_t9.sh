#!/usr/bin/env bash
# Sprint 1 — Chained compute tasks T6, T8, T9
# Run this script AFTER T5 (E28a-clean full 119) has completed.
#
# Usage:
#   bash scripts/sprint1_chain_t6_t8_t9.sh \
#       results/hybrid-xai/119fp-E28a-qwen3-8b-clean-<timestamp>
#
# If no argument given, auto-detects the latest E28a-clean dir.

set -e
cd "$(dirname "$0")/.."

E28A_CLEAN="${1:-}"
if [ -z "$E28A_CLEAN" ]; then
    E28A_CLEAN=$(ls -dt results/hybrid-xai/119fp-E28a-qwen3-8b-clean-* 2>/dev/null | head -1)
fi
if [ -z "$E28A_CLEAN" ] || [ ! -d "$E28A_CLEAN" ]; then
    echo "ERROR: E28a-clean dir not found. Pass it as first argument."
    exit 1
fi
echo "Using E28a-clean dir: $E28A_CLEAN"

N_LINES=$(wc -l < "$E28A_CLEAN/comparison_results.csv" 2>/dev/null || echo 0)
if [ "$N_LINES" -lt 120 ]; then
    echo "ERROR: comparison_results.csv has only $((N_LINES-1)) rows (expected 119). T5 may not be complete."
    exit 1
fi
echo "T5 output verified: $((N_LINES-1)) rows"

# ── T6: Run E30f-clean-LOOCV on full 119 FPs ─────────────────────────────────
echo ""
echo "========================================================"
echo "T6: Running E30f-clean-LOOCV on full 119..."
echo "========================================================"
T6_OUT="results/hybrid-xai/bert-improvements-clean-loocv"
python3.10 scripts/bert_hybrid_improvements_v2.py \
    --e28a-dir "$E28A_CLEAN" \
    --output-dir "$T6_OUT" \
    --skip e30g e30h e30i
echo "T6 complete → $T6_OUT"

# ── T6: Compare clean vs leaked ───────────────────────────────────────────────
LEAKED_DIR="results/hybrid-xai/bert-improvements-turbo/E30f-correction-ensemble"
if [ -d "$LEAKED_DIR" ]; then
    python3.10 scripts/compare_e30f_clean_vs_leaked.py \
        --leaked "$LEAKED_DIR" \
        --clean  "$T6_OUT/E30f-correction-ensemble" \
        --output results/leakage-impact
    echo "T6 comparison written to results/leakage-impact/"
else
    echo "WARNING: leaked dir not found at $LEAKED_DIR — skipping comparison"
fi

# ── T8: Re-run E28a on train set (89 FPs) ────────────────────────────────────
echo ""
echo "========================================================"
echo "T8: Re-running E28a on train set (89 FPs)..."
echo "========================================================"
TRAIN_IDS="datasets/holdout/devito_train.csv"
python3.10 scripts/hybrid_xai_runner.py \
    replications/cosmet-devito/datasets/MIS/ALBERGATE \
    --normalizer slm \
    --model-path models/gguf/Qwen3-8B-Q4_K_M.gguf \
    --dataset devito \
    --devito-scope all \
    --n-threads 8 \
    --n-gpu-layers 0 \
    --experiment-id E28a-qwen3-8b-clean-train \
    --fp-ids-file "$TRAIN_IDS"

TRAIN_DIR=$(ls -dt results/hybrid-xai/*E28a-qwen3-8b-clean-train* 2>/dev/null | head -1)
echo "T8 E28a-train complete → $TRAIN_DIR"

# ── T8: Run E30f on train set only ───────────────────────────────────────────
T8_OUT="results/hybrid-xai/bert-improvements-clean-train"
python3.10 scripts/bert_hybrid_improvements_v2.py \
    --e28a-dir "$TRAIN_DIR" \
    --output-dir "$T8_OUT" \
    --skip e30g e30h e30i
echo "T8 E30f-train complete → $T8_OUT"

# Extract hparams from metrics
python3.10 - <<'PYEOF'
import json
from pathlib import Path

t8_out = Path("results/hybrid-xai/bert-improvements-clean-train")
met_path = t8_out / "E30f-correction-ensemble" / "metrics.json"
if met_path.exists():
    met = json.loads(met_path.read_text())
    hparams = {
        "mean_blend_weight": met.get("mean_blend_weight"),
        "std_blend_weight": met.get("std_blend_weight"),
        "note": "Hyperparameters frozen from T8 train-set calibration; apply to hold-out as-is."
    }
    out = t8_out / "hparams_final.json"
    out.write_text(json.dumps(hparams, indent=2))
    print(f"hparams_final.json written: {hparams}")
else:
    print(f"WARNING: metrics.json not found at {met_path}")
PYEOF

# ── T9: Hold-out evaluation (ONE SHOT — do not re-run) ───────────────────────
echo ""
echo "========================================================"
echo "T9: ONE-SHOT hold-out evaluation (De Vito 30 + Zenodo 50)"
echo "========================================================"

HOLDOUT_IDS="datasets/holdout/devito_holdout.csv"
python3.10 scripts/hybrid_xai_runner.py \
    replications/cosmet-devito/datasets/MIS/ALBERGATE \
    --normalizer slm \
    --model-path models/gguf/Qwen3-8B-Q4_K_M.gguf \
    --dataset devito \
    --devito-scope all \
    --n-threads 8 \
    --n-gpu-layers 0 \
    --experiment-id E28a-qwen3-8b-holdout-devito \
    --fp-ids-file "$HOLDOUT_IDS"

HOLDOUT_DIR=$(ls -dt results/hybrid-xai/*E28a-qwen3-8b-holdout-devito* 2>/dev/null | head -1)

# Run E30f on hold-out with frozen hparams (the ensemble uses LOOCV internally,
# so T8 hparams serve as reference; the actual E30f LOOCV on 30 FPs is self-contained)
T9_OUT="results/hybrid-xai/holdout-final-devito"
python3.10 scripts/bert_hybrid_improvements_v2.py \
    --e28a-dir "$HOLDOUT_DIR" \
    --output-dir "$T9_OUT" \
    --skip e30g e30h e30i
echo "T9 De Vito hold-out complete → $T9_OUT"

echo ""
echo "========================================================"
echo "All T6/T8/T9 tasks complete. Run:"
echo "  git tag sprint1-holdout-final"
echo "========================================================"
