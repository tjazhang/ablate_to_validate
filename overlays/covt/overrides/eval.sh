#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LMUData="${LMUData:-${SCRIPT_DIR}/.cache/LMUData}"
mkdir -p "$LMUData"

if command -v module >/dev/null 2>&1; then
    module load gcc/11.2.0 || true
fi

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

###############################################################################
# CONFIGURATION - Edit these parameters to control evaluation behavior
###############################################################################
#
# Usage:
#   1. Set BASE_MODELS list (model names that match keys in config.py)
#   2. Set ABLATION_MODES to control which ablation modes to run:
#      - "none"         : Normal inference (visual-thinking tokens used as-is)
#      - "zero"         : Replace visual-thinking token embeddings with zeros
#      - "random"       : Replace visual-thinking token embeddings with random vectors
#      - "same"         : Replace visual-thinking token embeddings with themselves
#      - "random-dist"  : Replace with random vectors matched to original per-token mean/std
#      - "first-repeat" : Use first model anchor embedding for all remaining anchor positions
#   3. Set BENCHMARKS list
#   4. Run: ./eval.sh
#
###############################################################################

BASE_MODELS=(
    "CoVT-7B-depth"
    # "CoVT-LLaVA-13B-depth"
    # "CoVT-7B-seg"
    # "CoVT-7B-seg_depth_dino"
    # "CoVT-7B-seg_depth_dino_edge"
)

ABLATION_MODES=(
    # "none"
    # "zero"
    # "random"
    # "same"
    # "random-dist"
    "first-repeat"
)

BENCHMARKS=(
    "CV-Bench-2D"
    "CV-Bench-3D"
)

EVAL_MODE="${COVT_EVAL_MODE:-all}"
WORK_DIR="${COVT_WORK_DIR:-outputs}"
VERBOSE="${COVT_VERBOSE:-true}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLMEVAL_DIR="${VLMEVAL_DIR:-${SCRIPT_DIR}/VLMEvalKit}"

if [[ ! -d "$VLMEVAL_DIR" ]]; then
    echo "ERROR: VLMEvalKit checkout not found: $VLMEVAL_DIR"
    exit 1
fi

cd "$VLMEVAL_DIR"

BENCH_ARGS="${BENCHMARKS[*]}"
EXTRA_FLAGS=(--mode "$EVAL_MODE" --work-dir "$WORK_DIR")
if [[ "$VERBOSE" == "true" ]]; then
    EXTRA_FLAGS+=(--verbose)
fi

echo "================================================================================"
echo "CoVT VLMEvalKit Evaluation"
echo "================================================================================"
echo "Repo root:        ${SCRIPT_DIR}"
echo "VLMEvalKit:       ${VLMEVAL_DIR}"
echo "Base models:      ${BASE_MODELS[*]}"
echo "Ablation modes:   ${ABLATION_MODES[*]}"
echo "Benchmarks:       ${BENCHMARKS[*]}"
echo "Mode:             ${EVAL_MODE}"
echo "Output dir:       ${VLMEVAL_DIR}/${WORK_DIR}"
echo "LMUData:          ${LMUData}"
echo "================================================================================"
echo ""

for ABLATION in "${ABLATION_MODES[@]}"; do
    RUN_MODELS=()
    for BASE in "${BASE_MODELS[@]}"; do
        if [[ "$ABLATION" == "none" ]]; then
            RUN_MODELS+=("$BASE")
        else
            RUN_MODELS+=("${BASE}-${ABLATION}")
        fi
    done

    echo "================================================================================"
    echo "ABLATION MODE: ${ABLATION}"
    echo "Models: ${RUN_MODELS[*]}"
    echo "================================================================================"
    echo ""

    "$PYTHON_BIN" run.py \
        --data $BENCH_ARGS \
        --model "${RUN_MODELS[@]}" \
        "${EXTRA_FLAGS[@]}"

    echo ""
    echo "Ablation '${ABLATION}' complete."
    echo ""
done

echo "================================================================================"
echo "ALL EVALUATIONS COMPLETE. Results saved to: ${VLMEVAL_DIR}/${WORK_DIR}"
echo "================================================================================"
echo "Generating overall eval+ablation summary..."
"$PYTHON_BIN" "${VLMEVAL_DIR}/../summarize_eval_ablation_results.py"
