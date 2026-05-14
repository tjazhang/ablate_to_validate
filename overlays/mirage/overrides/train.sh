#!/usr/bin/env bash
set -euo pipefail

# Aurora overlay depth-training wrapper.
# Dataset flow:
# 1. `src/generate_gt_images.py` renders reasoning-path helper images.
# 2. The JSONL referenced by MIRAGE_DEPTH_TRAIN_JSONL stores the RGB image path,
#    prompt/answer text, and the paired reasoning image path.
# 3. `src/task_new.py` loads that paired reasoning image during `depth-reasoning`
#    training, so stage1/stage2 both consume the same processed JSONL.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

CACHE_DIR="${MIRAGE_CACHE_DIR:-${SCRIPT_DIR}/checkpoints}"
TRAIN_STAGE="${MIRAGE_TRAIN_STAGE:-stage2}"
BASE_MODEL="${MIRAGE_BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EPOCHS="${MIRAGE_DEPTH_EPOCHS:-5}"
LATENT_SIZE="${MIRAGE_LATENT_SIZE:-4}"
DATA_PATH="${MIRAGE_DEPTH_TRAIN_JSONL:-${SCRIPT_DIR}/data/depth_reasoning/train_depth_long.jsonl}"
LOG_DIR="${MIRAGE_LOG_DIR:-${SCRIPT_DIR}/logs/train}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRIPT_DIR}/.cache/triton}"
WANDB_PROJECT="${WANDB_PROJECT:-mirage-depth-reasoning}"
WANDB_DIR="${WANDB_DIR:-${SCRIPT_DIR}/wandb}"

STAGE1_OUTPUT="${MIRAGE_STAGE1_OUTPUT:-${CACHE_DIR}/depth_model_stage1_5ep}"
STAGE2_INPUT="${MIRAGE_STAGE2_INPUT:-${STAGE1_OUTPUT}/checkpoint-96395}"
STAGE2_OUTPUT="${MIRAGE_STAGE2_OUTPUT:-${CACHE_DIR}/depth_model_stage2_5ep}"

mkdir -p "$CACHE_DIR" "$LOG_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TRITON_CACHE_DIR
export WANDB_PROJECT
export WANDB_DIR

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: depth-reasoning JSONL not found: $DATA_PATH"
    echo "Set MIRAGE_DEPTH_TRAIN_JSONL to the processed train JSONL produced for the depth task."
    exit 1
fi

echo "================================================================================"
echo "Mirage depth training"
echo "================================================================================"
echo "Stage:          ${TRAIN_STAGE}"
echo "Base model:     ${BASE_MODEL}"
echo "Train JSONL:    ${DATA_PATH}"
echo "Checkpoint dir: ${CACHE_DIR}"
echo "Wandb dir:      ${WANDB_DIR}"
echo "================================================================================"
echo ""

case "$TRAIN_STAGE" in
    stage1)
        python3 src/main.py \
            --model "$BASE_MODEL" \
            --epochs "$EPOCHS" \
            --task depth-reasoning \
            --latent_size "$LATENT_SIZE" \
            --stage stage1 \
            --data_path "$DATA_PATH" \
            --log_file "${LOG_DIR}/depth_model_stage1.txt" \
            --cache_dir "$CACHE_DIR" \
            --save_model_path "$STAGE1_OUTPUT"
        ;;
    stage2)
        if [[ ! -d "$STAGE2_INPUT" ]]; then
            echo "ERROR: stage2 checkpoint not found: $STAGE2_INPUT"
            echo "Run stage1 first or override MIRAGE_STAGE2_INPUT."
            exit 1
        fi
        python3 src/main.py \
            --model "$BASE_MODEL" \
            --epochs "$EPOCHS" \
            --task depth-reasoning \
            --latent_size "$LATENT_SIZE" \
            --stage stage2 \
            --data_path "$DATA_PATH" \
            --log_file "${LOG_DIR}/depth_model_stage2.txt" \
            --cache_dir "$CACHE_DIR" \
            --load_model_path "$STAGE2_INPUT" \
            --save_model_path "$STAGE2_OUTPUT"
        ;;
    *)
        echo "ERROR: unsupported MIRAGE_TRAIN_STAGE=${TRAIN_STAGE} (expected stage1 or stage2)"
        exit 1
        ;;
esac
