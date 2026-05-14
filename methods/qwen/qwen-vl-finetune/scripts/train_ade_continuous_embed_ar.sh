#!/bin/bash
# NEW: Aurora-only file; not present in upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Aurora path: qwen-vl-finetune/scripts/train_ade_continuous_embed_ar.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
QWEN_METHOD_DIR="${REPO_ROOT}/methods/qwen"
QWEN_FINETUNE_DIR="${QWEN_METHOD_DIR}/qwen-vl-finetune"

if command -v nvidia-smi >/dev/null 2>&1; then
    DEFAULT_NPROC="$(nvidia-smi --list-gpus | wc -l)"
else
    DEFAULT_NPROC=1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_JSON="${TRAIN_JSON:?Set TRAIN_JSON to the continuous-depth training JSON.}"
DEPTH_ENCODER_NAME="${DEPTH_ENCODER_NAME:-google/siglip2-large-patch16-256}"
ENCODER_CONFIG="${ENCODER_CONFIG:-${REPO_ROOT}/methods/llava/data/encoder_config.json}"
NEW_TOKENS_FILE="${NEW_TOKENS_FILE:-${QWEN_METHOD_DIR}/New_tokens_continuous.txt}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${SCRIPT_DIR}/zero3.json}"
ENTRY_FILE="${ENTRY_FILE:-${QWEN_FINETUNE_DIR}/qwenvl/train/train_qwen.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QWEN_FINETUNE_DIR}/output}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
REPORT_TO="${REPORT_TO:-none}"

DEPTH_HEAD_TYPE="${DEPTH_HEAD_TYPE:-linear}"
DEPTH_PROJECTOR_TYPE="${DEPTH_PROJECTOR_TYPE:-linear}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-mse}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-1.0}"
DEPTH_EMBED_AR_GT_RATIO="${DEPTH_EMBED_AR_GT_RATIO:-0.0}"

if [[ "${REPORT_TO}" == "wandb" ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-qwen-vl-ade-depth}"
    export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
    export WANDB_WATCH="${WANDB_WATCH:-false}"
fi

encoder_slug="$(echo "${DEPTH_ENCODER_NAME}" | sed 's|/|_|g' | sed 's|-|_|g')"
RUN_NAME="${RUN_NAME:-qwen2.5vl-3b-ade-continuous-embed_ar-${NUM_EPOCHS}ep-${encoder_slug}-${DEPTH_HEAD_TYPE}_${DEPTH_PROJECTOR_TYPE}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --deepspeed "${DEEPSPEED_CFG}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --dataset_use "${TRAIN_JSON}"
    --config_path "${ENCODER_CONFIG}"
    --new_tokens_file "${NEW_TOKENS_FILE}"
    --use_discrete_depth_tokens False
    --depth_encoder_name "${DEPTH_ENCODER_NAME}"
    --lambda_depth "${LAMBDA_DEPTH}"
    --depth_head_type "${DEPTH_HEAD_TYPE}"
    --depth_projector_type "${DEPTH_PROJECTOR_TYPE}"
    --depth_loss_type "${DEPTH_LOSS_TYPE}"
    --use_depth_embed_ar True
    --depth_embed_ar_gt_ratio "${DEPTH_EMBED_AR_GT_RATIO}"
    --reinitialization_method random
    --data_flatten False
    --data_sequential False
    --data_packing False
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --tune_embeddings True
    --bf16
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --save_strategy epoch
    --save_total_limit 1
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
)

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${ENTRY_FILE}" "${ARGS[@]}"
