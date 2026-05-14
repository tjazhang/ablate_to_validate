#!/bin/bash
# needs to be run from the Video-R1 root directory
# bash src/eval_bench.sh

# Load newer GLIBC module for flash-attention compatibility
# module load mamslab/glibc/2.38

model_paths=(
 "array/Qwen2.5-VL-Mull"
  # "/home/jupyter/data_files/Qwen2.5-VL-7B-Instruct" # Qwen baseline
  # "/home/jupyter/data_files/Qwen2.5-VL-7B-COT-SFT" # Video R1 trained baseline
  # "/home/jupyter/Video-R1/src/r1-v/log/sat_vidr1_sft/checkpoint-18000",
  # "/home/jupyter/checkpoints/sat_vidr1_sft_pause/checkpoint-8000"
)

FILE_NAME="video_r1_eval"
DATASETS_LIST=("sat" "vsibench" "mmvu")

export DECORD_EOF_RETRY_MAX=20480
mkdir -p ./logs/eval

TOTAL_MODELS=${#model_paths[@]}

for ((MODEL_IDX=0; MODEL_IDX<TOTAL_MODELS; MODEL_IDX++)); do
  MODEL_PATH="${model_paths[$MODEL_IDX]}"
  MODEL_TAG=$(basename "$MODEL_PATH")
  RUN_FILE_NAME="${FILE_NAME}_${MODEL_TAG}"
  LOG_PATH="./logs/eval/${RUN_FILE_NAME}.log"

  echo "=========================================="
  echo "Running eval model [$((MODEL_IDX + 1))/$TOTAL_MODELS]: $MODEL_PATH"
  echo "Run file name: $RUN_FILE_NAME"
  echo "Datasets: ${DATASETS_LIST[*]}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=0 python3 ./src/eval_bench.py \
    --model_path "$MODEL_PATH" \
    --file_name "$RUN_FILE_NAME" \
    --dataset_names "${DATASETS_LIST[@]}" \
    2>&1 | tee "$LOG_PATH"
done

echo "Generating overall eval+ablation summary..."
python3 ./src/summarize_eval_ablation_results.py
