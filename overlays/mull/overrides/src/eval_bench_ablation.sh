#!/usr/bin/env bash
set -euo pipefail
# Ablation evaluation script for Video-R1 latent tokens
# Run from the mull root directory: bash src/eval_bench_ablation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ========================================
# CONFIGURATION: Control what to run
# ========================================
# Uncomment/add ablation types you want to run:
RUN=(
    "baseline"                    # Normal latent tokens
    "zero_latent"                 # All zeros
    "random_latent"               # Random with predicted distribution
    "same_latent"                 # Identity replacement sanity check
    "first_latent_repeat"         # First model latent repeated for remaining latent tokens
    "random_latent_same_dist"     # Random matched to replaced vector distribution
    # "random_latent_model_dist"  # Random matched to model output distribution
    # "random_latent_gt_dist"     # Random with GT distribution (optional)
    # "gt_latent"                 # GT latent (optional, requires GT images)
)

# Configuration
# MODEL_PATHS=("array/Qwen2.5-VL-Mull")  # UPDATE THIS
MODEL_PATHS=(
    "array/Qwen2.5-VL-Mull"
)
FILE_NAME="video_r1_ablation"
DATASETS="sat"  # Options: "sat" "vsibench" "mmvu" "blink"
# Set this to a real directory of GT auxiliary images before enabling gt_latent/random_latent_gt_dist
GT_IMAGE_DIR="/path/to/gt/images"

export DECORD_EOF_RETRY_MAX=20480

TOTAL_MODELS=${#MODEL_PATHS[@]}

for ((MODEL_IDX=0; MODEL_IDX<TOTAL_MODELS; MODEL_IDX++)); do
MODEL_PATH="${MODEL_PATHS[$MODEL_IDX]}"
MODEL_TAG=$(basename "$MODEL_PATH")
RUN_FILE_NAME="${FILE_NAME}_${MODEL_TAG}"
RUN_ID="${RUN_FILE_NAME}_${DATASETS}"
LOG_PREFIX="./logs/ablation/${RUN_ID}"
echo "=========================================="
echo "Video-R1 Latent Token Ablation Experiments"
echo "=========================================="
echo "Model Progress: [$((MODEL_IDX + 1))/$TOTAL_MODELS]"
echo "Model: $MODEL_PATH"
echo "Model tag: $MODEL_TAG"
echo "Datasets: $DATASETS"
echo "Run ID: $RUN_ID"
echo "Enabled ablations: ${RUN[@]}"
echo ""

# Create results directory
mkdir -p ./src/r1-v/eval_results
mkdir -p ./logs/ablation

# Helper function to check if ablation type should run
should_run() {
    local ablation_type=$1
    for item in "${RUN[@]}"; do
        if [ "$item" = "$ablation_type" ]; then
            return 0
        fi
    done
    return 1
}

# 1. Baseline (Normal mode - no ablation)
if should_run "baseline"; then
    echo "=========================================="
    echo "1. Running BASELINE (Normal latent tokens)"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        2>&1 | tee "${LOG_PREFIX}_baseline.log"

    echo ""
    echo "Baseline complete!"
    echo ""
fi

# 2. Zero Latent Ablation
if should_run "zero_latent"; then
    echo "=========================================="
    echo "2. Running ZERO LATENT ablation"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-zero-latent \
        2>&1 | tee "${LOG_PREFIX}_zero_latent.log"

    echo ""
    echo "Zero latent ablation complete!"
    echo ""
fi

# 3. Random Latent Ablation (Predicted Distribution)
if should_run "random_latent"; then
    echo "=========================================="
    echo "3. Running RANDOM LATENT ablation (predicted dist)"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-random-latent \
        2>&1 | tee "${LOG_PREFIX}_random_latent.log"

    echo ""
    echo "Random latent ablation complete!"
    echo ""
fi

# 4. Same Latent Ablation (Identity)
if should_run "same_latent"; then
    echo "=========================================="
    echo "4. Running SAME LATENT ablation (identity)"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-same-latent \
        2>&1 | tee "${LOG_PREFIX}_same_latent.log"

    echo ""
    echo "Same latent ablation complete!"
    echo ""
fi

# 5. Random Latent Ablation (Same Distribution)
if should_run "random_latent_same_dist"; then
    echo "=========================================="
    echo "5. Running RANDOM LATENT ablation (same dist)"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-random-latent-same-dist \
        2>&1 | tee "${LOG_PREFIX}_random_latent_same_dist.log"

    echo ""
    echo "Random latent (same dist) ablation complete!"
    echo ""
fi

# 6. First Latent Repeat Ablation
if should_run "first_latent_repeat"; then
    echo "=========================================="
    echo "6. Running FIRST LATENT REPEAT ablation"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-first-latent-repeat \
        2>&1 | tee "${LOG_PREFIX}_first_latent_repeat.log"

    echo ""
    echo "First latent repeat ablation complete!"
    echo ""
fi

# 7. Random Latent Ablation (GT Distribution) - OPTIONAL
if should_run "random_latent_gt_dist"; then
    if [ ! -d "$GT_IMAGE_DIR" ]; then
        echo "ERROR: GT_IMAGE_DIR does not exist: $GT_IMAGE_DIR"
        exit 1
    fi
    echo "=========================================="
    echo "7. Running RANDOM LATENT ablation (GT dist)"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-random-latent-gt-dist \
        --gt-image-dir "$GT_IMAGE_DIR" \
        2>&1 | tee "${LOG_PREFIX}_random_latent_gt_dist.log"

    echo ""
    echo "Random latent (GT dist) ablation complete!"
    echo ""
fi

# 8. GT Latent Ablation - OPTIONAL
if should_run "gt_latent"; then
    if [ ! -d "$GT_IMAGE_DIR" ]; then
        echo "ERROR: GT_IMAGE_DIR does not exist: $GT_IMAGE_DIR"
        exit 1
    fi
    echo "=========================================="
    echo "8. Running GT LATENT ablation"
    echo "=========================================="
    python3 ./src/eval_bench_ablation.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-gt-latent \
        --gt-image-dir "$GT_IMAGE_DIR" \
        2>&1 | tee "${LOG_PREFIX}_gt_latent.log"
 
    echo ""
    echo "GT latent ablation complete!"
    echo ""
fi

echo "=========================================="
echo "All ablation experiments complete!"
echo "=========================================="
echo "Results saved to: ./src/r1-v/eval_results/"
echo "Logs saved to: ./logs/ablation/"
echo ""
echo "To analyze results, check the JSON files in eval_results/"
echo "Compare accuracy across different ablation modes:"
echo "  - baseline: Normal latent tokens"
echo "  - zero_latent: All zeros"
echo "  - random_latent: Random with predicted distribution"
echo "  - same_latent: Identity replacement sanity check"
echo "  - first_latent_repeat: Repeat first model latent for all latent steps"
echo "  - random_latent_same_dist: Random matched to replaced vector distribution"
echo "  - random_latent_model_dist: Random matched to model output distribution"
echo "  - random_latent_gt_dist: Random matched to GT distribution (requires GT images)"
echo "  - gt_latent: GT latent replacement (requires GT images)"
echo ""
done

echo "Generating overall eval+ablation summary..."
python3 ./src/summarize_eval_ablation_results.py
