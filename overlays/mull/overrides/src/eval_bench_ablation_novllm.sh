#!/usr/bin/env bash
set -euo pipefail
# Ablation evaluation script for Video-R1 latent tokens (no vLLM, pure transformers)
# Run from the mull root directory: bash src/eval_bench_ablation_novllm.sh

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
    "array/Qwen2.5-VL-MullGRPO"
)
FILE_NAME="video_r1_ablation"
DATASETS_LIST=("sat" "blink")  # Options: "sat" "vsibench" "mmvu" "blink"
# Set this to a real directory of GT auxiliary images before enabling gt_latent/random_latent_gt_dist
GT_IMAGE_DIR="/path/to/gt/images"

TOTAL_MODELS=${#MODEL_PATHS[@]}

for ((MODEL_IDX=0; MODEL_IDX<TOTAL_MODELS; MODEL_IDX++)); do
MODEL_PATH="${MODEL_PATHS[$MODEL_IDX]}"
MODEL_TAG=$(basename "$MODEL_PATH")
RUN_FILE_NAME="${FILE_NAME}_${MODEL_TAG}"
for DATASETS in "${DATASETS_LIST[@]}"; do
    RUN_ID="${RUN_FILE_NAME}_${DATASETS}"
    LOG_PREFIX="./logs/ablation/${RUN_ID}"
    echo "******************************************"
    echo "STARTING EVALUATION FOR MODEL [$((MODEL_IDX + 1))/$TOTAL_MODELS]: $MODEL_PATH"
    echo "STARTING EVALUATION FOR DATASET: $DATASETS"
    echo "RUN ID: $RUN_ID"
    echo "******************************************"

# Set to 1 to enable no-cache mode (deletes existing results before each run)
NO_CACHE=0

# Enable verbose debugging
# set -x  # Print each command before executing

export DECORD_EOF_RETRY_MAX=20480

echo "=========================================="
echo "Video-R1 Latent Token Ablation Experiments"
echo "(No vLLM - Pure Transformers)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Model tag: $MODEL_TAG"
echo "Datasets: $DATASETS"
echo "Run ID: $RUN_ID"
echo "No-Cache Mode: $NO_CACHE (1=enabled, 0=disabled)"
echo "Enabled ablations: ${RUN[@]}"
echo "Current Time: $(date)"
echo "Script PID: $$"
echo ""

# Create results directory
mkdir -p ./src/r1-v/eval_results
mkdir -p ./logs/ablation

# Helper function to clear cache for a specific ablation mode
clear_cache() {
    local ablation_suffix=$1
    local ablation_name=$2
    
    echo ""
    echo "--- Cache Clearing for $ablation_name ---"
    if [ "$NO_CACHE" -eq 1 ]; then
        for dataset in $DATASETS; do
            local output_file="./src/r1-v/eval_results/eval_${dataset}_${RUN_FILE_NAME}${ablation_suffix}_greedy_output.json"
            if [ -f "$output_file" ]; then
                echo "DELETING CACHE: $output_file"
                rm -f "$output_file"
                echo "Cache deleted successfully"
            else
                echo "No cache file found: $output_file"
            fi
        done
    else
        echo "Cache preservation enabled - skipping deletion"
    fi
    echo "--- End Cache Clearing ---"
    echo ""
}

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
    echo "DEBUG: Starting BASELINE experiment"
    echo "DEBUG: Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DEBUG: Ablation flags: NONE (baseline mode)"
    echo "DEBUG: Model path: $MODEL_PATH"
    echo "DEBUG: Datasets: $DATASETS"
    echo "DEBUG: Output file suffix: (none)"
    echo "=========================================="

    clear_cache "" "BASELINE"

    echo "DEBUG: Invoking Python script for BASELINE..."
    python3 ./src/eval_bench_ablation_novllm.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "${DATASETS}" \
        2>&1 | tee "${LOG_PREFIX}_baseline.log"

    echo ""
    echo "DEBUG: BASELINE experiment completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Baseline complete!"
    echo ""
fi

# 2. Zero Latent Ablation
if should_run "zero_latent"; then
    echo "=========================================="
    echo "2. Running ZERO LATENT ablation"
    echo "=========================================="
    echo "DEBUG: Starting ZERO LATENT experiment"
    echo "DEBUG: Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DEBUG: Ablation flags: --use-zero-latent"
    echo "DEBUG: Model path: $MODEL_PATH"
    echo "DEBUG: Datasets: $DATASETS"
    echo "DEBUG: Output file suffix: _zero_latent"
    echo "=========================================="

    clear_cache "_zero_latent" "ZERO LATENT"

    echo "DEBUG: Invoking Python script for ZERO LATENT..."
    python3 ./src/eval_bench_ablation_novllm.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-zero-latent \
        2>&1 | tee "${LOG_PREFIX}_zero_latent.log"

    echo ""
    echo "DEBUG: ZERO LATENT experiment completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Zero latent ablation complete!"
    echo ""
fi

# 3. Random Latent Ablation (Predicted Distribution)
if should_run "random_latent"; then
    echo "=========================================="
    echo "3. Running RANDOM LATENT ablation (predicted dist)"
    echo "=========================================="
    echo "DEBUG: Starting RANDOM LATENT experiment"
    echo "DEBUG: Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DEBUG: Ablation flags: --use-random-latent"
    echo "DEBUG: Model path: $MODEL_PATH"
    echo "DEBUG: Datasets: $DATASETS"
    echo "DEBUG: Output file suffix: _random_latent"
    echo "=========================================="

    clear_cache "_random_latent" "RANDOM LATENT"

    echo "DEBUG: Invoking Python script for RANDOM LATENT..."
    python3 ./src/eval_bench_ablation_novllm.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-random-latent \
        2>&1 | tee "${LOG_PREFIX}_random_latent.log"

    echo ""
    echo "DEBUG: RANDOM LATENT experiment completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Random latent ablation complete!"
    echo ""
fi

# 4. Random Latent Ablation (GT Distribution) - OPTIONAL
if should_run "same_latent"; then
    echo "=========================================="
    echo "4. Running SAME LATENT ablation (identity)"
    echo "=========================================="

    clear_cache "_same_latent" "SAME LATENT"

    python3 ./src/eval_bench_ablation_novllm.py \
        --model_path "$MODEL_PATH" \
        --file_name "$RUN_FILE_NAME" \
        --dataset_names "$DATASETS" \
        --use-same-latent \
        2>&1 | tee "${LOG_PREFIX}_same_latent.log"

    echo ""
    echo "Same latent ablation complete!"
    echo ""
fi

if should_run "random_latent_same_dist"; then
    echo "=========================================="
    echo "5. Running RANDOM LATENT ablation (same dist)"
    echo "=========================================="

    clear_cache "_random_latent_same_dist" "RANDOM LATENT (SAME DIST)"

    python3 ./src/eval_bench_ablation_novllm.py \
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
    echo "DEBUG: Starting FIRST LATENT REPEAT experiment"
    echo "DEBUG: Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "DEBUG: Ablation flags: --use-first-latent-repeat"
    echo "DEBUG: Model path: $MODEL_PATH"
    echo "DEBUG: Datasets: $DATASETS"
    echo "DEBUG: Output file suffix: _first_latent_repeat"
    echo "=========================================="

    clear_cache "_first_latent_repeat" "FIRST LATENT REPEAT"

    python3 ./src/eval_bench_ablation_novllm.py \
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
    python3 ./src/eval_bench_ablation_novllm.py \
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
    python3 ./src/eval_bench_ablation_novllm.py \
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
echo "DEBUG: All experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
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
echo "DEBUG: Listing all result files:"
ls -lh ./src/r1-v/eval_results/eval_*_${RUN_FILE_NAME}*_greedy_output.json 2>/dev/null || echo "No result files found"
    echo ""
    echo "DEBUG: Comparing accuracies from all experiments:"
    for log_file in ./logs/ablation/*.log; do
        if [ -f "$log_file" ]; then
            echo "--- $(basename $log_file) ---"
            grep -E "(mean_acc|mean_mra|Final accuracy)" "$log_file" | tail -5 || echo "No accuracy found"
            echo ""
        fi
    done
    echo ""
done
done

echo "Generating overall eval+ablation summary..."
python3 ./src/summarize_eval_ablation_results.py
