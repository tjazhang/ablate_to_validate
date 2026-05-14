#!/usr/bin/env bash
set -euo pipefail

# Aurora overlay: keep the Mirage ablation entrypoint separate from upstream
# and avoid machine-specific absolute paths in the copied repo checkout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "${HOME}/.bashrc" ]]; then
    # Preserve the old workflow for environments that configure CUDA/modules in bashrc.
    # The script still runs without it if the environment is already active.
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

CACHE_DIR="${MIRAGE_CACHE_DIR:-${SCRIPT_DIR}/checkpoints}"
BASE_MODEL="${MIRAGE_BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOAD_MODEL_PATH="${MIRAGE_MODEL_PATH:-Miiche/vsp_spatial_planning_direct_sft}"
TASK="${MIRAGE_TASK:-vsp-spatial-planning}"
EPOCHS="${MIRAGE_EPOCHS:-15}"

read -r -a SPLITS <<< "${MIRAGE_SPLITS:-test}"

RUN=(
    # "baseline"
    # "gt_latent"
    # "gt_latent_count_matched"
    # "random_latent"
    # "random_latent_model_dist"
    # "random_latent_gt_dist"
    # "zero_latent"
    # "model_latent"
    "first_latent_repeat"
)

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results"

should_run() {
    local ablation_type=$1
    for item in "${RUN[@]}"; do
        if [[ "$item" == "$ablation_type" ]]; then
            return 0
        fi
    done
    return 1
}

run_eval() {
    local split=$1
    local label=$2
    local log_suffix=$3
    local extra_flag=${4:-}

    echo "Running ${label} on split=${split}"
    USE_MIRAGE_NEW=1 python3 src/test_new.py \
        --model "$BASE_MODEL" \
        --epochs "$EPOCHS" \
        --task "$TASK" \
        --split "$split" \
        --load_model_path "$LOAD_MODEL_PATH" \
        --cache_dir "$CACHE_DIR" \
        --log_file "${SCRIPT_DIR}/logs/log_${split}_${log_suffix}.txt" \
        ${extra_flag:+$extra_flag}
}

for split in "${SPLITS[@]}"; do
    echo "========================================"
    echo "Running Mirage ablations on split=${split}"
    echo "Model checkpoint: ${LOAD_MODEL_PATH}"
    echo "Enabled ablations: ${RUN[*]}"
    echo "========================================"

    if should_run "baseline"; then
        run_eval "$split" "baseline" "baseline"
    fi
    if should_run "gt_latent"; then
        run_eval "$split" "gt_latent" "gt_latent" "--use-gt-latent"
    fi
    if should_run "gt_latent_count_matched"; then
        run_eval "$split" "gt_latent_count_matched" "gt_latent_count_matched" "--use-gt-latent-count-matched"
    fi
    if should_run "random_latent"; then
        run_eval "$split" "random_latent" "random_latent" "--use-random-latent"
    fi
    if should_run "random_latent_model_dist"; then
        run_eval "$split" "random_latent_model_dist" "random_latent_model_dist" "--use-random-latent-model-dist"
    fi
    if should_run "random_latent_gt_dist"; then
        run_eval "$split" "random_latent_gt_dist" "random_latent_gt_dist" "--use-random-latent-gt-dist"
    fi
    if should_run "zero_latent"; then
        run_eval "$split" "zero_latent" "zero_latent" "--use-zero-latent"
    fi
    if should_run "model_latent"; then
        run_eval "$split" "model_latent" "model_latent" "--use-model-latent"
    fi
    if should_run "first_latent_repeat"; then
        run_eval "$split" "first_latent_repeat" "first_latent_repeat" "--use-first-latent-repeat"
    fi
done

echo "========================================"
echo "All Mirage ablation runs completed"
echo "Logs:    ${SCRIPT_DIR}/logs"
echo "Results: ${SCRIPT_DIR}/results"
echo "========================================"
echo "Generating overall eval+ablation summary..."
python3 "${SCRIPT_DIR}/summarize_eval_ablation_results.py"
