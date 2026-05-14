#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# MIRAGE DEPTH REASONING EVALUATION ON HARDBLINK
###############################################################################
#
# Aurora overlay notes:
# - This wrapper stays separate from upstream Mirage eval so the HardBlink
#   depth-ablation flow is easy to bootstrap and review.
# - All local paths are configurable via environment variables instead of
#   hard-coded user-specific scratch directories.
#
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# CONFIGURATION
# ============================================================================

AURORA_ROOT="${AURORA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CACHE_DIR="${MIRAGE_CACHE_DIR:-${SCRIPT_DIR}/checkpoints}"
MODEL_PATH="${MIRAGE_MODEL_PATH:-${CACHE_DIR}/depth_model_stage2_5ep}"
LATENT_SIZE="${MIRAGE_LATENT_SIZE:-4}"
MAX_NEW_TOKENS="${MIRAGE_MAX_NEW_TOKENS:-2048}"

ABLATION_MODES=(
    # "baseline"
    # "gt_latent"
    # "gt_latent_count_matched"
    # "random_latent"
    # "zero_latent"
    # "model_latent"
    "first_latent_repeat"
    # "random_latent_gt_dist"
    # "random_latent_model_dist"
)

DEPTH_IMAGE_FOLDER="${MIRAGE_DEPTH_IMAGE_FOLDER:-}"
DEPTH_IMAGE_FOLDERS=(
    "${MIRAGE_DEPTH_IMAGE_FOLDER_BLINK3:-}"
    "${MIRAGE_DEPTH_IMAGE_FOLDER_BLINK4:-}"
    "${MIRAGE_DEPTH_IMAGE_FOLDER_BLINK5:-}"
)

VERBOSE="${MIRAGE_VERBOSE:-false}"
FORCE_REGENERATE="${MIRAGE_FORCE_REGENERATE:-true}"

# Aurora-specific eval resources live outside Mirage itself. Defaults are
# repo-relative placeholders; override them in the environment when data sits elsewhere.
HARDBLINK_ROOT="${MIRAGE_HARDBLINK_ROOT:-${SCRIPT_DIR}/data/hardblink}"
QUESTION_DIR="${MIRAGE_QUESTION_DIR:-${HARDBLINK_ROOT}/questions}"
IMAGE_ROOT="${MIRAGE_IMAGE_ROOT:-${HARDBLINK_ROOT}/images}"
GT_DIR="${MIRAGE_GT_DIR:-${HARDBLINK_ROOT}/answers}"
OUTPUT_ROOT="${MIRAGE_OUTPUT_ROOT:-${HARDBLINK_ROOT}/answers_mirage}"
EVAL_ANSWERS_SCRIPT="${MIRAGE_EVAL_ANSWERS_SCRIPT:-${AURORA_ROOT}/methods/llava/eval_answers.py}"

QUESTION_FILES=(
    "${QUESTION_DIR}/blink_3pointscenter_questions_long.jsonl"
    "${QUESTION_DIR}/blink_4pointscenter_questions_long.jsonl"
    "${QUESTION_DIR}/blink_5pointscenter_questions_long.jsonl"
)

IMAGE_FOLDERS=(
    "${IMAGE_ROOT}/blink3pointscenter"
    "${IMAGE_ROOT}/blink4pointscenter"
    "${IMAGE_ROOT}/blink5pointscenter"
)

echo "================================================================================"
echo "Mirage Depth Reasoning Evaluation on HardBlink"
echo "================================================================================"
echo "Aurora root:          ${AURORA_ROOT}"
echo "Model:                ${MODEL_PATH}"
echo "Latent size:          ${LATENT_SIZE}"
echo "Ablation modes:       ${ABLATION_MODES[*]}"
echo "HardBlink root:       ${HARDBLINK_ROOT}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Eval answers script:  ${EVAL_ANSWERS_SCRIPT}"
echo "Force regenerate:     ${FORCE_REGENERATE}"
echo "================================================================================"
echo ""

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi
if [[ ! -f "$EVAL_ANSWERS_SCRIPT" ]]; then
    echo "ERROR: eval_answers.py not found: $EVAL_ANSWERS_SCRIPT"
    exit 1
fi

echo "Validating question files..."
for question_file in "${QUESTION_FILES[@]}"; do
    if [[ ! -f "$question_file" ]]; then
        echo "ERROR: Question file does not exist: $question_file"
        exit 1
    fi
    echo "  OK: $(basename "$question_file")"
done

echo "Validating image folders..."
for image_folder in "${IMAGE_FOLDERS[@]}"; do
    if [[ ! -d "$image_folder" ]]; then
        echo "ERROR: Image folder does not exist: $image_folder"
        exit 1
    fi
    echo "  OK: $(basename "$image_folder")"
done
echo ""

MODEL_NAME=$(basename "$MODEL_PATH")
mkdir -p "${SCRIPT_DIR}/logs"

for ABLATION_MODE in "${ABLATION_MODES[@]}"; do
    echo "================================================================================"
    echo "ABLATION MODE: ${ABLATION_MODE}"
    echo "================================================================================"

    ABLATION_FLAG=""
    SUFFIX=""
    case "$ABLATION_MODE" in
        baseline)
            ;;
        gt_latent)
            SUFFIX="_gt_latent"
            ABLATION_FLAG="--use-gt-latent"
            ;;
        gt_latent_count_matched)
            SUFFIX="_gt_latent_count_matched"
            ABLATION_FLAG="--use-gt-latent-count-matched"
            ;;
        random_latent)
            SUFFIX="_random_latent"
            ABLATION_FLAG="--use-random-latent"
            ;;
        zero_latent)
            SUFFIX="_zero_latent"
            ABLATION_FLAG="--use-zero-latent"
            ;;
        model_latent)
            SUFFIX="_model_latent"
            ABLATION_FLAG="--use-model-latent"
            ;;
        first_latent_repeat)
            SUFFIX="_first_latent_repeat"
            ABLATION_FLAG="--use-first-latent-repeat"
            ;;
        random_latent_gt_dist)
            SUFFIX="_random_latent_gt_dist"
            ABLATION_FLAG="--use-random-latent-gt-dist"
            ;;
        random_latent_model_dist)
            SUFFIX="_random_latent_model_dist"
            ABLATION_FLAG="--use-random-latent-model-dist"
            ;;
        *)
            echo "WARNING: Unknown ablation mode '${ABLATION_MODE}', skipping..."
            continue
            ;;
    esac

    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}${SUFFIX}"
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"

    ANSWER_FILENAMES=()
    GT_FILENAMES=()

    echo "=== GENERATING ANSWERS (${ABLATION_MODE}) ==="
    for index in "${!QUESTION_FILES[@]}"; do
        question_file="${QUESTION_FILES[$index]}"
        image_folder="${IMAGE_FOLDERS[$index]}"
        filename=$(basename "$question_file")
        answer_filename="${filename/questions/answers}"
        gt_filename="${filename/_long/}"
        gt_filename="${gt_filename/questions/answers}"
        answer_file="${OUTPUT_DIR}/${answer_filename}"

        ANSWER_FILENAMES+=("$answer_filename")
        GT_FILENAMES+=("$gt_filename")

        echo "  Question file: $question_file"
        echo "  Image folder:  $image_folder"
        echo "  Answer file:   $answer_file"

        if [[ -f "$answer_file" && "$FORCE_REGENERATE" != "true" ]]; then
            echo "  Skipping existing answer file (set MIRAGE_FORCE_REGENERATE=true to overwrite)"
            continue
        fi

        depth_image_folder_for_dataset=""
        if [[ -n "$DEPTH_IMAGE_FOLDER" ]]; then
            depth_image_folder_for_dataset="$DEPTH_IMAGE_FOLDER"
        elif [[ ${#DEPTH_IMAGE_FOLDERS[@]} -gt $index && -n "${DEPTH_IMAGE_FOLDERS[$index]}" ]]; then
            depth_image_folder_for_dataset="${DEPTH_IMAGE_FOLDERS[$index]}"
        else
            depth_image_folder_for_dataset="$image_folder"
        fi

        cmd=(
            env USE_MIRAGE_NEW=1 python3 "${SCRIPT_DIR}/src/test_depth.py"
            --load_model_path "$MODEL_PATH"
            --question_file "$question_file"
            --image_folder "$image_folder"
            --answers_file "$answer_file"
            --latent_size "$LATENT_SIZE"
            --max_new_tokens "$MAX_NEW_TOKENS"
            --cache_dir "$CACHE_DIR"
            --log_file "${SCRIPT_DIR}/logs/log_depth_eval_${ABLATION_MODE}.txt"
        )

        if [[ -n "$ABLATION_FLAG" ]]; then
            cmd+=("$ABLATION_FLAG")
        fi
        if [[ -n "$depth_image_folder_for_dataset" ]]; then
            cmd+=(--depth_image_folder "$depth_image_folder_for_dataset")
            echo "  Depth image folder: ${depth_image_folder_for_dataset}"
        fi
        if [[ "$VERBOSE" == "true" ]]; then
            cmd+=(--verbose)
        fi

        echo "  Running inference..."
        "${cmd[@]}"
    done

    echo ""
    echo "=== EVALUATING ANSWERS (${ABLATION_MODE}) ==="
    python3 "$EVAL_ANSWERS_SCRIPT" \
        --gt_dir "$GT_DIR" \
        --model_dir "$OUTPUT_DIR" \
        --gt_files "${GT_FILENAMES[@]}" \
        --files "${ANSWER_FILENAMES[@]}" \
        --summary_file "${OUTPUT_DIR}/evaluation_summary_long.json"

    echo "=== COMPLETED: ${ABLATION_MODE} ==="
    echo ""
done

echo "================================================================================"
echo "ALL MIRAGE DEPTH ABLATION EVALUATIONS COMPLETED"
echo "Results saved to: ${OUTPUT_ROOT}"
echo "================================================================================"
echo "Generating overall eval+ablation summary..."
export MIRAGE_HARDBLINK_ANSWERS_DIR="${OUTPUT_ROOT}"
python3 "${SCRIPT_DIR}/summarize_eval_ablation_results.py"
