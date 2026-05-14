#!/bin/bash
# NEW: Aurora-only file; not present in upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Aurora path: qwen-vl-finetune/eval_qwen.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
QWEN_FINETUNE_DIR="${REPO_ROOT}/methods/qwen/qwen-vl-finetune"
LLAVA_DIR="${REPO_ROOT}/methods/llava"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a checkpoint directory or Hugging Face model id.}"
HARDBLINK_ROOT="${HARDBLINK_ROOT:?Set HARDBLINK_ROOT to a dataset root containing questions/, images/, and answers/.}"

QUESTION_FORMAT="${QUESTION_FORMAT:-long}"
DEPTH_MODES="${DEPTH_MODES:-original}"
DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
VERBOSE="${VERBOSE:-false}"
FORCE_REGENERATE="${FORCE_REGENERATE:-false}"

QUESTION_DIR="${QUESTION_DIR:-${HARDBLINK_ROOT}/questions}"
IMAGE_ROOT="${IMAGE_ROOT:-${HARDBLINK_ROOT}/images}"
GT_DIR="${GT_DIR:-${HARDBLINK_ROOT}/answers}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HARDBLINK_ROOT}/answers_qwen/output}"

IFS=',' read -r -a DEPTH_MODE_LIST <<< "${DEPTH_MODES}"

if [[ "${QUESTION_FORMAT}" == "long" ]]; then
    QUESTION_FILES=(
        "${QUESTION_DIR}/blink_3pointscenter_questions_long.jsonl"
        "${QUESTION_DIR}/blink_4pointscenter_questions_long.jsonl"
        "${QUESTION_DIR}/blink_5pointscenter_questions_long.jsonl"
    )
    ANSWER_FILENAMES=(
        "blink_3pointscenter_answers_long.jsonl"
        "blink_4pointscenter_answers_long.jsonl"
        "blink_5pointscenter_answers_long.jsonl"
    )
    GT_FILENAMES=(
        "blink_3pointscenter_answers.jsonl"
        "blink_4pointscenter_answers.jsonl"
        "blink_5pointscenter_answers.jsonl"
    )
    SUMMARY_NAME="evaluation_summary_long.json"
else
    QUESTION_FILES=(
        "${QUESTION_DIR}/blink_3pointscenter_questions.jsonl"
        "${QUESTION_DIR}/blink_4pointscenter_questions.jsonl"
        "${QUESTION_DIR}/blink_5pointscenter_questions.jsonl"
    )
    ANSWER_FILENAMES=(
        "blink_3pointscenter_answers_short.jsonl"
        "blink_4pointscenter_answers_short.jsonl"
        "blink_5pointscenter_answers_short.jsonl"
    )
    GT_FILENAMES=(
        "blink_3pointscenter_answers.jsonl"
        "blink_4pointscenter_answers.jsonl"
        "blink_5pointscenter_answers.jsonl"
    )
    SUMMARY_NAME="evaluation_summary.json"
fi

IMAGE_FOLDERS=(
    "${IMAGE_ROOT}/blink3pointscenter"
    "${IMAGE_ROOT}/blink4pointscenter"
    "${IMAGE_ROOT}/blink5pointscenter"
)

for required_path in "${QUESTION_FILES[@]}" "${IMAGE_FOLDERS[@]}" "${GT_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

model_name="$(basename "${MODEL_PATH}")"
base_output_dir="${OUTPUT_ROOT}/${model_name}"

mode_to_suffix() {
    case "$1" in
        original) echo "" ;;
        gt_depth) echo "_gt_depth" ;;
        random) echo "_random_depth" ;;
        zero) echo "_zero_depth" ;;
        model) echo "_model_depth" ;;
        first_repeat) echo "_first_depth_repeat" ;;
        random_depth_gt_dist) echo "_random_depth_gt_dist" ;;
        *)
            echo "Unsupported depth mode: $1" >&2
            return 1
            ;;
    esac
}

mode_to_flag() {
    case "$1" in
        original) echo "" ;;
        gt_depth) echo "--use-gt-depth" ;;
        random) echo "--use-random-depth" ;;
        zero) echo "--use-zero-depth" ;;
        model) echo "--use-model-depth" ;;
        first_repeat) echo "--use-first-depth-repeat" ;;
        random_depth_gt_dist) echo "--use-random-depth-gt-dist" ;;
        *)
            echo "Unsupported depth mode: $1" >&2
            return 1
            ;;
    esac
}

for depth_mode in "${DEPTH_MODE_LIST[@]}"; do
    suffix="$(mode_to_suffix "${depth_mode}")"
    ablation_flag="$(mode_to_flag "${depth_mode}")"
    output_dir="${base_output_dir}${suffix}"
    summary_path="${output_dir}/${SUMMARY_NAME}"

    mkdir -p "${output_dir}"

    for i in "${!QUESTION_FILES[@]}"; do
        question_file="${QUESTION_FILES[$i]}"
        image_folder="${IMAGE_FOLDERS[$i]}"
        answer_file="${output_dir}/${ANSWER_FILENAMES[$i]}"

        if [[ -f "${answer_file}" && "${FORCE_REGENERATE}" != "true" ]]; then
            expected_lines="$(wc -l < "${question_file}")"
            actual_lines="$(wc -l < "${answer_file}")"
            if [[ "${actual_lines}" -ge "${expected_lines}" ]]; then
                echo "[skip] ${answer_file} (${actual_lines}/${expected_lines} lines)"
                continue
            fi
        fi

        CMD=(
            python "${QWEN_FINETUNE_DIR}/model_vqa_qwen.py"
            --model-path "${MODEL_PATH}"
            --question-file "${question_file}"
            --image-folder "${image_folder}"
            --answers-file "${answer_file}"
            --max-new-tokens "${MAX_NEW_TOKENS}"
            --dtype "${DTYPE}"
        )
        if [[ -n "${ablation_flag}" ]]; then
            CMD+=("${ablation_flag}")
        fi
        if [[ "${VERBOSE}" == "true" ]]; then
            CMD+=(--verbose)
        fi

        "${CMD[@]}"
    done

    python "${LLAVA_DIR}/eval_answers.py" \
        --gt_dir "${GT_DIR}" \
        --model_dir "${output_dir}" \
        --gt_files "${GT_FILENAMES[@]}" \
        --files "${ANSWER_FILENAMES[@]}" \
        --summary_file "${summary_path}"
done
