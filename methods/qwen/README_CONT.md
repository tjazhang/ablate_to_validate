<!--
NEW: Aurora-only file; not present in upstream QWEN.
NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
NEW: Aurora path: README_CONT.md
-->

# Qwen2.5-VL Continuous Depth Reasoning

This repository contains modifications to Qwen2.5-VL to support hybrid discrete/continuous depth reasoning.

For the release-oriented upstream comparison and cleaned train/eval entrypoints, see `AURORA_RELEASE.md`.

## Architecture Changes

We modify the Qwen2.5-VL architecture to support:

1.  **Discrete Depth Tokens (Aurora-style):** Standard vocabulary expansion with `<DEPTH_0>` to `<DEPTH_127>`.
2.  **Continuous Depth Tokens (MetaMorph-style):** A new mechanism where the model outputs continuous vectors for `K` steps between `<DEPTH_START>` and `<DEPTH_END>`.

### Key Files

*   `qwen-vl-finetune/qwenvl/configuration_qwen2_5_vl.py`: Configuration class with added parameters (`depth_input_dim`, `continuous_K`, etc.).
*   `qwen-vl-finetune/qwenvl/modeling_qwen2_5_vl.py`: Model implementation with:
    *   `depth_head`: Predicts continuous depth vectors.
    *   `depth_projector`: Injects depth vectors into the model input.
    *   Modified `forward` pass for input injection and depth loss computation.
*   `qwen-vl-finetune/qwenvl/train/train_qwen.py`: Training script updated to use the local model definition and handle special tokens.
*   `qwen-vl-finetune/qwenvl/data/data_qwen.py`: Data pipeline updated to preprocess prompts with placeholders and handle `depth_tensors`.

## Usage

### Training

To train the model, ensure your data contains `depth_tensors` and use the provided training scripts:

*   **Discrete Depth Training**: `qwen-vl-finetune/scripts/train_ade_discrete.sh`
*   **Continuous Depth Training**: `qwen-vl-finetune/scripts/train_ade_continuous.sh`

The data loader will automatically handle the insertion of placeholders and alignment of labels.

### Configuration

You can control the continuous depth parameters via the model configuration:

*   `depth_input_dim`: Dimension of the continuous depth vectors (default: 768).
*   `continuous_K`: Number of continuous depth steps (default: 64).
*   `lambda_depth`: Weight for the depth loss (default: 1.0).

## Note

Changes in `qwenvl/configuration_qwen2_5_vl.py` and `qwenvl/modeling_qwen2_5_vl.py` are wrapped in `<NEW>` tags for clarity.
