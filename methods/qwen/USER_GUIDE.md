<!--
NEW: Aurora-only file; not present in upstream QWEN.
NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
NEW: Aurora path: USER_GUIDE.md
-->

# Aurora Qwen User Guide

This guide covers the retained public Qwen surface in `methods/qwen`.

## What Is Included

- baseline Qwen2.5-VL finetuning
- discrete depth-token finetuning
- continuous depth-embedding finetuning
- continuous embed-AR finetuning
- HardBlink-style evaluation with optional depth ablations

## Important Paths

- Qwen method root: `methods/qwen`
- finetune code: `methods/qwen/qwen-vl-finetune`
- shared encoder registry: `methods/llava/data/encoder_config.json`
- release notes: `methods/qwen/AURORA_RELEASE.md`
- embed-AR notes: `methods/qwen/qwen-vl-finetune/EMBED_AR_CHANGES.md`

## Environment Setup

Use the shared env guide in `../../docs/ENV_SETUP.md`. The recommended env for this method is `qwen_vl`.

The checked-in env YAML is now a populated snapshot of the original `qwen_vl` env. The recommended path is:

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/qwen.environment.yml
conda activate qwen_vl
pip install -e methods/qwen/qwen-vl-utils
```

If you prefer a lighter manual install instead of the exported env snapshot, the minimum checked-in dependency list is still in `methods/qwen/requirements.txt`:

```bash
pip install -r methods/qwen/requirements.txt
pip install -e methods/qwen/qwen-vl-utils
```

If you want the Gradio demos, also install:

```bash
pip install -r methods/qwen/requirements_web_demo.txt
```

If you want to use `hf download` or `hf upload-large-folder` in this env, also install:

```bash
pip install "huggingface_hub[cli]"
```

## Data Expectations

### Download From Hugging Face

The released Aurora training and eval datasets are expected under the shared LLaVA data tree.

Download the mixed-depth training bundle:

```bash
hf download agianbig/mixed_depth \
  --repo-type dataset \
  --local-dir methods/llava/data/ADE20K/mixed_depth
```

Download the HardBlink eval bundle:

```bash
hf download agianbig/hardblink_eval \
  --repo-type dataset \
  --local-dir methods/llava/data/evals/hardblink
```

After that, the most common paths are:

- continuous-depth training JSON: `methods/llava/data/ADE20K/mixed_depth/mixed_depth_long.json`
- training images: `methods/llava/data/ADE20K/mixed_depth/images`
- eval root: `methods/llava/data/evals/hardblink`

### Training JSON

All cleaned training scripts expect `TRAIN_JSON` to point to a JSON or JSONL dataset file.

- baseline mode: normal image-text supervision, no depth embeddings required
- discrete mode: answers include regular depth tokens such as `<DEPTH_0> ... <DEPTH_127>`
- continuous mode: samples must provide `depth_tensors` or embedding paths compatible with the Aurora data loader

The loader also supports direct JSON paths passed through `--dataset_use`, so you do not need to register a dataset alias in code for normal use.

### Encoder Registry

Continuous training and GT-depth evaluation use the shared encoder registry:

- `methods/llava/data/encoder_config.json`

This file maps encoder names to feature dimension and patch/grid metadata.  
The cleaned scripts auto-detect it by default.

### HardBlink Evaluation Root

The eval script expects a dataset root like:

```text
HARDBLINK_ROOT/
  questions/
    blink_3pointscenter_questions_long.jsonl
    blink_4pointscenter_questions_long.jsonl
    blink_5pointscenter_questions_long.jsonl
  images/
    blink3pointscenter/
    blink4pointscenter/
    blink5pointscenter/
  answers/
    blink_3pointscenter_answers.jsonl
    blink_4pointscenter_answers.jsonl
    blink_5pointscenter_answers.jsonl
```

The released `agianbig/hardblink_eval` dataset matches this layout when downloaded to `methods/llava/data/evals/hardblink`.

## Training

Run all commands from the repo root.

### 1. Baseline

```bash
TRAIN_JSON=/path/to/baseline_train.json \
bash methods/qwen/qwen-vl-finetune/scripts/train_ade_baseline.sh
```

Useful overrides:

- `MODEL_NAME_OR_PATH`
- `NUM_EPOCHS`
- `PER_DEVICE_BATCH_SIZE`
- `GRAD_ACCUM_STEPS`
- `OUTPUT_ROOT`
- `REPORT_TO=wandb`

### 2. Discrete Depth

```bash
TRAIN_JSON=/path/to/discrete_depth_train.json \
bash methods/qwen/qwen-vl-finetune/scripts/train_ade_discrete.sh
```

This uses `methods/qwen/New_tokens.txt` by default.

### 3. Continuous Depth

```bash
TRAIN_JSON=/path/to/continuous_depth_train.json \
DEPTH_ENCODER_NAMES=google/siglip2-large-patch16-256,openai/clip-vit-large-patch14-336,facebook/dinov2-base \
bash methods/qwen/qwen-vl-finetune/scripts/train_ade_continuous.sh
```

Important continuous-mode overrides:

- `DEPTH_ENCODER_NAMES`
- `DEPTH_HEAD_TYPE=linear|mlp|mlp2x_gelu`
- `DEPTH_PROJECTOR_TYPE=linear|mlp|mlp2x_gelu`
- `DEPTH_LOSS_TYPE=mse|cosine|softmax`
- `LAMBDA_DEPTH`

### 4. Continuous Embed-AR

```bash
TRAIN_JSON=/path/to/continuous_depth_train.json \
DEPTH_ENCODER_NAME=google/siglip2-large-patch16-256 \
DEPTH_EMBED_AR_GT_RATIO=0.0 \
bash methods/qwen/qwen-vl-finetune/scripts/train_ade_continuous_embed_ar.sh
```

`DEPTH_EMBED_AR_GT_RATIO` controls scheduled sampling:

- `1.0`: full teacher forcing
- `0.0`: pure embed-AR

## Evaluation

Evaluate a checkpoint or model id with:

```bash
MODEL_PATH=/path/to/checkpoint_or_model \
HARDBLINK_ROOT=methods/llava/data/evals/hardblink \
bash methods/qwen/qwen-vl-finetune/eval_qwen.sh
```

Useful overrides:

- `QUESTION_FORMAT=long|short`
- `DEPTH_MODES=original`
- `DEPTH_MODES=original,random,zero,gt_depth,model,first_repeat,random_depth_gt_dist`
- `OUTPUT_ROOT=/path/to/output_dir`
- `DTYPE=auto|bfloat16|float16`
- `MAX_NEW_TOKENS=2048`
- `FORCE_REGENERATE=true`
- `VERBOSE=true`

For discrete GT ablation, also pass a codebook path through the underlying model script if needed:

- `--gt-depth-codebook /path/to/codebook.npy`

The evaluation script writes predictions under:

- `HARDBLINK_ROOT/answers_qwen/output/<model_name>...`

## Optional Web Demo

Local multimodal Gradio demo:

```bash
python methods/qwen/web_demo_mm.py \
  -c /path/to/checkpoint_or_model \
  --server-name 0.0.0.0 \
  --server-port 7860
```

Streaming demo:

```bash
python methods/qwen/web_demo_streaming/app.py \
  -c /path/to/checkpoint_or_model \
  --server-name 0.0.0.0 \
  --server-port 7860
```

## Outputs

Training outputs default to:

- `methods/qwen/qwen-vl-finetune/output/`

Evaluation outputs default to:

- `HARDBLINK_ROOT/answers_qwen/output/`

## Troubleshooting

- If continuous training fails on encoder metadata, check `methods/llava/data/encoder_config.json`.
- If embed-AR training is enabled, expect gradient checkpointing behavior to differ; see `EMBED_AR_CHANGES.md`.
- If depth tokens are missing, verify the selected `New_tokens*.txt` file matches the training mode.
- If evaluation cannot run GT-depth ablations, confirm the encoder registry and any discrete GT codebook path are available.
