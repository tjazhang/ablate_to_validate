# Aurora LLaVA User Guide

This guide is the curated Aurora entry point for `methods/llava`.

Use the shared env guide in [`../../repo_docs/ENV_SETUP.md`](../../repo_docs/ENV_SETUP.md). The recommended env for this method is `llava`.

## What Is Included

- standard LLaVA single-image and batch VQA inference
- Aurora continuous-depth evaluation
- Aurora discrete-depth evaluation
- standard LLaVA finetuning scripts
- Aurora two-stage depth training

## Important Paths

- method root: `methods/llava`
- main package: `methods/llava/llava`
- shared encoder registry: `methods/llava/data/encoder_config.json`
- upstream custom-data note: `methods/llava/docs/Finetune_Custom_Data.md`
- Aurora depth-training note: `methods/llava/docs/two_stage_depth_training.md`

## Working Directory

Run LLaVA commands from:

```bash
cd /path/to/Ablate-to-Validate/methods/llava
```

## Dataset Download

If the Hugging Face CLI is not installed in your `llava` env yet:

```bash
pip install "huggingface_hub[cli]"
```

Download the released training bundle into the repo-local LLaVA data tree:

```bash
hf download agianbig/mixed_depth \
  --repo-type dataset \
  --local-dir data/ADE20K/mixed_depth
```

That bundle gives you:

- training JSON: `data/ADE20K/mixed_depth/mixed_depth_long.json`
- training images: `data/ADE20K/mixed_depth/images`

Download the released HardBlink eval bundle with questions and images:

```bash
hf download agianbig/hardblink_eval \
  --repo-type dataset \
  --local-dir data/evals/hardblink
```

That eval bundle gives you:

- questions: `data/evals/hardblink/questions/blink_*pointscenter_questions_long.jsonl`
- images: `data/evals/hardblink/images/blink3pointscenter`, `blink4pointscenter`, `blink5pointscenter`

## Canonical Entry Points

### 1. Single-image inference

```bash
python -m llava.eval.run_llava \
  --model-path /path/to/checkpoint_or_hf_model \
  --image-file /path/to/image.png \
  --query "Describe the scene."
```

### 2. Batch eval for original checkpoints

```bash
python -m llava.eval.model_vqa \
  --model-path /path/to/checkpoint \
  --image-folder /path/to/images \
  --question-file /path/to/questions.jsonl \
  --answers-file /path/to/answers.jsonl \
  --conv-mode llava_v1
```

### 3. Batch eval for Aurora continuous-depth checkpoints

`model_vqa_depth_continuous.py` is the main Aurora eval path. It can run `original`, `continuous`, or `discrete` checkpoints, but it is most useful for continuous-depth models.

```bash
python model_vqa_depth_continuous.py \
  --model-path /path/to/checkpoint \
  --image-folder data/evals/hardblink/images/blink3pointscenter \
  --question-file data/evals/hardblink/questions/blink_3pointscenter_questions_long.jsonl \
  --answers-file /path/to/answers.jsonl \
  --conv-mode llava_v1 \
  --encoder-config data/encoder_config.json
```

Useful ablations:

- `--use-gt-depth-embeddings`
- `--use-random-depth`
- `--use-zero-depth`
- `--use-model-depth`
- `--use-first-depth-repeat`
- `--use-random-depth-gt-dist`
- `--use-gt-depth-permuted` (slot shuffle: the oracle span with its slots reordered by one fixed permutation, recorded in `_permutation.json` beside the answers)

The oracle-based arms (`--use-gt-depth-embeddings`, `--use-gt-depth-permuted`, `--use-random-depth-gt-dist`) encode a depth map per image with the checkpoint's depth encoder:

- `--gt-depth-map-dir DIR` encodes `DIR/<image base name>_depth.png`, the operand of the paper's oracle. Without it the driver encodes the RGB image and prints a warning; that operand is not the paper's.
- For a checkpoint whose name carries `_interploate_K`, pass `--gt-depth-target-num-patches K` (64 for the paper's SigLIP2 cell); the driver stops otherwise.

In the paper every depth map, training target and oracle alike, is a Depth Anything (ViT-S) estimate; reading the maps at the marked points reproduces 362 of the 372 HardBLINK answer labels.

### 4. Batch eval for Aurora discrete-depth checkpoints

```bash
python -m llava.eval.model_vqa_depth_discrete \
  --model-path /path/to/checkpoint \
  --image-folder data/evals/hardblink/images/blink3pointscenter \
  --question-file data/evals/hardblink/questions/blink_3pointscenter_questions_long.jsonl \
  --answers-file /path/to/answers.jsonl \
  --conv-mode llava_v1
```

For GT discrete-token injection:

```bash
python -m llava.eval.model_vqa_depth_discrete \
  --model-path /path/to/checkpoint \
  --image-folder data/evals/hardblink/images/blink3pointscenter \
  --question-file data/evals/hardblink/questions/blink_3pointscenter_questions_long.jsonl \
  --answers-file /path/to/answers_gt.jsonl \
  --use-gt-depth-embeddings \
  --gt-depth-codebook /path/to/codebook.npy
```

Add `--use-gt-depth-permuted-discrete` to that command for the slot shuffle: the same GT codes, with the slots of the 10x10 code grid reordered by one fixed permutation.

If the checkpoint's `config.json` has no depth fields, the depth token ids load as 0 and no forced span can engage. The driver then stops for `--use-gt-depth-embeddings`, `--use-gt-depth-permuted-discrete`, `--use-random-depth` and `--use-zero-depth`; add `--resync-discrete-depth-ids` to rebuild the ids from the tokenizer.

## Training

### Standard LLaVA finetuning

Start from the upstream-style scripts:

- `scripts/v1_5/finetune_task.sh`
- `scripts/v1_5/finetune_task_lora.sh`

Before running them, update the dataset and output paths inside the script or copy the command and replace:

- `--data_path`
- `--image_folder`
- `--output_dir`

The dataset format is documented in `docs/Finetune_Custom_Data.md`.

If you use the released Aurora training bundle above, the matching pair is:

- `--data_path data/ADE20K/mixed_depth/mixed_depth_long.json`
- `--image_folder data/ADE20K/mixed_depth/images`

### Aurora two-stage depth training

Aurora depth training is driven by `llava/train/train.py`.

Stage 1 example:

```bash
python -m llava.train.train \
  --model_name_or_path /path/to/base_model \
  --data_path data/ADE20K/mixed_depth/mixed_depth_long.json \
  --image_folder data/ADE20K/mixed_depth/images \
  --config_path data/encoder_config.json \
  --depth_data True \
  --depth_training_stage 1 \
  --depth_mode continuous \
  --output_dir /path/to/stage1_out
```

Stage 2 example:

```bash
python -m llava.train.train \
  --model_name_or_path /path/to/stage1_out \
  --data_path data/ADE20K/mixed_depth/mixed_depth_long.json \
  --image_folder data/ADE20K/mixed_depth/images \
  --config_path data/encoder_config.json \
  --depth_data True \
  --depth_training_stage 2 \
  --depth_mode continuous \
  --output_dir /path/to/stage2_out
```

For the full recipe and tuning knobs, see `docs/two_stage_depth_training.md`.

### Settings of the paper's models

The paper reports these settings for its LLaVA-13B depth models, each trained for 10 epochs:

| Setting | Continuous (SigLIP2, K=64) | Discrete (K=100) |
| --- | --- | --- |
| Trained parameters | LoRA on the LLM (r=128, α=256); vision tower frozen | LoRA on the LLM; vision tower frozen |
| Learning rates | base/LoRA 2e-4, multimodal projector 2e-5 | depth projector/head 1e-5 |
| Depth loss | cosine, normalization enabled, λ_depth = 1.0 | λ_depth = 1.0 |
| Batch | 16 per device on 8 GPUs (effective 128) | effective 128 |
| Other | max length 2048, AdamW, cosine schedule, BF16 + TF32, weight decay 0 | max length 2048, AdamW, cosine schedule, BF16 + TF32 |

Several `llava.train.train` defaults differ (for example `--depth_coef 0.1`, `--lora_r 64`, `--lora_alpha 16`), so pass these values explicitly.

## Reproducing the Paper's TRT Numbers

No trained checkpoints are released, so every arm needs a model trained with this repository. Some arms also need inputs that are not in the public datasets:

| Paper arm (HardBLINK) | Driver flags | Extra input | Status |
| --- | --- | --- | --- |
| Continuous identity, random, zero, first-repeat | none, `--use-random-depth`, `--use-zero-depth`, `--use-first-depth-repeat` | none | runnable |
| Continuous oracle, slot shuffle | `--use-gt-depth-embeddings` or `--use-gt-depth-permuted`, with `--gt-depth-map-dir` | HardBLINK depth maps | not reproducible from public artifacts yet |
| Discrete identity, random, zero | none, `--use-random-depth`, `--use-zero-depth` | none | runnable |
| Discrete oracle, slot shuffle | `--use-gt-depth-embeddings` (plus `--use-gt-depth-permuted-discrete`), with `--gt-depth-codebook` | HardBLINK VQ-VAE code file | not reproducible from public artifacts yet |
| Matched-budget control (appendix) | not in this release | VQ-VAE weights | not reproducible from public artifacts yet |

## Notes

- `data/encoder_config.json` is shared with Qwen continuous-depth workflows.
- If you only need upstream LLaVA behavior, the original upstream `README.md` is still present.
- For Aurora-specific behavior, this file and `docs/two_stage_depth_training.md` are the intended entry points.
