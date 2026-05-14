# Mirage User Guide

This guide is for the Aurora overlay around the external Mirage repo.

## What Mirage Is Used For

Use Mirage when you want either:

- the Aurora Mirage eval ablation flow, or
- the Aurora depth-reasoning training and HardBlink depth-eval flow

This is the overlay that carries the extra Aurora depth-training workflow.

## Environment

Use the `mirage` conda env for this overlay. The full setup flow is documented in `../../docs/ENV_SETUP.md`.

## First-Time Setup

From the repo root:

```bash
cd /path/to/Ablate-to-Validate
./tools/bootstrap_overlay.py mirage
```

That materializes the upstream repo into:

```text
external/mirage/upstream
```

## Main Entry Points

General Mirage ablation eval:

```bash
./tools/eval_mirage.sh
```

HardBlink depth ablation eval:

```bash
./tools/eval_mirage_depth.sh
```

Depth training:

```bash
./tools/train_mirage_depth.sh
```

## Supported Ablations

The Mirage depth-eval path supports:

- `baseline`
- `gt_latent`
- `gt_latent_count_matched`
- `random_latent`
- `zero_latent`
- `model_latent`
- `first_latent_repeat`
- `random_latent_gt_dist`
- `random_latent_model_dist`

The GT-based modes require depth images through `MIRAGE_DEPTH_IMAGE_FOLDER` or the per-split `MIRAGE_DEPTH_IMAGE_FOLDER_BLINK3`, `MIRAGE_DEPTH_IMAGE_FOLDER_BLINK4`, and `MIRAGE_DEPTH_IMAGE_FOLDER_BLINK5` overrides.

## Depth Training Data Flow

Aurora’s depth training expects paired RGB images and reasoning images.

1. `src/generate_gt_images.py` reads the dataset JSONL and renders reasoning-path images from `map_desc`.
2. The processed JSONL stores the RGB image path, prompt/answer text, and the paired reasoning-image path or enough information to derive it.
3. `src/task_new.py` loads both the RGB input and reasoning image for `depth-reasoning`.
4. `train.sh` runs stage1 or stage2 using that processed JSONL.

## Common Environment Overrides

General:

- `MIRAGE_CACHE_DIR`
- `MIRAGE_MODEL_PATH`
- `MIRAGE_BASE_MODEL`

Depth eval:

- `MIRAGE_HARDBLINK_ROOT`
- `MIRAGE_QUESTION_DIR`
- `MIRAGE_IMAGE_ROOT`
- `MIRAGE_GT_DIR`
- `MIRAGE_OUTPUT_ROOT`
- `MIRAGE_DEPTH_IMAGE_FOLDER`

Depth train:

- `MIRAGE_DEPTH_TRAIN_JSONL`
- `MIRAGE_TRAIN_STAGE`
- `MIRAGE_STAGE1_OUTPUT`
- `MIRAGE_STAGE2_INPUT`
- `MIRAGE_STAGE2_OUTPUT`

## Typical Depth-Eval Example

```bash
export MIRAGE_MODEL_PATH=/path/to/depth_model_stage2
export MIRAGE_HARDBLINK_ROOT=/path/to/hardblink
./tools/eval_mirage_depth.sh
```

## Typical Depth-Train Example

Stage 1:

```bash
export MIRAGE_TRAIN_STAGE=stage1
export MIRAGE_DEPTH_TRAIN_JSONL=/path/to/train_depth_long.jsonl
./tools/train_mirage_depth.sh
```

Stage 2:

```bash
export MIRAGE_TRAIN_STAGE=stage2
export MIRAGE_DEPTH_TRAIN_JSONL=/path/to/train_depth_long.jsonl
export MIRAGE_STAGE2_INPUT=/path/to/stage1/checkpoint
./tools/train_mirage_depth.sh
```

## Results

General Mirage eval JSON summaries are written under:

```text
external/mirage/upstream/results
```

HardBlink depth eval outputs default to:

```text
external/mirage/upstream/data/hardblink/answers_mirage
```

The summary helper is:

```bash
python3 external/mirage/upstream/summarize_eval_ablation_results.py
```

## Key Overlay Files

- `README.md`: technical overview of Aurora-specific changes
- `overrides/test.sh`: general eval wrapper
- `overrides/test_depth.sh`: HardBlink depth eval wrapper
- `overrides/train.sh`: depth training wrapper
- `overrides/src/test_depth.py`: depth eval runner
- `overrides/src/generate_gt_images.py`: reasoning-image generation helper
