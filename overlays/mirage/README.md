# Mirage Overlay

Pinned upstream:

- repo: `https://github.com/UMass-Embodied-AGI/Mirage.git`
- commit: `53f26de2682025e146781c5b198ec93bdfe4c4d6`

User-facing setup and run instructions live in `USER_GUIDE.md`.

This overlay carries Aurora’s Mirage eval ablations plus the depth-reasoning train/eval flow.

## What is added against upstream

- `overrides/test.sh`
  - Aurora ablation wrapper for the general Mirage eval path
  - repo-relative defaults plus env overrides instead of user-specific scratch paths
- `overrides/test_depth.sh`
  - dedicated HardBlink depth-ablation wrapper
  - separates the depth reasoning eval flow from the generic Mirage eval script
- `overrides/train.sh`
  - dedicated depth-reasoning training wrapper
  - stage1/stage2 selection and repo-relative cache/log defaults
- `overrides/src/test_depth.py`
  - HardBlink depth eval runner with Mirage-style latent ablations
- `overrides/src/task_new.py`
  - reasoning-image lookup logic used by the depth task
- `overrides/src/generate_gt_images.py`
  - helper that renders reasoning-path supervision images from `map_desc`
- `overrides/summarize_eval_ablation_results.py`
  - collects both Mirage JSON logs and HardBlink evaluation summaries

## Depth dataset processing

The depth-reasoning training path depends on paired RGB images and reasoning images.

1. `src/generate_gt_images.py` reads the training JSONL, reconstructs the path from `map_desc`, and writes helper images such as `map_reasoning_path.png` or `<id>_reasoning_path.png`.
2. The training JSONL keeps the RGB image path, prompt/answer text, and either an explicit `image_output` path or enough information for `src/task_new.py` to derive the reasoning-image filename.
3. `src/task_new.py` loads both the RGB input and the reasoning image when `--task depth-reasoning` is used.
4. `train.sh` runs stage1 or stage2 on that processed JSONL.

## Canonical entrypoints

- `./tools/eval_mirage.sh`
  - general Mirage ablation eval
- `./tools/eval_mirage_depth.sh`
  - HardBlink depth eval
- `./tools/train_mirage_depth.sh`
  - depth-reasoning training

## Supported ablation modes

The Mirage depth-eval override supports:

- `baseline`
- `gt_latent`
- `gt_latent_count_matched`
- `random_latent`
- `zero_latent`
- `model_latent`
- `first_latent_repeat`
- `random_latent_gt_dist`
- `random_latent_model_dist`

The GT-based modes require depth-image inputs through `MIRAGE_DEPTH_IMAGE_FOLDER` or the per-split `MIRAGE_DEPTH_IMAGE_FOLDER_BLINK*` overrides.

## Important environment overrides

- `MIRAGE_CACHE_DIR`
- `MIRAGE_MODEL_PATH`
- `MIRAGE_HARDBLINK_ROOT`
- `MIRAGE_IMAGE_ROOT`
- `MIRAGE_GT_DIR`
- `MIRAGE_OUTPUT_ROOT`
- `MIRAGE_DEPTH_TRAIN_JSONL`
- `MIRAGE_TRAIN_STAGE`
