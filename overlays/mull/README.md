# Mull Overlay

Pinned upstream:

- repo: `https://github.com/arijitray1993/Video-R1.git`
- commit: `97b7bc27f32842c0ef62eaa83f78e3602fc63c25`

User-facing setup and run instructions live in `USER_GUIDE.md`.

Aurora-specific changes in this overlay are limited to evaluation, not training.

## What is added against upstream

- `overrides/src/eval_bench_ablation.py`
  - vLLM-based latent ablation eval entrypoint
  - adds Aurora comments around the `text_config=None` vLLM config workaround
- `overrides/src/eval_bench_ablation_novllm.py`
  - canonical no-vLLM eval path used by `tools/eval_mull.sh`
  - loads the custom Mull model class for ablation support
- `overrides/src/aurora_eval_config.py`
  - Aurora-only helper that replaces hard-coded machine-local dataset mounts with CLI/env configuration
- `overrides/src/summarize_eval_ablation_results.py`
  - Aurora-only summary script added because the ablation shell wrappers call it
- `overrides/src/eval_bench_ablation.sh`
- `overrides/src/eval_bench_ablation_novllm.sh`
  - robust shell wrappers that `cd` to the repo root before launching the ablation scripts

## Canonical entrypoints

- `./tools/eval_mull.sh`
  - boots into `external/mull/upstream/src/eval_bench_ablation_novllm.sh`
- `external/mull/upstream/src/eval_bench_ablation.sh`
  - optional vLLM path if you specifically want the vLLM-backed ablation run

## Supported ablation modes

The Mull override code supports these latent ablations:

- `baseline`
- `zero_latent`
- `random_latent`
- `same_latent`
- `first_latent_repeat`
- `random_latent_same_dist`
- `random_latent_model_dist`
- `random_latent_gt_dist`
- `gt_latent`

The last two require GT auxiliary images via `--gt-image-dir`.

## Dataset configuration

- `blink` and `sat` default to their Hugging Face dataset ids.
- `vsibench` requires:
  - `MULL_VSIBENCH_LOCATION` or `--vsibench-location`
  - `MULL_VSIBENCH_ROOT` or `--vsibench-root`
- `mmvu` requires:
  - `MULL_MMVU_LOCATION` or `--mmvu-location`
  - `MULL_MMVU_ROOT` or `--mmvu-root`

This overlay intentionally does not carry depth-training code. The extra Aurora depth-training flow lives in the Mirage overlay, where the repo already contains the paired reasoning-image training pipeline.
