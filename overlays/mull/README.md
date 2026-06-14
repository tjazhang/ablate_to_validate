# Mull Overlay

Pinned upstream:

- repo: `https://github.com/arijitray1993/Video-R1.git`
- commit: `97b7bc27f32842c0ef62eaa83f78e3602fc63c25`

User-facing setup and run instructions live in `USER_GUIDE.md`.

Aurora-specific changes in this overlay are limited to evaluation, not training.

## What is added against upstream

- `overrides/src/eval_bench_ablation.py`
  - vLLM-based latent ablation eval entrypoint
  - **WARNING: known no-op ablation.** vLLM ignores the custom `use_zero_latent` /
    replacement flags and runs stock Qwen2.5-VL, so ablated outputs are identical to
    baseline (Δ≈0). Do not use this path to measure latent-token utilization — use the
    no-vLLM path below.
  - adds Aurora comments around the `text_config=None` vLLM config workaround
- `overrides/src/eval_bench_ablation_novllm.py`
  - canonical no-vLLM eval path used by `tools/eval_mull.sh`
  - loads the custom Mull model class for ablation support
- `overrides/models/mmlatent_qwen_vl_sample_imonly.py`
  - the custom Mull model class imported by the no-vLLM path
    (`from mmlatent_qwen_vl_sample_imonly import ...`). Not present in upstream, so it is
    vendored here and bootstrapped into `external/mull/upstream/models/`.
- `overrides/dataloaders/custom_datasets.py`
  - provides `EvalDataset`, loaded by the no-vLLM path via
    `sys.path.append("dataloaders/")`. Not present in upstream, so it is vendored here and
    bootstrapped into `external/mull/upstream/dataloaders/`.
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
  - vLLM path — **NOT a faithful ablation.** vLLM ignores the latent-replacement flags,
    so ablated outputs == baseline (Δ≈0). Kept only for plain (non-ablation) vLLM
    inference; for the faithful ablation always use `./tools/eval_mull.sh`.

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
