# Mull User Guide

This guide is for the Aurora overlay around the external Mull / Video-R1 repo.

## What Mull Is Used For

Use Mull when you want to run the Aurora latent-token evaluation and ablation flow on top of the pinned Video-R1 upstream checkout.

This overlay is eval-focused. It does not carry the extra Aurora depth-training workflow.

## Environment

Use the `mull` conda env for this overlay. The full setup flow is documented in `../../docs/ENV_SETUP.md`.

## First-Time Setup

From the repo root:

```bash
cd /path/to/Ablate-to-Validate
./tools/bootstrap_overlay.py mull
```

That materializes the upstream repo into:

```text
external/mull/upstream
```

## Main Entry Point

Run the standard Aurora Mull eval wrapper from the repo root:

```bash
./tools/eval_mull.sh
```

That launches:

```text
external/mull/upstream/src/eval_bench_ablation_novllm.sh
```

The default path is the no-vLLM ablation runner because it is the most self-contained overlay path.

## Supported Ablations

The no-vLLM Mull override supports:

- `baseline`
- `zero_latent`
- `random_latent`
- `same_latent`
- `first_latent_repeat`
- `random_latent_same_dist`
- `random_latent_model_dist`
- `random_latent_gt_dist`
- `gt_latent`

`random_latent_gt_dist` and `gt_latent` require GT auxiliary images through the wrapper's `GT_IMAGE_DIR` or the underlying `--gt-image-dir` flag.

## Dataset Configuration

Some datasets are referenced by Hugging Face id and some require local paths.

Defaults:

- `blink`: `BLINK-Benchmark/BLINK`
- `sat`: `array/SAT-v2`

Required local-path env vars for other datasets:

- `MULL_VSIBENCH_LOCATION`
- `MULL_VSIBENCH_ROOT`
- `MULL_MMVU_LOCATION`
- `MULL_MMVU_ROOT`

Optional overrides:

- `MULL_BLINK_LOCATION`
- `MULL_SAT_LOCATION`
- `MULL_SAT_SPLIT`

Example:

```bash
export MULL_VSIBENCH_LOCATION=/path/to/eval_vsibench.json
export MULL_VSIBENCH_ROOT=/path/to/VSI-Bench
export MULL_MMVU_LOCATION=/path/to/eval_mmvu.json
export MULL_MMVU_ROOT=/path/to/MMVU
./tools/eval_mull.sh
```

## Results

Generated eval JSON files are written under the materialized upstream repo:

```text
external/mull/upstream/src/r1-v/eval_results
```

The overlay also includes a summary helper:

```bash
python3 external/mull/upstream/src/summarize_eval_ablation_results.py
```

## Key Overlay Files

- `README.md`: technical overview of Aurora-specific changes
- `overrides/src/eval_bench_ablation.py`: vLLM ablation path
- `overrides/src/eval_bench_ablation_novllm.py`: default no-vLLM ablation path
- `overrides/src/aurora_eval_config.py`: dataset-path resolution
- `overrides/src/summarize_eval_ablation_results.py`: result summary
