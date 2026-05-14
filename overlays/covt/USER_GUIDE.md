# CoVT User Guide

This guide is for the Aurora overlay around the external CoVT repo.

## What CoVT Is Used For

Use CoVT when you want to run Aurora’s eval-only CoVT ablation flow through VLMEvalKit.

This overlay does not add a training workflow. It adds eval wrappers, ablation model registrations, and summary tooling.

## Environment

Use the `covt` conda env for this overlay. The full setup flow is documented in `../../docs/ENV_SETUP.md`.

## First-Time Setup

From the repo root:

```bash
cd /path/to/Ablate-to-Validate
./tools/bootstrap_overlay.py covt
```

That materializes the upstream repo into:

```text
external/covt/upstream
```

## Main Entry Point

Run:

```bash
./tools/eval_covt.sh
```

That launches the materialized overlay wrapper:

```text
external/covt/upstream/eval.sh
```

## Choosing Models and Ablations

Edit the arrays inside `external/covt/upstream/eval.sh` or the overlay source copy in:

```text
overlays/covt/overrides/eval.sh
```

Base models include examples such as:

- `CoVT-7B-depth`
- `CoVT-7B-seg`
- `CoVT-7B-seg_depth_dino`
- `CoVT-7B-seg_depth_dino_edge`
- `CoVT-LLaVA-13B-depth`

Ablation modes include:

- `none`
- `zero`
- `random`
- `same`
- `random-dist`
- `first-repeat`

## Useful Environment Overrides

- `LMUData`
- `VLMEVAL_DIR`
- `COVT_EVAL_MODE`
- `COVT_WORK_DIR`
- `COVT_VERBOSE`
- `PYTHON_BIN`

Example:

```bash
export LMUData=/path/to/LMUData
export COVT_WORK_DIR=outputs_depth
./tools/eval_covt.sh
```

## Results

VLMEvalKit outputs are written under:

```text
external/covt/upstream/VLMEvalKit/outputs
```

The overlay also includes a summary helper:

```bash
python3 external/covt/upstream/summarize_eval_ablation_results.py
```

## Key Overlay Files

- `README.md`: technical overview of Aurora-specific changes
- `overrides/eval.sh`: top-level CoVT eval wrapper
- `overrides/VLMEvalKit/vlmeval/config.py`: ablation model registration
- `overrides/VLMEvalKit/vlmeval/vlm/covt_qwen/model.py`: Qwen ablation hook
- `overrides/VLMEvalKit/vlmeval/vlm/covt_llava/model.py`: LLaVA ablation hook
