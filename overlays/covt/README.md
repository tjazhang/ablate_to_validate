# CoVT Overlay

Pinned upstream:

- repo: `https://github.com/Wakals/CoVT.git`
- commit: `3a3253ee462ad170cefeb10814530bf15329dd46`

User-facing setup and run instructions live in `USER_GUIDE.md`.

This overlay keeps CoVT external while carrying Aurora’s eval-only additions.

## What is added against upstream

- `overrides/eval.sh`
  - canonical CoVT eval launcher
  - repo-relative defaults for `VLMEvalKit` and `LMUData`
- `overrides/summarize_eval_ablation_results.py`
  - summarizes `*_acc.csv` outputs emitted by VLMEvalKit
- `overrides/VLMEvalKit/vlmeval/config.py`
  - registers Aurora ablation variants such as `*-zero`, `*-random`, `*-same`, `*-random-dist`, and `*-first-repeat`
- `overrides/VLMEvalKit/vlmeval/vlm/covt_qwen/model.py`
  - installs the Qwen CoVT anchor-token ablation hook
- `overrides/VLMEvalKit/vlmeval/vlm/covt_llava/model.py`
  - installs the matching LLaVA CoVT ablation hook

## Canonical entrypoint

- `./tools/eval_covt.sh`
  - launches `external/covt/upstream/eval.sh`

## Important environment overrides

- `LMUData`
- `VLMEVAL_DIR`
- `COVT_EVAL_MODE`
- `COVT_WORK_DIR`
