# CoVT overrides

These files overwrite exact upstream CoVT / VLMEvalKit paths when
`tools/bootstrap_overlay.py covt` is run.

Key Aurora-specific override files:

- `eval.sh`
- `summarize_eval_ablation_results.py`
- `VLMEvalKit/vlmeval/config.py`
- `VLMEvalKit/vlmeval/vlm/covt_qwen/model.py`
- `VLMEvalKit/vlmeval/vlm/covt_llava/model.py`

See [`../README.md`](../README.md) for the higher-level description of the eval-only overlay.
