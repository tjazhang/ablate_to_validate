# Mirage overrides

These files overwrite exact upstream Mirage paths when `tools/bootstrap_overlay.py mirage` is run.

Key Aurora-specific override files:

- `test.sh`
- `test_depth.sh`
- `train.sh`
- `summarize_eval_ablation_results.py`
- `src/test_depth.py`
- `src/task_new.py`
- `src/generate_gt_images.py`

See [`../README.md`](../README.md) for the depth dataset-processing notes and wrapper entrypoints.
