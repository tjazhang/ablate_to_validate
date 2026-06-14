# Mull overrides

These files overwrite (or add) files at exact upstream paths when `tools/bootstrap_overlay.py mull` is run.

Key Aurora-specific override files:

- `src/eval_bench_ablation.py` — **NOT faithful (no-op);** vLLM ignores the ablation flags. Use the no-vLLM path for real ablations.
- `src/eval_bench_ablation_novllm.py` — faithful ablation path
- `src/eval_bench_ablation.sh`
- `src/eval_bench_ablation_novllm.sh`
- `src/aurora_eval_config.py`
- `src/summarize_eval_ablation_results.py`
- `models/mmlatent_qwen_vl_sample_imonly.py` — custom Mull model class for the no-vLLM path (added; absent upstream)
- `dataloaders/custom_datasets.py` — `EvalDataset` for the no-vLLM path (added; absent upstream)

See [`../README.md`](../README.md) for the higher-level rationale and required dataset env vars.
