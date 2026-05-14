<!--
NEW: Aurora-only file; not present in upstream QWEN.
NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
NEW: Aurora path: AURORA_RELEASE.md
-->

# Aurora Qwen Release Notes

Baseline repo: `https://github.com/QwenLM/Qwen2.5-VL.git`  
Baseline commit: `96588727e44c78b25ba03ea03b8e12f7e64fd0da`

This release snapshot is a vendored Qwen fork under `methods/qwen`; it is not a separate git branch inside this repo.

## What Changed Relative To Upstream Qwen

### 1. Training entrypoint is extended for depth-aware finetuning

Upstream `qwen-vl-finetune/qwenvl/train/train_qwen.py` only wires the stock finetune flow.  
Aurora extends it to:

- auto-resolve the shared encoder registry in `methods/llava/data/encoder_config.json`
- derive `continuous_K` and `depth_input_dim` from the selected encoder
- load a local Qwen model/config fork instead of relying only on the stock `transformers` implementation
- register depth token ids after tokenizer resize
- initialize continuous-depth `depth_head` / `depth_projector` variants
- support discrete depth tokens, continuous depth loss, and embed-AR training

Direct upstream diff against the baseline finetune file:

- `qwenvl/train/train_qwen.py`: `434` insertions, `70` deletions
- `qwenvl/train/argument.py`: `49` insertions, `11` deletions

### 2. Aurora vendors the Qwen model internals locally

Upstream finetune code expects Qwen model internals to come from `transformers`.  
Aurora adds local copies so the depth bottleneck can live inside the model implementation:

- `qwen-vl-finetune/qwenvl/configuration_qwen2_5_vl.py`
- `qwen-vl-finetune/qwenvl/modeling_qwen2_5_vl.py`

These files add:

- `depth_input_dim`, `continuous_K`, `lambda_depth`
- depth boundary / placeholder token ids
- `depth_head_type` and `depth_projector_type`
- depth normalization and alternate depth losses
- embed-AR flags and the custom `greedy_decode_depth_ar()` path
- continuous and discrete ablation hooks for evaluation

### 3. The data pipeline is replaced to support depth placeholders and depth tensors

Aurora adds `qwen-vl-finetune/qwenvl/data/data_qwen.py` and uses it instead of upstream `data_processor.py`.

Behavioral differences:

- expands a single textual `<DEPTH_TOKEN>` placeholder into `K` positions for continuous training
- substitutes encoder-specific embedding file stubs in dataset JSON
- carries `depth_tensors` alongside the normal multimodal prompt processing
- keeps the dataset interface compatible with direct JSON/JSONL paths through `--dataset_use`

Aurora also keeps a legacy dataset alias registry in `qwen-vl-finetune/qwenvl/data/__init__.py`, but the private absolute paths were replaced with environment-variable placeholders for release.

### 4. Aurora adds a dedicated Qwen VQA / ablation entrypoint

`qwen-vl-finetune/model_vqa_qwen.py` is Aurora-only.

It adds:

- evaluation on the HardBlink-style JSONL format
- continuous-depth ablations: random, zero, GT, model, first-repeat, random-matched-to-GT-distribution
- discrete-depth GT token forcing
- auto-detection of the vendored `methods/llava` GT-depth provider
- release-safe resolution of `encoder_config.json` instead of a user-specific absolute path

### 5. Release-facing train/eval scripts were rewritten as reusable examples

The following scripts now derive repo paths automatically and require only external dataset/model inputs:

- `qwen-vl-finetune/scripts/train_ade_baseline.sh`
- `qwen-vl-finetune/scripts/train_ade_discrete.sh`
- `qwen-vl-finetune/scripts/train_ade_continuous.sh`
- `qwen-vl-finetune/scripts/train_ade_continuous_embed_ar.sh`
- `qwen-vl-finetune/eval_qwen.sh`

They no longer contain usernames, scratch paths, or assumptions about datasets being bundled in the repo.

## Inline Comment Coverage

Release-oriented inline comments were added or tightened in the main divergence points:

- `qwen-vl-finetune/qwenvl/train/argument.py`
- `qwen-vl-finetune/qwenvl/train/train_qwen.py`
- `qwen-vl-finetune/qwenvl/data/__init__.py`
- `qwen-vl-finetune/model_vqa_qwen.py`

The large model fork already had extensive inline comments in the Aurora-only depth sections of `qwen-vl-finetune/qwenvl/modeling_qwen2_5_vl.py`, so those were kept as the primary inline documentation there.

## Release Usage

The public launch surface is:

- baseline text/VQA finetuning: `qwen-vl-finetune/scripts/train_ade_baseline.sh`
- discrete depth finetuning: `qwen-vl-finetune/scripts/train_ade_discrete.sh`
- continuous depth finetuning: `qwen-vl-finetune/scripts/train_ade_continuous.sh`
- continuous embed-AR finetuning: `qwen-vl-finetune/scripts/train_ade_continuous_embed_ar.sh`
- evaluation: `qwen-vl-finetune/eval_qwen.sh`

Each script expects external dataset paths through environment variables such as `TRAIN_JSON`, `MODEL_PATH`, and `HARDBLINK_ROOT`.

## Related Files

- `UPSTREAM_DIFF.md`: raw file inventory against upstream
- `USER_GUIDE.md`: end-user setup and run instructions
- `README_CONT.md`: high-level continuous-depth overview
- `qwen-vl-finetune/EMBED_AR_CHANGES.md`: embed-AR-specific behavior
- `REMOVE_CANDIDATES.md`: cleanup log for files removed from the release snapshot
