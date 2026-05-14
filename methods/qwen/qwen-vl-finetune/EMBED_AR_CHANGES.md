<!--
NEW: Aurora-only file; not present in upstream QWEN.
NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
NEW: Aurora path: qwen-vl-finetune/EMBED_AR_CHANGES.md
-->

# Embed-AR: Embedding-Space Autoregression for Depth Tokens

## Overview

Embed-AR closes the train/test gap for continuous depth tokens. Instead of injecting ground-truth (GT) projected depth embeddings during training, the model learns to predict depth from its own hidden states — the same way it operates at inference time.

---

## What Changed

### 1. Configuration & Flags

| File | Change |
|---|---|
| `configuration_qwen2_5_vl.py` | Added `use_depth_embed_ar` (bool, default `False`) and `depth_embed_ar_gt_ratio` (float, default `1.0`) |
| `argument.py` | Added matching `TrainingArguments` fields: `--use_depth_embed_ar`, `--depth_embed_ar_gt_ratio` |
| `train_qwen.py` | Wires both flags from training args → model config; logs them in the config dump |

### 2. Training Forward Pass (`modeling_qwen2_5_vl.py`)

Two things happen in `forward()` when `depth_embed_ar=True`:

**a) Input-embedding setup (before autoregressive unroll):**

| Mode | `depth_embed_ar` | `gt_ratio` | Behaviour |
|---|---|---|---|
| **Original** | `False` | — | All `<DEPTH_TOKEN>` positions replaced with GT projected depth (teacher forcing) |
| **Pure Embed-AR** | `True` | `0.0` | No GT injection; all depth positions are model-predicted during unroll |
| **Scheduled Sampling** | `True` | `0.0–1.0` | Each depth position independently uses GT with probability `gt_ratio`; the rest use model prediction |

**b) Sequential autoregressive tail unroll (exact train-time matching):**

When `depth_embed_ar=True`, training no longer does a single full-sequence forward followed by post-hoc replacement.  
Instead, from the first `<DEPTH_TOKEN>` onward, it runs autoregressively:

```
embed_t = depth_projector(depth_head(hidden_{t-1}))   # for depth positions selected for model prediction
hidden_t = LLM(embed_t, past)
```

This makes depth positions sequentially dependent exactly like generation, and recomputes downstream tokens under those modified states.

Gradient checkpointing is incompatible with exact KV-cache sequential unroll.  
`train_qwen.py` auto-disables gradient checkpointing when `use_depth_embed_ar=True`. If checkpointing is still forced from another path, the model warns and falls back to the older approximate non-sequential replacement.

The depth loss (`h[t-1] → depth_head → target`) is always computed regardless of mode.

### 3. Custom Greedy Decode (`modeling_qwen2_5_vl.py`)

Added `greedy_decode_depth_ar()` method on `Qwen2_5_VLForConditionalGeneration`. This is a custom generation loop (like LLaVA's `greedy_decode`) with:

- **KV cache** with proper mRoPE position tracking via `cache_position` and `rope_deltas`.
- **State machine** with four states:
  - **Text mode**: argmax → embed → feed as next input.
  - **Enter depth**: sees `<DEPTH_START>` → switch to depth mode.
  - **Depth mode** (K steps): `h_last → depth_head → normalize → depth_projector` → feed projected hidden state as next input (not a token embedding).
  - **Exit depth**: after K steps, inject `<DEPTH_END>` token embedding → resume text.
- **`<DEPTH_TOKEN>` IDs emitted** for each of the K depth steps so downstream token-to-embedding alignment works.
- **Ablation support**: GT / random / zero / model overrides applied in depth space (D-dim) before projection, matching LLaVA's `override_depth_vec` path.

### 4. Inference Integration (`model_vqa_qwen.py`)

`generate_greedy()` now has three code paths:

| Path | Condition | Method |
|---|---|---|
| **Embed-AR** | `use_depth_embed_ar=True`, continuous mode | Manually builds `inputs_embeds` with vision tokens, calls `model.greedy_decode_depth_ar()` |
| **LogitsProcessor** | Continuous mode, non-embed-AR | HF `generate()` with `ContinuousDepthLogitsProcessor` (existing) |
| **Standard** | Discrete or baseline | Plain `model.generate()` (existing) |

New CLI flags:
- `--use-depth-embed-ar` — force enable embed-AR even if model config says otherwise.
- `--no-depth-embed-ar` — force disable embed-AR even if model config has it enabled.

Ablation mode (`--use-random-depth`, `--use-zero-depth`, `--use-gt-depth`, `--use-model-depth`) is passed through to the custom decode loop.

**Safety guard:** If `use_depth_embed_ar=True` and the user calls the standard HF `model.generate()` path (instead of `greedy_decode_depth_ar`), a warning is logged on the first call explaining that depth embedding replacement will not happen.

### 5. Training Script

Added `scripts/train_ade_continuous_embed_ar.sh` — mirrors `train_ade_continuous.sh` with:
- `--use_depth_embed_ar True`
- `--depth_embed_ar_gt_ratio 0.0` (pure embed-AR; edit to e.g. `0.7` for scheduled sampling)
- Output dirs include `embed_ar` in the name.

---

## Scheduled Sampling

`depth_embed_ar_gt_ratio` controls curriculum learning:

```
gt_ratio=1.0  →  All positions get GT (identical to non-embed-AR)
gt_ratio=0.7  →  70% GT, 30% model-predicted
gt_ratio=0.0  →  No GT at all (pure embed-AR, fully model-predicted)
```

Each depth token position is sampled independently per training step. To anneal, change the ratio across training runs (not currently annealed within a single run).

---

## File Summary

| File | Lines changed |
|---|---|
| `qwenvl/configuration_qwen2_5_vl.py` | +2 (new config fields) |
| `qwenvl/train/argument.py` | +5 (new training args) |
| `qwenvl/train/train_qwen.py` | +4 (wiring + logging) |
| `qwenvl/modeling_qwen2_5_vl.py` | ~300 (refactored GT injection, added `greedy_decode_depth_ar`) |
| `model_vqa_qwen.py` | ~100 (embed-AR path in `generate_greedy`, CLI flags, plumbing) |
| `scripts/train_ade_continuous_embed_ar.sh` | New file (147 lines) |
