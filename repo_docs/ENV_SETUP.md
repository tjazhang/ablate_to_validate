# Environment Setup

This repo uses the same conda env names as the original working setup:

- `llava`
- `qwen_vl`
- `mirage`
- `mull`
- `covt`

On this machine, those envs were present when checked on 2026-03-29. If they already work for you, reuse them with `conda activate <env>`. If not, recreate them from the files under `envs/`.

The `envs/*.environment.yml` files are now populated snapshots exported from those existing envs with machine-local `prefix` lines removed. They capture the third-party package set, but you should still run the repo-local editable install steps below so the env points at the Aurora code in this repo.

## Rules

- Vendored methods (`llava`, `qwen_vl`) can be installed directly from `methods/`.
- Overlay methods (`mirage`, `mull`, `covt`) need `./tools/bootstrap_overlay.py <method>` first, because their real install targets live under `external/<method>/upstream`.
- `methods/qwen/qwen-vl-utils` is shared across Qwen and the three overlay methods. Prefer installing it as a local editable package in those envs.

## LLaVA

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/llava.environment.yml
conda activate llava
pip install -e methods/llava
# Optional if you want `hf download` / `hf upload-large-folder`:
# pip install "huggingface_hub[cli]"
```

Use this env for `methods/llava`.

## Qwen

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/qwen.environment.yml
conda activate qwen_vl
pip install -e methods/qwen/qwen-vl-utils
# Optional if you want `hf download` / `hf upload-large-folder`:
# pip install "huggingface_hub[cli]"
```

Use this env for `methods/qwen`.

If you want the Gradio demos on top of the exported env snapshot:

```bash
pip install -r methods/qwen/requirements_web_demo.txt
```

## Mirage

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/mirage.environment.yml
conda activate mirage
./tools/bootstrap_overlay.py mirage
pip install -e external/mirage/upstream/transformers
pip install -e methods/qwen/qwen-vl-utils
```

Use this env for `overlays/mirage` and the generated checkout under `external/mirage/upstream`.

## Mull

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/mull.environment.yml
conda activate mull
./tools/bootstrap_overlay.py mull
(cd external/mull/upstream && bash setup.sh)
# Reinstall the vendored local qwen_vl_utils if you want the Aurora repo copy to win:
# pip install -e methods/qwen/qwen-vl-utils[decord]
```

Use this env for `overlays/mull` and `external/mull/upstream`.

## CoVT

```bash
cd /path/to/Ablate-to-Validate
conda env create -f envs/covt.environment.yml
conda activate covt
./tools/bootstrap_overlay.py covt
pip install -e external/covt/upstream/VLMEvalKit
pip install -e methods/qwen/qwen-vl-utils
# Optional if you want CoVT-LLaVA variants:
# pip install -e methods/llava
```

Use this env for `overlays/covt` and `external/covt/upstream`.

## Quick Map

| Method | Env | Setup doc |
| --- | --- | --- |
| LLaVA | `llava` | [`methods/llava/USER_GUIDE.md`](../methods/llava/USER_GUIDE.md) |
| Qwen | `qwen_vl` | [`methods/qwen/USER_GUIDE.md`](../methods/qwen/USER_GUIDE.md) |
| Mirage | `mirage` | [`overlays/mirage/USER_GUIDE.md`](../overlays/mirage/USER_GUIDE.md) |
| Mull | `mull` | [`overlays/mull/USER_GUIDE.md`](../overlays/mull/USER_GUIDE.md) |
| CoVT | `covt` | [`overlays/covt/USER_GUIDE.md`](../overlays/covt/USER_GUIDE.md) |
