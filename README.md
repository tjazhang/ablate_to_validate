# Ablate-to-Validate: Testing Whether Vision-Language Models Actually Use Latent Visual Tokens

\[[Read our arXiv Paper](TODO: add arXiv link)\] &nbsp; \[[Project Page](https://tjazhang.github.io/ablate_to_validate/)\]

Tianyi Zhang, [Mahtab Bigverdi](https://mahtabbigverdi.github.io/), [Ranjay Krishna](https://www.ranjaykrishna.com/)

### Introduction

Vision-language models (VLMs) are increasingly augmented with continuous or latent non-textual tokens meant to support visual reasoning, but higher accuracy alone does not show that models truly use those tokens as reasoning content. We introduce **Ablate-to-Validate**, a diagnostic principle for testing whether latent-token content is genuinely utilized, and instantiate it with the **Token Replacement Test (TRT)**, a standardized suite of content-replacement ablations. Across controlled relative-depth reasoning experiments on **LLaVA** and **Qwen2.5-VL**, TRT shows that much of the gain can remain even when latent-token content is corrupted or replaced, exposing a gap between having a latent channel and actually using it as an information bottleneck.

<p align="center">
  <a href="assets/AtV_figure.png">
    <img src="assets/AtV_figure.png" alt="Intro figure" width="700">
  </a>
</p>

<p align="center">
  <video src="docs/assets/atv_teaser.mp4" autoplay loop muted playsinline controls width="700">
    Your browser does not render embedded video. <a href="docs/assets/atv_teaser.mp4">Watch the 46&nbsp;s animated explainer (MP4)</a>.
  </video>
</p>

<p align="center">
  <em>▶ <a href="docs/assets/atv_teaser.mp4">46&nbsp;s animated explainer</a> · <a href="https://tjazhang.github.io/ablate_to_validate/">project page</a> · <a href="assets/teaser_animation/">animation source</a></em>
</p>

---

## Repository

This repository has five runnable surfaces:

- `methods/llava`: vendored LLaVA snapshot with Aurora depth work
- `methods/qwen`: vendored Qwen snapshot with Aurora depth work
- `overlays/mirage`: external upstream + Aurora override layer
- `overlays/mull`: external upstream + Aurora override layer
- `overlays/covt`: external upstream + Aurora override layer

If you are new here, use the docs in this order:

1. [`docs/ENV_SETUP.md`](docs/ENV_SETUP.md)
2. The guide for the method you want to run
3. [`docs/structure.md`](docs/structure.md) if you need repo internals

Maintenance notes such as `UPSTREAM_DIFF.md`, `REMOVE_CANDIDATES.md`, and `POLISHED_*.md` are not the primary runbooks.

## Method Index

| Component | Kind | Conda env | Bootstrap needed | Canonical entrypoints | User guide |
| --- | --- | --- | --- | --- | --- |
| LLaVA | vendored | `llava` | no | `python -m llava.eval.run_llava`, `python model_vqa_depth_continuous.py`, `python -m llava.eval.model_vqa_depth_discrete` | [`methods/llava/USER_GUIDE.md`](methods/llava/USER_GUIDE.md) |
| Qwen | vendored | `qwen_vl` | no | `bash methods/qwen/qwen-vl-finetune/scripts/train_ade_*.sh`, `bash methods/qwen/qwen-vl-finetune/eval_qwen.sh`, `python methods/qwen/web_demo_mm.py` | [`methods/qwen/USER_GUIDE.md`](methods/qwen/USER_GUIDE.md) |
| Mirage | overlay | `mirage` | `./tools/bootstrap_overlay.py mirage` | `./tools/eval_mirage.sh`, `./tools/eval_mirage_depth.sh`, `./tools/train_mirage_depth.sh` | [`overlays/mirage/USER_GUIDE.md`](overlays/mirage/USER_GUIDE.md) |
| Mull | overlay | `mull` | `./tools/bootstrap_overlay.py mull` | `./tools/eval_mull.sh` | [`overlays/mull/USER_GUIDE.md`](overlays/mull/USER_GUIDE.md) |
| CoVT | overlay | `covt` | `./tools/bootstrap_overlay.py covt` | `./tools/eval_covt.sh` | [`overlays/covt/USER_GUIDE.md`](overlays/covt/USER_GUIDE.md) |

## First-Time Setup

From the repo root:

```bash
cd /path/to/Ablate-to-Validate
```

If you will use an overlay method, materialize its upstream checkout first:

```bash
./tools/bootstrap_overlay.py --dry-run mirage
./tools/bootstrap_overlay.py mirage
```

Then activate the matching env and follow that method's guide.

## Repo Layout

- `methods/`
  - vendored runnable source trees for LLaVA and Qwen
- `overlays/`
  - Aurora-owned patch/override layers for Mirage, Mull, and CoVT
- `external/`
  - generated upstream checkouts for the overlay methods
  - intentionally git-ignored
- `envs/`
  - exported conda env snapshots using the same env names as the original working setup
- `tools/bootstrap_overlay.py`
  - clones pinned overlay repos, checks out the locked commit, applies patches, and copies override files

## Additional Docs

- [`docs/ENV_SETUP.md`](docs/ENV_SETUP.md): shared environment setup and recreation guide
- [`docs/structure.md`](docs/structure.md): repo organization
- [`third_party/THIRD_PARTY_NOTICES.md`](third_party/THIRD_PARTY_NOTICES.md): upstream provenance and notices
