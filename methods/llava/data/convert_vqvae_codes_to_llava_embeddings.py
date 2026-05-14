#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/convert_vqvae_codes_to_llava_embeddings.py

"""Convert monolithic VQ-VAE code dict into per-image .npy embeddings for LLaVA.

Expected input format:
  np.load(codebook_path, allow_pickle=True).item()
  -> dict[str, np.ndarray], where keys look like "ADE_train_00000001_depth.png"
     and values are [num_depth_tokens, feature_dim] float arrays.

Output format:
  <embedding_root>/<encoder_stub>/<base_image_name>.npy
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, Tuple

import numpy as np


DEFAULT_CODEBOOK_PATH = os.environ.get("LLaVA_VQVAE_CODEBOOK")
DEFAULT_ENCODER_NAME = "mahtab/ait-vqvae-continuous-100x512"


def encoder_stub(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def normalize_base_name(base_img_name: str) -> str:
    """Match the normalization behavior used by preprocessing.py."""
    if "ADE_" in base_img_name:
        i = base_img_name.find("ADE_")
        if i > 0:
            return base_img_name[i:]
    if "ADE_train_" in base_img_name:
        i = base_img_name.find("ADE_train_")
        if i > 0:
            return base_img_name[i:]
    return re.sub(r"^[0-9]+pts_", "", base_img_name)


def code_key_to_base_name(key: str) -> str:
    base = os.path.basename(str(key))
    if base.endswith("_depth.png"):
        base = base[: -len("_depth.png")]
    elif base.endswith(".png") or base.endswith(".npy"):
        base = os.path.splitext(base)[0]
    return normalize_base_name(base)


def load_codebook(path: str) -> Dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        loaded = loaded.item()
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict-like codebook, got {type(loaded)} from {path}")
    return loaded


def iter_items_with_limit(items: Iterable[Tuple[str, np.ndarray]], limit: int):
    if limit <= 0:
        yield from items
        return
    for i, item in enumerate(items):
        if i >= limit:
            break
        yield item


def validate_array(value: np.ndarray, expected_tokens: int, expected_dim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [tokens, dim], got shape={arr.shape}")
    if expected_tokens is not None and arr.shape[0] != expected_tokens:
        raise ValueError(f"Token count mismatch: got {arr.shape[0]}, expected {expected_tokens}")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(f"Feature dim mismatch: got {arr.shape[1]}, expected {expected_dim}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Expected numeric dtype, got {arr.dtype}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def derive_expected_stems_from_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    stems = set()
    for sample in data:
        embedding_path = sample.get("embedding")
        if embedding_path:
            stem = os.path.splitext(os.path.basename(embedding_path))[0]
        else:
            image_path = sample.get("image", "")
            stem = normalize_base_name(os.path.splitext(os.path.basename(image_path))[0])
        if stem:
            stems.add(stem)
    return stems, len(data)


def resolve_embedding_root(config: Dict[str, object], config_path: str) -> str:
    embed_root = os.path.expanduser(str(config["embedding_root"]))
    if not os.path.isabs(embed_root):
        embed_root = os.path.join(os.path.dirname(os.path.abspath(config_path)), embed_root)
    return os.path.abspath(embed_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-path", default=DEFAULT_CODEBOOK_PATH)
    parser.add_argument(
        "--config-path",
        default=os.path.join(os.path.dirname(__file__), "encoder_config.json"),
        help="Path to shared encoder config (used for defaults and output root).",
    )
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory; default is <embedding_root>/<encoder_stub> from config.",
    )
    parser.add_argument(
        "--expected-tokens",
        type=int,
        default=None,
        help="Expected token count per sample. Defaults to grid_size**2 from encoder config when available.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="Expected feature dim per sample. Defaults to feature_dim from encoder config when available.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process first N entries only (0 = all).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-jsons",
        nargs="*",
        default=[],
        help="Optional dataset JSON paths for post-conversion file coverage checks.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    if not args.codebook_path:
        raise ValueError("No VQ-VAE codebook was provided. Pass --codebook-path or set LLaVA_VQVAE_CODEBOOK.")

    model_cfg = config.get("models", {}).get(args.encoder_name, {})
    expected_tokens = args.expected_tokens
    expected_dim = args.expected_dim
    if expected_tokens is None:
        grid_size = model_cfg.get("grid_size")
        if isinstance(grid_size, int) and grid_size > 0:
            expected_tokens = grid_size * grid_size
    if expected_dim is None:
        feature_dim = model_cfg.get("feature_dim")
        if isinstance(feature_dim, int) and feature_dim > 0:
            expected_dim = feature_dim

    if args.output_dir is None:
        embed_root = resolve_embedding_root(config, args.config_path)
        args.output_dir = os.path.join(embed_root, encoder_stub(args.encoder_name))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading codebook: {args.codebook_path}")
    codebook = load_codebook(args.codebook_path)
    print(f"[INFO] Loaded entries: {len(codebook)}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Validation constraints: expected_tokens={expected_tokens}, expected_dim={expected_dim}")

    converted = 0
    skipped_existing = 0
    failed = 0
    shape_hist = Counter()

    for key, value in iter_items_with_limit(codebook.items(), args.limit):
        out_name = f"{code_key_to_base_name(key)}.npy"
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path) and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            arr = validate_array(value, expected_tokens, expected_dim)
            shape_hist[arr.shape] += 1
            if not args.dry_run:
                np.save(out_path, arr)
            converted += 1
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"[WARN] Failed key={key}: {exc}")

    print(
        f"[SUMMARY] converted={converted}, skipped_existing={skipped_existing}, "
        f"failed={failed}, dry_run={args.dry_run}, total_seen={converted + skipped_existing + failed}"
    )
    if shape_hist:
        print(f"[SUMMARY] shape_hist={dict(shape_hist)}")

    for dataset_path in args.validate_jsons:
        stems, num_samples = derive_expected_stems_from_dataset(dataset_path)
        missing = 0
        for stem in stems:
            if not os.path.exists(os.path.join(args.output_dir, f"{stem}.npy")):
                missing += 1
        print(
            f"[VALIDATE] {dataset_path}: samples={num_samples}, unique_stems={len(stems)}, "
            f"missing_files={missing}"
        )


if __name__ == "__main__":
    main()
