#!/usr/bin/env python
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/create_dummy_from_training_subset.py

"""create_dummy_from_training_subset.py

Builds a minimal toy dataset by sampling a few records from an existing
*ADE20K/train_annealing_data* JSON file so the directory structure + embedding
paths match the real training corpus.

Example usage
-------------
python create_dummy_from_training_subset.py \
        --src ADE20K/train_annealing_data_full.json \
        --out_dir ADE20K/train_annealing_data \
        --n_depth 60 --n_text 60 --mode both

Output directory will contain
  • dummy_subset/dummy_subset.json   – subset dataset
  • dummy_subset/embeddings/…        – copied or zero-filled .npy files

You can then point --data_path to the new JSON for quick smoke tests.
"""

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import random

PLACEHOLDER = "<DEPTH_START><DEPTH_TOKEN><DEPTH_END>"



def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("Dataset root must be a list of objects")
    return samples


def categorise(samples):
    """Split dataset into three groups:

    1. depth_only_idxs  – conversation consists solely of the placeholder string.
    2. mixed_idxs       – placeholder present *and* there is additional non-whitespace text.
    3. text_only_idxs   – no placeholder at all.
    """
    depth_only_idxs, mixed_idxs, text_only_idxs = [], [], []
    for idx, s in enumerate(samples):
        conv_text = "\n".join(
            msg.get("value", "")
            for msg in s.get("conversations", [])
            if msg.get("from") == "gpt"
        )

        if PLACEHOLDER not in conv_text:
            text_only_idxs.append(idx)
        else:
            residual = conv_text.replace(PLACEHOLDER, "").strip()

        if residual:
            mixed_idxs.append(idx)
        else:
            depth_only_idxs.append(idx)

    return depth_only_idxs, mixed_idxs, text_only_idxs


def sample_indices(indices: List[int], k: int) -> List[int]:
    return indices[:k] if len(indices) >= k else indices


# No-op helper retained for backward compatibility – simply returns original path.
def keep_embedding(src_path: Path, *_):
    return src_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ADE20K/train_annealing_data.json", help="Path to original training JSON file")
    parser.add_argument("--out_dir", default="ADE20K/train_annealing_data", help="Output directory")
    parser.add_argument("--n_depth", type=int, default=32, help="Number of depth-supervised samples to include if depth mode is selected")
    parser.add_argument("--n_text", type=int, default=32, help="Number of text-only samples to include if text mode is selected")
    parser.add_argument("--n_mixed", type=int, default=64, help="Number of mixed (placeholder+text) samples to include if mixed or all mode is selected")
    parser.add_argument("--mode", choices=["all", "depth", "mixed", "text", "both"], default="all",
                        help="Subset type to build: depth-only, mixed (placeholder+text), text-only, or all (default). 'both' is kept as alias for 'all' for backward compatibility")
    args = parser.parse_args()

    src_json = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(src_json)
    depth_only_idxs, mixed_idxs, text_only_idxs = categorise(samples)

    # Initialize selection lists
    depth_sel, mixed_sel, text_sel = [], [], []

    if args.mode == "depth":
        depth_sel = sample_indices(depth_only_idxs, args.n_depth)
    elif args.mode == "mixed":
        mixed_sel = sample_indices(mixed_idxs, args.n_mixed)
    elif args.mode == "text":
        text_sel = sample_indices(text_only_idxs, args.n_text)
    else:  # args.mode in ('all', 'both')
        depth_sel = sample_indices(depth_only_idxs, args.n_depth)
        mixed_sel = sample_indices(mixed_idxs, args.n_mixed)
        text_sel = sample_indices(text_only_idxs, args.n_text)
        # fill shortfall as before
        shortfall = args.n_text - len(text_sel)
        if shortfall > 0:
            additional = [idx for idx in mixed_idxs if idx not in text_sel][:shortfall]
            text_sel.extend(additional)

    # Combine
    selected_indices = depth_sel + mixed_sel + text_sel

    # Combine and shuffle selected indices to randomize output order
    random.shuffle(selected_indices)

    new_samples = []
    for new_id, orig_idx in enumerate(selected_indices):
        s = samples[orig_idx].copy()
        s["id"] = f"dummy_{new_id}"

        # Keep original embedding path; assume file already exists.
        # (Optionally sanity-check existence here.)

        new_samples.append(s)

    out_json = out_dir / "ADE20K/train_annealing_data_test.json"
    out_json.write_text(json.dumps(new_samples, indent=2))
    print(f"Wrote {len(new_samples)} samples to {out_json}")


if __name__ == "__main__":
    main() 