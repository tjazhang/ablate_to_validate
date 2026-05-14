# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: debug_loss.py

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Re-use training utilities
from llava.train.train import (
    DataArguments,
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# ------------------------------------------------------------------
# Patch: ensure llava.train.train has `local_rank` defined so that
# `rank0_print` works when datasets/logging are used outside original
# training driver.
# ------------------------------------------------------------------
import llava.train.train as _train_mod  # noqa: E402
if not hasattr(_train_mod, "local_rank"):
    _train_mod.local_rank = 0  # type: ignore


DEFAULT_ENCODER_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "encoder_config.json"


@torch.no_grad()
def evaluate_loss(
    model,
    dataloader: DataLoader,
    device: torch.device,
    log_f=None,
):
    """Iterate over dataloader and report losses.

    If *log_f* is a writable file object, each line is also saved there.
    """
    model.eval()
    # Pre-compute subset indices if available (e.g. when using torch.utils.data.Subset).
    subset_indices: List[int] | None = getattr(dataloader.dataset, "indices", None)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Move tensors to device; images may be list or tensor
        batch_on_device = {}
        for k, v in batch.items():
            if v is None:
                batch_on_device[k] = None
            elif isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
            else:
                # images could be list of tensors with different shapes
                if k == "images":
                    batch_on_device[k] = (
                        torch.stack([img.to(device) for img in v])
                        if isinstance(v, list)
                        else v.to(device)
                    )
                else:
                    batch_on_device[k] = v  # leave as-is

        # Ensure dtype consistency for vision & depth inputs
        images_in = batch_on_device.get("images", None)
        if images_in is not None:
            if isinstance(images_in, torch.Tensor):
                images_in = images_in.to(dtype=model.dtype)
            else:  # list
                images_in = [img.to(dtype=model.dtype) for img in images_in]

        depth_in = batch_on_device.get("depth_embeddings", None)
        if depth_in is not None and isinstance(depth_in, torch.Tensor):
            depth_in = depth_in.to(dtype=model.dtype)

        outputs = model(
            input_ids=batch_on_device["input_ids"],
            attention_mask=batch_on_device["attention_mask"],
            labels=batch_on_device["labels"],
            images=images_in,
            depth_embeddings=depth_in,
            return_dict=True,
        )

        loss_depth = getattr(outputs, "loss_depth", None)
        loss_total = getattr(outputs, "loss_total", None) if hasattr(outputs, "loss_total") else outputs.loss

        # ------------------------------------------------------------------
        # Compose log line with dataset index range **and exact IDs** for this batch.
        # ------------------------------------------------------------------
        batch_size_actual = batch_on_device["input_ids"].shape[0]

        if subset_indices is not None:
            # The dataloader is likely operating on a torch.utils.data.Subset.
            start_pos = batch_idx * dataloader.batch_size
            ids_this_batch = subset_indices[start_pos : start_pos + batch_size_actual]
        else:
            # Fallback: assume sequential ordering over the base dataset.
            start_global = batch_idx * dataloader.batch_size
            ids_this_batch = list(range(start_global, start_global + batch_size_actual))

        # Resolve dataset-specific 'id' values if available in the original JSON.
        base_dataset = getattr(dataloader.dataset, "dataset", dataloader.dataset)
        if hasattr(base_dataset, "list_data_dict"):
            list_data = base_dataset.list_data_dict  # type: ignore
            json_ids = []
            for idx in ids_this_batch:
                try:
                    json_ids.append(list_data[idx].get("id", idx))  # fallback to idx
                except Exception:
                    json_ids.append(idx)
        else:
            json_ids = ids_this_batch

        # Build human-readable ID strings.
        if json_ids:
            ids_str = ",".join(str(i) for i in json_ids)
        else:
            ids_str = "-"

        line_ids = f"Batch {batch_idx:03d} ids [{ids_str}]"
        line_loss = (
            f"Batch {batch_idx:03d} loss_depth = {loss_depth.item() if loss_depth is not None else 'N/A':>8}, "
            f"loss_total = {loss_total.item():>8}"
        )

        # Only print losses to stdout; IDs are written to the log file for detailed inspection.
        print(line_loss)
        if log_f is not None:
            print(line_ids, file=log_f, flush=True)
            print(line_loss, file=log_f, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Debug loss spikes for specific dataset indices.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory (folder containing config.json or sub-artifacts).",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Path or HF id of the base model (e.g. 'liuhaotian/llava-v1.5-13b'). If provided it helps the loader reconstruct LoRA checkpoints.",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON used in training.")
    parser.add_argument("--image_folder", type=str, required=True, help="Root directory containing images referenced in the dataset.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(DEFAULT_ENCODER_CONFIG_PATH),
        help="Shared encoder config JSON (same as --config_path during training).",
    )
    parser.add_argument(
        "--depth_encoder_name",
        type=str,
        default="facebook/dinov2-base",
        help="Depth encoder name used during training.",
    )
    parser.add_argument("--start_index", type=int, required=True, help="Start (inclusive) index in dataset.")
    parser.add_argument("--end_index", type=int, required=True, help="End (inclusive) index in dataset.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples per evaluation batch (should equal per_device_batch_size * grad_accum).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu).")

    # Additional common training flags (optional, mostly ignored but kept for CLI compatibility).
    parser.add_argument("--image_aspect_ratio", type=str, default="pad", help="Image aspect ratio handling (pad or square).")
    parser.add_argument("--custom_image_size", type=int, default=336, help="Custom image resize size (pixels).")
    parser.add_argument("--lazy_preprocess", action="store_true", help="Match training's lazy_preprocess flag (ignored).")
    parser.add_argument("--mm_use_im_start_end", action="store_true", help="Whether to wrap image token with <im_start><im_end> like training.")
    parser.add_argument("--log_file", type=str, default=None, help="If set, save batch losses to this file.")

    # Accept & ignore extra args so that the same command line from training works here.
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer (automatically sets correct config).
    # ------------------------------------------------------------------
    disable_torch_init()

    # Make transformers work purely offline (no remote fetch) if the environment permits.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.checkpoint_dir,
        model_base=args.model_base,
        model_name=os.path.basename(args.checkpoint_dir),
        device="cuda" if args.device.startswith("cuda") else args.device,
        local_files_only=True,
    )
    model.to(args.device)

    # Ensure depth tokens exist in tokenizer (expected IDs 32000-32002)
    depth_tokens = ["<DEPTH_START>", "<DEPTH_END>", "<DEPTH_TOKEN>"]
    missing = [t for t in depth_tokens if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_tokens(depth_tokens)
        # Do NOT resize model embeddings – we assume checkpoint already contains them.

    # ------------------------------------------------------------------
    # 2. Prepare DataArguments mirroring training configuration.
    # ------------------------------------------------------------------
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        config_path=args.config_path,
        depth_encoder_name=args.depth_encoder_name,
        is_multimodal=True,
        image_aspect_ratio=args.image_aspect_ratio,
        custom_image_size=args.custom_image_size,
        lazy_preprocess=args.lazy_preprocess,
    )

    # Dynamically attach flag used by preprocess_multimodal (mirrors training script)
    data_args.mm_use_im_start_end = args.mm_use_im_start_end

    # Vision processor from the loaded model
    data_args.image_processor = image_processor

    # ------------------------------------------------------------------
    # 2b. Derive depth_encoder_stub for correct embedding path substitution
    #      (mirrors logic in train.py)
    # ------------------------------------------------------------------
    if not os.path.exists(data_args.config_path):
        raise FileNotFoundError(f"Config file not found: {data_args.config_path}")

    with open(data_args.config_path, "r") as cf:
        shared_cfg = json.load(cf)

    depth_encoder_name = data_args.depth_encoder_name or shared_cfg.get("default_encoder")
    if depth_encoder_name not in shared_cfg.get("models", {}):
        raise ValueError(
            f"Encoder '{depth_encoder_name}' not found in config models list (config path: {data_args.config_path})."
        )

    depth_encoder_stub = depth_encoder_name.replace("/", "_").replace("-", "_")
    data_args.depth_encoder_stub = depth_encoder_stub

    # ------------------------------------------------------------------
    # 3. Build Dataset & Subset for requested indices.
    # ------------------------------------------------------------------
    full_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=args.data_path, data_args=data_args
    )
    indices = list(range(args.start_index, args.end_index + 1))
    subset_dataset = Subset(full_dataset, indices)

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate losses.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Determine log file path (always create one). If --log_file is not
    # provided, fall back to ./debug_loss.log (with timestamp+encoder dirs).
    # ------------------------------------------------------------------
    if args.log_file:
        input_log_path = args.log_file  # user-specified
    else:
        input_log_path = "debug_loss.log"

    parent_dir = os.path.dirname(input_log_path)
    base_stem = os.path.splitext(os.path.basename(input_log_path))[0]

    # ------------------------------------------------------------------
    # Change: group logs by *model* timestamp (i.e. the parent checkpoint
    # folder name) instead of the current script wall-clock time. This makes
    # repeated evaluations on different checkpoints of the *same* model land
    # in a single directory.
    # ------------------------------------------------------------------

    # Parent directory of the checkpoint: e.g.
    #   .../checkpoints/depth/train_annealing_data_1_llava-v1.5-13b-task-lora-facebook_dinov2_base/20250705_222709_encoder_facebook_dinov2_base/checkpoint-250
    # We want the segment `20250705_222709_encoder_facebook_dinov2_base`.
    cp_parts = os.path.normpath(args.checkpoint_dir).split(os.sep)
    model_dir_name = cp_parts[-2] if len(cp_parts) >= 2 else "model"

    encoder_stub = f"encoder_{depth_encoder_stub}"

    # Final sub-directory name (stable across different checkpoints for the
    # same model run).
    subdir_name = f"{base_stem}_{model_dir_name}_{encoder_stub}"
    full_dir = os.path.join(parent_dir if parent_dir else ".", subdir_name)

    # Final log filename combines model directory + checkpoint + indices for clarity.
    checkpoint_dir_norm = os.path.normpath(args.checkpoint_dir)
    cp_parts = checkpoint_dir_norm.split(os.sep)
    if len(cp_parts) >= 2 and cp_parts[-1].startswith("checkpoint-"):
        model_name_for_path = f"{cp_parts[-2]}_{cp_parts[-1]}"
    else:
        model_name_for_path = cp_parts[-1]
    indices_suffix = f"_idx{args.start_index}-{args.end_index}"
    log_path = os.path.join(full_dir, f"{model_name_for_path}{indices_suffix}.log")

    try:
        os.makedirs(full_dir, exist_ok=True)
    except PermissionError:
        print(f"[WARN] Cannot write to directory '{full_dir}'. Falling back to local file.")
        log_path = f"{model_name_for_path}.log"

    try:
        log_handle = open(log_path, "w")
    except PermissionError as e:
        print(f"[WARN] Failed to open log file '{log_path}' for writing ({e}). Logging disabled.")
        log_handle = None
    try:
        evaluate_loss(model, dataloader, torch.device(args.device), log_f=log_handle)
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main() 
