# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/preprocessing.py

"""
Generalized embedding extraction script for depth images using multiple vision encoders.

Changes (2025‑05‑30)
--------------------
* Added **skip_cls** flag per‑encoder so we keep *patch* tokens only and never accidentally
  capture the model's classification token or logits layer. For models that don't use a
  CLS token (e.g. SigLIP‑2) set `skip_cls=False` so **all** tokens are preserved.
* Refactored patch‑embedding logic.
* Fixed SigLIP model loading to use `SiglipVisionModel` instead of the default multimodal class.
"""

import os
import json
from pathlib import Path
import re
import argparse
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoConfig, AutoModelForDepthEstimation
import torch.nn.functional as F

# Load shared encoder configuration
CONFIG_PATH = Path(__file__).resolve().with_name("encoder_config.json")
CONFIG_DIR = CONFIG_PATH.parent

# Raise error if config is missing
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Shared config file not found at {CONFIG_PATH}. Please create it.")

# Parse encoder config and extract model definitions
with open(CONFIG_PATH, "r") as cf:
    _CFG = json.load(cf)

def _resolve_from_config_dir(path_value: str) -> str:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = CONFIG_DIR / candidate
    return str(candidate.resolve())


EMBED_ROOT: str = _resolve_from_config_dir(_CFG["embedding_root"])  # Root path to save embeddings
MODELS: Dict[str, Dict[str, Any]] = _CFG["models"]     # Dictionary of encoder configurations

# Repo-relative defaults. Override via --depth_map_dir if your depth maps live elsewhere.
DEPTH_MAP_DIR = str((CONFIG_DIR / "ADE20K" / "depth_maps").resolve())
INPUT_JSONS = [
    str((CONFIG_DIR / "ADE20K" / "mixed_depth" / "mixed_depth_long.json").resolve()),
    str((CONFIG_DIR / "ADE20K" / "train_annealing_data" / "train_annealing_data.json").resolve()),
]

# CLI to optionally override depth directory
parser = argparse.ArgumentParser(description="Extract depth embeddings for multiple dataset JSONs.")
parser.add_argument("--depth_map_dir", type=str, default=os.environ.get("DEPTH_MAP_DIR", DEPTH_MAP_DIR),
                    help="Directory containing depth PNGs named <base>_depth.png")
args = parser.parse_args()

DEPTH_MAP_DIR = args.depth_map_dir

# Load all datasets and combine them for processing
all_datasets = []
for json_path in INPUT_JSONS:
    with open(json_path, "r") as f:
        dataset = json.load(f)
        # Track which JSON each item came from
        for item in dataset:
            item["_source_json"] = json_path
        all_datasets.append(dataset)
        print(f"Loaded {len(dataset)} items from {json_path}")

# Combine all data for processing
data = []
for dataset in all_datasets:
    data.extend(dataset)
print(f"Total items to process: {len(data)}")

# Function to load model and its processor with proper patch resizing and architecture-specific fixes
def get_model_and_processor(model_name: str, conf: Dict[str, Any]):
    from transformers import CLIPVisionModel, SiglipVisionModel
    
    # Special handling for DepthAnything V2
    if conf.get("model_type") == "depth_anything":
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return processor, model

    processor = AutoImageProcessor.from_pretrained(model_name)
    inp_res = conf["input_resolution"]

    # Resize config logic (different for SigLIP and other models)
    if "siglip" in model_name.lower():
        if hasattr(processor, "size"):
            processor.size = {"height": inp_res, "width": inp_res}
    else:
        if hasattr(processor, "size"):
            processor.size = {"shortest_edge": inp_res}
        if hasattr(processor, "crop_size"):
            processor.crop_size = {"height": inp_res, "width": inp_res}

    # Handle patch adjustment logic
    if conf.get("adjust_patch", False):
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.patch_size = inp_res // conf["grid_size"]
        model = AutoModel.from_config(cfg)
    elif "clip" in model_name.lower():
        model = CLIPVisionModel.from_pretrained(model_name)
    elif "siglip" in model_name.lower():
        model = SiglipVisionModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    model.eval()  # Set model to evaluation mode
    if torch.cuda.is_available():
        model.cuda()  # Move to GPU if available
    return processor, model

# Function to extract features from DepthAnything V2's backbone
def get_depth_anything_features(model, pixel_values):
    """Extract features from DepthAnything V2's DINOv2 backbone"""
    # Access the backbone directly
    if hasattr(model, 'backbone'):
        # Forward through the backbone
        features = model.backbone(pixel_values, output_hidden_states=True)
        if hasattr(features, 'hidden_states'):
            # Use the last hidden state from backbone
            return features.hidden_states[-1]
        elif hasattr(features, 'last_hidden_state'):
            return features.last_hidden_state
    
    # Fallback to full model with hidden states
    outputs = model(pixel_values, output_hidden_states=True)
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        return outputs.hidden_states[-1]
    
    # Final fallback: use predicted depth as feature (not ideal but works)
    outputs = model(pixel_values)
    depth_pred = outputs.predicted_depth
    # Reshape depth to patch-like format
    b, h, w = depth_pred.shape
    patch_size = 14  # Typical patch size
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Reshape and pool to get patch-level features
    depth_patches = depth_pred.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    depth_patches = depth_patches.contiguous().view(b, num_patches_h, num_patches_w, patch_size * patch_size)
    depth_patches = depth_patches.view(b, num_patches, patch_size * patch_size)
    
    # Pool each patch to get a single feature
    patch_features = depth_patches.mean(dim=-1, keepdim=True)
    # Expand to match expected feature dimension
    feature_dim = 1024  # DepthAnything V2 large feature dimension
    patch_features = patch_features.expand(-1, -1, feature_dim)
    
    return patch_features

# Normalize base image names by removing prefixes like '5pts_' so depth/embedding files align
def normalize_base_name(base_img_name: str) -> str:
    # Prefer cutting directly to the ADE* portion if present
    if "ADE_" in base_img_name:
        i = base_img_name.find("ADE_")
        if i > 0:
            return base_img_name[i:]
    if "ADE_train_" in base_img_name:
        i = base_img_name.find("ADE_train_")
        if i > 0:
            return base_img_name[i:]
    # Fallback: strip patterns like '5pts_' or '10pts_'
    return re.sub(r"^[0-9]+pts_", "", base_img_name)

# Depth scalar sampling from maps using bilinear sampling
def sample_depth_scalars(D: torch.Tensor, coords: torch.Tensor):
    """
    D: [B,1,H,W] depth maps; coords: [B,K,2] with x,y in [0,1]
    returns scalars [B,K]
    """
    B, _, H, W = D.shape
    K = coords.shape[1]
    gx = coords[..., 0] * 2 - 1
    gy = coords[..., 1] * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(B, K, 1, 2)
    vals = F.grid_sample(D, grid, mode='bilinear', align_corners=True)  # [B,1,K,1]
    return vals.view(B, K)

# Main processing loop per model
for model_name, conf in MODELS.items():
    print(f"\n>>> Processing encoder: {model_name}")

    # Precomputed-code entries are materialized by a separate converter script.
    if conf.get("model_type") == "precomputed_codes":
        print(f"Skipping precomputed encoder entry '{model_name}' (no feature extraction needed).")
        continue

    # Create output directory for model-specific embeddings
    np_folder = os.path.join(EMBED_ROOT, model_name.replace("/", "_").replace("-", "_"))
    os.makedirs(np_folder, exist_ok=True)

    # Load model and preprocessing pipeline
    processor, model = get_model_and_processor(model_name, conf)
    
    # Determine if this model uses original images or depth maps
    use_original = conf.get("use_original_image", False)
    is_depth_anything = conf.get("model_type") == "depth_anything"

    # Track shape stats for debugging
    shape_counts = {}
    shape_examples = {}
    
    # Track failed items
    failed_items = []
    
    # Loop through all dataset items
    for item in tqdm(data, desc=os.path.basename(np_folder)):
        raw_base_name, _ = os.path.splitext(os.path.basename(item["image"]))
        base_img_name = normalize_base_name(raw_base_name)
        
        # Choose image path based on model type
        if use_original:
            # Use original image for DepthAnything
            img_path = item["image"]
        else:
            # Use depth map for other models
            img_path = os.path.join(DEPTH_MAP_DIR, f"{base_img_name}_depth.png")
            
        np_path = os.path.join(np_folder, f"{base_img_name}.npy")

        # If embedding not already saved, compute and save it
        patch_embedding = None
        if not os.path.exists(np_path):
            try:
                # Check if source image exists
                if not os.path.exists(img_path):
                    failed_items.append((base_img_name, f"Source image not found: {img_path}"))
                    continue
                
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                px = inputs["pixel_values"].cuda() if torch.cuda.is_available() else inputs["pixel_values"]

                with torch.no_grad():
                    if is_depth_anything:
                        # DepthAnything V2 feature extraction
                        last_hidden = get_depth_anything_features(model, px)
                    else:
                        # Original logic for other models
                        outputs = model(pixel_values=px)
                        
                        # Handle different output structuring
                        if hasattr(outputs, "last_hidden_state"):
                            last_hidden = outputs.last_hidden_state
                        elif hasattr(outputs, "vision_model_output"):
                            last_hidden = outputs.vision_model_output.last_hidden_state
                        else:
                            last_hidden = outputs[0]

                # Extract patch tokens and remove CLS if specified
                seq = last_hidden.squeeze(0)
                if conf["skip_cls"] and not is_depth_anything:
                    seq = seq[1:]
                patch_embedding = seq.cpu().numpy()

                # Save embedding
                np.save(np_path, patch_embedding)

                # Track embedding shape statistics
                shape = str(patch_embedding.shape)
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
                if len(shape_examples.get(shape, [])) < 3:
                    shape_examples.setdefault(shape, []).append(item["id"])
            
            except Exception as e:
                failed_items.append((base_img_name, str(e)))
                continue
        
        # Sample depth scalars if enabled for this encoder (process every sample, not just new embeddings)
        if conf.get("include_depth_scalars", False) and not use_original:
            # Load embedding if not already in memory
            if patch_embedding is None:
                patch_embedding = np.load(np_path)
            # Create depth scalars directory
            depth_scalars_folder = os.path.join(EMBED_ROOT, "depth_scalars", model_name.replace("/", "_").replace("-", "_"))
            os.makedirs(depth_scalars_folder, exist_ok=True)
            depth_scalar_path = os.path.join(depth_scalars_folder, f"{base_img_name}.npy")
            
            if not os.path.exists(depth_scalar_path):
                try:
                    # Load depth map
                    depth_img_path = os.path.join(DEPTH_MAP_DIR, f"{base_img_name}_depth.png")
                    if os.path.exists(depth_img_path):
                        depth_img = Image.open(depth_img_path).convert("L")  # Grayscale
                        depth_array = np.array(depth_img).astype(np.float32) / 255.0  # Normalize to [0,1]
                        
                        # Convert to tensor [1, 1, H, W]
                        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)
                        
                        # Get or generate coordinates matching the embedding patches
                        K = patch_embedding.shape[0]  # Number of patches
                        
                        # Generate normalized grid coordinates (same as train.py fallback)
                        def _factor_pair(n: int):
                            h = int(np.floor(np.sqrt(n)))
                            best_h, best_w, best_gap = 1, n, n
                            for hh in range(1, h+1):
                                if n % hh == 0:
                                    ww = n // hh
                                    gap = abs(ww - hh)
                                    if gap < best_gap:
                                        best_h, best_w, best_gap = hh, ww, gap
                            return best_h, best_w
                        
                        h, w = _factor_pair(K)
                        ys = (torch.arange(h, dtype=torch.float32) + 0.5) / max(h, 1)
                        xs = (torch.arange(w, dtype=torch.float32) + 0.5) / max(w, 1)
                        Y, X = torch.meshgrid(ys, xs, indexing='ij')
                        coords = torch.stack([X, Y], dim=-1).reshape(K, 2)  # [K, 2] in [0,1]
                        
                        # Add batch dimension [1, K, 2]
                        coords_batch = coords.unsqueeze(0)
                        
                        # Sample depth scalars
                        scalars = sample_depth_scalars(depth_tensor, coords_batch)  # [1, K]
                        scalars_np = scalars.squeeze(0).numpy()  # [K]
                        
                        # Save depth scalars
                        np.save(depth_scalar_path, scalars_np)
                except Exception as e:
                    if item == data[0]:  # Only print for first item to avoid spam
                        print(f"Warning: Failed to sample depth scalars for {base_img_name}: {e}")

    # Print shape statistics for the encoder
    print(f"Shape statistics for {model_name}:")
    for s, c in shape_counts.items():
        print(f"  {s}: {c} images  | examples: {shape_examples[s]}")
    print(f"Saved embeddings to: {np_folder}")
    
    # Report failed items
    if failed_items:
        print(f"\n⚠️  Failed to process {len(failed_items)} items:")
        for base_name, error in failed_items[:10]:  # Show first 10
            print(f"  - {base_name}: {error}")
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more")
    
    # Print depth scalars info if enabled for this encoder
    if conf.get("include_depth_scalars", False) and not use_original:
        depth_scalars_folder = os.path.join(EMBED_ROOT, "depth_scalars", model_name.replace("/", "_").replace("-", "_"))
        print(f"Saved depth scalars to: {depth_scalars_folder}")

    # Release memory
    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Create unified dataset JSON files using placeholder paths
print(f"\n>>> Creating unified dataset files...")

# Check if any encoder has depth scalars enabled
any_encoder_has_depth_scalars = any(conf.get("include_depth_scalars", False) for conf in MODELS.values())

# Get a reference encoder (skip precomputed-code entries).
reference_encoder = None
for encoder_name, encoder_conf in MODELS.items():
    if encoder_conf.get("model_type") != "precomputed_codes":
        reference_encoder = encoder_name
        break
if reference_encoder is None:
    raise ValueError("No extractable encoders found in config. Add at least one non-precomputed encoder entry.")

reference_folder = os.path.join(EMBED_ROOT, reference_encoder.replace("/", "_").replace("-", "_"))

# Update all items with embedding paths and depth scalars, but only keep items with valid embeddings
valid_data = []
missing_count = 0

for item in data:
    raw_base_name, _ = os.path.splitext(os.path.basename(item["image"]))
    base_img_name = normalize_base_name(raw_base_name)

    # Check if embedding exists for this item
    reference_embedding_path = os.path.join(reference_folder, f"{base_img_name}.npy")
    if not os.path.exists(reference_embedding_path):
        missing_count += 1
        if missing_count <= 5:  # Print first 5 missing items
            print(f"Warning: Skipping item {item.get('id', 'unknown')} - embedding not found: {base_img_name}.npy")
        continue

    # Placeholder used for generalization across encoders
    placeholder_path = os.path.join(EMBED_ROOT, "<encoder_stub>", f"{base_img_name}.npy")
    item["embedding"] = placeholder_path
    
    # Add depth scalars path if any encoder has it enabled
    if any_encoder_has_depth_scalars:
        depth_scalar_placeholder = os.path.join(EMBED_ROOT, "depth_scalars", "<encoder_stub>", f"{base_img_name}.npy")
        item["depth_scalars"] = depth_scalar_placeholder

    # Replace raw depth tokenized content with a fixed placeholder
    conv_val = item["conversations"][1]["value"]
    if "<DEPTH_START>" in conv_val and "<DEPTH_END>" in conv_val:
        a = conv_val.find("<DEPTH_START>") + len("<DEPTH_START>")
        b = conv_val.find("<DEPTH_END>")
        item["conversations"][1]["value"] = conv_val[:a] + "<DEPTH_TOKEN>" + conv_val[b:]
    
    valid_data.append(item)

if missing_count > 0:
    print(f"\n>>> Skipped {missing_count} items with missing embeddings")
    print(f">>> Keeping {len(valid_data)}/{len(data)} items with valid embeddings")

# Replace data with validated data
data = valid_data

# Group items back by source JSON and save separately
for json_path in INPUT_JSONS:
    # Filter items from this source
    source_data = [item for item in data if item.get("_source_json") == json_path]
    
    # Remove the temporary _source_json field
    for item in source_data:
        del item["_source_json"]
    
    # Create output directory based on original JSON name
    dir_name = os.path.basename(json_path.replace(".json", ""))
    dir_path = os.path.join(EMBED_ROOT, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Save unified JSON
    unified_json = os.path.join(dir_path, f"{dir_name}.json")
    with open(unified_json, "w") as f:
        json.dump(source_data, f, indent=2)
    
    print(f"Saved {len(source_data)} items to: {unified_json}")

print("All unified datasets saved with <encoder_stub> substitution support")

# List which encoders have depth scalars enabled
if any_encoder_has_depth_scalars:
    encoders_with_scalars = [name for name, conf in MODELS.items() if conf.get("include_depth_scalars", False)]
    print(f"Depth scalars enabled for {len(encoders_with_scalars)} encoder(s): {', '.join(encoders_with_scalars)}")
    print("Depth scalars will be loaded based on encoder selection at training time")

print("\nAll encoders processed")
