# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: eval_generated_vs_gt.py

import argparse
import os
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

METHOD_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("LLAVA_DATA_ROOT", str(METHOD_ROOT / "data"))).expanduser()
HARDBLINK_ROOT = Path(os.environ.get("HARDBLINK_ROOT", str(DATA_ROOT / "evals" / "hardblink"))).expanduser()
HARDBLINK_IMAGE_ROOT = Path(os.environ.get("HARDBLINK_IMAGE_ROOT", str(HARDBLINK_ROOT / "images"))).expanduser()
DEFAULT_ENCODER_CONFIG_PATH = DATA_ROOT / "encoder_config.json"

QUESTION_FILES = {
    "3": str(HARDBLINK_ROOT / "questions" / "blink_3pointscenter_questions_long.jsonl"),
    "4": str(HARDBLINK_ROOT / "questions" / "blink_4pointscenter_questions_long.jsonl"),
    "5": str(HARDBLINK_ROOT / "questions" / "blink_5pointscenter_questions_long.jsonl"),
}

IMAGE_FOLDERS = {
    "3": str(HARDBLINK_IMAGE_ROOT / "blink3pointscenter"),
    "4": str(HARDBLINK_IMAGE_ROOT / "blink4pointscenter"),
    "5": str(HARDBLINK_IMAGE_ROOT / "blink5pointscenter"),
}

ANSWER_BASENAMES = {
    "3": "blink_3pointscenter_answers_long.jsonl",
    "4": "blink_4pointscenter_answers_long.jsonl",
    "5": "blink_5pointscenter_answers_long.jsonl",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_encoder_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Encoder config.json not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def parse_encoder_from_answers_dir(answers_dir: str) -> Optional[str]:
    """
    Extract encoder id from directory suffix like *_enc_google_siglip2_large_patch16_256_interploate_16
    → google/siglip2-large-patch16-256-interploate-16
    """
    base = os.path.basename(answers_dir)
    if "_enc_" not in base:
        base = os.path.basename(os.path.dirname(answers_dir))
    if "_enc_" not in base:
        return None
    enc_part = base.split("_enc_")[-1]
    enc_part = enc_part.split(os.sep)[0]
    parts = enc_part.split("_")
    if len(parts) < 2:
        return None
    org = parts[0]
    model_name = "-".join(parts[1:])
    return f"{org}/{model_name}"


def try_read_mm_vision_tower_nearby(answers_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(answers_dir, "config.json"),
        os.path.join(os.path.dirname(answers_dir), "config.json"),
    ]
    for cfg in candidates:
        if os.path.isfile(cfg):
            try:
                with open(cfg, "r") as f:
                    cfg_json = json.load(f)
                val = cfg_json.get("mm_vision_tower")
                if isinstance(val, str) and len(val) > 0:
                    return val
            except Exception:
                pass
    return None


def find_encoder_key_in_config(models: Dict[str, Any], encoder_id: str) -> Optional[str]:
    # Exact match
    if encoder_id in models:
        return encoder_id

    # Try to ignore trailing "-interploate-*"
    base = re.sub(r"-interploate-\d+", "", encoder_id)
    for k in models.keys():
        if k == base:
            return k

    # Fuzzy: match by suffix after org/
    try:
        org, name = encoder_id.split("/", 1)
    except ValueError:
        name = encoder_id
    for k in models.keys():
        if k.endswith(name):
            return k

    # Last resort: case-insensitive contains of core tokens
    key_tokens = re.split(r"[-_/]", name)
    for k in models.keys():
        score = sum(1 for t in key_tokens if t and t.lower() in k.lower())
        if score >= max(2, len([t for t in key_tokens if t])) - 1:
            return k
    return None


def load_questions_map(question_file: str) -> Dict[int, Dict[str, Any]]:
    qmap: Dict[int, Dict[str, Any]] = {}
    with open(question_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = data["question_id"]
            if isinstance(qid, str) and qid.isdigit():
                qid = int(qid)
            qmap[qid] = data
    return qmap


def get_model_and_processor(model_name: str, conf: Dict[str, Any]):
    from transformers import AutoImageProcessor, AutoModel, AutoConfig, AutoModelForDepthEstimation
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


def interpolate_features_linear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate a sequence of patch features to target_len using 1-D linear interpolation.
    
    Args:
        features: Tensor of shape [N, D].
        target_len: Desired number of patch tokens.
    
    Returns:
        Tensor of shape [target_len, D].
    """
    # [N, D] ➔ [1, D, N] so that interpolate works on the sequence dimension
    features_1d = features.unsqueeze(0).permute(0, 2, 1)
    interpolated = F.interpolate(features_1d, size=target_len, mode="linear", align_corners=False)
    return interpolated.squeeze(0).permute(1, 0)  # ➔ [target_len, D]


def interpolate_features_bilinear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate patch features on a 2-D spatial grid using bilinear mode.
    
    The input tokens are assumed to form a square grid (H × W). The function reshapes the
    sequence back to the grid, applies F.interpolate with mode='bilinear', and then
    flattens it back to a sequence.
    """
    num_tokens, dim = features.shape
    
    h = w = int(np.sqrt(num_tokens))
    if h * w != num_tokens:
        raise ValueError(
            f"Input token length {num_tokens} is not a perfect square – cannot use bilinear interpolation.")
    
    target_h = target_w = int(np.sqrt(target_len))
    if target_h * target_w != target_len:
        raise ValueError(
            f"Target token length {target_len} is not a perfect square – cannot use bilinear interpolation.")
    
    # [N, D] ➔ [H, W, D] ➔ [1, D, H, W]
    features_2d = features.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)
    
    # Interpolate on the spatial dimensions (H, W)
    interpolated = F.interpolate(
        features_2d.to(torch.float32),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(features.dtype)
    
    # Back to [target_len, D]
    interpolated = interpolated.squeeze(0).permute(1, 2, 0).contiguous().view(target_len, dim)
    return interpolated


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


def extract_patch_tokens(image_path: str, processor, model, conf: Dict[str, Any]) -> torch.Tensor:
    """Extract patch tokens from an image using the encoder, matching preprocessing.py logic"""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    px = inputs["pixel_values"].cuda() if torch.cuda.is_available() else inputs["pixel_values"]
    
    is_depth_anything = conf.get("model_type") == "depth_anything"

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
    patch_embedding = seq.cpu()
    
    return patch_embedding


def to_square_grid(tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    t, d = tokens.shape
    h = int(round(math.sqrt(t)))
    w = h
    if h * w != t:
        if t < h * w:
            pad = h * w - t
            pad_vec = tokens[-1:, :].repeat(pad, 1)
            tokens = torch.cat([tokens, pad_vec], dim=0)
        else:
            tokens = tokens[: h * w, :]
    grid = tokens.view(h, w, d)
    return grid, h, w


def resize_tokens_to_match(src_tokens: torch.Tensor, target_token_count: int, mode: str = "auto") -> torch.Tensor:
    """
    Resize a [T, D] token sequence to target_token_count.
    mode:
      - "auto": use 2D bilinear if both sides are perfect squares, else 1D linear fallback
      - "linear": force 1D linear interpolation
      - "bilinear": force 2D bilinear interpolation (requires perfect squares; falls back to linear)
    """
    if src_tokens.shape[0] == target_token_count:
        return src_tokens
    if mode == "linear":
        return interpolate_features_linear(src_tokens, target_token_count)
    if mode == "bilinear":
        try:
            return interpolate_features_bilinear(src_tokens, target_token_count)
        except ValueError:
            return interpolate_features_linear(src_tokens, target_token_count)
    # auto
    tgt_side = int(round(math.sqrt(target_token_count)))
    if tgt_side * tgt_side == target_token_count:
        try:
            return interpolate_features_bilinear(src_tokens, target_token_count)
        except ValueError:
            return interpolate_features_linear(src_tokens, target_token_count)
    # fallback: linear interpolation over sequence to arbitrary length
    idx = torch.linspace(0, src_tokens.shape[0] - 1, steps=target_token_count)
    low = torch.floor(idx).long()
    high = torch.clamp(low + 1, max=src_tokens.shape[0] - 1)
    alpha = (idx - low.float()).unsqueeze(1)
    return (1 - alpha) * src_tokens[low] + alpha * src_tokens[high]


def l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def mean_cosine_similarity(a_tokens: torch.Tensor, b_tokens: torch.Tensor) -> float:
    t = min(a_tokens.shape[0], b_tokens.shape[0])
    a = a_tokens[:t]
    b = b_tokens[:t]
    return float((a * b).sum(dim=1).mean().item())


def l2_mse_loss(a_tokens: torch.Tensor, b_tokens: torch.Tensor) -> float:
    """Calculate L2 MSE loss between two token sequences."""
    t = min(a_tokens.shape[0], b_tokens.shape[0])
    a = a_tokens[:t]
    b = b_tokens[:t]
    return float(F.mse_loss(a, b).item())


def load_generated_embedding(npy_path: str) -> torch.Tensor:
    arr = np.load(npy_path)
    ten = torch.from_numpy(arr).float()
    if ten.dim() == 1:
        ten = ten.unsqueeze(0)
    return ten


def list_embedding_files(emb_dir: str) -> List[str]:
    if not os.path.isdir(emb_dir):
        return []
    return sorted([str(p) for p in Path(emb_dir).glob("depth_embeddings_*.npy")])


def extract_qid_from_filename(path: str) -> Optional[int]:
    m = re.search(r"depth_embeddings_(\d+)\.npy$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def run_dataset(
    name: str,
    answers_dir: str,
    question_file: str,
    image_folder: str,
    processor,
    model,
    conf: Dict[str, Any],
    k: int,
    max_samples: Optional[int],
    seed: int,
    device: torch.device,
    interp_mode: str,
    target_num_patches: Optional[int],
    print_first_n: int,
) -> Dict[str, float]:
    answers_basename = ANSWER_BASENAMES[name]
    emb_dir = os.path.join(
        answers_dir,
        os.path.splitext(answers_basename)[0] + "_embeddings",
    )
    qmap = load_questions_map(question_file)
    emb_files = list_embedding_files(emb_dir)
    if not emb_files:
        print(f"[WARN] No embeddings found in {emb_dir}; skipping dataset {name}")
        return {"num": 0, "hits": 0, "hit_rate": 0.0}

    qids: List[int] = []
    by_qid: Dict[int, str] = {}
    for p in emb_files:
        qid = extract_qid_from_filename(p)
        if qid is None:
            continue
        if qid not in qmap:
            continue
        qids.append(qid)
        by_qid[qid] = p

    if not qids:
        print(f"[WARN] No overlapping question ids between {emb_dir} and {question_file}")
        return {
            "num": 0,
            "cosine": {"hits": 0, "hit_rate": 0.0},
            "l2_mse": {"hits": 0, "hit_rate": 0.0}
        }

    rng = random.Random(seed)
    if max_samples is not None:
        rng.shuffle(qids)
        qids = qids[:max_samples]

    # Infer generated token count and dim
    first_gen = load_generated_embedding(by_qid[qids[0]]).to(device)
    gen_token_count = first_gen.shape[0]
    dim = first_gen.shape[1]

    hits_cosine = 0
    hits_l2_mse = 0
    total = 0
    printed = 0

    for qid in qids:
        gen_path = by_qid[qid]
        gen = load_generated_embedding(gen_path).to(device)  # (Tg, Dg)
        if gen.shape[1] != dim:
            dmin = min(dim, gen.shape[1])
            gen = gen[:, :dmin]
            dim = dmin

        image_rel = qmap[qid]["image"]
        image_path = os.path.join(image_folder, image_rel)
        gt_tokens = extract_patch_tokens(image_path, processor, model, conf)  # (Te, De)
        if gt_tokens.shape[1] != dim:
            dmin = min(dim, gt_tokens.shape[1])
            gt_tokens = gt_tokens[:, :dmin]
            gen = gen[:, :dmin]
            dim = dmin

        # Apply interpolation
        if target_num_patches is not None:
            # Force both to target length
            if gen.shape[0] != target_num_patches:
                gen = resize_tokens_to_match(gen, target_num_patches, mode=interp_mode)
            if gt_tokens.shape[0] != target_num_patches:
                gt_tokens = resize_tokens_to_match(gt_tokens, target_num_patches, mode=interp_mode)
        else:
            # Default: resize GT to GEN length
            gt_tokens = resize_tokens_to_match(gt_tokens, gen.shape[0], mode=interp_mode)

        # Calculate both cosine similarity and L2 MSE metrics
        gen_n = l2_normalize(gen, dim=1)
        gt_n = l2_normalize(gt_tokens.to(device), dim=1)
        
        # Cosine similarity metric
        sim_gt = mean_cosine_similarity(gen_n, gt_n)
        loss_gt_cosine = 1.0 - sim_gt
        
        # L2 MSE metric (no normalization needed)
        loss_gt_l2_mse = l2_mse_loss(gen, gt_tokens.to(device))

        neg_pool = [p for p in emb_files if p != gen_path]
        if len(neg_pool) < k:
            negs = [rng.choice(neg_pool) for _ in range(k)] if neg_pool else []
        else:
            negs = rng.sample(neg_pool, k)

        neg_losses_cosine: List[float] = []
        neg_losses_l2_mse: List[float] = []
        neg_sims: List[float] = []
        for npy in negs:
            neg = load_generated_embedding(npy).to(device)
            if neg.shape[1] != dim:
                dmin = min(dim, neg.shape[1])
                neg = neg[:, :dmin]
            neg = resize_tokens_to_match(neg, gen.shape[0], mode=interp_mode)
            
            # Cosine similarity
            neg_n = l2_normalize(neg, dim=1)
            sim = mean_cosine_similarity(gen_n, neg_n)
            neg_sims.append(sim)
            neg_losses_cosine.append(1.0 - sim)
            
            # L2 MSE
            neg_losses_l2_mse.append(l2_mse_loss(gen, neg))

        # Determine hits for both metrics
        all_losses_cosine = [loss_gt_cosine] + neg_losses_cosine
        best_idx_cosine = int(np.argmin(all_losses_cosine))
        if best_idx_cosine == 0:
            hits_cosine += 1
            
        all_losses_l2_mse = [loss_gt_l2_mse] + neg_losses_l2_mse
        best_idx_l2_mse = int(np.argmin(all_losses_l2_mse))
        if best_idx_l2_mse == 0:
            hits_l2_mse += 1
            
        total += 1

        # Debug prints for first N examples
        if printed < print_first_n:
            try:
                print(f"[DEBUG] Example {printed + 1} qid={qid}")
                print(f"        image: {image_path}")
                print(f"        gen shape: {tuple(gen.shape)}, gt shape: {tuple(gt_tokens.shape)}, dim: {dim}")
                if target_num_patches is not None:
                    print(f"        interpolation: both -> {target_num_patches} tokens, mode={interp_mode}")
                else:
                    print(f"        interpolation: gt -> {gen.shape[0]} tokens, mode={interp_mode}")
                top_neg_sims = sorted(neg_sims, reverse=True)[:3] if neg_sims else []
                top_neg_l2_mse = sorted(neg_losses_l2_mse)[:3] if neg_losses_l2_mse else []
                print(f"        [Cosine] loss(gt)={loss_gt_cosine:.4f}, top neg sims={', '.join(f'{s:.4f}' for s in top_neg_sims)}, hit={best_idx_cosine == 0}")
                print(f"        [L2 MSE] loss(gt)={loss_gt_l2_mse:.6f}, top neg losses={', '.join(f'{l:.6f}' for l in top_neg_l2_mse)}, hit={best_idx_l2_mse == 0}")
            except Exception:
                pass
            printed += 1

    hit_rate_cosine = (hits_cosine / total) if total > 0 else 0.0
    hit_rate_l2_mse = (hits_l2_mse / total) if total > 0 else 0.0
    return {
        "num": total,
        "cosine": {"hits": hits_cosine, "hit_rate": hit_rate_cosine},
        "l2_mse": {"hits": hits_l2_mse, "hit_rate": hit_rate_l2_mse}
    }


def main():
    parser = argparse.ArgumentParser(description="Check if generated embedding is closest to GT vs k negatives using both Cosine Similarity and L2 MSE metrics")
    parser.add_argument("--answers_dir", type=str, required=True,
                        help="Path to model answers dir containing blink_*answers_long.jsonl and *_embeddings/")
    parser.add_argument("--k", type=int, default=10, help="# of negative embeddings per query")
    parser.add_argument("--datasets", type=str, default="3,4,5", help="Comma-separated subset of {3,4,5}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_dataset", type=int, default=None, help="Optional cap on #queries per dataset")
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder id (must exist in encoder_config.json)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder_config", type=str,
                        default=str(DEFAULT_ENCODER_CONFIG_PATH),
                        help="Path to shared encoder_config.json (used in preprocessing.py)")
    parser.add_argument("--interp_mode", type=str, default="auto", choices=["auto", "linear", "bilinear"],
                        help="Interpolation mode when resizing tokens (auto uses bilinear for square lengths else linear).")
    parser.add_argument("--target_num_patches", type=int, default=None,
                        help="If set, interpolate both GEN and GT to this token count before comparison.")
    parser.add_argument("--print_first_n", type=int, default=3,
                        help="Print debug info for the first N examples compared.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    answers_dir = os.path.abspath(args.answers_dir)

    # Determine encoder id
    cfg = load_encoder_config(args.encoder_config)
    models = cfg.get("models", {})
    enc_id = args.encoder or try_read_mm_vision_tower_nearby(answers_dir) or parse_encoder_from_answers_dir(answers_dir)
    if enc_id is None:
        raise ValueError("Could not infer encoder from answers_dir; please pass --encoder")

    enc_key = find_encoder_key_in_config(models, enc_id)
    if enc_key is None:
        raise ValueError(f"Encoder '{enc_id}' not found in encoder_config.json. Available: {list(models.keys())}")

    conf = models[enc_key]
    print(f"[INFO] Using encoder: {enc_key}")

    # Load model and processor using preprocessing.py logic
    processor, model = get_model_and_processor(enc_key, conf)
    model.to(device)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip() in {"3", "4", "5"}]
    if not ds_list:
        raise ValueError("No valid datasets specified; choose from 3,4,5")

    overall_hits_cosine = 0
    overall_hits_l2_mse = 0
    overall_total = 0

    for name in ds_list:
        res = run_dataset(
            name=name,
            answers_dir=answers_dir,
            question_file=QUESTION_FILES[name],
            image_folder=IMAGE_FOLDERS[name],
            processor=processor,
            model=model,
            conf=conf,
            k=args.k,
            max_samples=args.max_per_dataset,
            seed=args.seed,
            device=device,
            interp_mode=args.interp_mode,
            target_num_patches=args.target_num_patches,
            print_first_n=args.print_first_n,
        )
        print(f"blink_{name}pointscenter: processed={res['num']}")
        print(f"  [Cosine Similarity] hits={res['cosine']['hits']}, hit@1={res['cosine']['hit_rate']:.2%}")
        print(f"  [L2 MSE]            hits={res['l2_mse']['hits']}, hit@1={res['l2_mse']['hit_rate']:.2%}")
        overall_hits_cosine += res["cosine"]["hits"]
        overall_hits_l2_mse += res["l2_mse"]["hits"]
        overall_total += res["num"]

    if overall_total > 0:
        print(f"\nOverall: {overall_total} queries")
        print(f"  [Cosine Similarity] hits={overall_hits_cosine}, hit@1={overall_hits_cosine/overall_total:.2%}")
        print(f"  [L2 MSE]            hits={overall_hits_l2_mse}, hit@1={overall_hits_l2_mse/overall_total:.2%}")
    else:
        print("No queries processed.")


if __name__ == "__main__":
    main()

