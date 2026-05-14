# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: model_vqa_depth_continuous.py

import argparse
import os
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL
import shortuuid
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEFAULT_METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER_CONFIG_PATH = DEFAULT_METHOD_ROOT / "data" / "encoder_config.json"
DEFAULT_GT_DEPTH_CODEBOOK = os.environ.get("GT_DEPTH_CODEBOOK")


def parse_encoder_name_from_model_path(model_path: str) -> Optional[str]:
    """
    Convert suffix like *_enc_openai_clip_vit_large_patch14_336 to canonical encoder id
    (e.g., openai/clip-vit-large-patch14-336).
    """
    if "_enc_" not in model_path:
        return None
    enc_part = model_path.split("_enc_")[-1]
    enc_part = enc_part.split("/")[0]
    parts = [p for p in enc_part.split("_") if p]
    if len(parts) < 2:
        return None
    org = parts[0]
    model_name = "-".join(parts[1:])
    return f"{org}/{model_name}"


def get_encoder_grid_size(model_path, encoder_config_path: Optional[str] = None):
    """
    Extract encoder name from model path and get its grid_size from encoder_config.json
    Returns grid_size or None if not found
    """
    try:
        encoder_config_path = encoder_config_path or str(DEFAULT_ENCODER_CONFIG_PATH)
        # Load encoder config
        with open(encoder_config_path, 'r') as f:
            encoder_config = json.load(f)
        
        encoder_name = parse_encoder_name_from_model_path(model_path)
        if encoder_name:
            if encoder_name in encoder_config["models"]:
                grid_size = encoder_config["models"][encoder_name]["grid_size"]
                print(f"[INFO] Detected encoder: {encoder_name}, grid_size: {grid_size}")
                return grid_size
            else:
                print(f"[WARNING] Encoder {encoder_name} not found in config, trying fuzzy match...")
                org = encoder_name.split("/")[0]
                parts = encoder_name.split("/")[-1].split("-")
                for config_encoder in encoder_config["models"]:
                    if org in config_encoder and any(p in config_encoder for p in parts):
                        grid_size = encoder_config["models"][config_encoder]["grid_size"]
                        print(f"[INFO] Matched encoder: {config_encoder}, grid_size: {grid_size}")
                        return grid_size
        
        print("[WARNING] Could not detect encoder from model path, using default grid_size=16")
        return 16  # Default fallback
        
    except Exception as e:
        print(f"[WARNING] Error loading encoder config: {e}, using default grid_size=16")
        return 16  # Default fallback

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_encoder_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Encoder config.json not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def find_encoder_key_in_config(models: Dict[str, Any], encoder_id: str) -> Optional[str]:
    if encoder_id in models:
        return encoder_id

    base = re.sub(r"-interploate-\d+", "", encoder_id)
    if base in models:
        return base

    try:
        org, name = encoder_id.split("/", 1)
    except ValueError:
        org = None
        name = encoder_id

    for key in models.keys():
        if org and not key.startswith(org):
            continue
        if key.endswith(name):
            return key

    key_tokens = re.split(r"[-_/]", name)
    filtered = [t for t in key_tokens if t]
    for key in models.keys():
        score = sum(1 for t in filtered if t.lower() in key.lower())
        if score >= max(2, len(filtered)) - 1:
            return key
    return None


def interpolate_features_linear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    features_1d = features.unsqueeze(0).permute(0, 2, 1)
    interpolated = F.interpolate(features_1d, size=target_len, mode="linear", align_corners=False)
    return interpolated.squeeze(0).permute(1, 0)


def interpolate_features_bilinear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    num_tokens, dim = features.shape
    h = w = int(np.sqrt(num_tokens))
    if h * w != num_tokens:
        raise ValueError(f"Input token length {num_tokens} is not a perfect square.")

    target_h = target_w = int(np.sqrt(target_len))
    if target_h * target_w != target_len:
        raise ValueError(f"Target token length {target_len} is not a perfect square.")

    features_2d = features.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)
    interpolated = F.interpolate(
        features_2d.to(torch.float32),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(features.dtype)
    interpolated = interpolated.squeeze(0).permute(1, 2, 0).contiguous().view(target_len, dim)
    return interpolated


def resize_tokens_to_match(src_tokens: torch.Tensor, target_token_count: int, mode: str = "auto") -> torch.Tensor:
    if src_tokens.shape[0] == target_token_count:
        return src_tokens
    if mode == "linear":
        return interpolate_features_linear(src_tokens, target_token_count)
    if mode == "bilinear":
        try:
            return interpolate_features_bilinear(src_tokens, target_token_count)
        except ValueError:
            return interpolate_features_linear(src_tokens, target_token_count)
    tgt_side = int(round(math.sqrt(target_token_count)))
    if tgt_side * tgt_side == target_token_count:
        try:
            return interpolate_features_bilinear(src_tokens, target_token_count)
        except ValueError:
            return interpolate_features_linear(src_tokens, target_token_count)
    idx = torch.linspace(0, src_tokens.shape[0] - 1, steps=target_token_count)
    low = torch.floor(idx).long()
    high = torch.clamp(low + 1, max=src_tokens.shape[0] - 1)
    alpha = (idx - low.float()).unsqueeze(1)
    return (1 - alpha) * src_tokens[low] + alpha * src_tokens[high]


def get_model_and_processor(model_name: str, conf: Dict[str, Any], device: Optional[torch.device] = None):
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForDepthEstimation,
        CLIPVisionModel,
        SiglipVisionModel,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    processor = AutoImageProcessor.from_pretrained(model_name)
    inp_res = conf["input_resolution"]

    if "siglip" in model_name.lower():
        if hasattr(processor, "size"):
            processor.size = {"height": inp_res, "width": inp_res}
    else:
        if hasattr(processor, "size"):
            processor.size = {"shortest_edge": inp_res}
        if hasattr(processor, "crop_size"):
            processor.crop_size = {"height": inp_res, "width": inp_res}

    if conf.get("adjust_patch", False):
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.patch_size = inp_res // conf["grid_size"]
        model = AutoModel.from_config(cfg)
    elif "clip" in model_name.lower():
        model = CLIPVisionModel.from_pretrained(model_name)
    elif "siglip" in model_name.lower():
        model = SiglipVisionModel.from_pretrained(model_name)
    elif conf.get("model_type") == "depth_anything":
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    model.eval()
    model.to(device)
    return processor, model


def extract_patch_tokens(
    image_path: str,
    processor,
    model,
    conf: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    is_depth_anything = conf.get("model_type") == "depth_anything"

    with torch.no_grad():
        if is_depth_anything and hasattr(model, "backbone"):
            features = model.backbone(pixel_values, output_hidden_states=True)
            if hasattr(features, "hidden_states") and features.hidden_states:
                last_hidden = features.hidden_states[-1]
            else:
                last_hidden = model(pixel_values, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = model(pixel_values=pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, "vision_model_output"):
                last_hidden = outputs.vision_model_output.last_hidden_state
            else:
                last_hidden = outputs[0]

    seq = last_hidden.squeeze(0)
    if conf.get("skip_cls", False) and not conf.get("model_type") == "depth_anything":
        seq = seq[1:]
    return seq.cpu().float()


class GroundTruthDepthProvider:
    def __init__(
        self,
        encoder_id: str,
        encoder_config_path: str,
        interp_mode: str = "auto",
        target_num_patches: Optional[int] = None,
        encoder_device: Optional[str] = None,
    ):
        self.encoder_id = encoder_id
        self.encoder_config_path = encoder_config_path
        self.interp_mode = interp_mode
        self.target_num_patches = target_num_patches
        device = encoder_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_device = torch.device(device)

        cfg = load_encoder_config(encoder_config_path)
        models = cfg.get("models", {})
        enc_key = find_encoder_key_in_config(models, encoder_id)
        if enc_key is None:
            raise ValueError(f"Encoder '{encoder_id}' not found in encoder_config.json.")
        self.encoder_key = enc_key
        self.encoder_conf = models[enc_key]
        self.processor, self.encoder_model = get_model_and_processor(enc_key, self.encoder_conf, self.encoder_device)
        self.cache: Dict[Tuple[str, int], torch.Tensor] = {}
        print(f"[INFO] GroundTruthDepthProvider initialized with encoder {enc_key} on {self.encoder_device}")

    def get_embeddings(self, image_path: str, fallback_tokens: int) -> torch.Tensor:
        target_len = self.target_num_patches or fallback_tokens
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        cache_key = (image_path, target_len)
        if cache_key in self.cache:
            return self.cache[cache_key]
        tokens = extract_patch_tokens(
            image_path,
            self.processor,
            self.encoder_model,
            self.encoder_conf,
            device=self.encoder_device,
        )
        tokens = resize_tokens_to_match(tokens, target_len, mode=self.interp_mode)
        self.cache[cache_key] = tokens
        return tokens


class DiscreteGroundTruthDepthProvider:
    """Provider for ground truth discrete depth tokens from pre-computed codebook."""
    
    def __init__(self, codebook_path: str, discrete_depth_token_ids: Optional[list] = None):
        """
        Initialize discrete GT depth provider.
        
        Args:
            codebook_path: Path to .npy file containing discrete token sequences
            discrete_depth_token_ids: List of token IDs for discrete depth levels (REQUIRED)
        """
        if not os.path.isfile(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        
        if discrete_depth_token_ids is None or len(discrete_depth_token_ids) == 0:
            raise ValueError("discrete_depth_token_ids must be provided and non-empty")
        
        # Load codebook (dictionary: {image_id_depth.png: token_sequence_string})
        self.codebook = np.load(codebook_path, allow_pickle=True).item()
        self.discrete_depth_token_ids = discrete_depth_token_ids
        self.cache: Dict[str, list] = {}
        
        print(f"[INFO] DiscreteGroundTruthDepthProvider initialized")
        print(f"  Codebook: {codebook_path}")
        print(f"  Entries: {len(self.codebook)}")
        print(f"  Discrete token ID mapping: {len(discrete_depth_token_ids)} levels")
        print(f"  Token ID range: {min(discrete_depth_token_ids)}-{max(discrete_depth_token_ids)}")
        
        # Analyze one entry to show structure
        if self.codebook:
            example_key = list(self.codebook.keys())[0]
            example_value = self.codebook[example_key]
            tokens = self._parse_token_sequence(example_value)
            print(f"  Example: {example_key} -> {len(tokens)} depth levels")
            print(f"  Depth level range: {min(tokens)}-{max(tokens)}")
            # Verify levels are within token ID mapping range
            max_level = max(tokens)
            if max_level >= len(discrete_depth_token_ids):
                raise ValueError(f"Codebook contains level {max_level} but only {len(discrete_depth_token_ids)} token IDs provided")
    
    def _parse_token_sequence(self, token_string: str) -> list:
        """Parse token sequence string into list of token level integers."""
        # Extract numbers from <DEPTH_N> tokens (excluding START/END)
        matches = re.findall(r'<DEPTH_(\d+)>', token_string)
        return [int(m) for m in matches]
    
    def _image_filename_to_codebook_key(self, image_filename: str) -> str:
        """Convert image filename to codebook key."""
        # Extract base name without extension and add _depth.png
        base_name = os.path.splitext(os.path.basename(image_filename))[0]
        return f"{base_name}_depth.png"
    
    def get_token_ids(self, image_filename: str) -> list:
        """
        Get ground truth discrete token IDs for an image.
        
        Args:
            image_filename: Image filename (e.g., "0.png", "path/to/42.png")
        
        Returns:
            List of token IDs corresponding to discrete depth levels
        """
        cache_key = image_filename
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert filename to codebook key
        codebook_key = self._image_filename_to_codebook_key(image_filename)
        
        if codebook_key not in self.codebook:
            raise KeyError(f"Image key '{codebook_key}' not found in codebook (from '{image_filename}')")
        
        # Parse token sequence to get depth levels
        token_string = self.codebook[codebook_key]
        depth_levels = self._parse_token_sequence(token_string)
        
        # Convert depth levels to actual token IDs using the mapping
        # discrete_depth_token_ids[level] gives the vocabulary token ID for that depth level
        token_ids = [self.discrete_depth_token_ids[level] for level in depth_levels]
        
        self.cache[cache_key] = token_ids
        return token_ids

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    encoder_config_path = os.path.expanduser(args.encoder_config)
    model_name = get_model_name_from_path(model_path)
    
    # Check config.json to see if this is a discrete token model
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        print(f"\n{'='*80}")
        print(f"CHECKING SAVED CONFIG: {config_path}")
        print(f"{'='*80}")
        print(f"use_discrete_depth_tokens: {saved_config.get('use_discrete_depth_tokens', 'NOT FOUND')}")
        print(f"num_discrete_depth_levels: {saved_config.get('num_discrete_depth_levels', 'NOT FOUND')}")
        print(f"depth_start_id: {saved_config.get('depth_start_id', 'NOT FOUND')}")
        print(f"depth_end_id: {saved_config.get('depth_end_id', 'NOT FOUND')}")
        has_discrete_token_ids = 'discrete_depth_token_ids' in saved_config
        saved_discrete_token_ids = saved_config.get('discrete_depth_token_ids')
        print(f"discrete_depth_token_ids present: {has_discrete_token_ids}")
        if isinstance(saved_discrete_token_ids, list):
            print(f"discrete_depth_token_ids (first 5): {saved_discrete_token_ids[:5]}")
        else:
            print(f"discrete_depth_token_ids value: {saved_discrete_token_ids}")
        print(f"{'='*80}\n")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Get encoder grid_size and calculate max_new_tokens dynamically
    grid_size = get_encoder_grid_size(model_path, encoder_config_path)
    # For each embedding, we get grid_size ** 2 depth embeddings (2D grid)
    # Add 512 tokens buffer for text response
    depth_token_target = grid_size ** 2
    max_new_tokens = depth_token_target + 512
    print(f"[INFO] Using max_new_tokens={max_new_tokens} (grid_size={grid_size}, depth_tokens={depth_token_target}, text_buffer=512)")

    use_discrete_depth_tokens = getattr(model.config, 'use_discrete_depth_tokens', False)
    depth_mode = getattr(model.config, 'depth_mode', None)
    if depth_mode not in {"original", "continuous", "discrete"}:
        if use_discrete_depth_tokens:
            depth_mode = "discrete"
        elif getattr(model.config, "depth_token_id", None) is not None:
            depth_mode = "continuous"
        else:
            depth_mode = "original"
    is_original_mode = depth_mode == "original"
    start_depth_token_id = getattr(model.config, 'depth_start_id', None)
    end_depth_token_id = getattr(model.config, 'depth_end_id', None)
    
    # Debug: Print model depth configuration
    print(f"\n{'='*80}")
    print("MODEL DEPTH CONFIGURATION")
    print(f"{'='*80}")
    print(f"depth_mode: {depth_mode}")
    print(f"use_discrete_depth_tokens: {use_discrete_depth_tokens}")
    print(f"depth_start_id: {start_depth_token_id}")
    print(f"depth_end_id: {end_depth_token_id}")
    if use_discrete_depth_tokens:
        num_levels = getattr(model.config, 'num_discrete_depth_levels', None)
        token_ids = getattr(model.config, 'discrete_depth_token_ids', None)
        print(f"num_discrete_depth_levels: {num_levels}")
        print(f"discrete_depth_token_ids: {token_ids[:5] if token_ids else None}... (first 5)")
    else:
        num_depth_tokens = getattr(model.config, 'num_depth_tokens', None)
        print(f"num_depth_tokens (continuous): {num_depth_tokens}")
    print(f"{'='*80}\n")

    # Validate ablation flags - only one can be active at a time
    ablation_flags = [
        args.use_gt_depth_embeddings,
        args.use_random_depth,
        args.use_zero_depth,
        args.use_model_depth,
        args.use_first_depth_repeat,
        args.use_random_depth_gt_dist,
    ]
    if sum(ablation_flags) > 1:
        print("[ERROR] Only one ablation mode can be active: --use-gt-depth-embeddings, --use-random-depth, --use-zero-depth, --use-model-depth, --use-first-depth-repeat, or --use-random-depth-gt-dist")
        return

    if is_original_mode and sum(ablation_flags) > 0:
        print("[WARNING] Depth ablation flags were provided for an original-mode checkpoint; they will be ignored.")
        args.use_gt_depth_embeddings = False
        args.use_random_depth = False
        args.use_zero_depth = False
        args.use_model_depth = False
        args.use_first_depth_repeat = False
        args.use_random_depth_gt_dist = False
    
    gt_depth_provider: Optional[GroundTruthDepthProvider] = None
    discrete_gt_depth_provider: Optional[DiscreteGroundTruthDepthProvider] = None

    if args.use_random_depth_gt_dist:
        # Distribution-matched random depth requires the GT encoder to get per-sample stats.
        encoder_name = args.gt_depth_encoder or parse_encoder_name_from_model_path(model_path)
        if encoder_name is None:
            print("[WARNING] Could not infer encoder name from model path; disabling random-depth-gt-dist ablation.")
            args.use_random_depth_gt_dist = False
        else:
            target_len = args.gt_depth_target_num_patches or depth_token_target
            try:
                gt_depth_provider = GroundTruthDepthProvider(
                    encoder_id=encoder_name,
                    encoder_config_path=encoder_config_path,
                    interp_mode=args.gt_depth_interp_mode,
                    target_num_patches=target_len,
                    encoder_device=args.gt_depth_device,
                )
                print(f"[INFO] GT depth provider initialized for random-depth-gt-dist mode.")
            except Exception as exc:
                print(f"[WARNING] Failed to initialize GroundTruthDepthProvider for distribution-matched mode: {exc}")
                gt_depth_provider = None
                args.use_random_depth_gt_dist = False

    if args.use_gt_depth_embeddings:
        if use_discrete_depth_tokens:
            # Initialize discrete GT depth provider
            codebook_path = args.gt_depth_codebook or DEFAULT_GT_DEPTH_CODEBOOK
            discrete_depth_token_ids = getattr(model.config, 'discrete_depth_token_ids', None)
            
            if not codebook_path:
                print("[WARNING] GT discrete depth requested but no codebook was provided. Pass --gt-depth-codebook or set GT_DEPTH_CODEBOOK.")
                discrete_gt_depth_provider = None
            elif discrete_depth_token_ids is None:
                print("[WARNING] discrete_depth_token_ids not found in model config; cannot use GT discrete depth")
                discrete_gt_depth_provider = None
            else:
                print(f"[INFO] Model has {len(discrete_depth_token_ids)} discrete depth token IDs")
                print(f"[INFO] Token ID range: {min(discrete_depth_token_ids)} - {max(discrete_depth_token_ids)}")
                try:
                    discrete_gt_depth_provider = DiscreteGroundTruthDepthProvider(
                        codebook_path=codebook_path,
                        discrete_depth_token_ids=discrete_depth_token_ids,
                    )
                    print(f"[INFO] GT depth ablation enabled for discrete tokens (codebook: {codebook_path})")
                except Exception as exc:
                    print(f"[WARNING] Failed to initialize DiscreteGroundTruthDepthProvider: {exc}")
                    discrete_gt_depth_provider = None
        else:
            # Initialize continuous GT depth provider (existing code)
            encoder_name = args.gt_depth_encoder or parse_encoder_name_from_model_path(model_path)
            if encoder_name is None:
                print("[WARNING] Could not infer encoder name from model path; disabling GT depth ablation.")
            else:
                target_len = args.gt_depth_target_num_patches or depth_token_target
                try:
                    gt_depth_provider = GroundTruthDepthProvider(
                        encoder_id=encoder_name,
                        encoder_config_path=encoder_config_path,
                        interp_mode=args.gt_depth_interp_mode,
                        target_num_patches=target_len,
                        encoder_device=args.gt_depth_device,
                    )
                except Exception as exc:
                    print(f"[WARNING] Failed to initialize GroundTruthDepthProvider: {exc}")
                    gt_depth_provider = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # for line in tqdm(questions):
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        ###### NEW #######
        image = image.resize((336,336), resample = PIL.Image.NEAREST)
        ###### NEW #######
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Ensure tensors live on the same device as the model (supports multi-GPU via HF `device_map="auto"`).
        device = next(model.parameters()).device

        gt_depth_tensor: Optional[torch.Tensor] = None
        gt_discrete_token_ids: Optional[list] = None
        
        # Fetch GT depth (continuous embeddings)
        gt_depth_mean = None
        gt_depth_std = None
        if gt_depth_provider is not None:
            gt_image_path = os.path.join(args.image_folder, image_file)
            try:
                gt_tokens = gt_depth_provider.get_embeddings(gt_image_path, depth_token_target)
                gt_depth_tensor = gt_tokens.to(torch.float32)
                print(f"[GT DEPTH] Extracted GT embeddings: shape={gt_depth_tensor.shape}, dtype={gt_depth_tensor.dtype}, target_len={depth_token_target}")
                print(f"[GT DEPTH] Stats: min={gt_depth_tensor.min():.4f}, max={gt_depth_tensor.max():.4f}, mean={gt_depth_tensor.mean():.4f}")
                gt_depth_mean = gt_depth_tensor.mean().item()
                gt_depth_std = gt_depth_tensor.std().item()
                if args.use_random_depth_gt_dist:
                    print(f"[RANDOM DEPTH GT DIST] GT distribution: mean={gt_depth_mean:.4f}, std={gt_depth_std:.4f}")
            except Exception as exc:
                print(f"[WARNING] Failed to fetch GT depth embeddings for {gt_image_path}: {exc}")
                gt_depth_tensor = None
        
        # Fetch GT depth (discrete token IDs)
        if discrete_gt_depth_provider is not None:
            try:
                gt_discrete_token_ids = discrete_gt_depth_provider.get_token_ids(image_file)
                print(f"[GT DEPTH DISCRETE] Extracted GT token IDs: count={len(gt_discrete_token_ids)}")
                print(f"[GT DEPTH DISCRETE] Token IDs (first 20): {gt_discrete_token_ids[:20]}")
                print(f"[GT DEPTH DISCRETE] Token IDs (last 20): {gt_discrete_token_ids[-20:]}")
            except Exception as exc:
                print(f"[WARNING] Failed to fetch GT discrete token IDs for {image_file}: {exc}")
                gt_discrete_token_ids = None

        with torch.inference_mode():
            # print(f"Generating for question: {qs[:10]}...")
            
            # Print input token length
            input_length = input_ids.shape[1]
            print(f"[INFO] Input tokens: {input_length}, Max new tokens: {max_new_tokens}")
            
            if is_original_mode:
                # Original mode: standard text generation only.
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    use_customize_greedy=False,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            elif use_discrete_depth_tokens:
                # For discrete tokens: call generate once with use_customize_greedy=False
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    output_depth=True,
                    use_gt_depth_embeddings=False,
                    # Pass GT discrete token IDs if available
                    gt_discrete_token_ids=gt_discrete_token_ids,
                    # Pass random depth flag
                    use_random_depth=args.use_random_depth,
                    # Pass zero depth flag
                    use_zero_depth=args.use_zero_depth,
                    # Pass model depth flag (sanity check)
                    use_model_depth=args.use_model_depth,
                    # Pass first-depth-repeat flag (continuous mode only; model validates mode)
                    use_first_depth_repeat=args.use_first_depth_repeat,
                    # Disable custom greedy decoding for discrete tokens
                    use_customize_greedy=False,
                    # Pass raw hidden state parameter
                    raw_hidden_state=args.raw_hidden_state,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            else:
                # For continuous tokens: call generate once WITH custom decoding (use_customize_greedy=True)
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    output_depth=True,
                    # Pass depth token IDs for custom decoding
                    start_depth_token_id=start_depth_token_id,
                    end_depth_token_id=end_depth_token_id,
                    use_gt_depth_embeddings=bool(gt_depth_tensor is not None and not args.use_random_depth_gt_dist),
                    gt_depth_embeddings=gt_depth_tensor if not args.use_random_depth_gt_dist else None,
                    # Pass random depth flag
                    use_random_depth=args.use_random_depth,
                    # Pass zero depth flag
                    use_zero_depth=args.use_zero_depth,
                    # Pass model depth flag (sanity check)
                    use_model_depth=args.use_model_depth,
                    # Pass first-depth-repeat flag
                    use_first_depth_repeat=args.use_first_depth_repeat,
                    # Pass distribution-matched random depth flag
                    use_random_depth_gt_dist=args.use_random_depth_gt_dist,
                    gt_depth_mean=gt_depth_mean,
                    gt_depth_std=gt_depth_std,
                    # Enable custom greedy decoding for continuous tokens
                    use_customize_greedy=True,
                    # Pass raw hidden state parameter
                    raw_hidden_state=args.raw_hidden_state,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            # Process generation output
            if isinstance(generate_out, tuple) and len(generate_out) == 2:
                output_ids, depth_embeddings = generate_out
            else:
                output_ids = generate_out
                depth_embeddings = torch.empty(0)  # Empty tensor as fallback
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Print token length statistics
            # Handle both tensor and list cases
            if isinstance(output_ids, list):
                # greedy_decode returns list with only generated tokens (no input)
                generated_length = len(output_ids[0]) if output_ids else 0
                output_length = input_length + generated_length
            else:
                # Standard generate returns tensor with input + generated
                output_length = output_ids.shape[1]
                generated_length = output_length - input_length
            print(f"[INFO] Generated {generated_length} tokens (Total: {output_length}, Input: {input_length})")
            print(f"[INFO] Generated text: {outputs[:200]}...")
            if depth_embeddings.numel() > 0:
                print(f"[INFO] Depth embeddings shape: {depth_embeddings.shape}")
        
        # Save depth embeddings as numpy array only if not empty
        if depth_embeddings.numel() > 0:
            depth_embeddings_np = depth_embeddings.cpu().numpy()
            
            # Create embeddings directory based on answer file name
            answer_file_basename = os.path.splitext(os.path.basename(answers_file))[0]
            embeddings_dir = os.path.join(os.path.dirname(answers_file), f"{answer_file_basename}_embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            
            depth_embeddings_path = os.path.join(embeddings_dir, f"depth_embeddings_{idx}.npy")
            np.save(depth_embeddings_path, depth_embeddings_np)
        else:
            depth_embeddings_path = None
        
        ans_id = shortuuid.uuid()
        ans_data = {"question_id": idx,
                   "prompt": cur_prompt,
                   "text": outputs,
                   "answer_id": ans_id,
                   "model_id": model_name,
                   "metadata": {
                       "raw_hidden_state": args.raw_hidden_state,
                       "use_gt_depth": args.use_gt_depth_embeddings,
                       "use_random_depth": args.use_random_depth,
                       "use_zero_depth": args.use_zero_depth,
                       "use_model_depth": args.use_model_depth,
                       "use_first_depth_repeat": args.use_first_depth_repeat,
                       "use_random_depth_gt_dist": args.use_random_depth_gt_dist,
                   }}
        
        # Only add depth_embeddings_path if it's not None
        if depth_embeddings_path is not None:
            ans_data["depth_embeddings_path"] = depth_embeddings_path
        
        ans_file.write(json.dumps(ans_data) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--raw-hidden-state", action="store_true", help="Use raw hidden states for depth token generation")
    parser.add_argument("--encoder-config", type=str,
                        default=str(DEFAULT_ENCODER_CONFIG_PATH),
                        help="Path to encoder_config.json used for depth token metadata.")
    parser.add_argument("--use-gt-depth-embeddings", action="store_true",
                        help="Inject ground truth depth embeddings during decoding (continuous depth tokens only).")
    parser.add_argument("--use-random-depth", action="store_true",
                        help="Use random depth tokens/embeddings instead of model predictions (ablation mode).")
    parser.add_argument("--use-zero-depth", action="store_true",
                        help="Use zero depth tokens/embeddings (ablation mode for discrete/continuous depth models).")
    parser.add_argument("--use-model-depth", action="store_true",
                        help="Capture and inject model's own predictions back via override mechanism (tests injection pipeline).")
    parser.add_argument("--use-first-depth-repeat", action="store_true",
                        help="Use the first model-predicted continuous depth vector and repeat it for all remaining depth positions (ablation mode).")
    parser.add_argument("--use-random-depth-gt-dist", action="store_true",
                        help="Use random depth embeddings whose mean/std are matched to the GT depth embedding distribution (ablation mode).")
    parser.add_argument("--gt-depth-interp-mode", type=str, default="auto", choices=["auto", "linear", "bilinear"],
                        help="Interpolation mode to resize GT embeddings to the target patch count.")
    parser.add_argument("--gt-depth-target-num-patches", type=int, default=None,
                        help="Override target number of depth tokens. Defaults to encoder grid_size^2.")
    parser.add_argument("--gt-depth-encoder", type=str, default=None,
                        help="Explicit encoder id (e.g., google/siglip2-large-patch16-256). Defaults to parsing model path.")
    parser.add_argument("--gt-depth-device", type=str, default=None,
                        help="Device for the GT encoder model (e.g., cuda, cuda:1, cpu). Defaults to auto.")
    parser.add_argument("--gt-depth-codebook", type=str, default=None,
                        help="Path to discrete token codebook (.npy file). Required for GT discrete-depth injection unless GT_DEPTH_CODEBOOK is set.")
    args = parser.parse_args()

    eval_model(args)

  
