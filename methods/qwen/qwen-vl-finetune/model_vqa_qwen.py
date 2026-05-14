#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Aurora path: qwen-vl-finetune/model_vqa_qwen.py

"""
VQA inference script for Qwen2.5-VL models.
Adapted from LLaVA's model_vqa_depth.py to work with Qwen models.

Supports depth embedding ablation modes:
  --use-random-depth         Replace depth embeddings with random vectors
  --use-zero-depth           Replace depth embeddings with zeros
  --use-gt-depth             Inject ground truth depth embeddings from an encoder
  --use-model-depth          Identity sanity check (same as normal inference)
  --use-first-depth-repeat   Use first model-predicted depth vector for all remaining depth steps
  --use-random-depth-gt-dist Replace depth embeddings with random vectors matched to GT distribution
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from qwen_vl_utils import process_vision_info

# Add qwenvl to path for custom model
sys.path.insert(0, str(Path(__file__).parent))

# Import custom model with depth support
from qwenvl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwenvl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


DEFAULT_GT_DEPTH_CODEBOOK = None
CONTINUOUS_ABLATION_MODES = {"random", "zero", "gt", "model", "first_repeat", "random_gt_dist"}


# ---------------------------------------------------------------------------
# Ground-truth depth provider (reuses LLaVA encoder infrastructure)
# ---------------------------------------------------------------------------

def _try_import_gt_depth_provider():
    """Lazily import GroundTruthDepthProvider from LLaVA codebase."""
    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        llava_root = repo_root / "methods" / "llava"
    else:
        llava_root = Path(__file__).resolve().parents[2] / "LLaVA"
    if str(llava_root) not in sys.path:
        sys.path.insert(0, str(llava_root))
    from model_vqa_depth_continuous import GroundTruthDepthProvider  # type: ignore
    return GroundTruthDepthProvider


def _find_repo_root(start: Path) -> Optional[Path]:
    """Locate the vendored repo root without hardcoding a checkout path."""
    for parent in [start, *start.parents]:
        if (parent / "methods" / "llava" / "data" / "encoder_config.json").exists():
            return parent
    return None


def _resolve_encoder_config_path(config_path: Optional[str]) -> str:
    repo_root = _find_repo_root(Path(__file__).resolve())
    candidates = []

    if config_path:
        requested = Path(config_path).expanduser()
        candidates.append(requested)
        if not requested.is_absolute():
            candidates.append((Path.cwd() / requested).resolve())
            if repo_root is not None:
                candidates.append((repo_root / requested).resolve())
    elif repo_root is not None:
        candidates.append((repo_root / "methods" / "llava" / "data" / "encoder_config.json").resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(candidate) for candidate in candidates) or "<auto-detect failed>"
    raise FileNotFoundError(f"Could not resolve encoder_config.json. Checked: {searched}")


def parse_encoder_name_from_model_path(model_path: str) -> Optional[str]:
    """Infer depth encoder name from model checkpoint path."""
    # Patterns: "google_siglip2_large_patch16_256" or "openai_clip_vit_large_patch14_336"
    patterns = [
        (r'google_siglip2_large_patch16_256', 'google/siglip2-large-patch16-256'),
        (r'openai_clip_vit_large_patch14_336', 'openai/clip-vit-large-patch14-336'),
    ]
    for pattern, name in patterns:
        if re.search(pattern, model_path):
            return name
    return None


def parse_interp_size_from_model_path(model_path: str) -> Optional[int]:
    """Extract interpolation target size from model path (e.g. 'interploate_64' -> 64)."""
    m = re.search(r'interploate_(\d+)', model_path)
    if m:
        return int(m.group(1))
    return None


class ContinuousDepthLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that forces K depth tokens after <DEPTH_START> for continuous mode.
    
    This ensures the model generates exactly K <DEPTH_TOKEN>s between <DEPTH_START> 
    and <DEPTH_END>, allowing the autoregressive hidden state mechanism to work properly.
    The model's prepare_inputs_for_generation will detect these tokens and use the
    previous hidden state as the embedding (lines 2738-2743 in modeling_qwen2_5_vl.py).
    """
    
    def __init__(
        self,
        depth_start_token_id: int,
        depth_token_id: int,
        depth_end_token_id: int,
        continuous_K: int,
    ):
        self.depth_start_id = depth_start_token_id
        self.depth_token_id = depth_token_id
        self.depth_end_id = depth_end_token_id
        self.continuous_K = continuous_K
        
        # State tracking
        self.in_depth_section = False
        self.depth_token_count = 0
        self.depth_end_generated = False  # Track if we've already generated DEPTH_END
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modify logits to force depth token generation pattern.
        
        Args:
            input_ids: [batch_size, seq_len] - generated tokens so far
            scores: [batch_size, vocab_size] - logits for next token
        
        Returns:
            Modified scores that force the desired token
        """
        # Check the last generated token
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()  # Assume batch_size=1 for simplicity
            
            # Just generated <DEPTH_START> - enter depth section
            if last_token == self.depth_start_id:
                self.in_depth_section = True
                self.depth_token_count = 0
                self.depth_end_generated = False  # Reset for new depth section
            
            # Just generated <DEPTH_END> - mark it and exit depth section
            elif last_token == self.depth_end_id:
                self.in_depth_section = False
                self.depth_token_count = 0
                self.depth_end_generated = True  # Mark that we've used DEPTH_END
            
            # Currently in depth section - force the correct token sequence
            if self.in_depth_section:
                if self.depth_token_count < self.continuous_K:
                    # Force <DEPTH_TOKEN> generation
                    scores[:, :] = float('-inf')
                    scores[:, self.depth_token_id] = 0.0
                    self.depth_token_count += 1
                else:
                    # Force <DEPTH_END> generation (only once per section)
                    scores[:, :] = float('-inf')
                    scores[:, self.depth_end_id] = 0.0
                    self.in_depth_section = False
                    self.depth_token_count = 0
                    self.depth_end_generated = True
            else:
                # Not in depth section - suppress <DEPTH_END> to prevent repetition
                # Model can only generate <DEPTH_END> after <DEPTH_START> + K tokens
                if self.depth_end_generated or not self.in_depth_section:
                    scores[:, self.depth_end_id] = float('-inf')
        
        return scores


class DiscreteGroundTruthDepthProvider:
    """Provide GT discrete depth token IDs from a serialized codebook."""

    def __init__(self, codebook_path: str, discrete_depth_token_ids: List[int]):
        if not os.path.isfile(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        if not discrete_depth_token_ids:
            raise ValueError("discrete_depth_token_ids must be provided for GT depth injection.")

        self.codebook_path = codebook_path
        self.codebook = np.load(codebook_path, allow_pickle=True).item()
        self.discrete_depth_token_ids = discrete_depth_token_ids
        print(f"[GT DEPTH DISCRETE] Loaded codebook '{codebook_path}' with {len(self.codebook)} entries.")

    def _parse_token_sequence(self, token_string: str) -> List[int]:
        matches = re.findall(r"<DEPTH_(\d+)>", token_string)
        return [int(m) for m in matches]

    def _image_key(self, image_filename: str) -> str:
        base = os.path.splitext(os.path.basename(image_filename))[0]
        return f"{base}_depth.png"

    def get_token_ids(self, image_filename: str) -> List[int]:
        key = self._image_key(image_filename)
        if key not in self.codebook:
            raise KeyError(f"Image key '{key}' missing in codebook {self.codebook_path}")
        depth_levels = self._parse_token_sequence(self.codebook[key])
        if not depth_levels:
            raise ValueError(f"No discrete depth levels found for '{key}'")
        if max(depth_levels) >= len(self.discrete_depth_token_ids):
            raise ValueError(
                f"Depth level {max(depth_levels)} exceeds token mapping ({len(self.discrete_depth_token_ids)} tokens)."
            )
        return [self.discrete_depth_token_ids[level] for level in depth_levels]


class GTDiscreteDepthLogitsProcessor(LogitsProcessor):
    """Force a fixed discrete GT token sequence between depth boundary tokens."""

    def __init__(self, gt_token_ids: List[int], depth_start_id: int, depth_end_id: int):
        self.gt_token_ids = gt_token_ids
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.in_depth_mode = False
        self.depth_token_idx = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
            if self.in_depth_mode and self.depth_token_idx >= len(self.gt_token_ids):
                scores[:, :] = float("-inf")
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores

        if self.in_depth_mode and self.depth_token_idx < len(self.gt_token_ids):
            gt_token_id = self.gt_token_ids[self.depth_token_idx]
            scores[:, :] = float("-inf")
            scores[:, gt_token_id] = 0.0
            self.depth_token_idx += 1

        return scores


class RandomDiscreteDepthLogitsProcessor(LogitsProcessor):
    """Force a random discrete token sequence between depth boundary tokens."""

    def __init__(
        self,
        discrete_depth_token_ids: List[int],
        target_num_tokens: int,
        depth_start_id: int,
        depth_end_id: int,
    ):
        self.random_token_ids = torch.tensor(discrete_depth_token_ids)[
            torch.randint(0, len(discrete_depth_token_ids), (target_num_tokens,))
        ].tolist()
        self.target_num_tokens = target_num_tokens
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.in_depth_mode = False
        self.depth_token_idx = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
            if self.in_depth_mode and self.depth_token_idx >= self.target_num_tokens:
                scores[:, :] = float("-inf")
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores

        if self.in_depth_mode and self.depth_token_idx < self.target_num_tokens:
            token_id = self.random_token_ids[self.depth_token_idx]
            scores[:, :] = float("-inf")
            scores[:, token_id] = 0.0
            self.depth_token_idx += 1

        return scores


class ZeroDiscreteDepthLogitsProcessor(LogitsProcessor):
    """Force <DEPTH_0> between depth boundary tokens."""

    def __init__(self, depth_zero_token_id: int, target_num_tokens: int, depth_start_id: int, depth_end_id: int):
        self.depth_zero_token_id = depth_zero_token_id
        self.target_num_tokens = target_num_tokens
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.in_depth_mode = False
        self.depth_token_idx = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
            if self.in_depth_mode and self.depth_token_idx >= self.target_num_tokens:
                scores[:, :] = float("-inf")
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores

        if self.in_depth_mode and self.depth_token_idx < self.target_num_tokens:
            scores[:, :] = float("-inf")
            scores[:, self.depth_zero_token_id] = 0.0
            self.depth_token_idx += 1

        return scores


def get_discrete_depth_token_ids(tokenizer, num_levels: int = 128) -> List[int]:
    """Return the tokenizer IDs for <DEPTH_0>..<DEPTH_{num_levels-1}>."""
    token_ids: List[int] = []
    missing: List[str] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for i in range(num_levels):
        token = f"<DEPTH_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_id:
            missing.append(token)
        else:
            token_ids.append(int(token_id))
    if missing:
        raise ValueError(f"Missing discrete depth tokens in tokenizer: {missing[:5]}")
    return token_ids


def ensure_continuous_generation_ready(model, ablation_mode: str) -> None:
    """Fail loudly if a continuous ablation request cannot be honored."""
    if ablation_mode not in CONTINUOUS_ABLATION_MODES:
        return
    if getattr(model.config, "use_discrete_depth_tokens", False):
        raise RuntimeError(
            f"Continuous ablation mode '{ablation_mode}' was requested for a discrete-depth model."
        )

    missing = []
    continuous_k = getattr(model.config, "continuous_K", None)
    depth_start_id = getattr(model.config, "depth_start_token_id", None)
    depth_token_id = getattr(model.config, "depth_token_id", None)
    depth_end_id = getattr(model.config, "depth_end_token_id", None)

    if continuous_k is None or int(continuous_k) <= 0:
        missing.append("continuous_K")
    if depth_start_id is None:
        missing.append("depth_start_token_id")
    if depth_token_id is None:
        missing.append("depth_token_id")
    if depth_end_id is None:
        missing.append("depth_end_token_id")

    if missing:
        raise RuntimeError(
            f"Continuous ablation mode '{ablation_mode}' requested but model config is missing: {', '.join(missing)}"
        )


def ensure_continuous_ablation_state(model, ablation_mode: str) -> None:
    """Verify the expected continuous ablation flag is actually enabled on the model."""
    if ablation_mode not in CONTINUOUS_ABLATION_MODES:
        return

    expected_flag = {
        "random": "_depth_ablation_random",
        "zero": "_depth_ablation_zero",
        "gt": "_depth_ablation_gt",
        "model": "_depth_ablation_model",
        "first_repeat": "_depth_ablation_first_repeat",
        "random_gt_dist": "_depth_ablation_random_gt_dist",
    }[ablation_mode]

    if not getattr(model, expected_flag, False):
        raise RuntimeError(
            f"Continuous ablation mode '{ablation_mode}' requested but model flag '{expected_flag}' is not enabled."
        )


def validate_continuous_generation_output(
    generated_ids: torch.LongTensor,
    model,
    processor,
    ablation_mode: str,
) -> None:
    """Ensure continuous generation produced the expected visible placeholder sequence."""
    if ablation_mode not in CONTINUOUS_ABLATION_MODES:
        return
    if getattr(model.config, "use_discrete_depth_tokens", False):
        raise RuntimeError("Continuous output validation called for a discrete-depth model.")

    depth_start_id = getattr(model.config, "depth_start_token_id", None)
    depth_token_id = getattr(model.config, "depth_token_id", None)
    depth_end_id = getattr(model.config, "depth_end_token_id", None)
    continuous_k = int(getattr(model.config, "continuous_K", 0) or 0)

    start_count = int((generated_ids == depth_start_id).sum().item()) if depth_start_id is not None else 0
    token_count = int((generated_ids == depth_token_id).sum().item()) if depth_token_id is not None else 0
    end_count = int((generated_ids == depth_end_id).sum().item()) if depth_end_id is not None else 0

    if start_count != 1 or token_count != continuous_k or end_count != 1:
        decoded_text = processor.batch_decode(
            [generated_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        raise RuntimeError(
            "Continuous generation output does not match expected depth placeholder pattern "
            f"for ablation='{ablation_mode}': start={start_count}, token={token_count}, "
            f"end={end_count}, expected_token_count={continuous_k}. "
            f"Decoded prefix: {decoded_text[:200]!r}"
        )


def load_model_and_processor(
    model_id_or_path: str,
    lora_adapter: Optional[str] = None,
    merge_lora: bool = False,
    device_map: str = "auto",
    dtype: torch.dtype | str = "auto",
):
    """
    Load Qwen2.5-VL model and processor.
    
    Args:
        model_id_or_path: Path to finetuned model or HF repo
        lora_adapter: Path to LoRA adapter if not merged
        merge_lora: Whether to merge LoRA weights
        device_map: Device map for model
        dtype: Data type for model weights
    """
    print(f"Loading model from: {model_id_or_path}")
    
    # Load config first to preserve custom depth settings
    config = Qwen2_5_VLConfig.from_pretrained(model_id_or_path)
    
    # Load model with custom config
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id_or_path,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if lora_adapter:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed, but lora_adapter provided.")
        model = PeftModel.from_pretrained(model, lora_adapter)
        if merge_lora:
            model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(model_id_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)

    processor.tokenizer = tokenizer
    processor.chat_template = tokenizer.chat_template

    # Verify tokenizer vocabulary size matches model embeddings
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    
    print(f"Model loaded successfully on device: {model.device}")
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    
    if model_vocab_size != tokenizer_vocab_size:
        print(f"WARNING: Vocab size mismatch! Model has {model_vocab_size} tokens but tokenizer has {tokenizer_vocab_size}")
    
    # Display depth mode configuration
    print("\n=== DEPTH CONFIGURATION ===")
    use_discrete = getattr(model.config, 'use_discrete_depth_tokens', None)
    if use_discrete is not None:
        mode = "DISCRETE" if use_discrete else "CONTINUOUS"
        print(f"Depth Mode: {mode}")
        
        if use_discrete:
            # Check discrete tokens
            depth_tokens = ["<DEPTH_START>", "<DEPTH_0>", "<DEPTH_64>", "<DEPTH_127>", "<DEPTH_END>"]
            found_tokens = [token for token in depth_tokens if token in tokenizer.get_vocab()]
            print(f"✓ Discrete depth tokens found: {len(found_tokens)}/{len(depth_tokens)}")
            print(f"  Sample IDs: {[tokenizer.convert_tokens_to_ids(t) for t in found_tokens[:3]]}")
        else:
            # Check continuous tokens
            depth_token = "<DEPTH_TOKEN>"
            if depth_token in tokenizer.get_vocab():
                depth_token_id = tokenizer.convert_tokens_to_ids(depth_token)
                print(f"✓ Continuous <DEPTH_TOKEN> found: ID={depth_token_id}")
                print(f"  Continuous K (tokens per depth map): {getattr(model.config, 'continuous_K', 'N/A')}")
            else:
                print(f"✗ <DEPTH_TOKEN> not found in vocabulary!")
        
        # Display depth-related config
        if hasattr(model.config, 'depth_start_token_id'):
            print(f"  depth_start_token_id: {model.config.depth_start_token_id}")
            print(f"  depth_end_token_id: {getattr(model.config, 'depth_end_token_id', 'N/A')}")
        
        # Check if model has depth modules
        has_depth_head = hasattr(model, 'depth_head')
        has_depth_projector = hasattr(model, 'depth_projector')
        print(f"  Depth modules present: head={has_depth_head}, projector={has_depth_projector}")
        
        print("  Generation: LLaVA-style continuous rollout (standard generate path)")
    else:
        print("No depth configuration found (baseline model)")
    print("===========================\n")
    
    return model, processor


@torch.inference_mode()
def generate_greedy(model, processor, messages, max_new_tokens, verbose=False,
                    ablation_mode="none", gt_depth_embeddings=None,
                    gt_discrete_token_ids: Optional[List[int]] = None):
    """
    Generate answer using greedy decoding (deterministic).
    
    Supports two code-paths:
      1. **Continuous rollout**: HF ``generate()`` with
         ``ContinuousDepthLogitsProcessor``.
      2. **Standard**: plain ``model.generate()``.

    Args:
        model: Qwen2.5-VL model
        processor: Model processor
        messages: Chat-style list with mixed text and images
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print token statistics
        ablation_mode: One of "none", "random", "zero", "gt", "model", "first_repeat"
        gt_depth_embeddings: ``[K, D]`` tensor for GT ablation
    
    Returns:
        Generated answer text
    """
    # Prepare inputs using the official chat template and utils
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Check model configuration
    use_discrete = getattr(model.config, 'use_discrete_depth_tokens', False)
    continuous_K = getattr(model.config, 'continuous_K', None)
    depth_start_id = getattr(model.config, 'depth_start_token_id', None)
    depth_token_id = getattr(model.config, 'depth_token_id', None)
    depth_end_id = getattr(model.config, 'depth_end_token_id', None)

    if not use_discrete:
        ensure_continuous_generation_ready(model, ablation_mode)

    discrete_depth_token_ids = None
    if use_discrete:
        discrete_depth_token_ids = get_discrete_depth_token_ids(processor.tokenizer)

    discrete_target_tokens = int(continuous_K or 100)

    # ---- PATH 1: LogitsProcessor constrained generation (discrete token forcing) ----
    # For discrete ablations we must force token IDs, then rely on the model's
    # ordinary token embedding lookup for those IDs. Mixing this with model-side
    # embedding overrides would make the visible depth tokens diverge from the
    # embeddings actually consumed during generation.
    if use_discrete and depth_start_id is not None and depth_end_id is not None and ablation_mode in {"gt", "random", "zero"}:
        logits_processors = LogitsProcessorList()

        if ablation_mode == "gt":
            if not gt_discrete_token_ids:
                raise ValueError("Discrete GT ablation requires gt_discrete_token_ids for the current image.")
            if verbose:
                print(f"  [Using discrete GT token forcing: {len(gt_discrete_token_ids)} tokens]")
            logits_processors.append(
                GTDiscreteDepthLogitsProcessor(
                    gt_token_ids=gt_discrete_token_ids,
                    depth_start_id=depth_start_id,
                    depth_end_id=depth_end_id,
                )
            )
        elif ablation_mode == "random":
            if verbose:
                print(f"  [Using discrete random token forcing: {discrete_target_tokens} tokens]")
            logits_processors.append(
                RandomDiscreteDepthLogitsProcessor(
                    discrete_depth_token_ids=discrete_depth_token_ids,
                    target_num_tokens=discrete_target_tokens,
                    depth_start_id=depth_start_id,
                    depth_end_id=depth_end_id,
                )
            )
        else:
            if verbose:
                print(f"  [Using discrete zero token forcing: {discrete_target_tokens} tokens]")
            logits_processors.append(
                ZeroDiscreteDepthLogitsProcessor(
                    depth_zero_token_id=discrete_depth_token_ids[0],
                    target_num_tokens=discrete_target_tokens,
                    depth_start_id=depth_start_id,
                    depth_end_id=depth_end_id,
                )
            )

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            num_beams=1,
            use_cache=True,
            logits_processor=logits_processors,
        )

    # ---- PATH 2: LogitsProcessor constrained generation (continuous rollout) ----
    elif (not use_discrete and continuous_K is not None and 
        depth_start_id is not None and depth_token_id is not None and depth_end_id is not None):
        
        if verbose:
            print(f"  [Using LogitsProcessor: forcing {continuous_K} <DEPTH_TOKEN>s after <DEPTH_START>]")
        
        logits_processor = ContinuousDepthLogitsProcessor(
            depth_start_token_id=depth_start_id,
            depth_token_id=depth_token_id,
            depth_end_token_id=depth_end_id,
            continuous_K=continuous_K,
        )
        
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            temperature=0,        
            num_beams=1,
            use_cache=True,
            logits_processor=[logits_processor],
        )
    else:
        # ---- PATH 2: Standard generation ----
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            temperature=0,        
            num_beams=1,
            use_cache=True,
        )
    
    # Trim the prompt part before decoding
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]

    if not use_discrete and trimmed:
        validate_continuous_generation_output(trimmed[0], model, processor, ablation_mode)
    
    # Count depth tokens if verbose
    if verbose:
        tokenizer = processor.tokenizer
        generated_ids = trimmed[0]
        
        depth_start_id = getattr(model.config, 'depth_start_token_id', None)
        depth_end_id = getattr(model.config, 'depth_end_token_id', None)
        depth_token_id = getattr(model.config, 'depth_token_id', None)
        use_discrete = getattr(model.config, 'use_discrete_depth_tokens', None)
        
        if depth_start_id:
            start_count = (generated_ids == depth_start_id).sum().item()
            end_count = (generated_ids == depth_end_id).sum().item() if depth_end_id else 0
            
            if use_discrete:
                discrete_count = 0
                vocab = tokenizer.get_vocab()
                for i in range(128):
                    token_name = f"<DEPTH_{i}>"
                    if token_name in vocab:
                        token_id = vocab[token_name]
                        discrete_count += (generated_ids == token_id).sum().item()
                
                print(f"  [Token Stats] <DEPTH_START>: {start_count}, Discrete tokens: {discrete_count}, <DEPTH_END>: {end_count}")
            else:
                continuous_count = (generated_ids == depth_token_id).sum().item() if depth_token_id else 0
                print(f"  [Token Stats] <DEPTH_START>: {start_count}, <DEPTH_TOKEN>: {continuous_count}, <DEPTH_END>: {end_count}")
                
                decoded_text = processor.batch_decode([generated_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                text_start_count = decoded_text.count('<DEPTH_START>')
                text_token_count = decoded_text.count('<DEPTH_TOKEN>')
                text_end_count = decoded_text.count('<DEPTH_END>')
                
                if (start_count != text_start_count or continuous_count != text_token_count or end_count != text_end_count):
                    print(f"  [WARNING] Token ID counts don't match decoded text!")
                    print(f"    Decoded text shows: <DEPTH_START>: {text_start_count}, <DEPTH_TOKEN>: {text_token_count}, <DEPTH_END>: {text_end_count}")
                    print(f"    Token IDs used: START={depth_start_id}, TOKEN={depth_token_id}, END={depth_end_id}")
    
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return out_text[0]


def generate_answer(model, processor, image_path: str, prompt: str,
                    max_new_tokens: int = 2048, verbose: bool = False,
                    ablation_mode: str = "none",
                    gt_depth_embeddings=None,
                    gt_discrete_token_ids: Optional[List[int]] = None):
    """
    Generate answer for a single image-question pair.
    
    Args:
        model: Qwen2.5-VL model
        processor: Model processor
        image_path: Path to image file
        prompt: Question text
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print token statistics
        ablation_mode: Ablation mode string ("none"/"random"/"zero"/"gt"/"model"/"first_repeat")
        gt_depth_embeddings: Tensor [K, D] for GT ablation
    
    Returns:
        Generated answer text
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare messages in chat format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    
    return generate_greedy(model, processor, messages, max_new_tokens,
                           verbose=verbose, ablation_mode=ablation_mode,
                           gt_depth_embeddings=gt_depth_embeddings,
                           gt_discrete_token_ids=gt_discrete_token_ids)


def eval_model(args):
    """
    Main evaluation function.
    Reads questions from JSONL file, generates answers, and saves to output file.
    Supports depth embedding ablation modes.
    """
    # ---- Validate ablation flags ----
    ablation_flags = [
        getattr(args, 'use_random_depth', False),
        getattr(args, 'use_zero_depth', False),
        getattr(args, 'use_gt_depth', False),
        getattr(args, 'use_model_depth', False),
        getattr(args, 'use_first_depth_repeat', False),
        getattr(args, 'use_random_depth_gt_dist', False),
    ]
    if sum(ablation_flags) > 1:
        print("[ERROR] Only one ablation mode can be active: --use-random-depth, --use-zero-depth, --use-gt-depth, --use-model-depth, --use-first-depth-repeat, or --use-random-depth-gt-dist")
        return
    
    # Determine ablation mode string
    if args.use_random_depth:
        ablation_mode = "random"
    elif args.use_zero_depth:
        ablation_mode = "zero"
    elif args.use_gt_depth:
        ablation_mode = "gt"
    elif args.use_model_depth:
        ablation_mode = "model"
    elif args.use_first_depth_repeat:
        ablation_mode = "first_repeat"
    elif args.use_random_depth_gt_dist:
        ablation_mode = "random_gt_dist"
    else:
        ablation_mode = "none"
    
    print(f"\n{'='*80}")
    print(f"DEPTH ABLATION MODE: {ablation_mode.upper()}")
    print(f"{'='*80}\n")
    
    # Load model and processor
    dtype = torch.bfloat16 if torch.cuda.is_available() and args.dtype == "bfloat16" else "auto"
    model, processor = load_model_and_processor(
        args.model_path,
        lora_adapter=args.lora_adapter,
        merge_lora=args.merge_lora,
        dtype=dtype,
        device_map="auto"
    )
    
    # ---- Initialize GT depth provider if needed ----
    use_discrete = getattr(model.config, 'use_discrete_depth_tokens', False)
    if use_discrete and ablation_mode in ("first_repeat", "random_gt_dist"):
        print(f"[WARNING] Ablation mode '{ablation_mode}' is only supported for continuous depth models. Falling back to normal inference.")
        ablation_mode = "none"
    if use_discrete and ablation_mode == "model":
        print("[INFO] Discrete model-depth ablation is identical to normal inference. Using normal inference.")
        ablation_mode = "none"
    if use_discrete and ablation_mode in {"gt", "random", "zero"}:
        print("[INFO] Discrete ablation uses forced depth token IDs and the corresponding learned token embeddings.")
    if not use_discrete:
        ensure_continuous_generation_ready(model, ablation_mode)

    gt_depth_provider = None
    discrete_gt_provider = None
    if ablation_mode == "gt" and use_discrete:
        try:
            if not args.gt_depth_codebook:
                raise ValueError(
                    "Discrete GT ablation requires --gt-depth-codebook because the release snapshot "
                    "does not bake in a machine-local default."
                )
            discrete_depth_token_ids = get_discrete_depth_token_ids(processor.tokenizer)
            discrete_gt_provider = DiscreteGroundTruthDepthProvider(
                codebook_path=os.path.expanduser(args.gt_depth_codebook),
                discrete_depth_token_ids=discrete_depth_token_ids,
            )
            print(f"[GT DEPTH DISCRETE] Initialized provider: codebook={args.gt_depth_codebook}")
        except Exception as exc:
            print(f"[WARNING] Failed to initialize discrete GT depth provider: {exc}")
            print("[WARNING] Falling back to normal inference.")
            ablation_mode = "none"
    elif ablation_mode in ("gt", "random_gt_dist"):
        encoder_name = args.gt_depth_encoder or parse_encoder_name_from_model_path(args.model_path)
        if encoder_name is None:
            raise RuntimeError(
                f"Could not infer encoder name from model path for continuous ablation mode '{ablation_mode}'."
            )
        else:
            # Resolve the shared encoder registry from the vendored repo layout
            # so release users do not need to edit a user-specific absolute path.
            encoder_config_path = _resolve_encoder_config_path(args.gt_depth_encoder_config)
            interp_size = args.gt_depth_target_num_patches or parse_interp_size_from_model_path(args.model_path)
            # Fall back to continuous_K from model config
            if interp_size is None:
                interp_size = getattr(model.config, 'continuous_K', None)
            
            try:
                GroundTruthDepthProvider = _try_import_gt_depth_provider()
                gt_depth_provider = GroundTruthDepthProvider(
                    encoder_id=encoder_name,
                    encoder_config_path=encoder_config_path,
                    interp_mode=args.gt_depth_interp_mode,
                    target_num_patches=interp_size,
                    encoder_device=args.gt_depth_device,
                )
                print(f"[GT DEPTH] Initialized provider: encoder={encoder_name}, target_patches={interp_size}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize GroundTruthDepthProvider for continuous ablation mode '{ablation_mode}': {exc}"
                ) from exc
    
    # Legacy flags retained for CLI compatibility. Generation always uses the
    # standard LLaVA-style continuous rollout path.
    if getattr(args, 'use_depth_embed_ar', None) or getattr(args, 'no_depth_embed_ar', False):
        print("[INFO] Ignoring embed-AR override flags; using standard continuous rollout generation.")
    
    # ---- Set ablation mode on the model ----
    model.set_depth_ablation(mode="none" if use_discrete else ablation_mode)
    if use_discrete and ablation_mode in {"gt", "random", "zero"}:
        discrete_override_flags = [
            getattr(model, "_depth_ablation_random", False),
            getattr(model, "_depth_ablation_zero", False),
            getattr(model, "_depth_ablation_gt", False),
            getattr(model, "_depth_ablation_random_gt_dist", False),
        ]
        if any(discrete_override_flags):
            raise RuntimeError(
                "Discrete token-forcing ablations must not run with model-side embedding overrides enabled."
            )
    if not use_discrete:
        ensure_continuous_ablation_state(model, ablation_mode)
    
    # Load questions
    print(f"Loading questions from: {args.question_file}")
    questions = []
    with open(args.question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    
    # Process questions
    print(f"Generating answers...")
    answers = []
    
    for question in tqdm(questions, desc="Processing questions"):
        question_id = question['question_id']
        image_file = question['image']
        prompt = question['text']
        category = question.get('category', 'unknown')
        
        # Construct full image path
        image_path = os.path.join(args.image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # ---- Per-image GT depth extraction ----
        per_image_gt_depth = None
        per_image_gt_discrete_tokens = None
        per_image_gt_mean = None
        per_image_gt_std = None
        if ablation_mode == "gt" and use_discrete and discrete_gt_provider is not None:
            try:
                per_image_gt_discrete_tokens = discrete_gt_provider.get_token_ids(image_file)
                if args.verbose:
                    print(f"  [GT DEPTH DISCRETE] Loaded {len(per_image_gt_discrete_tokens)} tokens for {image_file}")
            except Exception as exc:
                print(f"  [WARNING] Failed to get discrete GT depth for {image_file}: {exc}")
                per_image_gt_discrete_tokens = None
        elif ablation_mode in ("gt", "random_gt_dist") and gt_depth_provider is not None:
            try:
                continuous_K = getattr(model.config, 'continuous_K', 64)
                gt_tokens = gt_depth_provider.get_embeddings(image_path, continuous_K)
                gt_tokens_f = gt_tokens.to(torch.float32)
                per_image_gt_mean = gt_tokens_f.mean().item()
                per_image_gt_std = gt_tokens_f.std().item()
                if ablation_mode == "gt":
                    per_image_gt_depth = gt_tokens_f
                    model._gt_depth_embeddings_seq = per_image_gt_depth
                    model._gt_depth_idx = 0
                else:
                    # random_gt_dist: store distribution stats, not the embeddings
                    model._gt_depth_mean = per_image_gt_mean
                    model._gt_depth_std = per_image_gt_std
                if args.verbose:
                    print(f"  [GT DEPTH] Extracted embeddings for {image_file}: shape={gt_tokens.shape}, mean={per_image_gt_mean:.4f}, std={per_image_gt_std:.4f}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to get GT depth for image '{image_file}' under continuous ablation mode '{ablation_mode}': {exc}"
                ) from exc
        
        try:
            # Generate answer
            answer_text = generate_answer(
                model, 
                processor, 
                image_path, 
                prompt, 
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
                ablation_mode=ablation_mode,
                gt_depth_embeddings=per_image_gt_depth,
                gt_discrete_token_ids=per_image_gt_discrete_tokens,
            )
            
            # Store answer
            answer_entry = {
                'question_id': question_id,
                'text': answer_text,
                'category': category,
                'image': image_file,
                'prompt': prompt if args.save_prompt else None,
            }
            
            # Add ablation metadata
            if ablation_mode != "none":
                answer_entry['ablation_mode'] = ablation_mode
            
            # Remove None values
            answer_entry = {k: v for k, v in answer_entry.items() if v is not None}
            
            answers.append(answer_entry)
            
            if args.verbose:
                print(f"\nQ{question_id}: {prompt[:100]}...")
                print(f"A: {answer_text[:200]}...")
        
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue
    
    # Clear ablation state
    model.clear_depth_ablation()
    
    # Save answers
    print(f"\nSaving {len(answers)} answers to: {args.answers_file}")
    with open(args.answers_file, 'w') as f:
        for answer in answers:
            f.write(json.dumps(answer) + '\n')
    
    print("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Run VQA evaluation on Qwen2.5-VL model")
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to finetuned Qwen2.5-VL model"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter (if not merged)"
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights before inference"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Model dtype"
    )
    
    # Data arguments
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to JSONL file containing questions"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="Path to save answers JSONL file"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print questions and answers"
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save prompts in answer file"
    )
    
    # Depth ablation arguments
    parser.add_argument(
        "--use-random-depth",
        action="store_true",
        help="Replace depth embeddings with random vectors (ablation mode)"
    )
    parser.add_argument(
        "--use-zero-depth",
        action="store_true",
        help="Replace depth embeddings with zeros (ablation mode)"
    )
    parser.add_argument(
        "--use-gt-depth",
        action="store_true",
        help="Inject ground truth depth embeddings from encoder (ablation mode)"
    )
    parser.add_argument(
        "--use-model-depth",
        action="store_true",
        help="Identity sanity check: use model's own predictions (should match baseline)"
    )
    parser.add_argument(
        "--use-first-depth-repeat",
        action="store_true",
        help="Use first model-predicted depth vector and repeat it for all remaining depth steps (continuous mode)"
    )
    parser.add_argument(
        "--use-random-depth-gt-dist",
        action="store_true",
        help="Replace depth embeddings with random vectors whose mean/std match the GT depth embedding distribution (ablation mode)"
    )
    
    # GT depth arguments (only used when --use-gt-depth or --use-random-depth-gt-dist is set)
    parser.add_argument(
        "--gt-depth-encoder",
        type=str,
        default=None,
        help="Explicit encoder id (e.g., google/siglip2-large-patch16-256). Defaults to parsing model path."
    )
    parser.add_argument(
        "--gt-depth-encoder-config",
        type=str,
        default=None,
        help="Optional path to encoder_config.json used for depth token metadata. "
             "If omitted, the script auto-detects methods/llava/data/encoder_config.json."
    )
    parser.add_argument(
        "--gt-depth-interp-mode",
        type=str,
        default="auto",
        choices=["auto", "linear", "bilinear"],
        help="Interpolation mode for GT depth embeddings."
    )
    parser.add_argument(
        "--gt-depth-target-num-patches",
        type=int,
        default=None,
        help="Override target number of depth tokens. Defaults to encoder grid_size^2 or continuous_K."
    )
    parser.add_argument(
        "--gt-depth-device",
        type=str,
        default=None,
        help="Device for the GT encoder model (e.g., cuda, cuda:1, cpu). Defaults to auto."
    )
    parser.add_argument(
        "--gt-depth-codebook",
        type=str,
        default=DEFAULT_GT_DEPTH_CODEBOOK,
        help="Path to the discrete GT depth token codebook (.npy) used for discrete GT ablation.",
    )
    
    # Legacy embed-AR overrides (kept for backward-compatible CLI parsing)
    parser.add_argument(
        "--use-depth-embed-ar",
        action="store_true",
        default=None,
        help="Legacy flag; ignored. Generation uses standard continuous rollout."
    )
    parser.add_argument(
        "--no-depth-embed-ar",
        action="store_true",
        default=False,
        help="Legacy flag; ignored. Generation uses standard continuous rollout."
    )
    
    args = parser.parse_args()
    
    eval_model(args)


if __name__ == "__main__":
    main()
