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
  --use-gt-depth-permuted    Inject GT depth embeddings with the slots shuffled by one fixed
                             permutation (slot shuffle; continuous models)
  --use-gt-depth-permuted-discrete
                             Force GT depth codes with the slots shuffled by one fixed
                             permutation (slot shuffle; discrete models)
  --disable-kv-cache         Disable the KV cache during generation
  --controlled-kv-off        Use the KV cache until <DEPTH_START>, then recompute every step
                             (continuous models)
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
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)
from qwen_vl_utils import process_vision_info

# Add qwenvl to path for custom model
sys.path.insert(0, str(Path(__file__).parent))

# Import custom model with depth support
from qwenvl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwenvl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
# Slot-permutation helpers for the slot-shuffle arms (same module as the LLaVA drivers').
from qwenvl import gt_spatial as _gts

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


DEFAULT_GT_DEPTH_CODEBOOK = None
CONTINUOUS_ABLATION_MODES = {"random", "zero", "gt", "model", "first_repeat", "random_gt_dist",
                             "gt_permuted"}

# Slot shuffle on the discrete span: the same operator as `gt_permuted`, applied to the
# 10x10 grid of VQ-VAE code ids that the discrete GT arm forces. A separate mode string,
# deliberately not in CONTINUOUS_ABLATION_MODES, so the continuous arm keeps being refused
# on a discrete checkpoint and vice versa.
DISCRETE_GT_PERMUTED_MODE = "gt_permuted_discrete"
# The discrete span is the VQ-VAE's 10x10 code grid. It is not `config.continuous_K`: on
# the discrete checkpoints that field is leftover template state (256).
DISCRETE_DEPTH_GRID = 10
DISCRETE_DEPTH_K = DISCRETE_DEPTH_GRID * DISCRETE_DEPTH_GRID
# The acceptance bar of the K=64 permutation (60 of 64 slots moved), scaled to this grid.
DISCRETE_PERM_MIN_MOVED = int(round(
    _gts.PERM_MIN_MOVED / _gts.K_DEFAULT * DISCRETE_DEPTH_K))
# Discrete arms whose span is forced token by token through a LogitsProcessor.
DISCRETE_FORCED_MODES = ("gt", "random", "zero", DISCRETE_GT_PERMUTED_MODE)


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


class StopOnTokenCriteria(StoppingCriteria):
    """Stop generation immediately after a specific token is emitted."""

    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] == 0:
            return False
        return input_ids[0, -1].item() == self.token_id


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
        "gt_permuted": "_depth_ablation_gt_permuted",
    }[ablation_mode]

    if not getattr(model, expected_flag, False):
        raise RuntimeError(
            f"Continuous ablation mode '{ablation_mode}' requested but model flag '{expected_flag}' is not enabled."
        )


def write_permutation_json(answers_file: str, gt_spatial: dict) -> str:
    """Record the fixed slot permutation of a slot-shuffle arm beside its answers.

    Chunked runs of one arm write into the same dir, so an existing file is compared
    rather than overwritten: runs that disagree would mean the arm's rows were not all
    produced under one permutation, and that is a hard failure.
    """
    out_dir = os.path.dirname(os.path.abspath(answers_file))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "_permutation.json")
    rec = {
        "mode": gt_spatial["mode"],
        "perm_seed": gt_spatial["perm_seed"],
        "perm": list(gt_spatial["perm"]),
        "perm_sha256": gt_spatial["perm_sha"],
        "grid": gt_spatial["grid"],
        "K": gt_spatial["K"],
        "builder": ("gt_spatial.build_slot_permutation"
                    "(k=K, seed=perm_seed, grid=grid, min_moved=min_moved_cheb, "
                    "min_cheb=min_chebyshev)"),
        "properties": gt_spatial["perm_facts"],
        # The acceptance bar the permutation was drawn under (60 for K=64; scaled for
        # the discrete 10x10 grid).
        "min_moved_cheb": gt_spatial.get("perm_min_moved", _gts.PERM_MIN_MOVED),
        "min_chebyshev": _gts.PERM_MIN_CHEBYSHEV,
    }
    if os.path.exists(path):
        prev = json.load(open(path))
        if (prev.get("perm_sha256") != rec["perm_sha256"]
                or prev.get("perm_seed") != rec["perm_seed"]):
            raise RuntimeError(
                f"{path} already records permutation seed {prev.get('perm_seed')} / sha "
                f"{prev.get('perm_sha256')}, but this run would inject seed "
                f"{rec['perm_seed']} / sha {rec['perm_sha256']}. All answers of one arm "
                "must be produced under one permutation.")
        return path
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[GT SPATIAL] wrote {path} (seed {rec['perm_seed']}, "
          f"sha {rec['perm_sha256'][:12]})")
    return path


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
                    gt_discrete_token_ids: Optional[List[int]] = None,
                    discrete_span_length: Optional[int] = None,
                    use_cache: bool = True):
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
        discrete_span_length: Number of depth codes to force between ``<DEPTH_START>``
            and ``<DEPTH_END>`` for the discrete random/zero ablations (the GT arms force
            their own sequence). Required for those modes; the caller sources it from the
            per-image GT code sequence or ``--discrete-span-length``.
        use_cache: Whether to use the generation KV cache
    
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

    # The discrete forced span length is supplied by the caller (per-image GT code
    # count, or --discrete-span-length). It is deliberately not read from
    # config.continuous_K: that field is continuous-only and on discrete checkpoints
    # it is leftover template state (256), which made the random/zero arms force
    # 2.56x more codes than identity/GT emit.
    discrete_target_tokens = int(discrete_span_length) if discrete_span_length else None

    # ---- PATH 1: LogitsProcessor constrained generation (discrete token forcing) ----
    # For discrete ablations we must force token IDs, then rely on the model's
    # ordinary token embedding lookup for those IDs. Mixing this with model-side
    # embedding overrides would make the visible depth tokens diverge from the
    # embeddings actually consumed during generation.
    if use_discrete and depth_start_id is not None and depth_end_id is not None and ablation_mode in DISCRETE_FORCED_MODES:
        logits_processors = LogitsProcessorList()

        if ablation_mode in {"random", "zero"} and not discrete_target_tokens:
            raise ValueError(
                f"Discrete '{ablation_mode}' ablation requires discrete_span_length "
                "(the per-image GT code count, or --discrete-span-length) so the forced "
                "span is length-matched to identity/GT."
            )

        # The slot-shuffle arm rides the GT processor: eval_model hands it the GT code
        # sequence already reordered by the fixed slot permutation.
        if ablation_mode in ("gt", DISCRETE_GT_PERMUTED_MODE):
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
            use_cache=use_cache,
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
            use_cache=use_cache,
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
            use_cache=use_cache,
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


@torch.inference_mode()
def generate_greedy_kv_off(
    model,
    processor,
    messages,
    max_new_tokens,
    verbose=False,
    ablation_mode="none",
    gt_depth_embeddings=None,
    controlled_kv_off=False,
):
    """Greedy decoding with the KV cache off: the full sequence is recomputed every step.

    Continuous depth models only. With ``controlled_kv_off`` the prefix through
    ``<DEPTH_START>`` is first generated with normal cached generation, and only the
    depth span and the answer are recomputed without the cache (``--controlled-kv-off``).
    Without it every step is recomputed (``--disable-kv-cache``). The depth-span
    ablations are applied in depth space and projected through the model's bottleneck,
    as on the cached path.
    """
    if getattr(model.config, "use_discrete_depth_tokens", False):
        raise ValueError("The KV-cache-off decode supports continuous depth models only.")

    # ---- 1. Prepare inputs using the standard processor pipeline ----
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    input_ids = inputs["input_ids"]               # [1, L]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    # ---- 2. Depth configuration ----
    continuous_K = int(getattr(model.config, "continuous_K", 0) or 0)
    depth_start_id = getattr(model.config, "depth_start_token_id", None)
    depth_token_id = getattr(model.config, "depth_token_id", None)
    depth_end_id = getattr(model.config, "depth_end_token_id", None)
    # Stop on the union of the config and generation_config eos ids, as model.generate
    # does (the config alone may carry only one of them).
    eos_token_id = []
    for _src in (model.config.eos_token_id,
                 getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if isinstance(_src, int):
            _src = [_src]
        for _t in (_src or []):
            if _t not in eos_token_id:
                eos_token_id.append(_t)

    # ---- 3. Cached prefix generation (controlled_kv_off only) ----
    # Run before any other GPU work so the cached prefix matches the standard cached
    # path (generate_greedy).
    cached_prefix_ids: list = []
    start_in_depth = False
    if controlled_kv_off:
        if depth_start_id is None:
            return generate_greedy(
                model,
                processor,
                messages,
                max_new_tokens,
                verbose=verbose,
                ablation_mode=ablation_mode,
                gt_depth_embeddings=gt_depth_embeddings,
                use_cache=True,
            )

        if verbose:
            print("[CONTROLLED_KV_OFF] Generating cached prefix through <DEPTH_START>")

        logits_processor = None
        if continuous_K and depth_token_id is not None and depth_end_id is not None:
            logits_processor = [ContinuousDepthLogitsProcessor(
                depth_start_token_id=depth_start_id,
                depth_token_id=depth_token_id,
                depth_end_token_id=depth_end_id,
                continuous_K=continuous_K,
            )]

        cached_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            num_beams=1,
            use_cache=True,
            logits_processor=logits_processor,
            stopping_criteria=StoppingCriteriaList([StopOnTokenCriteria(depth_start_id)]),
        )
        prompt_len = input_ids.shape[1]
        cached_generated = cached_out[0, prompt_len:]
        if cached_generated.numel() == 0:
            return ""

        cached_prefix_ids = cached_generated.tolist()
        if cached_prefix_ids[-1] != depth_start_id:
            if verbose:
                print("[CONTROLLED_KV_OFF] <DEPTH_START> was not generated; returning cached output")
            return processor.batch_decode(
                cached_generated.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        if hasattr(model, "reset_depth_token_counters"):
            model.reset_depth_token_counters()
        if hasattr(model, "_gt_depth_idx"):
            model._gt_depth_idx = 0
        if hasattr(model, "_first_depth_repeat_vec"):
            model._first_depth_repeat_vec = None

    # ---- 4. Compute mRoPE position_ids ----
    # attention_mask is passed by keyword: positionally it would land in
    # second_per_grid_ts.
    attn_2d = torch.ones_like(input_ids)
    position_ids, rope_deltas = model.get_rope_index(
        input_ids, image_grid_thw, attention_mask=attn_2d,
    )
    position_ids = position_ids.to(model.device)

    # ---- 5. Build inputs_embeds (encode images once) ----
    inputs_embeds = model.model.embed_tokens(input_ids)
    if pixel_values is not None:
        pv = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pv, grid_thw=image_grid_thw)
        n_img_tok = (input_ids == model.config.image_token_id).sum().item()
        n_img_feat = image_embeds.shape[0]
        if n_img_tok != n_img_feat:
            B, H = input_ids.shape[0], image_embeds.shape[-1]
            image_embeds = image_embeds.view(B, -1, H)
            image_embeds = image_embeds[:, : n_img_tok // B, :].contiguous()
            image_embeds = image_embeds.view(-1, H)
        img_mask = input_ids == model.config.image_token_id
        inputs_embeds = inputs_embeds.masked_scatter(
            img_mask.unsqueeze(-1).expand_as(inputs_embeds),
            image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )

    # ---- 6. Ablation vectors ----
    depth_dim = getattr(model, "depth_input_dim", None) or model.config.hidden_size
    if hasattr(model, "depth_projector") and model.depth_projector is not None:
        proj_dtype = next(model.depth_projector.parameters()).dtype
    else:
        proj_dtype = inputs_embeds.dtype

    gt_depth_seq = None
    if gt_depth_embeddings is not None:
        gt_depth_seq = gt_depth_embeddings.to(model.device).to(proj_dtype)

    ablation_zero_vec = None
    if ablation_mode == "zero" and continuous_K > 0:
        ablation_zero_vec = torch.zeros(1, depth_dim, device=model.device, dtype=proj_dtype)
    ablation_random_vecs = None
    if ablation_mode == "random" and continuous_K > 0:
        ablation_random_vecs = torch.rand(
            continuous_K, depth_dim, device=model.device, dtype=proj_dtype,
        ) * 2.0 - 1.0
    ablation_random_gt_dist_vecs = None
    if ablation_mode == "random_gt_dist" and continuous_K > 0:
        gt_mean = float(getattr(model, "_gt_depth_mean", 0.0))
        gt_std = float(getattr(model, "_gt_depth_std", 1.0))
        ablation_random_gt_dist_vecs = torch.randn(
            continuous_K, depth_dim, device=model.device, dtype=proj_dtype,
        ) * gt_std + gt_mean

    # ---- 7. Append the cached prefix to embeds/position_ids ----
    if cached_prefix_ids:
        prefix_tensor = torch.tensor([cached_prefix_ids], device=model.device, dtype=torch.long)
        prefix_embeds = model.model.embed_tokens(prefix_tensor)
        inputs_embeds = torch.cat([inputs_embeds, prefix_embeds], dim=1)
        pos_increments = torch.arange(
            1,
            len(cached_prefix_ids) + 1,
            device=model.device,
            dtype=position_ids.dtype,
        ).view(1, 1, -1)
        position_ids = torch.cat([position_ids, position_ids[:, :, -1:] + pos_increments], dim=-1)
        start_in_depth = cached_prefix_ids[-1] == depth_start_id

        if verbose:
            print(
                "[CONTROLLED_KV_OFF] Cached prefix complete; disabling KV cache "
                f"after {len(cached_prefix_ids)} generated token(s)"
            )

    # ---- 8. Decode loop (full recompute, no KV cache) ----
    generated_ids: list = list(cached_prefix_ids)
    in_depth = start_in_depth
    past_depth_end = False
    depth_count = 0
    depth_embeds_list: list = []
    random_depth_idx = 0
    random_gt_dist_idx = 0
    prev_pred_depth_vec = None
    first_repeat_vec = None

    tag = "CONTROLLED_KV_OFF" if controlled_kv_off else "NO_KV"
    if verbose:
        print(f"[{tag}] Prefill: {inputs_embeds.shape[1]} tokens, target_depth={continuous_K}")

    for step in range(len(generated_ids), max_new_tokens):
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device),
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )

        hidden_states = outputs.last_hidden_state          # [1, L, H]
        logits = model.lm_head(hidden_states[:, -1:, :])   # [1, 1, V]
        logits = logits.squeeze(1)                          # [1, V]

        # --- depth token forcing ---
        if in_depth and depth_count < continuous_K:
            next_token_id = depth_token_id
            depth_count += 1
        elif in_depth and depth_count >= continuous_K:
            next_token_id = depth_end_id
            in_depth = False
            past_depth_end = True
        else:
            # Outside the forced depth span, <DEPTH_END> should never be emitted.
            if depth_end_id is not None:
                logits[:, depth_end_id] = float("-inf")
            next_token_id = torch.argmax(logits, dim=-1).item()

        if next_token_id == depth_start_id and not in_depth and not past_depth_end:
            in_depth = True
            depth_count = 0

        generated_ids.append(next_token_id)

        # --- compute embedding for the new token ---
        if in_depth and depth_count > 0:
            h = hidden_states[:, -1, :]                # [1, H]
            override_depth_vec = None
            if ablation_mode == "zero" and ablation_zero_vec is not None:
                override_depth_vec = ablation_zero_vec
            elif ablation_mode == "random" and ablation_random_vecs is not None:
                idx = min(random_depth_idx, ablation_random_vecs.shape[0] - 1)
                override_depth_vec = ablation_random_vecs[idx : idx + 1]
                random_depth_idx += 1
            elif ablation_mode == "random_gt_dist" and ablation_random_gt_dist_vecs is not None:
                idx = min(random_gt_dist_idx, ablation_random_gt_dist_vecs.shape[0] - 1)
                override_depth_vec = ablation_random_gt_dist_vecs[idx : idx + 1]
                random_gt_dist_idx += 1
            elif ablation_mode == "gt" and gt_depth_seq is not None:
                gt_idx = depth_count - 1
                if gt_idx < gt_depth_seq.shape[0]:
                    override_depth_vec = gt_depth_seq[gt_idx : gt_idx + 1]
            elif ablation_mode == "model" and prev_pred_depth_vec is not None:
                override_depth_vec = prev_pred_depth_vec
            elif ablation_mode == "first_repeat" and first_repeat_vec is not None:
                override_depth_vec = first_repeat_vec

            if override_depth_vec is not None:
                projected, depth_vec = model._apply_depth_bottleneck(
                    h, override_depth_vec=override_depth_vec,
                )
            else:
                projected, depth_vec = model._apply_depth_bottleneck(h)

            if ablation_mode == "model":
                prev_pred_depth_vec = depth_vec.detach().clone()
            if ablation_mode == "first_repeat" and first_repeat_vec is None:
                first_repeat_vec = depth_vec.detach().clone()

            new_embed = projected.unsqueeze(1)              # [1, 1, H]
            depth_embeds_list.append(depth_vec.detach())
        else:
            tok = torch.tensor([[next_token_id]], device=model.device)
            new_embed = model.model.embed_tokens(tok)       # [1, 1, H]

        # --- extend sequences ---
        inputs_embeds = torch.cat([inputs_embeds, new_embed], dim=1)
        next_pos = position_ids[:, :, -1:] + 1
        position_ids = torch.cat([position_ids, next_pos], dim=-1)

        # --- stopping criteria ---
        if not in_depth and next_token_id in (eos_token_id or []):
            break
        if step >= max_new_tokens - 1:
            break

    if verbose:
        print(f"[{tag}] Done. generated={len(generated_ids)}, "
              f"depth_embeds={len(depth_embeds_list)}, "
              f"final_len={inputs_embeds.shape[1]}")

    gen_tensor = torch.tensor(generated_ids, device=model.device).unsqueeze(0)
    validate_continuous_generation_output(gen_tensor[0], model, processor, ablation_mode)
    out_text = processor.batch_decode(gen_tensor, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
    return out_text[0]


def generate_answer(model, processor, image_path: str, prompt: str,
                    max_new_tokens: int = 2048, verbose: bool = False,
                    ablation_mode: str = "none",
                    gt_depth_embeddings=None,
                    gt_discrete_token_ids: Optional[List[int]] = None,
                    discrete_span_length: Optional[int] = None,
                    disable_kv_cache: bool = False,
                    controlled_kv_off: bool = False):
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
        discrete_span_length: Depth codes to force for the discrete random/zero ablations
        disable_kv_cache: Disable the KV cache during generation
        controlled_kv_off: Use cached generation until <DEPTH_START>, then disable the KV cache
    
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
    
    use_discrete = getattr(model.config, "use_discrete_depth_tokens", False)
    if (disable_kv_cache or controlled_kv_off) and not use_discrete:
        return generate_greedy_kv_off(
            model, processor, messages, max_new_tokens,
            verbose=verbose, ablation_mode=ablation_mode,
            gt_depth_embeddings=gt_depth_embeddings,
            controlled_kv_off=controlled_kv_off and not disable_kv_cache,
        )

    # first_repeat: the model keeps the first predicted depth vector in
    # `_first_depth_repeat_vec` and resets it only when generation starts with
    # `past_key_values is None`, which the cached generate path does not guarantee, so
    # the vector could carry over from the previous row. Clear it at the row boundary so
    # each row repeats its own first vector. No other ablation mode reads it.
    if ablation_mode == "first_repeat" and hasattr(model, "_first_depth_repeat_vec"):
        model._first_depth_repeat_vec = None

    return generate_greedy(model, processor, messages, max_new_tokens,
                           verbose=verbose, ablation_mode=ablation_mode,
                           gt_depth_embeddings=gt_depth_embeddings,
                           gt_discrete_token_ids=gt_discrete_token_ids,
                           discrete_span_length=discrete_span_length,
                           use_cache=not disable_kv_cache)


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
        getattr(args, 'use_gt_depth_permuted', False),
        getattr(args, 'use_gt_depth_permuted_discrete', False),
    ]
    if sum(ablation_flags) > 1:
        print("[ERROR] Only one ablation mode can be active: --use-random-depth, --use-zero-depth, --use-gt-depth, --use-model-depth, --use-first-depth-repeat, --use-random-depth-gt-dist, --use-gt-depth-permuted, or --use-gt-depth-permuted-discrete")
        return
    if getattr(args, "disable_kv_cache", False) and getattr(args, "controlled_kv_off", False):
        print("[ERROR] Use only one KV override: --disable-kv-cache or --controlled-kv-off")
        return
    # The KV-cache-off decode (continuous models) has no slot-shuffle branch, so the arm
    # would silently run the model's own vectors under the arm's name.
    if getattr(args, 'use_gt_depth_permuted', False) and (
            getattr(args, 'disable_kv_cache', False) or getattr(args, 'controlled_kv_off', False)):
        raise ValueError(
            "--use-gt-depth-permuted is implemented on the cached generate path only; the "
            "KV-cache-off decode used by --disable-kv-cache / --controlled-kv-off has no "
            "slot-shuffle branch and would run an identity pass under the arm's name."
        )
    
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
    elif args.use_gt_depth_permuted:
        ablation_mode = "gt_permuted"
    elif args.use_gt_depth_permuted_discrete:
        ablation_mode = DISCRETE_GT_PERMUTED_MODE
    else:
        ablation_mode = "none"
    
    print(f"\n{'='*80}")
    print(f"DEPTH ABLATION MODE: {ablation_mode.upper()}")
    if args.controlled_kv_off:
        print("KV CACHE: CONTROLLED_OFF")
    elif args.disable_kv_cache:
        print("KV CACHE: OFF")
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
    # Falling back to normal inference here would write an identity run into the arm's
    # answers file, where it reads as a null ablation result, so these refuse instead.
    if use_discrete and ablation_mode in ("first_repeat", "random_gt_dist", "gt_permuted"):
        raise RuntimeError(
            f"Ablation mode '{ablation_mode}' is only supported for continuous depth models"
            + (" (use --use-gt-depth-permuted-discrete on a discrete checkpoint)."
               if ablation_mode == "gt_permuted" else ".")
        )
    if not use_discrete and ablation_mode == DISCRETE_GT_PERMUTED_MODE:
        raise RuntimeError(
            f"Ablation mode '{ablation_mode}' is only supported for discrete depth models; "
            "this checkpoint is continuous (use --use-gt-depth-permuted)."
        )
    if use_discrete and args.controlled_kv_off:
        raise RuntimeError(
            "--controlled-kv-off is implemented for continuous depth models only; on a "
            "discrete checkpoint it would leave the KV cache on. Use --disable-kv-cache."
        )
    if use_discrete and ablation_mode == "model":
        print("[INFO] Discrete model-depth ablation is identical to normal inference. Using normal inference.")
        ablation_mode = "none"
    if use_discrete and ablation_mode in DISCRETE_FORCED_MODES:
        print("[INFO] Discrete ablation uses forced depth token IDs and the corresponding learned token embeddings.")
    if not use_discrete:
        ensure_continuous_generation_ready(model, ablation_mode)

    gt_depth_provider = None
    discrete_gt_provider = None
    # The discrete random/zero arms force a span as long as this image's GT code
    # sequence (the sequence the GT arm forces), so all discrete arms emit the same
    # number of codes. --discrete-span-length overrides it; the codebook is then needed
    # only by the arms that inject GT content (gt, gt_permuted_discrete).
    discrete_span_override = args.discrete_span_length
    needs_discrete_codebook = use_discrete and (
        ablation_mode in ("gt", DISCRETE_GT_PERMUTED_MODE)
        or (discrete_span_override is None and ablation_mode in ("random", "zero"))
    )
    if needs_discrete_codebook:
        try:
            if not args.gt_depth_codebook:
                raise ValueError(
                    "Discrete GT / random / zero ablations require --gt-depth-codebook because the "
                    "release snapshot does not bake in a machine-local default."
                )
            discrete_depth_token_ids = get_discrete_depth_token_ids(processor.tokenizer)
            discrete_gt_provider = DiscreteGroundTruthDepthProvider(
                codebook_path=os.path.expanduser(args.gt_depth_codebook),
                discrete_depth_token_ids=discrete_depth_token_ids,
            )
            print(f"[GT DEPTH DISCRETE] Initialized provider: codebook={args.gt_depth_codebook}")
        except Exception as exc:
            if ablation_mode not in ("gt", DISCRETE_GT_PERMUTED_MODE):
                raise RuntimeError(
                    f"Discrete '{ablation_mode}' ablation needs the GT codebook to source the "
                    f"span length (pass --discrete-span-length to override): {exc}"
                ) from exc
            raise RuntimeError(
                f"Discrete '{ablation_mode}' ablation needs the GT codebook for its span "
                f"content, and the provider failed to initialize: {exc}"
            ) from exc
    elif ablation_mode in ("gt", "random_gt_dist", "gt_permuted"):
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
                    depth_map_dir=args.gt_depth_map_dir,
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
    
    # ---- Slot-shuffle arms: one fixed permutation for the whole arm ----
    # Built once from the fixed seed in gt_spatial.py, recorded in `_permutation.json`
    # beside the answers and in every row, so the slot map a table cites is the slot map
    # the run injected.
    gt_spatial_state = None
    discrete_perm = None
    if ablation_mode == "gt_permuted":
        _k = int(getattr(model.config, "continuous_K", 0) or 0)
        if _k <= 0:
            raise RuntimeError(
                f"ablation mode '{ablation_mode}' needs config.continuous_K; this "
                f"checkpoint reports {getattr(model.config, 'continuous_K', None)!r}")
        # build_slot_permutation requires k == grid*grid (an 8x8 span).
        _perm = _gts.build_slot_permutation(k=_k, seed=_gts.PERM_SEED, grid=_gts.GRID)
        gt_spatial_state = {
            "mode": ablation_mode, "grid": _gts.GRID, "K": _k,
            "perm": list(_perm), "perm_seed": _gts.PERM_SEED,
            "perm_sha": _gts.permutation_sha(_perm),
            "perm_facts": _gts.permutation_facts(_perm, _gts.GRID),
        }
        write_permutation_json(args.answers_file, gt_spatial_state)
    elif ablation_mode == DISCRETE_GT_PERMUTED_MODE:
        # K comes from the 10x10 code grid, not from config.continuous_K. Each row's GT
        # sequence is checked against this length below.
        _dperm = _gts.build_slot_permutation(
            k=DISCRETE_DEPTH_K, seed=_gts.PERM_SEED, grid=DISCRETE_DEPTH_GRID,
            min_moved=DISCRETE_PERM_MIN_MOVED, min_cheb=_gts.PERM_MIN_CHEBYSHEV)
        _dfacts = _gts.permutation_facts(_dperm, DISCRETE_DEPTH_GRID)
        if not (_dfacts["is_permutation"] and _dfacts["is_derangement"]):
            raise RuntimeError(
                f"the discrete slot permutation is not a derangement of "
                f"0..{DISCRETE_DEPTH_K - 1}: {_dfacts}")
        discrete_perm = _dperm
        gt_spatial_state = {
            "mode": ablation_mode, "grid": DISCRETE_DEPTH_GRID, "K": DISCRETE_DEPTH_K,
            "perm": list(_dperm), "perm_seed": _gts.PERM_SEED,
            "perm_sha": _gts.permutation_sha(_dperm), "perm_facts": _dfacts,
            "perm_min_moved": DISCRETE_PERM_MIN_MOVED,
            "operand": "vqvae_code_ids",
        }
        write_permutation_json(args.answers_file, gt_spatial_state)
        print(f"[GT SPATIAL] {ablation_mode}: {DISCRETE_DEPTH_GRID}x"
              f"{DISCRETE_DEPTH_GRID} code grid, slot s <- GT[perm[s]]; seed "
              f"{_gts.PERM_SEED}, sha {gt_spatial_state['perm_sha'][:12]}, derangement, "
              f"{_dfacts['n_moved_cheb_ge']}/{_dfacts['k']} slots moved >= "
              f"{_gts.PERM_MIN_CHEBYSHEV} cells (bar {DISCRETE_PERM_MIN_MOVED})")

    # ---- Set ablation mode on the model ----
    model.set_depth_ablation(mode="none" if use_discrete else ablation_mode,
                             gt_spatial=gt_spatial_state)
    if use_discrete and ablation_mode in DISCRETE_FORCED_MODES:
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
        per_image_gt_codes = None
        per_image_gt_discrete_tokens = None
        per_image_discrete_span = discrete_span_override
        per_image_gt_mean = None
        per_image_gt_std = None
        if ablation_mode in DISCRETE_FORCED_MODES and use_discrete and discrete_gt_provider is not None:
            try:
                per_image_gt_codes = discrete_gt_provider.get_token_ids(image_file)
                # random/zero take only the LENGTH of the GT sequence, never its content.
                if ablation_mode == "gt":
                    per_image_gt_discrete_tokens = per_image_gt_codes
                if per_image_discrete_span is None:
                    per_image_discrete_span = len(per_image_gt_codes)
                if args.verbose:
                    print(f"  [GT DEPTH DISCRETE] Loaded {len(per_image_gt_codes)} tokens for {image_file}")
            except Exception as exc:
                print(f"  [WARNING] Failed to get discrete GT depth for {image_file}: {exc}")
                per_image_gt_discrete_tokens = None
                per_image_gt_codes = None
        elif ablation_mode in ("gt", "random_gt_dist", "gt_permuted") and gt_depth_provider is not None:
            try:
                continuous_K = getattr(model.config, 'continuous_K', 64)
                gt_tokens = gt_depth_provider.get_embeddings(image_path, continuous_K)
                gt_tokens_f = gt_tokens.to(torch.float32)
                per_image_gt_mean = gt_tokens_f.mean().item()
                per_image_gt_std = gt_tokens_f.std().item()
                if ablation_mode in ("gt", "gt_permuted"):
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
        
        # The discrete shuffle. Outside the fetch's try/except above on purpose: there a
        # missing GT sequence degrades to a free span, which for this arm would be an
        # identity row recorded as a shuffled one, so here it stops the run instead.
        if ablation_mode == DISCRETE_GT_PERMUTED_MODE:
            if per_image_gt_codes is None:
                raise RuntimeError(
                    f"{ablation_mode}: no GT depth codes for {image_file}; a free span would be "
                    "indistinguishable from an identity row.")
            if len(per_image_gt_codes) != DISCRETE_DEPTH_K:
                raise RuntimeError(
                    f"{ablation_mode}: {image_file} has {len(per_image_gt_codes)} GT depth "
                    f"codes, but the permutation is over the "
                    f"{DISCRETE_DEPTH_GRID}x{DISCRETE_DEPTH_GRID} grid ({DISCRETE_DEPTH_K} slots).")
            # slot s receives GT[perm[s]], on code ids.
            per_image_gt_discrete_tokens = [per_image_gt_codes[discrete_perm[s]]
                                            for s in range(DISCRETE_DEPTH_K)]
            per_image_discrete_span = DISCRETE_DEPTH_K

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
                discrete_span_length=per_image_discrete_span,
                disable_kv_cache=args.disable_kv_cache,
                controlled_kv_off=args.controlled_kv_off,
            )
            
            # Store answer
            answer_entry = {
                'question_id': question_id,
                'text': answer_text,
                'category': category,
                'image': image_file,
                'prompt': prompt if args.save_prompt else None,
            }
            
            # Add ablation metadata. Stamped on every row, "none" included, so a run that
            # ended up on the identity path is visible in the answers file.
            answer_entry['ablation_mode'] = ablation_mode
            # Slot-shuffle provenance (the full perm list is in `_permutation.json`).
            if gt_spatial_state is not None:
                answer_entry['gt_spatial'] = {k: v for k, v in gt_spatial_state.items()
                                              if k not in ("perm", "perm_facts")}
            # The span actually forced, so the length match is auditable downstream.
            if use_discrete and ablation_mode in DISCRETE_FORCED_MODES:
                answer_entry['discrete_span_length'] = (
                    len(per_image_gt_discrete_tokens)
                    if ablation_mode in ("gt", DISCRETE_GT_PERMUTED_MODE)
                    else per_image_discrete_span
                )
            # Discrete slot shuffle: the unpermuted GT codes and the codes actually forced,
            # so injected[s] == gt[perm[s]] can be checked for every slot of every row.
            if ablation_mode == DISCRETE_GT_PERMUTED_MODE:
                answer_entry['gt_depth_codes'] = list(per_image_gt_codes)
                answer_entry['injected_depth_codes'] = list(per_image_gt_discrete_tokens)
            if args.disable_kv_cache:
                answer_entry['disable_kv_cache'] = True
            if args.controlled_kv_off:
                answer_entry['controlled_kv_off'] = True
            
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
    parser.add_argument(
        "--use-gt-depth-permuted",
        action="store_true",
        help="Slot shuffle: inject this image's GT depth embeddings, but slot s receives GT[perm[s]] "
             "for one fixed permutation shared by every row (seed "
             f"{_gts.PERM_SEED}, a derangement moving >= {_gts.PERM_MIN_MOVED} of K "
             f"slots by >= {_gts.PERM_MIN_CHEBYSHEV} cells). Continuous path only; uses the "
             "same GT provider flags as --use-gt-depth and writes _permutation.json beside the answers."
    )
    parser.add_argument(
        "--use-gt-depth-permuted-discrete",
        action="store_true",
        help="Slot shuffle on the discrete span: force this image's GT depth code ids over the "
             f"{DISCRETE_DEPTH_GRID}x{DISCRETE_DEPTH_GRID} grid, but slot s receives GT[perm[s]] for "
             f"one fixed permutation (seed {_gts.PERM_SEED}, a derangement moving >= "
             f"{DISCRETE_PERM_MIN_MOVED} of {DISCRETE_DEPTH_K} slots by >= "
             f"{_gts.PERM_MIN_CHEBYSHEV} cells). Same codes and span length as --use-gt-depth; only "
             "the slot order changes. Discrete path only; reads --gt-depth-codebook."
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
    parser.add_argument(
        "--discrete-span-length",
        type=int,
        default=None,
        help="Override the number of depth codes forced between <DEPTH_START> and <DEPTH_END> "
             "for the discrete random/zero ablations. Defaults to the per-image GT code-sequence "
             "length from --gt-depth-codebook, which keeps the forced arms length-matched to "
             "identity/GT. Does not affect the GT arms or continuous models.",
    )
    parser.add_argument(
        "--gt-depth-map-dir",
        type=str,
        default=None,
        help="Directory of depth-map PNGs named '<base>_depth.png'. When set, the continuous GT "
             "depth provider encodes these maps, as the paper's oracle does; when unset it encodes "
             "the RGB image (previous behaviour).",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable the KV cache during generation (continuous models: recompute every step).",
    )
    parser.add_argument(
        "--controlled-kv-off",
        action="store_true",
        help="Generate with the KV cache until <DEPTH_START>, then recompute every step over the "
             "span and the answer (continuous models only).",
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
