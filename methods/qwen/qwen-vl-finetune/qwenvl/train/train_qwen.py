# NEW: Aurora modification relative to upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Upstream-tracked path: qwen-vl-finetune/qwenvl/train/train_qwen.py

# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
import math
import re
from typing import Dict, List, Optional
import shutil
import sys
from pathlib import Path
import contextlib, torch
import torch.distributed as dist
import deepspeed
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

import qwenvl.train.trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class, QwenVLTrainer

# <NEW>
# Use local model definition
# from transformers import (
#     Qwen2VLForConditionalGeneration,
#     Qwen2_5_VLForConditionalGeneration,
# )
from qwenvl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwenvl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers import Qwen2VLForConditionalGeneration
# </NEW>

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from deepspeed import zero as ds_zero

local_rank = None
import contextlib, torch, torch.distributed as dist

import contextlib, torch
import torch.distributed as dist
from deepspeed import zero as ds_zero
def are_embeddings_tied(model):
    """Check if input and output embeddings are tied (share memory)."""
    return model.config.tie_word_embeddings
   

def reinitialize_new_tokens(model, old_len, new_len, method):
    """Reinitialize only the embeddings for new tokens."""
    is_tied =  are_embeddings_tied(model)
    if is_deepspeed_zero3_enabled():
        params_to_gather = [model.get_input_embeddings().weight]
        params_to_gather.append(model.get_output_embeddings().weight)
        
        with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
            if dist.get_rank() == 0:
                with torch.no_grad(): 
                    # Initialize only new tokens
                    rank0_print(
                        "[Token Init]",
                        f"tied={is_tied}",
                        f"method={method}",
                        f"input_shape={tuple(model.get_input_embeddings().weight.data.shape)}",
                        f"output_shape={tuple(model.get_output_embeddings().weight.data.shape)}",
                        f"old_len={old_len}",
                        f"new_len={new_len}",
                    )
                    if method == "random":
                        model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
                        if not is_tied:
                            model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
                    
                    elif method == "average":
                        mean_input_embedding = model.get_input_embeddings().weight.data[:old_len].mean(dim=0)
                        model.get_input_embeddings().weight.data[old_len:new_len] = mean_input_embedding.unsqueeze(0).repeat(new_len - old_len, 1)
                        if not is_tied:
                            mean_output_embedding = model.get_output_embeddings().weight.data[:old_len].mean(dim=0)
                            model.get_output_embeddings().weight.data[old_len:new_len] = mean_output_embedding.unsqueeze(0).repeat(new_len - old_len, 1)
                    
                    elif method == "adaptive":
                        mean_input_val = model.get_input_embeddings().weight.data[:old_len].mean()
                        std_input_val =  model.get_input_embeddings().weight.data[:old_len].std()
                        model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=mean_input_val, std=std_input_val)
                        if not is_tied:
                            mean_output_val = model.get_output_embeddings().weight.data[:old_len].mean()
                            std_output_val =  model.get_output_embeddings().weight.data[:old_len].std()
                            model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=mean_output_val, std=std_output_val)
                    else:
                        raise ValueError(f"Unsupported initialization method: {method}. Choose from 'random', 'average', or 'adaptive'.")

                    
    else:
        with torch.no_grad():
            if method == "random": 
                model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
                if not is_tied:
                    model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=0.0, std=0.02)
            elif method == "average":
                mean_input_embedding = model.get_input_embeddings().weight.data[:old_len].mean(dim=0)
                model.get_input_embeddings().weight.data[old_len:new_len] = mean_input_embedding.unsqueeze(0).repeat(new_len - old_len, 1)
                if not is_tied:
                    mean_output_embedding = model.get_output_embeddings().weight.data[:old_len].mean(dim=0)
                    model.get_output_embeddings().weight.data[old_len:new_len] = mean_output_embedding.unsqueeze(0).repeat(new_len - old_len, 1)
            elif method == "adaptive":
                mean_input_val = model.get_input_embeddings().weight.data[:old_len].mean()
                std_input_val =  model.get_input_embeddings().weight.data[:old_len].std()
                model.get_input_embeddings().weight.data[old_len:new_len].normal_(mean=mean_input_val, std=std_input_val)
                if not is_tied:
                    mean_output_val = model.get_output_embeddings().weight.data[:old_len].mean()
                    std_output_val =  model.get_output_embeddings().weight.data[:old_len].std()
                    model.get_output_embeddings().weight.data[old_len:new_len].normal_(mean=mean_output_val, std=std_output_val)
            else:
                raise ValueError(f"Unsupported initialization method: {method}. Choose from 'random', 'average', or 'adaptive'.")

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _find_repo_root(start: Path) -> Optional[Path]:
    """Locate the vendored repo root from this training entrypoint."""
    for parent in [start, *start.parents]:
        if (parent / "methods" / "llava" / "data" / "encoder_config.json").exists():
            return parent
    return None


def _resolve_encoder_config_path(config_path: Optional[str]) -> str:
    """Resolve encoder_config.json without assuming a user-specific checkout path."""
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

def _find_latest_valid_checkpoint(output_dir: str):
    """Return the latest checkpoint dir containing trainer_state.json, or None."""
    ckpt_dirs = []
    for p in pathlib.Path(output_dir).glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        ckpt_dirs.append((step, p))

    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
    for _, ckpt in ckpt_dirs:
        if (ckpt / "trainer_state.json").exists():
            return str(ckpt)

    return None


def load_new_tokens(tokens_file: str) -> List[str]:
    """Load new tokens from a text file (one token per line)."""
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith('#'):  # Skip empty lines and comments
                tokens.append(token)
    return tokens

def add_tokens_to_tokenizer(tokenizer, new_tokens: List[str]) -> int:
    """Add new tokens to tokenizer and return number of added tokens."""
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens_filtered = [token for token in new_tokens if token not in existing_tokens]
    
    if new_tokens_filtered:
        tokenizer.add_tokens(new_tokens_filtered)
        rank0_print(f"Added {len(new_tokens_filtered)} new tokens to tokenizer")
        rank0_print(f"Skipped {len(new_tokens) - len(new_tokens_filtered)} existing tokens")
    else:
        rank0_print("All tokens already exist in the vocabulary!")
    
    return len(new_tokens_filtered)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        model.lm_head.weight.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        model.lm_head.weight.requires_grad = False
    
    if model_args.tune_embeddings:
        model.get_input_embeddings().weight.requires_grad = True
    else:
        model.get_input_embeddings().weight.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    # ------------------------------------------------------------------
    # Shared encoder configuration (load once per training run)
    # Aurora keeps encoder metadata in the LLaVA side of the repo so training,
    # evaluation, and dataset expansion all agree on K and feature_dim.
    # ------------------------------------------------------------------
    data_args.config_path = _resolve_encoder_config_path(data_args.config_path)

    with open(data_args.config_path, 'r') as cf:
        shared_cfg = json.load(cf)
    
    # Determine encoder to use (either from CLI or default)
    depth_encoder_name = data_args.depth_encoder_name or shared_cfg.get("default_encoder")
    
    # ------------------------------------------------------------------
    # Handle special variants that specify an alternate patch/grid size via
    # a name suffix e.g. "google_siglip2_large_patch16_256_interploate_64".
    # In such cases we look up the *base* encoder (before the suffix) in the
    # shared config, but override the grid_size with the value after the
    # suffix.
    # ------------------------------------------------------------------
    grid_size_override = None
    interpolate_match = re.match(r"(.+)_interploate_(\d+)$", depth_encoder_name)
    if interpolate_match is not None:
        base_encoder_name = interpolate_match.group(1)
        grid_size_override = int(interpolate_match.group(2))
        lookup_encoder_name = base_encoder_name
    else:
        lookup_encoder_name = depth_encoder_name
    
    if lookup_encoder_name not in shared_cfg.get("models", {}):
        raise ValueError(f"Encoder '{lookup_encoder_name}' not found in config models list.")
    
    depth_encoder_conf = shared_cfg["models"][lookup_encoder_name]
    
    # Derive stub (filesystem-safe identifier) and store on data_args for dataset substitution
    depth_encoder_stub = depth_encoder_name.replace("/", "_").replace("-", "_")
    data_args.depth_encoder_stub = depth_encoder_stub
    
    # Depth embedding dimension expected by model
    depth_feature_dim = depth_encoder_conf.get("feature_dim", 768)
    
    # Calculate num_depth_tokens from grid_size
    grid_size = depth_encoder_conf.get("grid_size")
    
    if grid_size_override is not None:
        # The override value directly specifies the desired number of depth tokens
        num_depth_tokens = grid_size_override
    else:
        if grid_size is None:
            raise ValueError(f"grid_size must be specified for encoder '{depth_encoder_name}' in config.")
        num_depth_tokens = grid_size ** 2
    
    # Automatically set model_args and data_args based on encoder config
    model_args.depth_input_dim = depth_feature_dim
    model_args.continuous_K = num_depth_tokens
    data_args.continuous_K = num_depth_tokens  # CRITICAL: Data loader needs this!
    data_args.use_discrete_depth_tokens = training_args.use_discrete_depth_tokens  # Pass flag to data loader
    
    rank0_print("\n" + "="*80)
    rank0_print("[DEPTH ENCODER CONFIGURATION]")
    rank0_print("="*80)
    rank0_print(f"  Depth Encoder:            {depth_encoder_name}")
    rank0_print(f"  Depth Encoder Stub:       {depth_encoder_stub}")
    rank0_print(f"  Depth Feature Dim:        {depth_feature_dim}")
    rank0_print(f"  Num Depth Tokens (K):     {num_depth_tokens}")
    rank0_print(f"  Use Discrete Tokens:      {training_args.use_discrete_depth_tokens}")
    rank0_print(f"  Lambda Depth:             {training_args.lambda_depth}")
    rank0_print(f"  Depth Head Type:          {training_args.depth_head_type}")
    rank0_print(f"  Depth Projector Type:     {training_args.depth_projector_type}")
    rank0_print(f"  Depth Normalize:          {training_args.depth_normalize}")
    rank0_print(f"  Depth Softmax:            {training_args.depth_apply_softmax}")
    rank0_print(f"  Depth Temperature:        {training_args.depth_temperature}")
    rank0_print(f"  Depth Loss Type:          {training_args.depth_loss_type}")
    rank0_print(f"  Depth Loss Layer:         {training_args.depth_loss_layer}")
    rank0_print(f"  Depth Embed-AR:           {training_args.use_depth_embed_ar}")
    if training_args.use_depth_embed_ar:
        rank0_print(f"  Embed-AR GT Ratio:        {training_args.depth_embed_ar_gt_ratio}")
    rank0_print("="*80 + "\n")
    # ------------------------------------------------------------------
    
    # Apply monkey patch for flattened data if needed (must be done before model loading)
    if data_args.data_flatten:
        rank0_print("Applying flash attention monkey patches for flattened data mode")
        replace_qwen2_vl_attention_class()
    
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        # Load config first using local config class
        config = Qwen2_5_VLConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        # Aurora vendors the model definition locally because the upstream
        # finetune package relies on transformers' stock model internals,
        # which do not contain the depth bottleneck or embed-AR logic.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        # <NEW>
        # Update model config with depth reasoning parameters
        model.config.depth_input_dim = model_args.depth_input_dim
        model.config.continuous_K = model_args.continuous_K
        model.config.lambda_depth = training_args.lambda_depth
        model.config.use_discrete_depth_tokens = training_args.use_discrete_depth_tokens
        model.config.depth_head_type = training_args.depth_head_type
        model.config.depth_projector_type = training_args.depth_projector_type
        model.config.depth_normalize = training_args.depth_normalize
        model.config.depth_apply_softmax = training_args.depth_apply_softmax
        model.config.depth_temperature = training_args.depth_temperature
        model.config.depth_loss_type = training_args.depth_loss_type
        model.config.depth_loss_layer = training_args.depth_loss_layer
        if model.config.depth_loss_layer != -1:
            num_layers = int(getattr(model.config, "num_hidden_layers", 0))
            if model.config.depth_loss_layer < 0 or model.config.depth_loss_layer >= num_layers:
                raise ValueError(
                    f"Invalid --depth_loss_layer={model.config.depth_loss_layer}. "
                    f"Expected -1 or 0..{max(num_layers - 1, 0)}"
                )
        model.config.use_depth_embed_ar = training_args.use_depth_embed_ar
        model.config.depth_embed_ar_gt_ratio = training_args.depth_embed_ar_gt_ratio
        
        # Re-initialize depth modules only for continuous mode (not discrete)
        if not training_args.use_discrete_depth_tokens:
            depth_head_type = training_args.depth_head_type
            depth_projector_type = training_args.depth_projector_type
            depth_input_dim = model.config.depth_input_dim
            hidden_size = model.config.hidden_size
            
            if depth_head_type == 'linear' and depth_projector_type == 'linear':
                # Weight tying: only need depth_projector
                model.depth_projector = torch.nn.Linear(depth_input_dim, hidden_size).to(model.device, model.dtype)
                model.depth_head = None
                rank0_print(f"[Depth Init] Using weight tying: depth_head=None, depth_projector: {depth_input_dim}->{ hidden_size}")
            else:
                # Build depth_head
                if depth_head_type == 'linear':
                    model.depth_head = torch.nn.Linear(hidden_size, depth_input_dim).to(model.device, model.dtype)
                elif depth_head_type == 'mlp':
                    model.depth_head = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, depth_input_dim),
                    ).to(model.device, model.dtype)
                elif depth_head_type == 'mlp2x_gelu':
                    model.depth_head = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, depth_input_dim),
                    ).to(model.device, model.dtype)
                else:
                    model.depth_head = torch.nn.Linear(hidden_size, depth_input_dim).to(model.device, model.dtype)
                
                # Build depth_projector
                if depth_projector_type == 'linear':
                    model.depth_projector = torch.nn.Linear(depth_input_dim, hidden_size).to(model.device, model.dtype)
                elif depth_projector_type == 'mlp':
                    model.depth_projector = torch.nn.Sequential(
                        torch.nn.Linear(depth_input_dim, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, hidden_size),
                    ).to(model.device, model.dtype)
                elif depth_projector_type == 'mlp2x_gelu':
                    model.depth_projector = torch.nn.Sequential(
                        torch.nn.Linear(depth_input_dim, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_size, hidden_size),
                    ).to(model.device, model.dtype)
                else:
                    model.depth_projector = torch.nn.Linear(depth_input_dim, hidden_size).to(model.device, model.dtype)
                
                rank0_print(f"[Depth Init] depth_head_type={depth_head_type}, depth_projector_type={depth_projector_type}")
            
            # Store depth attributes on model
            model.depth_input_dim = depth_input_dim
            model.normalize_depth = training_args.depth_normalize
            model.apply_depth_softmax = training_args.depth_apply_softmax
            model.depth_temperature = training_args.depth_temperature
            model.depth_loss_type = training_args.depth_loss_type
            model.depth_loss_layer = training_args.depth_loss_layer
            
            rank0_print(
                f"[Depth Init] Loss config: type={model.depth_loss_type}, layer={model.depth_loss_layer}, "
                f"normalize={model.normalize_depth}, softmax={model.apply_depth_softmax}, temp={model.depth_temperature}"
            )
        # </NEW>
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    ## Newly added ##
    if model_args.new_tokens_file:
        old_len = len(tokenizer)
        new_tokens = load_new_tokens(model_args.new_tokens_file)
        
        # <NEW>
        # Require depth_encoder_stub only for continuous mode (not discrete)
        if not training_args.use_discrete_depth_tokens and not data_args.depth_encoder_stub:
            raise ValueError(
                f"depth_encoder_stub must be explicitly provided when using new_tokens_file in continuous mode. "
                f"Got new_tokens_file={model_args.new_tokens_file} but depth_encoder_stub is None. "
                f"Either provide depth_encoder_stub or set use_discrete_depth_tokens=True for discrete mode."
            )
        # </NEW>

        # Following LLaVA architecture:
        # - For BOTH modes: <DEPTH_START>, <DEPTH_END>, and all depth tokens are REGULAR tokens
        # - For continuous mode: <DEPTH_TOKEN> is a placeholder (but still regular token)
        # - No depth tokens should be special tokens (matching mmseek and LLaVA)
        num_added = add_tokens_to_tokenizer(tokenizer, new_tokens)
        
        # Ensure <DEPTH_START> and <DEPTH_END> are NOT treated as special tokens
        # by removing them from the additional_special_tokens list if they were added there
        if hasattr(tokenizer, "additional_special_tokens"):
            tokenizer.additional_special_tokens = [
                t for t in tokenizer.additional_special_tokens 
                if t not in ["<DEPTH_START>", "<DEPTH_END>"]
            ]
        
        rank0_print(f"Added {num_added} depth tokens as regular tokens (matching LLaVA/mmseek)")
        
        # Update config with token IDs for depth reasoning (if tokens exist)
        if "<DEPTH_START>" in tokenizer.get_vocab():
            model.config.depth_start_token_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
        if "<DEPTH_END>" in tokenizer.get_vocab():
            model.config.depth_end_token_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
        if "<DEPTH_TOKEN>" in tokenizer.get_vocab():
            model.config.depth_token_id = tokenizer.convert_tokens_to_ids("<DEPTH_TOKEN>")
            rank0_print(f"Set depth_token_id={model.config.depth_token_id} for <DEPTH_TOKEN> placeholder")
        
        # Verify that these tokens are NOT marked as special in the tokenizer's internal state
        for token_name in ["<DEPTH_START>", "<DEPTH_END>", "<DEPTH_TOKEN>"]:
            if token_name in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                # In newer transformers, added_tokens_decoder is the source of truth for special status
                if hasattr(tokenizer, "added_tokens_decoder") and token_id in tokenizer.added_tokens_decoder:
                    is_special = tokenizer.added_tokens_decoder[token_id].special
                    assert not is_special, f"Token {token_name} (ID {token_id}) should NOT be special!"
        rank0_print("Verified: Depth tokens are registered in model config and marked as regular in tokenizer.")
        
        new_len = len(tokenizer)
        num_emb = model.get_input_embeddings().num_embeddings
        
        if new_len > num_emb:
            model.resize_token_embeddings(new_len, pad_to_multiple_of=128)
            model.config.vocab_size = model.get_input_embeddings().num_embeddings
            rank0_print(f"Resized model embeddings from {num_emb} to {model.config.vocab_size}")
        rank0_print(
            "Token resize summary:",
            model_args.reinitialization_method.lower(),
            are_embeddings_tied(model),
            old_len,
            new_len,
        )
        if model_args.reinitialization_method.lower() != "none":
            reinitialize_new_tokens(model, old_len, new_len, model_args.reinitialization_method.lower())
        if are_embeddings_tied(model):
            model.tie_weights() 
    ## Newly added ##            

    set_model(model_args, model)
    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = QwenVLTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    resume_ckpt = _find_latest_valid_checkpoint(training_args.output_dir)
    if resume_ckpt is not None:
        logging.info(f"valid checkpoint found, resume training from {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.warning(
                "checkpoint directories exist but none has trainer_state.json; "
                "starting a fresh train() in this output_dir."
            )
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    if model_args.new_tokens_file:
        rank0_print("Saving updated tokenizer...")
        tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
