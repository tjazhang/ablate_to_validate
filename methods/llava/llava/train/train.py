# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/train/train.py

import os
import json
from datetime import datetime

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

import copy
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import numpy as np
import torch
import PIL
import transformers
from transformers import TrainerCallback
import tokenizers
import re

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image




def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Depth-related model configuration parameters
    depth_apply_l1: bool = field(default=False)
    depth_normalize: bool = field(default=False)
    depth_apply_softmax: bool = field(default=False)
    depth_temperature: float = field(default=0.07)
    # NEW: Metamorph-style depth architecture options
    depth_head_type: Optional[str] = field(
        default="linear",
        metadata={"help": "Type of depth head: linear, mlp, or mlp2x_gelu"}
    )
    depth_projector_type: Optional[str] = field(
        default="linear", 
        metadata={"help": "Type of depth projector: linear, mlp, or mlp2x_gelu"}
    )
    vision_guidance_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for vision guidance loss during depth training"}
    )
    # NEW: Loss calculation mode for distributed training
    loss_mode: str = field(
        default="local",
        metadata={"help": "Loss calculation mode: 'local' (original scaling) or 'global' (true global token-weighted)"}
    )
    # NEW: Simple vision freezing option (MetaMorph-style)
    freeze_vision: bool = field(
        default=False,
        metadata={"help": "Freeze vision tower parameters (similar to MetaMorph)"}
    )
    # NEW: Depth loss type configuration
    depth_loss_type: str = field(
        default="mse",
        metadata={"help": "Type of depth loss: 'mse', 'cosine', 'l1', etc."}
    )
    depth_loss_layer: int = field(
        default=-1,
        metadata={"help": "Decoder layer index for depth loss (0-based). Use -1 for final hidden state."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    custom_image_size: int = 336
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # NEW: Encoder selection and shared config file
    # Path to shared config JSON defining available encoders and paths.
    config_path: str = field(default_factory=lambda: str(pathlib.Path(__file__).resolve().parents[2] / "data" / "encoder_config.json"),
                             metadata={"help": "Path to the shared encoder config JSON."})
    # Name of encoder to use (must exist in config['models']). If None, fallback to config['default_encoder'].
    depth_encoder_name: Optional[str] = field(default=None)
    # Populated at runtime after loading config. Holds encoder stub string.
    depth_encoder_stub: Optional[str] = field(default=None, init=False)
    # Validation dataset configuration
    eval_data_path: Optional[str] = field(default=None,
                                          metadata={"help": "Path to the evaluation/validation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    depth_mode: str = field(
        default="auto",
        metadata={"help": "Depth mode: auto|original|continuous|discrete"}
    )
    lora_enable: bool = False
    lora_r: int = 64
    # NEW: Aurora dataset switches and tokenizer behavior. These flags decide
    # whether training stays upstream-original, uses continuous depth vectors,
    # or uses Aurora discrete depth tokens.
    depth_data: bool = False
    # Legacy Aurora dataset variants kept for older experiment configs.
    annealing_data: bool = False
    coordinates_data: bool = False
    # Aurora-style discrete depth tokens configuration
    use_discrete_depth_tokens: bool = field(
        default=False,
        metadata={"help": "Use discrete depth tokens (Aurora-style) instead of continuous tokens"}
    )
    discrete_depth_supervision_mode: str = field(
        default="ground_truth",
        metadata={"help": "Discrete depth CE supervision: ground_truth|random|ignore"}
    )
    num_discrete_depth_levels: int = field(
        default=128,
        metadata={"help": "Number of discrete depth levels when using discrete tokens"}
    )
    depth_token_reinit_method: str = field(
        default="none",
        metadata={"help": "New token embedding initialization: none|random|average|adaptive"}
    )
    # NEW: Two-stage depth optimization splits projector-only warmup from
    # later LM/depth-head finetuning.
    depth_training_stage: Optional[int] = field(
        default=None,
        metadata={
            "help": "Depth training stage: 1 = train only depth_projector, 2 = train lm_head + depth_head (freeze projector)"
        }
    )
    depth_projector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for depth_projector in stage 1"}
    )
    depth_head_lr: Optional[float] = field(
        default=None, 
        metadata={"help": "Separate learning rate for depth_head in stage 2"}
    )
    depth_projector_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Weight decay for depth_projector parameters"}
    )
    depth_head_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Weight decay for depth_head parameters"}
    )
    # NEW: Vision freezing and depth training options
    freeze_vision_for_depth: bool = field(
        default=False,
        metadata={"help": "Freeze vision components when training depth"}
    )
    depth_lr_multiplier: float = field(
        default=0.1,
        metadata={"help": "Learning rate multiplier for depth components when not using separate LRs"}
    )
    depth_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient for depth loss"}
    )
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_depth: bool = field(default=False,
                                metadata={"help": "Whether to group samples by presence of depth tokens"})
    use_context_gate: bool = field(default=False,
                                  metadata={"help": "Whether to use context gate in depth processing"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    # Add depth-related modules to exclusion list for LoRA [[memory:4968535]]
    depth_keywords = ['depth_head', 'depth_projector', 'depth_context_gate']
    
    for name, module in model.named_modules():
        # Skip multimodal modules
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        # Skip depth-related modules  
        if any(depth_keyword in name for depth_keyword in depth_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def resolve_depth_mode(training_args: "TrainingArguments") -> None:
    """Normalize depth flags from a single explicit mode when requested."""
    mode = (getattr(training_args, "depth_mode", "auto") or "auto").lower()
    valid_modes = {"auto", "original", "continuous", "discrete"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported --depth-mode '{mode}'. Choose from {sorted(valid_modes)}.")

    if mode == "auto":
        return
    if mode == "original":
        training_args.depth_data = False
        training_args.use_discrete_depth_tokens = False
    elif mode == "continuous":
        training_args.depth_data = True
        training_args.use_discrete_depth_tokens = False
    elif mode == "discrete":
        training_args.depth_data = True
        training_args.use_discrete_depth_tokens = True


def reinitialize_new_token_embeddings(model, old_len: int, new_len: int, method: str) -> None:
    """Reinitialize only newly added token rows in input/output embeddings."""
    method = (method or "none").lower()
    if method == "none" or new_len <= old_len:
        return
    if method not in {"random", "average", "adaptive"}:
        raise ValueError(f"Unsupported depth_token_reinit_method='{method}'. Use none|random|average|adaptive.")

    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    if input_embed is None:
        return

    with torch.no_grad():
        input_w = input_embed.weight.data
        output_w = output_embed.weight.data if output_embed is not None else None
        is_tied = bool(output_w is not None and input_w.data_ptr() == output_w.data_ptr())

        if method == "random":
            input_w[old_len:new_len].normal_(mean=0.0, std=0.02)
            if output_w is not None and not is_tied:
                output_w[old_len:new_len].normal_(mean=0.0, std=0.02)
            return

        if method == "average":
            input_mean = input_w[:old_len].mean(dim=0, keepdim=True)
            input_w[old_len:new_len] = input_mean.repeat(new_len - old_len, 1)
            if output_w is not None and not is_tied:
                output_mean = output_w[:old_len].mean(dim=0, keepdim=True)
                output_w[old_len:new_len] = output_mean.repeat(new_len - old_len, 1)
            return

        # adaptive
        input_mean = input_w[:old_len].mean()
        input_std = input_w[:old_len].std()
        input_w[old_len:new_len].normal_(mean=input_mean, std=input_std)
        if output_w is not None and not is_tied:
            output_mean = output_w[:old_len].mean()
            output_std = output_w[:old_len].std()
            output_w[old_len:new_len].normal_(mean=output_mean, std=output_std)


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
   
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 training_args=None):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args

        # Replace placeholder in embedding paths once up-front to avoid doing it per-sample
        depth_encoder_stub = getattr(self.data_args, 'depth_encoder_stub', None)
        if depth_encoder_stub is None:
            raise ValueError("data_args.depth_encoder_stub is not set. Ensure config processing ran before dataset init.")

        for sample in list_data_dict:
            if 'embedding' in sample and '<encoder_stub>' in sample['embedding']:
                sample['embedding'] = sample['embedding'].replace('<encoder_stub>', depth_encoder_stub)

        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        # Check if we should group by depth
        group_by_depth = getattr(self.training_args, 'group_by_depth', False) if self.training_args else False
        
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            
            # Check for depth tokens in GPT responses if grouping by depth is enabled
            has_depth_tokens = False
            if group_by_depth:
                for conv in sample['conversations']:
                    if conv.get('from') == 'gpt' and any(token in conv.get('value', '') for token in ['<DEPTH_START>', '<DEPTH_END>', '<DEPTH_TOKEN>']):
                        has_depth_tokens = True
                        break
            
            # Encoding scheme:
            # - Positive: has image, no depth tokens
            # - Negative: no image, no depth tokens  
            # - Add 1000000 (large offset): has depth tokens (only if group_by_depth is True)
            if 'image' in sample:
                if has_depth_tokens and group_by_depth:
                    # Has image AND depth tokens
                    cur_len = cur_len + 1000000
                else:
                    # Has image, no depth tokens (or depth grouping disabled)
                    cur_len = cur_len
            else:
                if has_depth_tokens and group_by_depth:
                    # No image but has depth tokens
                    cur_len = -cur_len + 1000000
                else:
                    # No image, no depth tokens (or depth grouping disabled)
                    cur_len = -cur_len
                    
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        # Store original index
        original_index = i
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # print("%%%%%%%", np.array(image).shape, image_file, i, local_rank)
            # NEW: Aurora resizes training images up front so the vision tower
            # and depth encoder see a consistent patch grid.
            image = image.resize((self.data_args.custom_image_size,self.data_args.custom_image_size), resample = PIL.Image.NEAREST )
            # print(np.array(image).shape)
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # Expansion of <DEPTH_TOKEN> is handled model-side in llava_arch.py
        # to ensure exact token count alignment without tokenization artifacts.

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        ### FIXED: Only load depth embeddings for samples that actually need them ###
        # Check if using discrete depth tokens
        use_discrete_depth_tokens = getattr(self.training_args, 'use_discrete_depth_tokens', False) if self.training_args else False
        
        if use_discrete_depth_tokens:
            # Aurora-style: discrete depth tokens don't need embeddings
            data_dict['needs_depth'] = False
            data_dict['depth_embedding'] = None
        else:
            # Continuous mode: check if this sample contains continuous depth tokens
            needs_depth = False
            for conv in self.list_data_dict[i]['conversations']:
                if any(token in conv.get('value', '') for token in ['<DEPTH_TOKEN>', '<DEPTH_START>', '<DEPTH_END>']):
                    needs_depth = True
                    break
            
            # ASSERTION: If sample has depth tokens, it MUST have embedding
            if needs_depth:
                if 'embedding' not in self.list_data_dict[i]:
                    # Print detailed information before failing
                    rank0_print(f"\n[ERROR] Sample {i} has depth tokens but no 'embedding' key!")
                    rank0_print(f"Sample data keys: {list(self.list_data_dict[i].keys())}")
                    rank0_print(f"Sample conversations:")
                    for conv_idx, conv in enumerate(self.list_data_dict[i]['conversations']):
                        rank0_print(f"  [{conv_idx}] from={conv.get('from')}: {conv.get('value', '')[:200]}")
                    if 'image' in self.list_data_dict[i]:
                        rank0_print(f"Image: {self.list_data_dict[i]['image']}")
                    raise AssertionError(
                        f"Sample {i} has depth tokens but no 'embedding' key! "
                        f"All samples with depth tokens must have corresponding embeddings."
                    )
                
                try:
                    embedding_path = self.list_data_dict[i]['embedding']
                    embedding = np.load(embedding_path)
                    
                    # --- Audit Bug #3: Missing Data Shape Verification ---
                    expected_tokens = self.data_args.num_depth_tokens
                    if embedding.shape[0] != expected_tokens:
                        raise ValueError(
                            f"Depth embedding shape mismatch at sample {i}: "
                            f"got {embedding.shape[0]}, expected {expected_tokens}. "
                            f"Check grid_size configuration matches embedding files."
                        )
                    # -----------------------------------------------------
                    
                    data_dict['depth_embedding'] = torch.from_numpy(embedding).float()
                    data_dict['needs_depth'] = True
                except Exception as e:
                    rank0_print(f"\n[ERROR] Sample {i} failed to load depth embedding!")
                    rank0_print(f"Embedding path: {self.list_data_dict[i]['embedding']}")
                    rank0_print(f"Error: {e}")
                    raise RuntimeError(
                        f"Sample {i} has depth tokens but failed to load embedding from "
                        f"'{self.list_data_dict[i]['embedding']}': {e}"
                    )
            else:
                data_dict['needs_depth'] = False
                data_dict['depth_embedding'] = None
        
        # --- Audit Bug #2: Tokenization Sanity Check ---
        if data_dict.get('needs_depth', False):
            depth_token_id = self.tokenizer.convert_tokens_to_ids('<DEPTH_TOKEN>')
            depth_count = (data_dict['input_ids'] == depth_token_id).sum().item()
            if depth_count != 1:
                 # Note: If you switched to expansion in train.py, this should be num_depth_tokens
                 raise AssertionError(f"Sample {i} has needs_depth=True but {depth_count} <DEPTH_TOKEN> tokens found (expected 1 for model-side expansion).")
        # -----------------------------------------------
        
        # Add original index to the output
        data_dict['original_index'] = original_index
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # Add batch composition logging
        batch_info = {
            'batch_size': len(instances),
            'has_image': [],
            'needs_depth': [],
            'has_depth_embedding': [],
            'original_indices': []
        }
        
        # Check each instance for various modalities
        for idx, instance in enumerate(instances):
            batch_info['has_image'].append('image' in instance)
            batch_info['needs_depth'].append(instance.get('needs_depth', False))
            batch_info['has_depth_embedding'].append(instance.get('depth_embedding') is not None)
            batch_info['original_indices'].append(instance.get('original_index', -1))
        
        # Log batch composition
        if local_rank == 0:  # Only log from main process
            print(f"\n[BATCH_MONITOR] Batch composition:")
            print(f"  - Size: {batch_info['batch_size']}")
            print(f"  - With images: {sum(batch_info['has_image'])}")
            print(f"  - Need depth: {sum(batch_info['needs_depth'])}")
            print(f"  - Have depth embeddings: {sum(batch_info['has_depth_embedding'])}")
            print(f"  - Original indices: {batch_info['original_indices']}")
            
            # Check for alignment issues
            depth_mismatch = sum(batch_info['needs_depth']) != sum(batch_info['has_depth_embedding'])
            if depth_mismatch:
                print(f"  [WARNING] Depth mismatch: {sum(batch_info['needs_depth'])} need depth but {sum(batch_info['has_depth_embedding'])} have embeddings!")
            
            # Check for mixed batches (which shouldn't happen with proper grouping)
            if len(set(batch_info['has_image'])) > 1:
                print(f"  [WARNING] Mixed image/text batch detected!")
            if len(set(batch_info['needs_depth'])) > 1:
                print(f"  [WARNING] Mixed depth/non-depth batch detected!")
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  
        )
        
        ### FIXED: Only include depth embeddings for samples that actually need them and are NOT truncated ###
        depth_embeddings = []
        depth_indices = []  # Track which batch indices have depth embeddings
        depth_token_id = self.tokenizer.convert_tokens_to_ids('<DEPTH_TOKEN>')

        for idx, instance in enumerate(instances):
            if instance.get('needs_depth', False) and instance.get('depth_embedding') is not None:
                # Check for truncation: ensure all depth tokens are within model_max_length
                inst_input_ids = instance['input_ids']
                depth_positions = (inst_input_ids == depth_token_id).nonzero(as_tuple=True)[0]
                
                if len(depth_positions) > 0:
                    max_depth_pos = depth_positions.max().item()
                    if max_depth_pos < self.tokenizer.model_max_length:
                        depth_embeddings.append(instance['depth_embedding'])
                        depth_indices.append(idx)
                    else:
                        if local_rank == 0:
                            print(f"[WARNING] Sample {instance.get('original_index')} depth tokens truncated (max_pos={max_depth_pos} >= {self.tokenizer.model_max_length}). Dropping depth supervision for this sample to avoid desync.")
                else:
                    # This could happen if the depth token was already missing or mis-tokenized
                    if local_rank == 0:
                        print(f"[WARNING] Sample {instance.get('original_index')} needs depth but no <DEPTH_TOKEN> found in input_ids!")

        # Only create depth_embeddings tensor if there are samples with depth and not truncated
        if depth_embeddings:
            # --- Audit Bug #4: Batching Crash with Mixed Grid Sizes ---
            first_shape = depth_embeddings[0].shape
            for d_idx, d_embed in enumerate(depth_embeddings):
                if d_embed.shape != first_shape:
                    raise ValueError(
                        f"Depth embedding shape mismatch in batch! "
                        f"Sample at index {depth_indices[d_idx]} has shape {d_embed.shape}, "
                        f"but first sample has shape {first_shape}. "
                        "All depth embeddings in a batch must have the same shape for torch.stack."
                    )
            # ----------------------------------------------------------
            
            batch['depth_embeddings'] = torch.stack(depth_embeddings)
            batch['depth_indices'] = torch.tensor(depth_indices, dtype=torch.long)
            if local_rank == 0:
                print(f"[DEPTH_INDICES] depth_indices={depth_indices}, depth_embeddings.shape={batch['depth_embeddings'].shape}")
        else:
            batch['depth_embeddings'] = None
            batch['depth_indices'] = None

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # Ensure all ranks produce the same keys; if depth embeddings are not
        # present for this batch, create an empty placeholder tensor so that
        # downstream code and distributed synchronisation remain consistent.
        # if 'depth_embeddings' not in batch:
        #     batch['depth_embeddings'] = None

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                training_args=training_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Create evaluation dataset if eval_data_path is provided
    eval_dataset = None
    if data_args.eval_data_path is not None:
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                             data_path=data_args.eval_data_path,
                                             data_args=data_args,
                                             training_args=training_args)
        if getattr(training_args, 'local_rank', -1) in [0, -1]:
            print(f"[VALIDATION] Created evaluation dataset with {len(eval_dataset)} samples")
    
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def get_llava_config(model_args, training_args, depth_feature_dim, num_depth_tokens, use_context_gate=False):
    """Loads full multimodal config using LlavaLlamaForCausalLM."""
    config = LlavaLlamaForCausalLM.config_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    # Inject depth params
    config.depth_embedding_dim = depth_feature_dim
    config.num_depth_tokens = num_depth_tokens
    config.use_context_gate = use_context_gate
    config.use_discrete_depth_tokens = bool(getattr(training_args, "use_discrete_depth_tokens", False))
    if config.use_discrete_depth_tokens:
        config.num_discrete_depth_levels = getattr(training_args, "num_discrete_depth_levels", 128)
        # Continuous placeholder id must be unset in discrete mode.
        config.depth_token_id = None
    
    # Transfer depth-related model arguments to config
    config.depth_apply_l1 = model_args.depth_apply_l1
    config.depth_normalize = model_args.depth_normalize
    config.depth_apply_softmax = model_args.depth_apply_softmax
    config.depth_temperature = model_args.depth_temperature
    
    # NEW: Transfer metamorph-style architecture options
    config.depth_head_type = model_args.depth_head_type
    config.depth_projector_type = model_args.depth_projector_type
    config.vision_guidance_weight = model_args.vision_guidance_weight
    
    # Transfer loss mode configuration
    config.loss_mode = model_args.loss_mode
    
    # Transfer depth loss type configuration
    config.depth_loss_type = model_args.depth_loss_type
    config.depth_loss_layer = int(model_args.depth_loss_layer)
    if config.depth_loss_layer != -1:
        num_layers = int(getattr(config, "num_hidden_layers", 0))
        if config.depth_loss_layer < 0 or config.depth_loss_layer >= num_layers:
            raise ValueError(
                f"Invalid depth_loss_layer={config.depth_loss_layer}. "
                f"Expected -1 or 0..{max(num_layers - 1, 0)}"
            )
    
    # Transfer loss coefficients from training args
    if hasattr(training_args, 'depth_coef'):
        config.depth_coef = training_args.depth_coef
    if hasattr(training_args, 'discrete_depth_supervision_mode'):
        config.discrete_depth_supervision_mode = training_args.discrete_depth_supervision_mode
    if hasattr(training_args, 'depth_mode'):
        config.depth_mode = training_args.depth_mode
    
    return config

def train(attn_implementation=None):
    global local_rank
    
    # Enable anomaly detection for debugging gradient explosions
    torch.autograd.set_detect_anomaly(True)
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    resolve_depth_mode(training_args)

    discrete_supervision_mode = training_args.discrete_depth_supervision_mode.lower()
    if discrete_supervision_mode not in {"ground_truth", "random", "ignore"}:
        raise ValueError(
            f"Unsupported --discrete-depth-supervision-mode '{training_args.discrete_depth_supervision_mode}'. "
            "Choose from ground_truth|random|ignore."
        )
    training_args.discrete_depth_supervision_mode = discrete_supervision_mode

    token_reinit_method = training_args.depth_token_reinit_method.lower()
    if token_reinit_method not in {"none", "random", "average", "adaptive"}:
        raise ValueError(
            f"Unsupported --depth-token-reinit-method '{training_args.depth_token_reinit_method}'. "
            "Choose from none|random|average|adaptive."
        )
    training_args.depth_token_reinit_method = token_reinit_method

    # ------------------------------------------------------------------
    # Shared encoder configuration (load once per training run)
    # ------------------------------------------------------------------
    if not os.path.exists(data_args.config_path):
        raise FileNotFoundError(f"Config file not found: {data_args.config_path}")

    with open(data_args.config_path, 'r') as cf:
        shared_cfg = json.load(cf)

    # Determine encoder to use (either from CLI or default)
    depth_encoder_name = data_args.depth_encoder_name or shared_cfg.get("default_encoder")

    # ------------------------------------------------------------------
    # Handle special variants that specify an alternate patch/grid size via
    # a name suffix e.g. "google/siglip2-large-patch16-256_interploate_64".
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
        # The override value directly specifies the desired number of depth tokens.
        num_depth_tokens = grid_size_override
    else:
        if grid_size is None:
            raise ValueError(f"grid_size must be specified for encoder '{depth_encoder_name}' in config.")
        num_depth_tokens = grid_size ** 2
    
    # Store num_depth_tokens in data_args for dataset access
    data_args.num_depth_tokens = num_depth_tokens

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # Log depth training configuration
    if training_args.depth_data:
        rank0_print("\n" + "="*80)
        rank0_print("[DEPTH TRAINING CONFIGURATION]")
        rank0_print("="*80)
        rank0_print(f"  Depth Mode:               {training_args.depth_mode}")
        rank0_print(f"  Depth Loss Type:          {model_args.depth_loss_type}")
        rank0_print(f"  Depth Loss Layer:         {model_args.depth_loss_layer}")
        rank0_print(f"  Depth Coefficient:        {training_args.depth_coef}")
        rank0_print(f"  Loss Mode:                {model_args.loss_mode}")
        rank0_print(f"  Depth Head Type:          {model_args.depth_head_type}")
        rank0_print(f"  Depth Projector Type:     {model_args.depth_projector_type}")
        rank0_print(f"  Vision Guidance Weight:   {model_args.vision_guidance_weight}")
        rank0_print(f"  Depth Encoder:            {depth_encoder_name}")
        rank0_print(f"  Depth Feature Dim:        {depth_feature_dim}")
        rank0_print(f"  Num Depth Tokens:         {num_depth_tokens}")
        rank0_print(f"  Use Discrete Tokens:      {training_args.use_discrete_depth_tokens}")
        if training_args.use_discrete_depth_tokens:
            rank0_print(f"  Discrete Depth Levels:    {training_args.num_discrete_depth_levels}")
            rank0_print(f"  Discrete CE Supervision:  {training_args.discrete_depth_supervision_mode}")
        rank0_print(f"  Token Reinit Method:      {training_args.depth_token_reinit_method}")
        if hasattr(training_args, 'depth_training_stage') and training_args.depth_training_stage:
            rank0_print(f"  Training Stage:           {training_args.depth_training_stage}")
        rank0_print("="*80 + "\n")
    else:
        rank0_print(f"[DEPTH MODE] Running without depth tokens (mode={training_args.depth_mode}).")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            print(f"[TRAIN] Setting use_context_gate to: {training_args.use_context_gate}")
            config = get_llava_config(model_args, training_args, depth_feature_dim, num_depth_tokens, training_args.use_context_gate)
            print(f"[DEPTH CONFIG] depth_loss_type: {model_args.depth_loss_type}")
            print(f"[DEPTH CONFIG] depth_loss_layer: {model_args.depth_loss_layer}")
            print(f"[DEPTH CONFIG] depth_coef: {training_args.depth_coef}")
            print(f"[DEPTH CONFIG] loss_mode: {model_args.loss_mode}")
            print(f"[DEPTH CONFIG] depth_head_type: {model_args.depth_head_type}")
            print(f"[DEPTH CONFIG] depth_projector_type: {model_args.depth_projector_type}")
            print(f"[DEPTH CONFIG] vision_guidance_weight: {model_args.vision_guidance_weight}")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                config=config,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
  
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        # Get LoRA target modules, excluding depth components
        target_modules = find_all_linear_names(model)
        
        # For stage training with LoRA, only apply LoRA to unfrozen components [[memory:4968535]]
        if hasattr(training_args, 'depth_training_stage') and training_args.depth_training_stage in [1, 2]:
            print(f"[LoRA] Configuring LoRA for stage {training_args.depth_training_stage} training")
            print(f"[LoRA]: Applying LoRA to {len(target_modules)} modules")
        
        if training_args.lora_enable:  # Check again in case we disabled it
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
            
            # Defer loading of LoRA and non-LoRA weights until after tokenizer resize and module init
            pending_lora_weight_path = getattr(training_args, 'lora_weight_path', None)
        

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )


    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    # NEW: Aurora grows the tokenizer here for continuous or discrete depth
    # control tokens, then pushes the resolved IDs into both the config and the
    # runtime model attributes before training starts.
    token_count_before_depth = len(tokenizer)
    token_count_after_depth = token_count_before_depth
    depth_setup_mode = "none"
    if training_args.depth_data:
        depth_setup_mode = "discrete" if training_args.use_discrete_depth_tokens else "continuous"
    rank0_print(f"[DEPTH TOKEN SETUP] mode={depth_setup_mode}")

    if depth_setup_mode == "discrete":
        num_levels = int(training_args.num_discrete_depth_levels)
        if num_levels <= 0:
            raise ValueError(f"num_discrete_depth_levels must be > 0, got {num_levels}")

        depth_tokens = ["<DEPTH_START>", "<DEPTH_END>"] + [f"<DEPTH_{num}>" for num in range(num_levels)]
        num_added_tokens = tokenizer.add_tokens(depth_tokens)
        print(f"Added {num_added_tokens} discrete depth tokens (Aurora-style)")

        depth_start_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
        depth_end_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
        discrete_depth_token_ids = [
            tokenizer.convert_tokens_to_ids(f"<DEPTH_{i}>")
            for i in range(num_levels)
        ]
        if depth_start_id == tokenizer.unk_token_id or depth_end_id == tokenizer.unk_token_id:
            raise AssertionError("Discrete depth boundary tokens were not added to tokenizer.")
        if any(tok_id == tokenizer.unk_token_id for tok_id in discrete_depth_token_ids):
            raise AssertionError("One or more discrete depth tokens were not added to tokenizer.")

        if hasattr(model, 'config'):
            model.config.use_discrete_depth_tokens = True
            model.config.num_discrete_depth_levels = num_levels
            model.config.depth_start_id = depth_start_id
            model.config.depth_end_id = depth_end_id
            model.config.depth_token_id = None
            model.config.discrete_depth_token_ids = discrete_depth_token_ids
            model.config.discrete_depth_supervision_mode = training_args.discrete_depth_supervision_mode
            model.config.depth_mode = "discrete"
            print(f"Discrete depth tokens: start={model.config.depth_start_id}, end={model.config.depth_end_id}")
            print(f"First few discrete tokens: {model.config.discrete_depth_token_ids[:5]}")

        if hasattr(model, 'depth_start_id'):
            model.depth_start_id = depth_start_id
        if hasattr(model, 'depth_end_id'):
            model.depth_end_id = depth_end_id
        if hasattr(model, 'depth_token_id'):
            model.depth_token_id = None

    elif depth_setup_mode == "continuous":
        num_added_tokens = tokenizer.add_tokens(["<DEPTH_START>", "<DEPTH_END>", "<DEPTH_TOKEN>"])
        print("Number of added tokens for depth (continuous): ", num_added_tokens)

        depth_start_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
        depth_end_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
        depth_token_id = tokenizer.convert_tokens_to_ids("<DEPTH_TOKEN>")
        assert depth_start_id != tokenizer.unk_token_id, "DEPTH_START not added to tokenizer!"
        assert depth_end_id != tokenizer.unk_token_id, "DEPTH_END not added to tokenizer!"
        assert depth_token_id != tokenizer.unk_token_id, "DEPTH_TOKEN not added to tokenizer!"

        if hasattr(model, 'depth_start_id'):
            model.depth_start_id = depth_start_id
        if hasattr(model, 'depth_end_id'):
            model.depth_end_id = depth_end_id
        if hasattr(model, 'depth_token_id'):
            model.depth_token_id = depth_token_id
        print(f"Updated depth token IDs: start={depth_start_id}, end={depth_end_id}, token={depth_token_id}")

        if hasattr(model, 'config'):
            model.config.depth_start_id = depth_start_id
            model.config.depth_end_id = depth_end_id
            model.config.depth_token_id = depth_token_id
            model.config.use_discrete_depth_tokens = False
            model.config.discrete_depth_token_ids = None
            model.config.depth_mode = "continuous"

    else:
        # Original mode / no depth training.
        if hasattr(model, 'config'):
            model.config.use_discrete_depth_tokens = False
            model.config.discrete_depth_token_ids = None
            model.config.depth_token_id = None
            model.config.depth_mode = "original"

    # Mode invariants: fail early if config fields are inconsistent.
    if hasattr(model, "config"):
        cfg_mode = getattr(model.config, "depth_mode", "original")
        if cfg_mode not in {"original", "continuous", "discrete"}:
            raise ValueError(f"Invalid model.config.depth_mode={cfg_mode}. Expected original|continuous|discrete.")
        if cfg_mode == "original":
            if bool(getattr(model.config, "use_discrete_depth_tokens", False)):
                raise ValueError("depth_mode=original requires use_discrete_depth_tokens=False.")
            if getattr(model.config, "depth_token_id", None) is not None:
                raise ValueError("depth_mode=original requires depth_token_id=None.")
        elif cfg_mode == "continuous":
            if bool(getattr(model.config, "use_discrete_depth_tokens", False)):
                raise ValueError("depth_mode=continuous requires use_discrete_depth_tokens=False.")
            if getattr(model.config, "depth_token_id", None) is None:
                raise ValueError("depth_mode=continuous requires depth_token_id to be set.")
        else:  # discrete
            if not bool(getattr(model.config, "use_discrete_depth_tokens", False)):
                raise ValueError("depth_mode=discrete requires use_discrete_depth_tokens=True.")
            discrete_ids = getattr(model.config, "discrete_depth_token_ids", None)
            if not discrete_ids:
                raise ValueError("depth_mode=discrete requires non-empty discrete_depth_token_ids.")

        # Keep runtime attributes synchronized with config invariants.
        if hasattr(model, "depth_mode"):
            model.depth_mode = cfg_mode
        if hasattr(model, "use_discrete_depth_tokens"):
            model.use_discrete_depth_tokens = bool(getattr(model.config, "use_discrete_depth_tokens", False))
        if hasattr(model, "depth_token_id"):
            model.depth_token_id = getattr(model.config, "depth_token_id", None)
        if hasattr(model, "depth_start_id"):
            model.depth_start_id = getattr(model.config, "depth_start_id", getattr(model, "depth_start_id", None))
        if hasattr(model, "depth_end_id"):
            model.depth_end_id = getattr(model.config, "depth_end_id", getattr(model, "depth_end_id", None))
    token_count_after_depth = len(tokenizer)
    if training_args.coordinates_data:
        num_added_tokens = tokenizer.add_tokens([f"<PIXEL_{str(num)}>" for num in range(336)])
        print("Number of added tokens for coor: ", num_added_tokens)
        token_count_after_depth = len(tokenizer)

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # Freeze vision tower if requested [[memory:4968535]]
        if training_args.freeze_vision_for_depth:
            print("[VISION] Freezing vision tower for depth training")
            vision_tower.requires_grad_(False)
            for p in vision_tower.parameters():
                p.requires_grad = False
        
        # NEW: MetaMorph-style simple vision freezing
        if model_args.freeze_vision:
            print("[VISION] Freezing vision tower (MetaMorph-style)")
            vision_tower_for_freezing = model.model.get_vision_tower()
            vision_tower_for_freezing.requires_grad_(False)
            # Ensure all parameters are frozen (double insurance like LLaVA original)
            for p in vision_tower_for_freezing.parameters():
                p.requires_grad = False
            # # Additional freezing for special projector types (like MetaMorph)
            if model_args.mm_projector_type == "mlpsoftmax":
                print("[VISION] Freezing first linear layer of mlpsoftmax projector")
                projector = model.get_model().mm_projector
                first_linear_layer = projector[0]
                for param in first_linear_layer.parameters():
                    param.requires_grad = False

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        # Save training stage info to model config for checkpoint management
        if hasattr(training_args, 'depth_training_stage'):
            model.config.depth_training_stage = training_args.depth_training_stage
            print(f"[CONFIG] Saved depth_training_stage={training_args.depth_training_stage} to model config")
        
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projesctor_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    # NEW: Resize embeddings only once after all Aurora tokenizer growth
    # decisions are finalized so new depth and coordinate tokens share the same
    # initialization path.
    if training_args.depth_data or training_args.coordinates_data:
        model.resize_token_embeddings(len(tokenizer))
        if token_count_after_depth > token_count_before_depth:
            reinitialize_new_token_embeddings(
                model,
                old_len=token_count_before_depth,
                new_len=token_count_after_depth,
                method=training_args.depth_token_reinit_method,
            )
            rank0_print(
                f"[TOKEN INIT] Reinitialized new token rows [{token_count_before_depth}:{token_count_after_depth}) "
                f"using method='{training_args.depth_token_reinit_method}'."
            )

        # Two-stage depth training logic [[memory:4968535]]
        if training_args.depth_training_stage == 1:
            # Stage 1: Train only depth_projector
            print("[STAGE 1] Training only depth_projector")
            # First freeze everything
            model.requires_grad_(False)
            
            # Then unfreeze only depth_projector
            for n, p in model.named_parameters():
                if 'depth_projector' in n:
                    p.requires_grad = True
                    print(f"[STAGE 1] Unfreezing: {n}, shape: {p.shape}")
                    
            # Also need to unfreeze embeddings if new tokens were added
            if training_args.depth_data:
                for n, p in model.named_parameters():
                    if 'embed_tokens' in n:
                        p.requires_grad = True
                        print(f"[STAGE 1] Unfreezing embeddings: {n}, shape: {p.shape}")
                        
        elif training_args.depth_training_stage == 2:
            # Stage 2: Train lm_head + depth_head + depth_projector
            print("[STAGE 2] Training lm_head + depth_head + depth_projector")
            # First freeze everything
            model.requires_grad_(False)
            
            # Unfreeze lm_head, depth_head, and depth_projector
            for n, p in model.named_parameters():
                if 'lm_head' in n or 'depth_head' in n or 'depth_projector' in n:
                    p.requires_grad = True
                    print(f"[STAGE 2] Unfreezing: {n}, shape: {p.shape}")
                    
            # Also need to handle embeddings if depth tokens exist
            if training_args.depth_data:
                for n, p in model.named_parameters():
                    if 'embed_tokens' in n:
                        p.requires_grad = True
                        print(f"[STAGE 2] Unfreezing embeddings: {n}, shape: {p.shape}")
                        
        else:
            # Default behavior: train all depth-related components
            print("[DEFAULT] Training all depth-related components")
            for n, p in model.named_parameters():
                if any(
                    [
                        x in n
                        for x in ["lm_head", "embed_tokens", 'depth_head', 'depth_projector']
                    ]
                ):
                    p.requires_grad = True
                    print("n: ", n, "p.shape: ", p.shape, p.requires_grad)

    # Ensure depth components are not included in LoRA targets
    assert 'depth_head' not in find_all_linear_names(model)
    assert 'depth_projector' not in find_all_linear_names(model)
    

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Deferred: load LoRA and non-LoRA weights after modules/tokens are initialized
    if training_args.lora_enable and 'pending_lora_weight_path' in locals() and pending_lora_weight_path:
        lora_dir = pending_lora_weight_path
        possible_files = [
            os.path.join(lora_dir, 'adapter_model.bin'),
            os.path.join(lora_dir, 'pytorch_model.bin'),
            os.path.join(lora_dir, 'adapter_model.safetensors'),
        ]
        lora_state_path = None
        for f in possible_files:
            if os.path.isfile(f):
                lora_state_path = f
                break
        if lora_state_path is not None:
            rank0_print(f"[LoRA] Loading adapter weights from: {lora_state_path}")
            if lora_state_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                lora_state = load_file(lora_state_path)
            else:
                lora_state = torch.load(lora_state_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(lora_state, strict=False)
            rank0_print(f"[LoRA] Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        else:
            rank0_print(f"[LoRA] No adapter weights found in {lora_dir}; starting from fresh LoRA init")

        # Load non-LoRA trainables (matching builder.py exactly)
        non_lora_path = os.path.join(lora_dir, 'non_lora_trainables.bin')
        if os.path.isfile(non_lora_path):
            rank0_print(f"[LoRA] Loading non-LoRA trainables from: {non_lora_path}")
            non_lora_state = torch.load(non_lora_path, map_location='cpu')
            
            # Debug: Check embeddings in non_lora_trainables (matching builder.py)
            rank0_print(f"Keys in non_lora_trainables: {list(non_lora_state.keys())[:10]}...")
            if 'model.embed_tokens.weight' in non_lora_state:
                rank0_print(f"  model.embed_tokens.weight shape: {non_lora_state['model.embed_tokens.weight'].shape}")
            if 'base_model.model.model.embed_tokens.weight' in non_lora_state:
                rank0_print(f"  base_model.model.model.embed_tokens.weight shape: {non_lora_state['base_model.model.model.embed_tokens.weight'].shape}")
            if 'lm_head.weight' in non_lora_state:
                rank0_print(f"  lm_head.weight shape: {non_lora_state['lm_head.weight'].shape}")
            if 'base_model.model.lm_head.weight' in non_lora_state:
                rank0_print(f"  base_model.model.lm_head.weight shape: {non_lora_state['base_model.model.lm_head.weight'].shape}")
            
            # Key transformation to match builder.py exactly
            non_lora_state = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_state.items()}
            if any(k.startswith('model.model.') for k in non_lora_state):
                non_lora_state = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_state.items()}
            
            rank0_print(f"After key transformation, checking for embedding keys:")
            if 'model.embed_tokens.weight' in non_lora_state:
                rank0_print(f"  model.embed_tokens.weight shape: {non_lora_state['model.embed_tokens.weight'].shape}")
            if 'lm_head.weight' in non_lora_state:
                rank0_print(f"  lm_head.weight shape: {non_lora_state['lm_head.weight'].shape}")
            
            missing, unexpected = model.load_state_dict(non_lora_state, strict=False)
            rank0_print(f"[LoRA] Loaded non_lora_trainables: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if missing:
                rank0_print(f"  Missing keys (first 10): {missing[:10]}")
            if unexpected:
                rank0_print(f"  Unexpected keys (first 10): {unexpected[:10]}")
        else:
            rank0_print(f"[LoRA] No non-LoRA trainables found at: {non_lora_path}")


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                           data_args=data_args,
                                           training_args=training_args)
 
        # Callback to log optimizer LR for depth head/projector groups
    from transformers.trainer_callback import TrainerCallback as HFTrainerCallback
    class DepthLrLoggingCallback(HFTrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
             optimizer = kwargs.get('optimizer')
             if optimizer is None:
                 return
             if getattr(args, 'local_rank', -1) not in [0, -1]:
                 return
             try:
                 logs = []
                 for idx, group in enumerate(optimizer.param_groups):
                     name = group.get('group_name', f'pg_{idx}')
                     if any(key in name for key in ['depth_projector', 'depth_head']):
                         logs.append(f"{name}={group.get('lr')}")
                 if logs:
                     print(f"[LR][step {state.global_step}] " + ", ".join(logs))
             except Exception:
                 pass
 
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[DepthLrLoggingCallback()],
                    **data_module)
    trainer.train()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
