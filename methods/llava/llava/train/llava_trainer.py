# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/train/llava_trainer.py

import os
import torch
import torch.nn as nn
import math
import logging
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import Sampler
from torch.utils.data import  SequentialSampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    # Decode the modality and depth information
    # Encoding: base length + 1000000 if has depth tokens
    # Positive base: has image, Negative base: no image
    DEPTH_OFFSET = 1000000
    
    # Separate samples into 4 categories
    image_no_depth = []
    image_with_depth = []
    text_no_depth = []
    text_with_depth = []
    
    for i, l in enumerate(lengths):
        has_depth = l > DEPTH_OFFSET or l < -DEPTH_OFFSET
        
        if has_depth:
            # Remove depth offset to get actual length
            actual_len = l - DEPTH_OFFSET if l > 0 else l + DEPTH_OFFSET
            if actual_len > 0:
                image_with_depth.append((i, actual_len))
            else:
                text_with_depth.append((i, -actual_len))
        else:
            if l > 0:
                image_no_depth.append((i, l))
            else:
                text_no_depth.append((i, -l))
    
    # Log grouping statistics
    total_samples = len(lengths)
    print(f"\n[MODALITY_GROUPING] Dataset composition:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Image without depth: {len(image_no_depth)} ({len(image_no_depth)/total_samples*100:.1f}%)")
    print(f"  - Image with depth: {len(image_with_depth)} ({len(image_with_depth)/total_samples*100:.1f}%)")
    print(f"  - Text without depth: {len(text_no_depth)} ({len(text_no_depth)/total_samples*100:.1f}%)")
    print(f"  - Text with depth: {len(text_with_depth)} ({len(text_with_depth)/total_samples*100:.1f}%)")
    
    # If all samples are in one category, use regular length grouping
    categories = [image_no_depth, image_with_depth, text_no_depth, text_with_depth]
    non_empty_categories = [cat for cat in categories if cat]
    
    if len(non_empty_categories) == 1:
        # All samples in same category
        indices, lengths_only = zip(*non_empty_categories[0]) if non_empty_categories[0] else ([], [])
        return get_length_grouped_indices(lengths_only, batch_size, world_size, generator=generator)
    
    # Group each category separately
    megabatch_size = world_size * batch_size
    all_megabatches = []
    
    for category in categories:
        if not category:
            continue
            
        indices, cat_lengths = zip(*category)
        # Sort by length within category
        sorted_indices = [indices[i] for i in get_length_grouped_indices(cat_lengths, batch_size, world_size, generator=None)]
        # Create megabatches for this category
        category_megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(sorted_indices), megabatch_size)]
        all_megabatches.extend(category_megabatches)
    
    # Shuffle megabatches while keeping samples within each megabatch together
    megabatch_indices = torch.randperm(len(all_megabatches), generator=generator)
    shuffled_megabatches = [all_megabatches[i] for i in megabatch_indices]
    
    # Flatten megabatches to get final indices
    return [i for megabatch in shuffled_megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

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


def get_inverse_cosine_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

         ##### NEW ########
        # Audit Bug #5: SequentialSampler disables shuffling.
        # This is intentional if the dataset is ordered by difficulty for curriculum learning.
        # If shuffling is desired with annealing, this block should be modified.
        if self.args.annealing_data:
            print("%%%%%%%%%%%%% SEQUENTIAL SAMPLER")
            return SequentialSampler((self.train_dataset))
        ##### NEW ########

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        print("Creating optimizer")
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Freeze vision components if depth training with vision guidance is enabled
            if hasattr(self.args, 'freeze_vision_for_depth') and self.args.freeze_vision_for_depth:
                print("[OPTIMIZER] Freezing vision components for depth training")
                for name, param in opt_model.named_parameters():
                    if 'vision_tower' in name or ('mm_projector' in name and not any(x in name for x in ['depth_projector', 'depth_head'])):
                        param.requires_grad = False
                        if getattr(self.args, 'local_rank', -1) in [0, -1]:
                            print(f"  - Frozen: {name}")
            
            # Handle two-stage depth training [[memory:4968535]]
            if hasattr(self.args, 'depth_training_stage') and self.args.depth_training_stage in [1, 2]:
                if self.args.depth_training_stage == 1:
                    # Stage 1: Optimize depth_projector with optional separate LR
                    print("[OPTIMIZER] Setting up Stage 1 optimizer for depth_projector")
                    depth_projector_params = [name for name, _ in opt_model.named_parameters() if "depth_projector" in name]
                    dp_weight_decay = getattr(self.args, 'depth_projector_weight_decay', None)
                    if dp_weight_decay is None:
                        dp_weight_decay = self.args.weight_decay
                    
                    if self.args.depth_projector_lr is not None:
                        # Use separate learning rate for depth_projector
                        optimizer_grouped_parameters = [
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and n not in depth_projector_params and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "group_name": "base_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and n not in depth_projector_params and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "group_name": "base_nodecay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and n in depth_projector_params and p.requires_grad)
                                ],
                                "weight_decay": dp_weight_decay,
                                "lr": self.args.depth_projector_lr,
                                "group_name": "depth_projector_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and n in depth_projector_params and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.depth_projector_lr,
                                "group_name": "depth_projector_nodecay",
                            },
                        ]
                    else:
                        # Use default learning rate
                        optimizer_grouped_parameters = [
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "group_name": "base_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "group_name": "base_nodecay",
                            },
                        ]
                        
                elif self.args.depth_training_stage == 2:
                    # Stage 2: Optimize lm_head and depth_head with optional separate LR
                    print("[OPTIMIZER] Setting up Stage 2 optimizer for lm_head + depth_head")
                    depth_head_params = [name for name, _ in opt_model.named_parameters() if "depth_head" in name]
                    dh_weight_decay = getattr(self.args, 'depth_head_weight_decay', None)
                    if dh_weight_decay is None:
                        dh_weight_decay = self.args.weight_decay
                    
                    if self.args.depth_head_lr is not None:
                        # Use separate learning rate for depth_head
                        optimizer_grouped_parameters = [
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and n not in depth_head_params and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "group_name": "base_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and n not in depth_head_params and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "group_name": "base_nodecay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and n in depth_head_params and p.requires_grad)
                                ],
                                "weight_decay": dh_weight_decay,
                                "lr": self.args.depth_head_lr,
                                "group_name": "depth_head_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and n in depth_head_params and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.depth_head_lr,
                                "group_name": "depth_head_nodecay",
                            },
                        ]
                    else:
                        # Use default learning rate
                        optimizer_grouped_parameters = [
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n in decay_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "group_name": "base_decay",
                            },
                            {
                                "params": [
                                    p for n, p in opt_model.named_parameters() 
                                    if (n not in decay_parameters and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "group_name": "base_nodecay",
                            },
                        ]
                        
            else:
                # Combined handling outside staged training:
                # If any of depth_projector_lr, depth_head_lr, or mm_projector_lr are provided,
                # create dedicated groups for them while putting the rest into base groups.
                dp_lr = getattr(self.args, 'depth_projector_lr', None)
                dh_lr = getattr(self.args, 'depth_head_lr', None)
                mp_lr = getattr(self.args, 'mm_projector_lr', None)
                
                dp_weight_decay = getattr(self.args, 'depth_projector_weight_decay', None)
                if dp_weight_decay is None:
                    dp_weight_decay = self.args.weight_decay
                    
                dh_weight_decay = getattr(self.args, 'depth_head_weight_decay', None)
                if dh_weight_decay is None:
                    dh_weight_decay = self.args.weight_decay

                if dp_lr is not None or dh_lr is not None or mp_lr is not None:
                    if getattr(self.args, 'local_rank', -1) in [0, -1]:
                        print("[OPTIMIZER] Using combined LR grouping (no staged training)")

                    depth_projector_names = {name for name, _ in opt_model.named_parameters() if 'depth_projector' in name}
                    depth_head_names = {name for name, _ in opt_model.named_parameters() if 'depth_head' in name}
                    projector_names = {name for name, _ in opt_model.named_parameters() if 'mm_projector' in name}

                    def should_exclude(name: str) -> bool:
                        if dp_lr is not None and name in depth_projector_names:
                            return True
                        if dh_lr is not None and name in depth_head_names:
                            return True
                        if mp_lr is not None and name in projector_names:
                            return True
                        return False

                    # Base groups: everything not in any special group (that has an override LR)
                    base_decay_params = [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad and not should_exclude(n))
                    ]
                    base_nodecay_params = [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad and not should_exclude(n))
                    ]

                    optimizer_grouped_parameters = []
                    if base_decay_params:
                        optimizer_grouped_parameters.append({
                            'params': base_decay_params,
                            'weight_decay': self.args.weight_decay,
                            'group_name': 'base_decay',
                        })
                    if base_nodecay_params:
                        optimizer_grouped_parameters.append({
                            'params': base_nodecay_params,
                            'weight_decay': 0.0,
                            'group_name': 'base_nodecay',
                        })

                    # Depth projector groups
                    if dp_lr is not None:
                        dp_decay = [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in depth_projector_names and p.requires_grad)]
                        dp_nodecay = [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in depth_projector_names and p.requires_grad)]
                        if dp_decay:
                            optimizer_grouped_parameters.append({
                                'params': dp_decay,
                                'weight_decay': dp_weight_decay,
                                'lr': dp_lr,
                                'group_name': 'depth_projector_decay',
                            })
                        if dp_nodecay:
                            optimizer_grouped_parameters.append({
                                'params': dp_nodecay,
                                'weight_decay': 0.0,
                                'lr': dp_lr,
                                'group_name': 'depth_projector_nodecay',
                            })

                    # Depth head groups
                    if dh_lr is not None:
                        dh_decay = [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in depth_head_names and p.requires_grad)]
                        dh_nodecay = [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in depth_head_names and p.requires_grad)]
                        if dh_decay:
                            optimizer_grouped_parameters.append({
                                'params': dh_decay,
                                'weight_decay': dh_weight_decay,
                                'lr': dh_lr,
                                'group_name': 'depth_head_decay',
                            })
                        if dh_nodecay:
                            optimizer_grouped_parameters.append({
                                'params': dh_nodecay,
                                'weight_decay': 0.0,
                                'lr': dh_lr,
                                'group_name': 'depth_head_nodecay',
                            })

                    # mm_projector groups
                    if mp_lr is not None:
                        mp_decay = [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_names and p.requires_grad)]
                        mp_nodecay = [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_names and p.requires_grad)]
                        if mp_decay:
                            optimizer_grouped_parameters.append({
                                'params': mp_decay,
                                'weight_decay': self.args.weight_decay,
                                'lr': mp_lr,
                                'group_name': 'mm_projector_decay',
                            })
                        if mp_nodecay:
                            optimizer_grouped_parameters.append({
                                'params': mp_nodecay,
                                'weight_decay': 0.0,
                                'lr': mp_lr,
                                'group_name': 'mm_projector_nodecay',
                            })
                else:
                    # Default optimizer setup
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "group_name": "base_decay",
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "group_name": "base_nodecay",
                        },
                    ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Rank-0 debug: print group learning rates and sizes prior to optimizer init
            if getattr(self.args, 'local_rank', -1) in [0, -1]:
                try:
                    print("[OPTIMIZER] Parameter groups (pre-init):")
                    for idx, group in enumerate(optimizer_grouped_parameters):
                        group_name = group.get("group_name", f"pg_{idx}")
                        group_lr = group.get("lr", optimizer_kwargs.get("lr", None))
                        num_params = sum(p.numel() for p in group["params"]) if isinstance(group.get("params"), list) else 0
                        print(f"  - {group_name}: lr={group_lr}, weight_decay={group.get('weight_decay')}, params={num_params}")
                except Exception as e:
                    print(f"[OPTIMIZER] failed to summarize groups before init: {e}")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Rank-0 debug: print group learning rates after optimizer init (will include scheduler-adjusted base later)
            if getattr(self.args, 'local_rank', -1) in [0, -1]:
                try:
                    print("[OPTIMIZER] Parameter groups (post-init):")
                    for idx, group in enumerate(self.optimizer.param_groups):
                        group_name = group.get("group_name", f"pg_{idx}")
                        print(f"  - {group_name}: lr={group.get('lr')}, weight_decay={group.get('weight_decay')}")

                    # Summarize depth parameter effective LRs even if sharing base groups
                    # Build id sets for depth params that are trainable
                    dp_ids = set()
                    dh_ids = set()
                    for n, p in opt_model.named_parameters():
                        if p.requires_grad:
                            if 'depth_projector' in n:
                                dp_ids.add(id(p))
                            if 'depth_head' in n:
                                dh_ids.add(id(p))

                    def summarize_ids(id_set, label):
                        matched = []
                        for group in self.optimizer.param_groups:
                            # Count how many params from this set are in the group
                            cnt = sum(1 for p in group.get('params', []) if id(p) in id_set)
                            if cnt > 0:
                                matched.append((group.get('group_name', None), group.get('lr'), cnt))
                        if matched:
                            for name, lr, cnt in matched:
                                print(f"[OPTIMIZER] {label}: lr={lr} (group={name}, params={cnt})")
                        else:
                            print(f"[OPTIMIZER] {label}: no trainable params in optimizer groups")

                    summarize_ids(dp_ids, 'depth_projector')
                    summarize_ids(dh_ids, 'depth_head')

                except Exception as e:
                    print(f"[OPTIMIZER] failed to summarize groups after init: {e}")

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # import pdb;pdb.set_trace()
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                os.makedirs(output_dir, exist_ok=True)
                state_dict = get_peft_state_maybe_zero_3(
                    self.model.named_parameters(), self.args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    self.model.named_parameters()
                )
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    print(f"save models to {output_dir} ")
                    self.model.config.save_pretrained(output_dir)
                    self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            else:
                super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
 
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create a learning rate scheduler that can handle annealing data better.
        """
        if optimizer is None:
            optimizer = self.optimizer
            
        # Check if we should use inverse cosine schedule for annealing data
        if getattr(self.args, 'annealing_data', False) and getattr(self.args, 'use_inverse_cosine', False):
            if getattr(self.args, 'local_rank', -1) in [0, -1]:
                print("[SCHEDULER] Using inverse cosine schedule for annealing data")
            
            # Calculate warmup steps
            if hasattr(self.args, 'warmup_steps'):
                num_warmup_steps = self.args.warmup_steps
            elif hasattr(self.args, 'warmup_ratio'):
                num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
            else:
                num_warmup_steps = 0
                
            min_lr_ratio = getattr(self.args, 'min_lr_ratio', 0.1)
            return get_inverse_cosine_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio)
        else:
            # Use default scheduler creation
            return super().create_scheduler(num_training_steps, optimizer)
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Custom evaluation loop that tracks depth MSE loss during validation.
        """
        # Call parent evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        
        # Compute aggregated depth MSE if we tracked it
        if hasattr(self, '_val_depth_losses') and len(self._val_depth_losses) > 0:
            # Compute local average
            local_sum = sum(self._val_depth_losses)
            local_count = len(self._val_depth_losses)
            
            # Synchronize across all processes if using distributed training
            if torch.distributed.is_initialized():
                # Convert to tensors for all_reduce
                sum_tensor = torch.tensor(local_sum, dtype=torch.float32, device=self.args.device)
                count_tensor = torch.tensor(local_count, dtype=torch.float32, device=self.args.device)
                
                # Reduce across all ranks
                torch.distributed.all_reduce(sum_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
                
                # Compute global average
                avg_depth_mse = (sum_tensor / count_tensor).item()
            else:
                # Single process - just compute local average
                avg_depth_mse = local_sum / local_count
            
            # Add to metrics
            if output.metrics is not None:
                output.metrics[f"{metric_key_prefix}_depth_mse"] = avg_depth_mse
            
            # Log the aggregated depth MSE (only on rank 0)
            if getattr(self.args, 'local_rank', -1) in [0, -1]:
                print(f"\n[VALIDATION] Average Depth MSE: {avg_depth_mse:.6f}")
                print(f"[VALIDATION] Total validation batches with depth: {local_count}")
            
            # Clear the tracked losses
            self._val_depth_losses = []
        elif getattr(self.args, 'local_rank', -1) in [0, -1]:
            # No depth samples in validation set
            print(f"\n[VALIDATION] No depth samples found in validation set")
        
        return output
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to compute validation losses including depth MSE.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        
        # Run forward pass
        with torch.no_grad():
            if has_labels:
                # Need to get outputs directly
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    loss = outputs.loss
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
        
        # Track depth MSE loss for aggregation
        if outputs is not None and hasattr(outputs, 'loss_depth_ar'):
            depth_mse_loss = outputs.loss_depth_ar
            if depth_mse_loss is not None and depth_mse_loss != 0.0:
                # Initialize list if needed
                if not hasattr(self, '_val_depth_losses'):
                    self._val_depth_losses = []
                # Store the loss value
                if isinstance(depth_mse_loss, torch.Tensor):
                    self._val_depth_losses.append(depth_mse_loss.item())
                else:
                    self._val_depth_losses.append(depth_mse_loss)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return outputs for metric computation
        logits = outputs.logits if hasattr(outputs, 'logits') else None
        labels = inputs.get("labels")
        
        return (loss, logits, labels)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through the full (possibly wrapped) model so gradients flow correctly
        try:
            # Add debug mode if needed
            debug_mode = getattr(self.args, 'debug_mode', False)
            if debug_mode:
                print(f"[DEBUG] compute_loss called with inputs keys: {list(inputs.keys())}")
                for key, value in inputs.items():
                    if hasattr(value, 'shape'):
                        print(f"[DEBUG] {key} shape: {value.shape}")
                    elif hasattr(value, '__len__'):
                        print(f"[DEBUG] {key} length: {len(value)}")
            
            outputs = model(**inputs)          # CausalLMOutputWithPast
            loss = outputs.loss                # what the optimiser needs

            # Pick up the extra components from outputs directly
            extra = {}
            for attr_name in ["loss_total", "loss_language", "loss_depth_ar"]:
                val = getattr(outputs, attr_name, None)
                if val is not None:
                    if hasattr(val, 'detach'):
                        extra[attr_name] = val.detach().item()  # Convert to scalar for clean logging
                    else:
                        extra[attr_name] = val  # Already a scalar

            # Add aliases for backward compatibility
            if "loss_language" in extra:
                extra["loss_ce"] = extra["loss_language"]  # Alias for backward compatibility
            if "loss_depth_ar" in extra:
                extra["loss_depth"] = extra["loss_depth_ar"]  # Alias for backward compatibility

            extra_prefixed = {f"train_{k}": v for k, v in extra.items()}
            
            if extra_prefixed:
                self.log(extra_prefixed)

            # Periodic LR logging for depth groups (approx per step), rank 0 only
            if getattr(self.args, 'local_rank', -1) in [0, -1]:
                if getattr(self, 'optimizer', None) is not None:
                    try:
                        self._lr_debug_counter = getattr(self, '_lr_debug_counter', 0) + 1
                        logging_every = max(1, getattr(self.args, 'logging_steps', 1))
                        if self._lr_debug_counter % logging_every == 0:
                            logs = []
                            for idx, group in enumerate(self.optimizer.param_groups):
                                name = group.get('group_name', f'pg_{idx}')
                                if ('depth_projector' in name) or ('depth_head' in name):
                                    logs.append(f"{name}={group.get('lr')}")
                            if logs:
                                step_val = getattr(self.state, 'global_step', self._lr_debug_counter)
                                print(f"[LR][step {step_val}] " + ", ".join(logs))
                    except Exception:
                        pass

            # Trainer will still see the main loss for back-prop
            return (loss, outputs) if return_outputs else loss
            
        except RuntimeError as e:
            if "CUDA" in str(e) and "illegal memory access" in str(e):
                print(f"CUDA memory error detected: {e}")
                print("This might be due to depth processing issues. Check depth_embeddings and depth_positions.")
                # Return a zero loss to prevent training from crashing
                if return_outputs:
                    return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True), None
                else:
                    return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
            else:
                raise e