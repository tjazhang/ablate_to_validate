# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/model/builder.py

#    Copyright 2023 Haotian Liu
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
import warnings
import shutil
from glob import glob

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, update_only_newtokens=False, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, use_discrete_depth_tokens=None, num_discrete_depth_levels=128, **kwargs):
    """Load an upstream or Aurora LLaVA checkpoint and reconcile depth-token state."""
    kwargs = {"device_map": device_map, **kwargs}

    model_path = os.path.expanduser(model_path)
    model_base = os.path.expanduser(model_base) if model_base is not None else None
    model_path_lower = model_path.lower()
    model_base_lower = model_base.lower() if model_base is not None else ""
    local_checkpoint_dir = os.path.isdir(model_path)

    # NEW: Aurora supports several checkpoint layouts, so we inspect the local
    # directory before deciding whether this is a merged model, a LoRA adapter,
    # or just an mm_projector drop-in.
    def _path_has_any(files):
        if not local_checkpoint_dir:
            return False
        return any(os.path.exists(os.path.join(model_path, fname)) for fname in files)

    def _path_has_glob(patterns):
        if not local_checkpoint_dir:
            return False
        return any(glob(os.path.join(model_path, pattern)) for pattern in patterns)

    adapter_markers = ['adapter_config.json', 'adapter_model.bin', 'adapter_model.safetensors', 'non_lora_trainables.bin']
    full_weight_markers = ['pytorch_model.bin', 'pytorch_model.bin.index.json', 'model.safetensors', 'model.safetensors.index.json']
    sharded_weight_patterns = ['pytorch_model-*.bin', 'model-*.safetensors', 'model-*-of-*.safetensors']

    has_adapter_markers = _path_has_any(adapter_markers)
    has_full_model_weights = _path_has_any(full_weight_markers) or _path_has_glob(sharded_weight_patterns)
    has_mm_projector = local_checkpoint_dir and os.path.exists(os.path.join(model_path, 'mm_projector.bin'))

    path_implies_lora = 'lora' in model_path_lower
    prefers_lora_checkpoint = model_base is not None and (has_adapter_markers or (path_implies_lora and not has_full_model_weights))
    prefers_mm_projector = model_base is not None and not prefers_lora_checkpoint and not has_full_model_weights and has_mm_projector

    # NEW: Depth mode can come from config, loader flags, or checkpoint naming.
    # Normalize that once so tokenizer growth and runtime attributes stay aligned.
    def infer_cfg_depth_mode(cfg=None):
        if cfg is None:
            return "discrete" if use_discrete_depth_tokens else "original"
        mode = getattr(cfg, "depth_mode", None)
        if mode in {"original", "continuous", "discrete"}:
            return mode
        if getattr(cfg, "use_discrete_depth_tokens", False):
            return "discrete"
        if getattr(cfg, "depth_token_id", None) is not None:
            return "continuous"
        return "original"

    def add_depth_tokens_to_tokenizer(tokenizer, cfg=None):
        depth_mode = infer_cfg_depth_mode(cfg)
        # Explicit override from loader arg.
        if use_discrete_depth_tokens:
            depth_mode = "discrete"

        if depth_mode == "discrete":
            levels = getattr(cfg, 'num_discrete_depth_levels', num_discrete_depth_levels)
            depth_tokens = ["<DEPTH_START>", "<DEPTH_END>"] + [f"<DEPTH_{i}>" for i in range(levels)]
            label = "discrete depth tokens (Aurora-style)"
            if cfg is not None:
                cfg.depth_mode = "discrete"
                cfg.use_discrete_depth_tokens = True
        elif depth_mode == "continuous":
            depth_tokens = ["<DEPTH_START>", "<DEPTH_END>", "<DEPTH_TOKEN>"]
            label = "continuous depth tokens"
            if cfg is not None:
                cfg.depth_mode = "continuous"
                cfg.use_discrete_depth_tokens = False
        else:
            print("[INFO] No depth token config detected; skipping depth token resize.")
            return 0
        num_added = tokenizer.add_tokens(depth_tokens)
        if cfg is not None and depth_mode == "continuous":
            if getattr(cfg, "depth_start_id", None) is None:
                cfg.depth_start_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
            if getattr(cfg, "depth_end_id", None) is None:
                cfg.depth_end_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
            if getattr(cfg, "depth_token_id", None) is None:
                cfg.depth_token_id = tokenizer.convert_tokens_to_ids("<DEPTH_TOKEN>")
        if num_added > 0:
            print(f"Added {num_added} {label}")
        else:
            print(f"{label} already present in tokenizer")
        return num_added

    # NEW: Some decode paths still read depth flags directly from the model
    # instance, so keep instance attributes synchronized after config mutation.
    def sync_model_depth_runtime(model):
        # Keep runtime attributes used in generate()/decode in sync with config.
        # We mutate config during load (tokenizer resize, mode inference), and several
        # decode branches read `self.use_discrete_depth_tokens` / `self.depth_token_id`
        # directly from the model instance rather than only `self.config`.
        if not hasattr(model, "config"):
            return
        cfg = model.config
        mode = infer_cfg_depth_mode(cfg)
        if mode not in {"original", "continuous", "discrete"}:
            mode = "original"
        cfg.depth_mode = mode
        if mode == "discrete":
            cfg.use_discrete_depth_tokens = True
            cfg.depth_token_id = None
        elif mode == "continuous":
            cfg.use_discrete_depth_tokens = False
        else:
            cfg.use_discrete_depth_tokens = False
            cfg.depth_token_id = None

        if hasattr(model, "depth_mode"):
            model.depth_mode = cfg.depth_mode
        if hasattr(model, "use_discrete_depth_tokens"):
            model.use_discrete_depth_tokens = bool(cfg.use_discrete_depth_tokens)
        if hasattr(model, "depth_start_id"):
            model.depth_start_id = getattr(cfg, "depth_start_id", getattr(model, "depth_start_id", None))
        if hasattr(model, "depth_end_id"):
            model.depth_end_id = getattr(cfg, "depth_end_id", getattr(model, "depth_end_id", None))
        if hasattr(model, "depth_token_id"):
            model.depth_token_id = getattr(cfg, "depth_token_id", None)
        if hasattr(model, "num_depth_tokens"):
            model.num_depth_tokens = int(getattr(cfg, "num_depth_tokens", getattr(model, "num_depth_tokens", 16)))
        if hasattr(model, "num_discrete_depth_levels"):
            model.num_discrete_depth_levels = int(getattr(cfg, "num_discrete_depth_levels", getattr(model, "num_discrete_depth_levels", 128)))
        if hasattr(model, "discrete_depth_token_ids"):
            model.discrete_depth_token_ids = getattr(cfg, "discrete_depth_token_ids", None)
        if hasattr(model, "discrete_depth_supervision_mode"):
            model.discrete_depth_supervision_mode = getattr(cfg, "discrete_depth_supervision_mode", getattr(model, "discrete_depth_supervision_mode", "ground_truth"))

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    is_llava_model = ('llava' in model_path_lower) or ('llava' in model_base_lower)
    if is_llava_model:
        # Load LLaVA model
        if model_base is None and (has_adapter_markers or (path_implies_lora and not has_full_model_weights)):
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        # NEW: Aurora LoRA checkpoints may add depth tokens and resize the
        # vocabulary, so this branch loads the base model first and then merges
        # adapter and non-LoRA trainables with extra verification.
        if prefers_lora_checkpoint:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            add_depth_tokens_to_tokenizer(tokenizer, lora_cfg_pretrained)
            
            # Update config with discrete depth settings if not already set
            use_discrete = getattr(lora_cfg_pretrained, "use_discrete_depth_tokens", False)
            if not use_discrete and ("depth_discrete" in model_path.lower() or use_discrete_depth_tokens):
                print(f"[INFO] Auto-detected discrete depth tokens from model path or parameter")
                lora_cfg_pretrained.use_discrete_depth_tokens = True
                num_levels = getattr(lora_cfg_pretrained, "num_discrete_depth_levels", num_discrete_depth_levels)
                lora_cfg_pretrained.num_discrete_depth_levels = num_levels
                lora_cfg_pretrained.depth_token_id = None
                lora_cfg_pretrained.depth_start_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
                lora_cfg_pretrained.depth_end_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
                lora_cfg_pretrained.discrete_depth_token_ids = [
                    tokenizer.convert_tokens_to_ids(f"<DEPTH_{i}>") 
                    for i in range(num_levels)
                ]
                lora_cfg_pretrained.num_depth_tokens = 100  # Grid size 10x10 = 100 tokens
                print(f"[DEPTH CONFIG] use_discrete_depth_tokens=True, depth_start_id={lora_cfg_pretrained.depth_start_id}, depth_end_id={lora_cfg_pretrained.depth_end_id}")
                print(f"[DEPTH CONFIG] num_depth_tokens={lora_cfg_pretrained.num_depth_tokens}, discrete_depth_token_ids (first 5): {lora_cfg_pretrained.discrete_depth_token_ids[:5]}")
            
            print('Loading LLaVA from base model...')
            print("tokenizer length ############## "+ str(len(tokenizer)))
            print(f"Config vocab_size from checkpoint: {lora_cfg_pretrained.vocab_size}")
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            print(f"After loading base model: lm_head.weight.shape={model.lm_head.weight.shape}, embed_tokens.weight.shape={model.model.embed_tokens.weight.shape}")
            print(f"Expected from config: token_num={token_num}, tokem_dim={tokem_dim}")
            if model.lm_head.weight.shape[0] != token_num:
                print(f"[WARNING] Vocab size mismatch! Creating empty parameters for {token_num} tokens. These will be filled from non_lora_trainables.bin")
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            else:
                print(f"[INFO] Vocab size matches! No need to resize embeddings.")
            
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            
            # Debug: Check embeddings in non_lora_trainables
            print(f"Keys in non_lora_trainables: {list(non_lora_trainables.keys())[:10]}...")
            if 'model.embed_tokens.weight' in non_lora_trainables:
                print(f"  model.embed_tokens.weight shape: {non_lora_trainables['model.embed_tokens.weight'].shape}")
            if 'base_model.model.model.embed_tokens.weight' in non_lora_trainables:
                print(f"  base_model.model.model.embed_tokens.weight shape: {non_lora_trainables['base_model.model.model.embed_tokens.weight'].shape}")
            if 'lm_head.weight' in non_lora_trainables:
                print(f"  lm_head.weight shape: {non_lora_trainables['lm_head.weight'].shape}")
            if 'base_model.model.lm_head.weight' in non_lora_trainables:
                print(f"  base_model.model.lm_head.weight shape: {non_lora_trainables['base_model.model.lm_head.weight'].shape}")
            
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            
            print(f"After key transformation, checking for embedding keys:")
            if 'model.embed_tokens.weight' in non_lora_trainables:
                print(f"  model.embed_tokens.weight shape: {non_lora_trainables['model.embed_tokens.weight'].shape}")
            if 'lm_head.weight' in non_lora_trainables:
                print(f"  lm_head.weight shape: {non_lora_trainables['lm_head.weight'].shape}")
            
            missing_keys, unexpected_keys = model.load_state_dict(non_lora_trainables, strict=False)
            print(f"Loaded non_lora_trainables: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
            if missing_keys:
                print(f"  Missing keys (first 10): {missing_keys[:10]}")
            if unexpected_keys:
                print(f"  Unexpected keys (first 10): {unexpected_keys[:10]}")

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            
            # Final verification
            print(f"\n{'='*80}")
            print("FINAL MODEL STATE VERIFICATION")
            print(f"{'='*80}")
            print(f"Tokenizer vocab size: {len(tokenizer)}")
            print(f"Model config vocab_size: {model.config.vocab_size}")
            print(f"Model lm_head.weight shape: {model.lm_head.weight.shape}")
            print(f"Model embed_tokens.weight shape: {model.model.embed_tokens.weight.shape}")
            print(f"Depth token IDs from config:")
            print(f"  depth_start_id: {getattr(model.config, 'depth_start_id', 'NOT SET')}")
            print(f"  depth_end_id: {getattr(model.config, 'depth_end_id', 'NOT SET')}")
            print(f"  depth_token_id: {getattr(model.config, 'depth_token_id', 'NOT SET')}")
            print(f"Tokenizer depth tokens:")
            print(f"  <DEPTH_START>: {tokenizer.convert_tokens_to_ids('<DEPTH_START>')}")
            print(f"  <DEPTH_END>: {tokenizer.convert_tokens_to_ids('<DEPTH_END>')}")
            print(f"  <DEPTH_TOKEN>: {tokenizer.convert_tokens_to_ids('<DEPTH_TOKEN>')}")
            
            # Verify alignment
            if len(tokenizer) != model.config.vocab_size:
                print(f"\n[ERROR] VOCAB SIZE MISMATCH!")
                print(f"  Tokenizer: {len(tokenizer)}")
                print(f"  Model config: {model.config.vocab_size}")
            elif len(tokenizer) != model.lm_head.weight.shape[0]:
                print(f"\n[ERROR] LM_HEAD SIZE MISMATCH!")
                print(f"  Tokenizer: {len(tokenizer)}")
                print(f"  lm_head: {model.lm_head.weight.shape[0]}")
            else:
                print(f"\n[SUCCESS] All vocab sizes are aligned: {len(tokenizer)}")
            print(f"{'='*80}\n")
            
        # NEW: Full Aurora checkpoints already contain merged depth-aware weights.
        elif has_full_model_weights or model_base is None:
            print('Loading full fine-tuned LLaVA checkpoint...')
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                add_depth_tokens_to_tokenizer(tokenizer, cfg_pretrained)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **kwargs
                )
                print(len(tokenizer), "tokenizeeer")
                
        # NEW: Some experiments only save the multimodal projector on top of a
        # base LLaVA model; keep that path separate from full checkpoints.
        elif prefers_mm_projector:
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                print(len(tokenizer), "tokenizeeer")
                
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None
    if is_llava_model:
        # NEW: Final tokenizer sync happens after load so image specials and
        # depth specials coexist in one consistent vocabulary.
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # NEW: Rebuild depth-token metadata after load so generation code can
        # infer the correct original|continuous|discrete runtime mode.
        use_discrete = getattr(model.config, "use_discrete_depth_tokens", False)
        
        # Auto-detect from path if not set in config
        if not use_discrete and "depth_discrete" in model_path.lower():
            print(f"[INFO] Auto-detected discrete depth tokens from model path (contains 'depth_discrete')")
            use_discrete = True
            model.config.use_discrete_depth_tokens = True
            # Set default num_levels if not in config
            if not hasattr(model.config, "num_discrete_depth_levels"):
                model.config.num_discrete_depth_levels = 128
        
        # Add discrete depth tokens if configured on the model
        if use_discrete:
            num_levels = getattr(model.config, "num_discrete_depth_levels", 128)
            depth_tokens = ["<DEPTH_START>", "<DEPTH_END>"] + [f"<DEPTH_{i}>" for i in range(num_levels)]
            num_added = tokenizer.add_tokens(depth_tokens)
            if num_added > 0:
                print(f"[INFO] Added {num_added} discrete depth tokens")
            else:
                print(f"[INFO] Discrete depth tokens already in tokenizer")
            
            # Build discrete_depth_token_ids mapping if missing
            if not hasattr(model.config, "discrete_depth_token_ids") or model.config.discrete_depth_token_ids is None:
                print("[INFO] Building discrete_depth_token_ids mapping from tokenizer")
                model.config.discrete_depth_token_ids = [
                    tokenizer.convert_tokens_to_ids(f"<DEPTH_{i}>") 
                    for i in range(num_levels)
                ]
                model.config.depth_start_id = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
                model.config.depth_end_id = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
                print(f"  depth_start_id: {model.config.depth_start_id}")
                print(f"  depth_end_id: {model.config.depth_end_id}")
                print(f"  Depth token ID range: {min(model.config.discrete_depth_token_ids)}-{max(model.config.discrete_depth_token_ids)}")
            else:
                print(f"[INFO] Using saved discrete_depth_token_ids from config: {len(model.config.discrete_depth_token_ids)} levels")
            # Explicitly clear continuous-only placeholder id in discrete mode.
            model.config.depth_token_id = None
            
            # Set num_depth_tokens if not already set (critical for ablations)
            if not hasattr(model.config, "num_depth_tokens") or model.config.num_depth_tokens is None or model.config.num_depth_tokens == 0:
                # Default to 100 tokens (10x10 grid)
                model.config.num_depth_tokens = 100
                print(f"[INFO] Set num_depth_tokens to {model.config.num_depth_tokens} (default 10x10 grid)")
            else:
                print(f"[INFO] Using saved num_depth_tokens from config: {model.config.num_depth_tokens}")
        else:
            # Ensure explicit non-discrete mode metadata is present.
            model.config.use_discrete_depth_tokens = False
            if getattr(model.config, "depth_mode", None) not in {"original", "continuous"}:
                model.config.depth_mode = "continuous" if getattr(model.config, "depth_token_id", None) is not None else "original"

        sync_model_depth_runtime(model)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
