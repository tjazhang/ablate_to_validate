# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/model/language_model/llava_llama.py

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

"""LLaVA-Llama Language Model with multimodal and depth prediction support."""

from typing import List, Optional, Tuple, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.generation import LogitsProcessor

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from torch.nn import CrossEntropyLoss, MSELoss

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX, DEPTH_TOKEN_ID, DEPTH_START_ID, DEPTH_END_ID, IMAGE_TOKEN_INDEX


# NEW: Aurora discrete-depth ablations are implemented as logits processors so
# generation can force GT, random, or zero-valued depth token spans without
# rewriting Hugging Face generation internals.
class GTDiscreteDepthLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that forces ground truth discrete depth tokens during generation.
    
    This processor overrides model predictions with GT token IDs when inside the depth
    generation region (between DEPTH_START and DEPTH_END tokens).
    """
    
    def __init__(
        self,
        gt_token_ids: List[int],
        depth_start_id: int,
        depth_end_id: int,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ):
        """
        Args:
            gt_token_ids: List of ground truth discrete depth token IDs
            depth_start_id: Token ID for <DEPTH_START>
            depth_end_id: Token ID for <DEPTH_END>
            eos_token_id: EOS token ID(s) to recognize sequence end
        """
        self.gt_token_ids = gt_token_ids
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        
        # State tracking
        self.in_depth_mode = False
        self.depth_token_idx = 0
        self.num_gt_tokens = len(gt_token_ids)
        
        print(f"[GT DEPTH DISCRETE] LogitsProcessor initialized with {self.num_gt_tokens} GT tokens")
        print(f"[GT DEPTH DISCRETE] depth_start_id={depth_start_id}, depth_end_id={depth_end_id}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to force GT tokens when in depth mode.
        
        Args:
            input_ids: [batch_size, seq_len] - generated token IDs so far
            scores: [batch_size, vocab_size] - logits for next token
        
        Returns:
            Modified scores that force the GT token
        """
        # Check the last generated token to track state
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            
            # Enter depth mode when we see DEPTH_START
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
                print(f"[GT DEPTH DISCRETE] Entered depth mode at position {input_ids.shape[1]}")
            
            # Exit depth mode when we've generated all GT tokens
            if self.in_depth_mode and self.depth_token_idx >= self.num_gt_tokens:
                print(f"[GT DEPTH DISCRETE] Exiting depth mode, forcing DEPTH_END token")
                # Force DEPTH_END token
                scores[:, :] = float('-inf')
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores
        
        # If in depth mode and have GT tokens left, force the next GT token
        if self.in_depth_mode and self.depth_token_idx < self.num_gt_tokens:
            gt_token_id = self.gt_token_ids[self.depth_token_idx]
            
            # Force this token by setting all others to -inf
            scores[:, :] = float('-inf')
            scores[:, gt_token_id] = 0.0
            
            self.depth_token_idx += 1
        
        return scores


class RandomDepthLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that forces random discrete depth tokens during generation.
    
    This processor overrides model predictions with randomly sampled token IDs when 
    inside the depth generation region (between DEPTH_START and DEPTH_END tokens).
    """
    
    def __init__(
        self,
        discrete_depth_token_ids: List[int],
        target_num_tokens: int,
        depth_start_id: int,
        depth_end_id: int,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            discrete_depth_token_ids: List of all available discrete depth token IDs
            target_num_tokens: Number of depth tokens to generate
            depth_start_id: Token ID for <DEPTH_START>
            depth_end_id: Token ID for <DEPTH_END>
            eos_token_id: EOS token ID(s) to recognize sequence end
            seed: Random seed for reproducibility (optional)
        """
        self.discrete_depth_token_ids = discrete_depth_token_ids
        self.target_num_tokens = target_num_tokens
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        
        # Generate random token sequence
        if seed is not None:
            torch.manual_seed(seed)
        self.random_token_ids = torch.tensor(discrete_depth_token_ids)[
            torch.randint(0, len(discrete_depth_token_ids), (target_num_tokens,))
        ].tolist()
        
        # State tracking
        self.in_depth_mode = False
        self.depth_token_idx = 0
        
        print(f"[RANDOM DEPTH DISCRETE] LogitsProcessor initialized with {self.target_num_tokens} random tokens")
        print(f"[RANDOM DEPTH DISCRETE] depth_start_id={depth_start_id}, depth_end_id={depth_end_id}")
        print(f"[RANDOM DEPTH DISCRETE] Random tokens (first 10): {self.random_token_ids[:10]}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to force random tokens when in depth mode.
        
        Args:
            input_ids: [batch_size, seq_len] - generated token IDs so far
            scores: [batch_size, vocab_size] - logits for next token
        
        Returns:
            Modified scores that force the random token
        """
        # Check the last generated token to track state
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            
            # Enter depth mode when we see DEPTH_START
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
                print(f"[RANDOM DEPTH DISCRETE] Entered depth mode at position {input_ids.shape[1]}")
            
            # Exit depth mode when we've generated all random tokens
            if self.in_depth_mode and self.depth_token_idx >= self.target_num_tokens:
                print(f"[RANDOM DEPTH DISCRETE] Exiting depth mode, forcing DEPTH_END token")
                # Force DEPTH_END token
                scores[:, :] = float('-inf')
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores
        
        # If in depth mode and have tokens left, force the next random token
        if self.in_depth_mode and self.depth_token_idx < self.target_num_tokens:
            random_token_id = self.random_token_ids[self.depth_token_idx]
            
            if self.depth_token_idx % 20 == 0:  # Log every 20th token
                print(f"[RANDOM DEPTH DISCRETE] Forcing token {self.depth_token_idx}/{self.target_num_tokens}: {random_token_id}")
            
            # Force this token by setting all others to -inf
            scores[:, :] = float('-inf')
            scores[:, random_token_id] = 0.0
            
            self.depth_token_idx += 1
        
        return scores


class ZeroDepthLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that forces all discrete depth tokens to be depth_0 during generation.
    
    This processor overrides model predictions with the depth_0 token ID when inside the depth
    generation region (between DEPTH_START and DEPTH_END tokens).
    """
    
    def __init__(
        self,
        depth_zero_token_id: int,
        target_num_tokens: int,
        depth_start_id: int,
        depth_end_id: int,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ):
        """
        Args:
            depth_zero_token_id: Token ID for <DEPTH_0>
            target_num_tokens: Number of depth tokens to generate
            depth_start_id: Token ID for <DEPTH_START>
            depth_end_id: Token ID for <DEPTH_END>
            eos_token_id: EOS token ID(s) to recognize sequence end
        """
        if depth_zero_token_id is None:
            raise ValueError("depth_zero_token_id must be provided for zero-depth ablation")
        if target_num_tokens <= 0:
            raise ValueError("target_num_tokens must be positive for zero-depth ablation")
        
        self.depth_zero_token_id = depth_zero_token_id
        self.target_num_tokens = target_num_tokens
        self.depth_start_id = depth_start_id
        self.depth_end_id = depth_end_id
        self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        
        # State tracking
        self.in_depth_mode = False
        self.depth_token_idx = 0
        
        print(f"[ZERO DEPTH DISCRETE] LogitsProcessor initialized with {self.target_num_tokens} zero tokens (all DEPTH_0)")
        print(f"[ZERO DEPTH DISCRETE] depth_start_id={depth_start_id}, depth_end_id={depth_end_id}, depth_zero_token_id={depth_zero_token_id}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to force DEPTH_0 token when in depth mode.
        
        Args:
            input_ids: [batch_size, seq_len] - generated token IDs so far
            scores: [batch_size, vocab_size] - logits for next token
        
        Returns:
            Modified scores that force the DEPTH_0 token
        """
        # Check the last generated token to track state
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            
            # Enter depth mode when we see DEPTH_START
            if last_token == self.depth_start_id and not self.in_depth_mode:
                self.in_depth_mode = True
                self.depth_token_idx = 0
                print(f"[ZERO DEPTH DISCRETE] Entered depth mode at position {input_ids.shape[1]}")
            
            # Exit depth mode when we've generated all zero tokens
            if self.in_depth_mode and self.depth_token_idx >= self.target_num_tokens:
                print(f"[ZERO DEPTH DISCRETE] Exiting depth mode, forcing DEPTH_END token")
                # Force DEPTH_END token
                scores[:, :] = float('-inf')
                scores[:, self.depth_end_id] = 0.0
                self.in_depth_mode = False
                return scores
        
        # If in depth mode and have tokens left, force DEPTH_0 token
        if self.in_depth_mode and self.depth_token_idx < self.target_num_tokens:
            if self.depth_token_idx % 20 == 0:  # Log every 20th token
                print(f"[ZERO DEPTH DISCRETE] Forcing DEPTH_0 token {self.depth_token_idx}/{self.target_num_tokens}")
            
            # Force DEPTH_0 token by setting all others to -inf
            scores[:, :] = float('-inf')
            scores[:, self.depth_zero_token_id] = 0.0
            
            self.depth_token_idx += 1
        
        return scores


class LlavaConfig(LlamaConfig):
    """Configuration class for LLaVA-Llama model."""
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """Base LLaVA-Llama model combining vision and language."""
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """LLaVA-Llama causal LM with multimodal and depth prediction."""
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # NEW: Normalize Aurora depth mode once during init so both training and
        # generation agree on whether this checkpoint is original, continuous, or discrete.
        self.depth_start_id = getattr(config, "depth_start_id", 32000)
        self.depth_end_id = getattr(config, "depth_end_id", 32001)
        self.use_discrete_depth_tokens = getattr(config, "use_discrete_depth_tokens", False)
        self.depth_mode = getattr(config, "depth_mode", None)
        if self.depth_mode not in {"original", "continuous", "discrete"}:
            if self.use_discrete_depth_tokens:
                self.depth_mode = "discrete"
            elif getattr(config, "depth_token_id", None) is not None:
                self.depth_mode = "continuous"
            else:
                self.depth_mode = "original"
        self.config.depth_mode = self.depth_mode
        # depth_token_id is continuous-mode only. Avoid defaulting to a fixed id
        # in discrete mode where that id can map to a valid <DEPTH_i> token.
        if self.use_discrete_depth_tokens or self.depth_mode == "original":
            self.depth_token_id = None
        else:
            self.depth_token_id = getattr(config, "depth_token_id", None)
        self.num_depth_tokens = getattr(config, "num_depth_tokens", 16)
        self.num_discrete_depth_levels = getattr(config, "num_discrete_depth_levels", 128)
        self.discrete_depth_token_ids = getattr(config, "discrete_depth_token_ids", None)
        self.discrete_depth_supervision_mode = getattr(config, "discrete_depth_supervision_mode", "ground_truth")
        if self.discrete_depth_supervision_mode not in {"ground_truth", "random", "ignore"}:
            raise ValueError(
                f"Unsupported discrete_depth_supervision_mode={self.discrete_depth_supervision_mode}. "
                "Use ground_truth|random|ignore."
            )
        
        # Depth loss configuration
        self.depth_coef = getattr(config, 'depth_coef', 1.0)
        self.normalize_depth = getattr(config, 'depth_normalize', False)
        self.apply_depth_softmax = getattr(config, 'depth_apply_softmax', False)
        self.depth_temperature = getattr(config, 'depth_temperature', 1.0)
        self.depth_loss_type = getattr(config, 'depth_loss_type', 'mse')
        
        # Validate loss configuration
        if self.apply_depth_softmax and not self.normalize_depth:
            raise ValueError("apply_depth_softmax=True requires normalize_depth=True")
        if self.depth_loss_type == "cosine" and not self.normalize_depth:
            raise ValueError("depth_loss_type='cosine' requires normalize_depth=True")
        if self.depth_loss_type == "softmax" and not self.apply_depth_softmax:
            raise ValueError("depth_loss_type='softmax' requires apply_depth_softmax=True")
        if self.depth_loss_type not in {"mse", "cosine", "softmax"}:
            raise ValueError(f"Unsupported depth_loss_type={self.depth_loss_type}")
        
        # NEW: Continuous Aurora depth uses a projector into LM hidden space and
        # an optional reverse head for autoregressive depth supervision.
        depth_embedding_dim = getattr(config, "depth_embedding_dim", 768)
        if depth_embedding_dim <= 0:
            depth_embedding_dim = 768
        
        if not self.use_discrete_depth_tokens:
            depth_head_type = getattr(config, 'depth_head_type', 'linear')
            depth_projector_type = getattr(config, "depth_projector_type", "linear")
            
            if depth_head_type == 'linear' and depth_projector_type == 'linear':
                # Weight tying: use depth_projector.weight transpose for depth_head
                self.depth_projector = nn.Linear(depth_embedding_dim, config.hidden_size)
                self.depth_head = None
                print(f"[INFO] Using weight tying for depth_head and depth_projector")
            else:
                if depth_head_type == 'linear':
                    self.depth_head = nn.Linear(config.hidden_size, depth_embedding_dim)
                elif depth_head_type == 'mlp':
                    self.depth_head = nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, depth_embedding_dim),
                    )
                elif depth_head_type == 'mlp2x_gelu':
                    self.depth_head = nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, depth_embedding_dim),
                    )
                else:
                    self.depth_head = nn.Linear(config.hidden_size, depth_embedding_dim)
                
                if depth_projector_type == "linear":
                    self.depth_projector = nn.Linear(depth_embedding_dim, config.hidden_size)
                elif depth_projector_type == "mlp":
                    self.depth_projector = nn.Sequential(
                        nn.Linear(depth_embedding_dim, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, config.hidden_size),
                    )
                elif depth_projector_type == "mlp2x_gelu":
                    self.depth_projector = nn.Sequential(
                        nn.Linear(depth_embedding_dim, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, config.hidden_size),
                    )
                else:
                    self.depth_projector = nn.Linear(depth_embedding_dim, config.hidden_size)
        else:
            self.depth_head = None
            self.depth_projector = None
        
        self.depth_embedding_dim = depth_embedding_dim
        self.loss_mode = getattr(config, "loss_mode", "local")
        self._depth_loss_layer_logged = False
        self.post_init()
        self._log_model_config(config, depth_embedding_dim)
    
    def _log_model_config(self, config, depth_embedding_dim):
        print(f"\n{'='*60}")
        print("LLaVA Model Initialized")
        print(f"{'='*60}")
        print(f"Architecture:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  embed_tokens shape: {self.model.embed_tokens.weight.shape}")
        print(f"  lm_head shape: {self.lm_head.weight.shape}")
        
        # Verify consistency
        if self.vocab_size != self.model.embed_tokens.weight.shape[0]:
            print(f"  [WARNING] vocab_size ({self.vocab_size}) != embed_tokens size ({self.model.embed_tokens.weight.shape[0]})")
        if self.vocab_size != self.lm_head.weight.shape[0]:
            print(f"  [WARNING] vocab_size ({self.vocab_size}) != lm_head size ({self.lm_head.weight.shape[0]})")
        
        print(f"\nDepth Token IDs:")
        print(f"  depth_start_id: {self.depth_start_id}")
        print(f"  depth_end_id: {self.depth_end_id}")
        if not self.use_discrete_depth_tokens:
            print(f"  depth_token_id: {self.depth_token_id}")
        
        print(f"\nDepth Configuration:")
        print(f"  mode: {self.depth_mode}")
        if self.depth_mode == "discrete":
            print(f"  num_discrete_levels: {self.num_discrete_depth_levels}")
            if hasattr(self, 'discrete_depth_token_ids') and self.discrete_depth_token_ids:
                print(f"  token_ids (first 5): {self.discrete_depth_token_ids[:5]}")
            print(f"  supervision_mode: {self.discrete_depth_supervision_mode}")
        elif self.depth_mode == "continuous":
            print(f"  depth_embedding_dim: {depth_embedding_dim}")
            print(f"  num_depth_tokens: {self.num_depth_tokens}")
            print(f"  depth_coef: {self.depth_coef}")
            print(f"  depth_loss_type: {self.depth_loss_type}")
            print(f"  depth_loss_layer: {getattr(config, 'depth_loss_layer', -1)}")
            print(f"  normalize_depth: {self.normalize_depth}")
            print(f"  apply_depth_softmax: {self.apply_depth_softmax}")
            print(f"  depth_temperature: {self.depth_temperature}")
        else:
            print("  depth disabled for original mode")
        print(f"\nTraining Configuration:")
        print(f"  loss_mode: {self.loss_mode}")

    def get_model(self):
        return self.model

    def _resolve_depth_loss_layer(self) -> int:
        layer_idx = int(getattr(self.config, "depth_loss_layer", -1))
        num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if layer_idx == -1:
            return layer_idx
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"Invalid depth_loss_layer={layer_idx}. Expected -1 or 0..{max(num_layers - 1, 0)}"
            )
        return layer_idx

    def _infer_depth_mode(self) -> str:
        mode = getattr(self.config, "depth_mode", None)
        if mode in {"original", "continuous", "discrete"}:
            return mode
        if bool(getattr(self.config, "use_discrete_depth_tokens", False)) or self.use_discrete_depth_tokens:
            return "discrete"
        if getattr(self.config, "depth_token_id", None) is not None:
            return "continuous"
        if getattr(self, "depth_token_id", None) is not None:
            return "continuous"
        return "original"

    def _extract_hidden_states_tuple(self, outputs, use_cache: bool):
        if hasattr(outputs, "hidden_states"):
            return outputs.hidden_states
        if not isinstance(outputs, tuple):
            return None
        hs_index = 2 if use_cache else 1
        if len(outputs) <= hs_index:
            return None
        return outputs[hs_index]

    def _select_depth_hidden_for_loss(self, outputs, final_hidden_states: torch.Tensor, use_cache: bool) -> torch.Tensor:
        # NEW: Aurora can supervise depth from an intermediate decoder block, not
        # just the final hidden state, so select that tensor here once per step.
        depth_loss_layer = self._resolve_depth_loss_layer()
        if depth_loss_layer == -1:
            selected_hidden = final_hidden_states
        else:
            all_hidden_states = self._extract_hidden_states_tuple(outputs, use_cache=use_cache)
            if all_hidden_states is None:
                raise RuntimeError(
                    "depth_loss_layer requires output hidden states, but none were returned by the decoder."
                )
            hs_idx = depth_loss_layer + 1  # hidden_states[0]=embeddings, hidden_states[k+1]=after block k
            if hs_idx >= len(all_hidden_states):
                raise RuntimeError(
                    f"Requested depth_loss_layer={depth_loss_layer}, but only {len(all_hidden_states)} hidden tensors were returned."
                )
            selected_hidden = all_hidden_states[hs_idx]

        if not self._depth_loss_layer_logged:
            num_layers = int(getattr(self.config, "num_hidden_layers", 0))
            print(
                f"[DEPTH CONFIG] depth_loss_layer={depth_loss_layer}, num_hidden_layers={num_layers}, "
                f"selected_hidden_shape={tuple(selected_hidden.shape)}, final_hidden_shape={tuple(final_hidden_states.shape)}"
            )
            self._depth_loss_layer_logged = True
        return selected_hidden

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        depth_embeddings: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        depth_indices: Optional[torch.LongTensor] = None,
        original_indices: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with multimodal inputs and depth prediction."""
        
        # # Print rank and original indices for each forward pass
        # import os
        # rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', -1)))
        # orig_idx = original_indices.tolist() if original_indices is not None else None
        # print(f"Rank {rank}: {orig_idx}")
        
        depth_positions = None
        depth_target_features = None
        
        if depth_embeddings is not None:
            if self.use_discrete_depth_tokens:
                raise ValueError("depth_embeddings should not be provided when using discrete depth tokens")
            if depth_embeddings.size(-1) != self.depth_embedding_dim:
                raise ValueError(f"Depth embedding dimension mismatch: expected {self.depth_embedding_dim}, got {depth_embeddings.size(-1)}")
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                depth_positions,
                depth_target_features
            ) = self.prepare_inputs_labels_for_multimodal(input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels, images, image_sizes, depth_embeds=depth_embeddings, depth_indices=depth_indices)

        return self.llm_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            depth_positions=depth_positions,
            depth_target_features=depth_target_features
        )

    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        depth_positions: Optional[torch.LongTensor] = None,
        depth_target_features: Optional[torch.FloatTensor] = None,
        decoding: bool = False,
        override_depth_vec: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Core forward with language modeling and depth prediction losses."""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        depth_loss_layer = self._resolve_depth_loss_layer()
        depth_supervision_active = (
            self.training
            and (not self.use_discrete_depth_tokens)
            and depth_positions is not None
            and depth_target_features is not None
        )
        needs_hidden_states = (
            decoding or
            (depth_supervision_active and depth_loss_layer != -1) or
            (output_hidden_states is True)
        )
        output_hidden_states = needs_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pred_depth_vec = None
        
        if decoding and not self.use_discrete_depth_tokens and self.depth_projector is not None:
            last_hidden_state = hidden_states[:, -1, :]
            
            if override_depth_vec is not None:
                pred_depth_vec = override_depth_vec
            
            elif self.depth_head is None:
                pred_depth_vec = last_hidden_state @ self.depth_projector.weight
                print(f"[DEPTH DEBUG] llm_forward: Computing depth from hidden state (weight tying)")
            else:
                pred_depth_vec = self.depth_head(last_hidden_state)
                print(f"[DEPTH DEBUG] llm_forward: Computing depth from hidden state (depth_head)")
            
            if self.normalize_depth:
                pred_depth_vec = F.normalize(pred_depth_vec, p=2, dim=-1)
            if self.apply_depth_softmax:
                pred_depth_vec = F.softmax(pred_depth_vec / self.depth_temperature, dim=-1)
            
            projected_depth = self.depth_projector(pred_depth_vec)
            hidden_states[:, -1, :] = projected_depth

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        self.loss_language = 0.0
        self.loss_depth_ar = 0.0
        self.loss_total = 0.0
        
        if labels is not None:
            # NEW: Language CE stays upstream-like, while Aurora optionally
            # swaps or masks depth-token labels depending on the supervision mode.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_labels = shift_labels
            IGNORE = IGNORE_INDEX
            
            # Safety check: ensure depth placeholder tokens are masked in labels
            if self.training and not self.use_discrete_depth_tokens and (self.depth_token_id is not None):
                if (shift_labels == self.depth_token_id).any():
                    print(f"\n[ASSERTION DEBUG] Found depth token {self.depth_token_id} in shift_labels!")
                    print(f"[ASSERTION DEBUG] shift_labels.shape: {shift_labels.shape}")
                    print(f"[ASSERTION DEBUG] use_discrete_depth_tokens: {self.use_discrete_depth_tokens}")
                    print(f"[ASSERTION DEBUG] depth_token_id: {self.depth_token_id}")
                    
                    # Find positions where depth token appears
                    for batch_idx in range(shift_labels.shape[0]):
                        depth_mask = (shift_labels[batch_idx] == self.depth_token_id)
                        if depth_mask.any():
                            positions = torch.where(depth_mask)[0]
                            print(f"[ASSERTION DEBUG] Batch {batch_idx}: depth tokens at positions {positions.tolist()}")
                            print(f"[ASSERTION DEBUG] Batch {batch_idx}: context around first position:")
                            first_pos = positions[0].item()
                            start = max(0, first_pos - 5)
                            end = min(shift_labels.shape[1], first_pos + 6)
                            print(f"[ASSERTION DEBUG]   shift_labels[{batch_idx}, {start}:{end}] = {shift_labels[batch_idx, start:end].tolist()}")
                    
                    assert False, "Depth placeholder tokens must be masked as IGNORE_INDEX in labels before CE loss"

            if self.use_discrete_depth_tokens and self.discrete_depth_supervision_mode != "ground_truth":
                ce_labels = shift_labels.clone()
                discrete_ids = self.discrete_depth_token_ids or []
                if len(discrete_ids) > 0:
                    discrete_ids_tensor = torch.tensor(discrete_ids, device=ce_labels.device, dtype=ce_labels.dtype)
                    depth_value_mask = (ce_labels[..., None] == discrete_ids_tensor).any(dim=-1)
                else:
                    depth_value_mask = torch.zeros_like(ce_labels, dtype=torch.bool)

                depth_boundary_mask = torch.zeros_like(ce_labels, dtype=torch.bool)
                if self.depth_start_id is not None:
                    depth_boundary_mask |= (ce_labels == self.depth_start_id)
                if self.depth_end_id is not None:
                    depth_boundary_mask |= (ce_labels == self.depth_end_id)

                if self.discrete_depth_supervision_mode == "ignore":
                    ce_labels[depth_value_mask | depth_boundary_mask] = IGNORE
                elif self.discrete_depth_supervision_mode == "random":
                    if len(discrete_ids) > 0 and depth_value_mask.any():
                        random_idx = torch.randint(
                            low=0,
                            high=len(discrete_ids),
                            size=(int(depth_value_mask.sum().item()),),
                            device=ce_labels.device,
                        )
                        ce_labels[depth_value_mask] = discrete_ids_tensor[random_idx]
            
            # Language CE loss
            ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            nll = ce_loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                             ce_labels.view(-1))
            valid = (ce_labels.view(-1) != IGNORE)
            nll = nll * valid
            ce_num = nll.sum()
            ce_den = valid.to(ce_num.dtype).sum()

            # Global loss reduction (Audit Bug #2)
            if self.loss_mode == "global" and torch.distributed.is_initialized():
                torch.distributed.all_reduce(ce_num, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(ce_den, op=torch.distributed.ReduceOp.SUM)
            
            ce_loss = ce_num / (ce_den.clamp_min(1.0))
            
            loss = ce_loss
            self.loss_language = ce_loss.item()
            
            if self.use_discrete_depth_tokens:
                self.loss_depth_ar = 0.0
            else:
                # NEW: Continuous depth loss is computed only at the positions
                # created from depth placeholders in llava_arch.py.
                depth_sum = hidden_states.new_zeros(())
                depth_count = torch.tensor(0, dtype=torch.float32, device=hidden_states.device)
                N = 0
                
                if depth_target_features is not None:
                    depth_hidden_states = self._select_depth_hidden_for_loss(
                        outputs=outputs,
                        final_hidden_states=hidden_states,
                        use_cache=use_cache,
                    )
                    if depth_positions is None:
                        assert False, "depth_positions required for depth loss"
                    next_is_depth = depth_positions[:, 1:].bool()
                    slots_per_sample = next_is_depth.sum(dim=1)
                    N = int(slots_per_sample.sum().item())
                    
                    B, T, H = depth_hidden_states.size()
                    
                    if N > 0:
                        hs_for_depth = depth_hidden_states[:, :-1, :]
                        h_at_slots = hs_for_depth[next_is_depth]
                        
                        if self.depth_head is None:
                            pred_depth = h_at_slots @ self.depth_projector.weight
                        else:
                            pred_depth = self.depth_head(h_at_slots)
                        
                        rank_flat = (next_is_depth.cumsum(dim=1) - 1)[next_is_depth]
                        b_idx = torch.arange(B, device=next_is_depth.device).unsqueeze(1).expand(B, T - 1)
                        b_flat = b_idx[next_is_depth]
                        if depth_target_features.size(0) == B:
                            tgt = depth_target_features[b_flat, rank_flat, :]
                        else:
                            valid_rows = (slots_per_sample > 0)
                            compact = torch.full((B,), -1, device=next_is_depth.device, dtype=torch.long)
                            compact[valid_rows] = torch.arange(valid_rows.sum(), device=compact.device, dtype=torch.long)
                            d_flat = compact[b_flat]
                            if not (d_flat >= 0).all():
                                assert False, "Depth target row mapping failed (mixed batch)"
                            tgt = depth_target_features[d_flat, rank_flat, :]
                        
                        tgt = tgt.to(dtype=pred_depth.dtype)
                        
                        if self.normalize_depth:
                            pred_depth = F.normalize(pred_depth, p=2, dim=-1)
                            tgt = F.normalize(tgt, p=2, dim=-1)
                        if getattr(self, "depth_loss_type", "mse") == "mse":
                            per_ex = (pred_depth - tgt).pow(2).mean(dim=-1)
                        elif self.depth_loss_type == "cosine":
                            per_ex = 1.0 - F.cosine_similarity(pred_depth, tgt, dim=-1)
                        elif self.depth_loss_type == "softmax":
                            if self.apply_depth_softmax:
                                pred_depth = F.softmax(pred_depth / self.depth_temperature, dim=-1)
                            eps = 1e-10
                            pred_prob = torch.clamp(pred_depth, min=eps)
                            per_ex = -(tgt * torch.log(pred_prob)).sum(dim=-1)
                        else:
                            if self.apply_depth_softmax:
                                pred_depth = F.softmax(pred_depth / self.depth_temperature, dim=-1)
                                per_ex = -(tgt * torch.log(pred_depth + 1e-10)).sum(dim=-1)
                            elif self.normalize_depth:
                                per_ex = 1.0 - F.cosine_similarity(pred_depth, tgt, dim=-1)
                            else:
                                per_ex = (pred_depth - tgt).pow(2).mean(dim=-1)
                        depth_sum = per_ex.sum()
                        depth_count = torch.tensor(per_ex.numel(), dtype=depth_sum.dtype, device=per_ex.device)

                # Global loss reduction (Audit Bug #2)
                if self.loss_mode == "global" and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(depth_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(depth_count, op=torch.distributed.ReduceOp.SUM)

                if depth_count.item() > 0:
                    loss_depth = depth_sum / depth_count.clamp_min(1)
                else:
                    loss_depth = hidden_states.new_zeros(())
                
                # Debug logging for loss scaling
                if self.training and getattr(self, "step_count", 0) < 10:
                    self.step_count = getattr(self, "step_count", 0) + 1
                    if torch.distributed.get_rank() == 0:
                        print(f"[LOSS_DEBUG] step={self.step_count} ce_den={ce_den.item()} depth_count={depth_count.item()} depth_coef={self.depth_coef}")

                loss += self.depth_coef * loss_depth
                self.loss_depth_ar = loss_depth.item()
            
            self.loss_total = loss.item()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if decoding:
            output_obj = CausalLMOutputWithPast(
                loss=pred_depth_vec,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )
            output_obj.pred_depth_vec = pred_depth_vec
            return output_obj

        output_obj = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output_obj.loss_language = getattr(self, "loss_language", None)
        output_obj.loss_depth_ar = getattr(self, "loss_depth_ar", None)
        output_obj.loss_total = getattr(self, "loss_total", None)
        return output_obj

    @torch.no_grad()
    def greedy_decode(
        self,
        position_ids,
        attention_mask, 
        inputs_embeds,
        start_depth_token_id=None,
        end_depth_token_id=None,
        eos_token_id=None,
        max_new_tokens=1024,
        use_cache=True,
        output_depth=False,
        raw_hidden_state=False,
        force_end_after_k=True,
        use_gt_depth_embeddings=False,
        gt_depth_embeddings: Optional[torch.Tensor] = None,
        use_random_depth=False,
        use_zero_depth=False,
        use_model_depth=False,
        use_first_depth_repeat=False,
        use_random_depth_gt_dist=False,
        gt_depth_mean: Optional[float] = None,
        gt_depth_std: Optional[float] = None,
        **kwargs
    ):
        """Custom greedy decoding with K-step gated autoregressive depth generation.
        
        Uses KV cache for O(1) per-step cost instead of recomputing the full sequence.
        On the first step (prefill), the entire inputs_embeds is processed and cached.
        On subsequent steps, only the last embedding (1 token) is passed along with
        the cached past_key_values.
        """
        
        vocab_size = int(self.lm_head.out_features)
        if start_depth_token_id is None:
            start_depth_token_id = getattr(self.config, 'depth_start_id', 32000)
        if end_depth_token_id is None:
            end_depth_token_id = getattr(self.config, 'depth_end_id', 32001)
        start_depth_token_valid = isinstance(start_depth_token_id, int) and 0 <= start_depth_token_id < vocab_size
        end_depth_token_valid = isinstance(end_depth_token_id, int) and 0 <= end_depth_token_id < vocab_size
        if not start_depth_token_valid or not end_depth_token_valid:
            print(
                f"[DEPTH DEBUG] Invalid depth boundary ids for vocab_size={vocab_size}: "
                f"start={start_depth_token_id}, end={end_depth_token_id}"
            )
        
        target_depth_count = int(self.num_depth_tokens)
        
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        
        past_key_values = None
        in_depth_mode = False
        depth_steps = 0
        generated_ids_list = []
        depth_embeds_list = []
        text_steps = 0
        use_cache = True
        is_prefill = True
        next_position_id = None
        depth_exit_pending = False
        
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)
        
        batch_size = inputs_embeds.size(0)
        if use_gt_depth_embeddings and batch_size != 1:
            raise ValueError("use_gt_depth_embeddings currently supports batch size 1.")
        if use_random_depth and batch_size != 1:
            raise ValueError("use_random_depth currently supports batch size 1.")
        if use_zero_depth and batch_size != 1:
            raise ValueError("use_zero_depth currently supports batch size 1.")
        if use_model_depth and batch_size != 1:
            raise ValueError("use_model_depth currently supports batch size 1.")
        if use_first_depth_repeat and batch_size != 1:
            raise ValueError("use_first_depth_repeat currently supports batch size 1.")
        
        # Validate ablation flags - only one can be active
        ablation_flags = [use_gt_depth_embeddings, use_random_depth, use_zero_depth, use_model_depth, use_first_depth_repeat, use_random_depth_gt_dist]
        if sum(ablation_flags) > 1:
            raise ValueError("Only one ablation mode can be active: GT depth, random depth, zero depth, model depth, first-depth-repeat, or random-depth-gt-dist.")
        
        # NEW: These ablation buffers let Aurora override model-predicted depth
        # vectors during greedy decode without touching the text-generation path.
        gt_depth_seq = None
        gt_depth_idx = 0
        if use_gt_depth_embeddings:
            if gt_depth_embeddings is None:
                raise ValueError("gt_depth_embeddings must be provided when use_gt_depth_embeddings=True.")
            if not isinstance(gt_depth_embeddings, torch.Tensor):
                gt_depth_embeddings = torch.as_tensor(gt_depth_embeddings)
            if gt_depth_embeddings.dim() == 3:
                if gt_depth_embeddings.size(0) != 1:
                    raise ValueError("gt_depth_embeddings batch dimension must equal 1.")
                gt_depth_seq = gt_depth_embeddings.squeeze(0)
            elif gt_depth_embeddings.dim() == 2:
                gt_depth_seq = gt_depth_embeddings
            else:
                raise ValueError("gt_depth_embeddings must have shape [T, D] or [1, T, D].")
            # Get dtype from depth_projector (handle both Linear and Sequential)
            if self.depth_projector is not None:
                if isinstance(self.depth_projector, nn.Linear):
                    target_dtype = self.depth_projector.weight.dtype
                else:
                    # For Sequential, get the first linear layer's dtype
                    target_dtype = next(self.depth_projector.parameters()).dtype
            else:
                target_dtype = inputs_embeds.dtype
            gt_depth_seq = gt_depth_seq.to(inputs_embeds.device).to(target_dtype)
            print(f"[GT DEPTH] Initialized GT sequence: shape={gt_depth_seq.shape}, dtype={gt_depth_seq.dtype}, device={gt_depth_seq.device}")
            print(f"[GT DEPTH] Target depth count: {target_depth_count}, GT sequence length: {gt_depth_seq.size(0)}")
            if gt_depth_seq.size(0) != target_depth_count:
                print(f"[DEPTH DEBUG] WARNING: GT depth embeddings len {gt_depth_seq.size(0)} != target_depth_count {target_depth_count}")
        
        # Initialize random depth embeddings
        random_depth_seq = None
        random_depth_idx = 0
        if use_random_depth:
            if self.depth_projector is not None:
                if isinstance(self.depth_projector, nn.Linear):
                    depth_dim = self.depth_projector.in_features
                    target_dtype = self.depth_projector.weight.dtype
                else:
                    first_layer = list(self.depth_projector.children())[0]
                    depth_dim = first_layer.in_features
                    target_dtype = next(self.depth_projector.parameters()).dtype
            else:
                depth_dim = inputs_embeds.size(-1)
                target_dtype = inputs_embeds.dtype
            
            random_depth_seq = torch.rand(target_depth_count, depth_dim, device=inputs_embeds.device, dtype=target_dtype) * 2.0 - 1.0
            print(f"[RANDOM DEPTH] Generated random sequence: shape={random_depth_seq.shape}, dtype={random_depth_seq.dtype}, device={random_depth_seq.device}")
            print(f"[RANDOM DEPTH] Stats: min={random_depth_seq.min():.4f}, max={random_depth_seq.max():.4f}, mean={random_depth_seq.mean():.4f}")

        # Initialize distribution-matched random depth embeddings (GT distribution)
        random_depth_gt_dist_seq = None
        random_depth_gt_dist_idx = 0
        if use_random_depth_gt_dist:
            if self.depth_projector is not None:
                if isinstance(self.depth_projector, nn.Linear):
                    depth_dim = self.depth_projector.in_features
                    target_dtype = self.depth_projector.weight.dtype
                else:
                    first_layer = list(self.depth_projector.children())[0]
                    depth_dim = first_layer.in_features
                    target_dtype = next(self.depth_projector.parameters()).dtype
            else:
                depth_dim = inputs_embeds.size(-1)
                target_dtype = inputs_embeds.dtype

            mean = gt_depth_mean if gt_depth_mean is not None else 0.0
            std = gt_depth_std if gt_depth_std is not None else 1.0
            random_depth_gt_dist_seq = torch.randn(target_depth_count, depth_dim, device=inputs_embeds.device, dtype=target_dtype) * std + mean
            print(f"[RANDOM DEPTH GT DIST] Generated distribution-matched random sequence: shape={random_depth_gt_dist_seq.shape}")
            print(f"[RANDOM DEPTH GT DIST] Target distribution: mean={mean:.4f}, std={std:.4f}")
            print(f"[RANDOM DEPTH GT DIST] Actual stats: mean={random_depth_gt_dist_seq.mean():.4f}, std={random_depth_gt_dist_seq.std():.4f}")
        
        # Initialize zero depth embeddings
        zero_depth_seq = None
        zero_depth_idx = 0
        if use_zero_depth:
            if self.depth_projector is not None:
                if isinstance(self.depth_projector, nn.Linear):
                    depth_dim = self.depth_projector.in_features
                    target_dtype = self.depth_projector.weight.dtype
                else:
                    first_layer = list(self.depth_projector.children())[0]
                    depth_dim = first_layer.in_features
                    target_dtype = next(self.depth_projector.parameters()).dtype
            else:
                depth_dim = inputs_embeds.size(-1)
                target_dtype = inputs_embeds.dtype
            
            zero_depth_seq = torch.zeros(target_depth_count, depth_dim, device=inputs_embeds.device, dtype=target_dtype)
            print(f"[ZERO DEPTH] Generated zero sequence: shape={zero_depth_seq.shape}, dtype={zero_depth_seq.dtype}, device={zero_depth_seq.device}")
            print(f"[ZERO DEPTH] All values are zero (max abs value: {zero_depth_seq.abs().max():.4f})")
        
        # Model depth mode: store model predictions and inject them back
        model_depth_seq = None
        model_depth_idx = 0
        prev_pred_depth_vec = None
        if use_model_depth:
            print(f"[MODEL DEPTH] Model depth ablation enabled (sanity check mode)")
            print(f"[MODEL DEPTH] Will copy model's predictions and inject them back to test injection pipeline")

        # First-depth-repeat mode: keep the first model-predicted depth vector and
        # repeat it for the rest of the depth span.
        first_repeat_depth_vec = None
        first_repeat_depth_idx = 0
        if use_first_depth_repeat:
            print(f"[FIRST DEPTH REPEAT] Enabled")
            print(f"[FIRST DEPTH REPEAT] First depth token uses model prediction; remaining depth tokens repeat that first vector")
        
        gt_depth_warned = False
        random_depth_warned = False
        zero_depth_warned = False
        random_depth_gt_dist_warned = False
        print(f"[DEPTH DEBUG] Starting generation (KV cache enabled): max_new_tokens={max_new_tokens}, target_depth_count={target_depth_count}, force_end_after_k={force_end_after_k}")
        print(f"[DEPTH DEBUG] Initial inputs_embeds shape: {inputs_embeds.shape}, attention_mask shape: {attention_mask.shape}")
        
        while True:
            override_depth_vec = None
            ablation_type = None
            
            # Priority: GT depth > Random depth > Zero depth > Model depth > Model prediction (no injection)
            if in_depth_mode and use_gt_depth_embeddings and gt_depth_seq is not None:
                if gt_depth_idx < gt_depth_seq.size(0):
                    override_depth_vec = gt_depth_seq[gt_depth_idx].unsqueeze(0)
                    ablation_type = "GT"
                elif not gt_depth_warned:
                    print("[DEPTH DEBUG] WARNING: Exhausted GT depth embeddings; falling back to model predictions.")
                    gt_depth_warned = True
            elif in_depth_mode and use_random_depth and random_depth_seq is not None:
                if random_depth_idx < random_depth_seq.size(0):
                    override_depth_vec = random_depth_seq[random_depth_idx].unsqueeze(0)
                    ablation_type = "RANDOM"
                    if random_depth_idx % 20 == 0:
                        print(f"[RANDOM DEPTH] Using random embedding at index {random_depth_idx}/{random_depth_seq.size(0)}, shape={override_depth_vec.shape}")
                elif not random_depth_warned:
                    print("[DEPTH DEBUG] WARNING: Exhausted random depth embeddings; falling back to model predictions.")
                    random_depth_warned = True
            elif in_depth_mode and use_random_depth_gt_dist and random_depth_gt_dist_seq is not None:
                if random_depth_gt_dist_idx < random_depth_gt_dist_seq.size(0):
                    override_depth_vec = random_depth_gt_dist_seq[random_depth_gt_dist_idx].unsqueeze(0)
                    ablation_type = "RANDOM_GT_DIST"
                    if random_depth_gt_dist_idx % 20 == 0:
                        print(f"[RANDOM DEPTH GT DIST] Using dist-matched random embedding at index {random_depth_gt_dist_idx}/{random_depth_gt_dist_seq.size(0)}")
                elif not random_depth_gt_dist_warned:
                    print("[DEPTH DEBUG] WARNING: Exhausted random-depth-gt-dist embeddings; falling back to model predictions.")
                    random_depth_gt_dist_warned = True
            elif in_depth_mode and use_zero_depth and zero_depth_seq is not None:
                if zero_depth_idx < zero_depth_seq.size(0):
                    override_depth_vec = zero_depth_seq[zero_depth_idx].unsqueeze(0)
                    ablation_type = "ZERO"
                    if zero_depth_idx % 20 == 0:
                        print(f"[ZERO DEPTH] Using zero embedding at index {zero_depth_idx}/{zero_depth_seq.size(0)}, shape={override_depth_vec.shape}")
                elif not zero_depth_warned:
                    print("[DEPTH DEBUG] WARNING: Exhausted zero depth embeddings; falling back to model predictions.")
                    zero_depth_warned = True
            elif in_depth_mode and use_model_depth:
                if prev_pred_depth_vec is not None:
                    override_depth_vec = prev_pred_depth_vec
                    ablation_type = "MODEL"
                    if model_depth_idx % 20 == 0:
                        print(f"[MODEL DEPTH] Using stored model prediction from previous iteration at index {model_depth_idx}")
                else:
                    ablation_type = "MODEL_INIT"
                    if model_depth_idx == 0:
                        print(f"[MODEL DEPTH] First depth token - capturing model's prediction")
            elif in_depth_mode and use_first_depth_repeat:
                if first_repeat_depth_vec is not None:
                    override_depth_vec = first_repeat_depth_vec
                    ablation_type = "FIRST_REPEAT"
                    if first_repeat_depth_idx % 20 == 0:
                        print(f"[FIRST DEPTH REPEAT] Reusing first depth vector at index {first_repeat_depth_idx}")
                else:
                    ablation_type = "FIRST_REPEAT_INIT"
                    if first_repeat_depth_idx == 0:
                        print(f"[FIRST DEPTH REPEAT] First depth token - capturing model prediction")
            
            if is_prefill:
                # First step: process full sequence, populate KV cache
                cur_inputs_embeds = inputs_embeds
                cur_attention_mask = attention_mask
                cur_position_ids = position_ids
            else:
                # Subsequent steps: only pass the last (new) embedding
                # KV cache already has all previous positions
                cur_inputs_embeds = next_input_embed
                # Attention mask must cover all positions (cached + new) for proper causal masking
                one = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, one], dim=1)
                cur_attention_mask = attention_mask
                # Position ID for the new token
                cur_position_ids = next_position_id.unsqueeze(0) if next_position_id is not None else None
            
            outputs = self.llm_forward(
                input_ids=None,
                attention_mask=cur_attention_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=cur_inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                output_hidden_states=True,
                decoding=in_depth_mode,
                override_depth_vec=override_depth_vec,
            )
            
            past_key_values = outputs.past_key_values
            is_prefill = False
            
            # Track position for next token
            if next_position_id is None:
                next_position_id = torch.tensor([inputs_embeds.size(1)], dtype=torch.long, device=inputs_embeds.device)
            else:
                next_position_id = next_position_id + 1
            
            # After depth mode ends, the last depth hidden state was just processed
            # to update the KV cache. Now inject the END token and resume text generation.
            if depth_exit_pending:
                depth_exit_pending = False
                if end_depth_token_valid:
                    generated_ids_list.append(end_depth_token_id)
                    end_token_tensor = torch.tensor([[end_depth_token_id]], device=inputs_embeds.device)
                    next_input_embed = self.model.embed_tokens(end_token_tensor)
                    text_steps += 1
                else:
                    print("[DEPTH DEBUG] Skipping END depth token injection due to invalid token id.")
                continue
            
            logits = outputs.logits[:, -1, :]
            
            if not in_depth_mode and end_depth_token_valid:
                logits[:, end_depth_token_id] = float('-inf')
            
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            next_token_id = int(next_token.item())
            next_token_embed = self.model.embed_tokens(next_token)
            
            if (not in_depth_mode) and start_depth_token_valid and next_token_id == start_depth_token_id:
                print(f"[DEPTH DEBUG] Entering depth mode at text_step {text_steps}, target_count={target_depth_count}")
                in_depth_mode = True
                depth_steps = 0
                generated_ids_list.append(next_token_id)
                next_input_embed = next_token_embed
                text_steps += 1
                continue
            
            if in_depth_mode and (depth_steps < target_depth_count):
                depth_steps += 1
                
                pdv = override_depth_vec if override_depth_vec is not None else getattr(outputs, 'pred_depth_vec', None)
                
                if use_model_depth and pdv is not None:
                    prev_pred_depth_vec = pdv.clone().detach()
                    if model_depth_idx == 0:
                        print(f"[MODEL DEPTH] Captured first depth vector, shape={prev_pred_depth_vec.shape}")
                        print(f"[MODEL DEPTH] Subsequent iterations will inject this through override mechanism")
                    model_depth_idx += 1

                if use_first_depth_repeat and pdv is not None:
                    if first_repeat_depth_vec is None:
                        first_repeat_depth_vec = pdv.clone().detach()
                        print(f"[FIRST DEPTH REPEAT] Captured first depth vector, shape={first_repeat_depth_vec.shape}")
                        print(f"[FIRST DEPTH REPEAT] Subsequent depth steps will reuse this vector")
                    first_repeat_depth_idx += 1
                
                if pdv is not None:
                    depth_embeds_list.append(pdv)
                    if ablation_type == "GT":
                        if depth_steps % 20 == 0:
                            print(f"[GT DEPTH] Depth step {depth_steps}/{target_depth_count}: Used GT embedding (idx={gt_depth_idx})")
                    elif ablation_type == "RANDOM":
                        if depth_steps % 20 == 0:
                            print(f"[RANDOM DEPTH] Depth step {depth_steps}/{target_depth_count}: Used random embedding (idx={random_depth_idx})")
                    elif ablation_type == "ZERO":
                        if depth_steps % 20 == 0:
                            print(f"[ZERO DEPTH] Depth step {depth_steps}/{target_depth_count}: Used zero embedding (idx={zero_depth_idx})")
                    elif ablation_type == "MODEL":
                        if depth_steps % 20 == 0:
                            print(f"[MODEL DEPTH] Depth step {depth_steps}/{target_depth_count}: Injected stored model prediction")
                    elif ablation_type == "MODEL_INIT":
                        if depth_steps % 20 == 0:
                            print(f"[MODEL DEPTH] Depth step {depth_steps}/{target_depth_count}: Using natural model prediction (capturing for next step)")
                    elif ablation_type == "FIRST_REPEAT":
                        if depth_steps % 20 == 0:
                            print(f"[FIRST DEPTH REPEAT] Depth step {depth_steps}/{target_depth_count}: Repeated first depth vector")
                    elif ablation_type == "FIRST_REPEAT_INIT":
                        if depth_steps % 20 == 0:
                            print(f"[FIRST DEPTH REPEAT] Depth step {depth_steps}/{target_depth_count}: Using first natural model prediction")
                    else:
                        if depth_steps % 20 == 0:
                            print(f"[DEPTH DEBUG] Depth step {depth_steps}/{target_depth_count}: Used model prediction")
                else:
                    print(f"[DEPTH DEBUG] WARNING: pred_depth_vec is None at depth step {depth_steps}")
                
                # Use last hidden state as the next input embedding (autoregressive depth)
                next_embed_from_hidden = outputs.hidden_states[:, -1, :].unsqueeze(1)
                next_input_embed = next_embed_from_hidden
                
                if depth_steps >= target_depth_count:
                    print(f"[DEPTH DEBUG] Target count reached. Forcing exit and appending END token.")
                    in_depth_mode = False
                    depth_exit_pending = True
                    
                # Increment the appropriate index counter
                if ablation_type == "GT" and gt_depth_idx < target_depth_count:
                    gt_depth_idx += 1
                elif ablation_type == "RANDOM" and random_depth_idx < target_depth_count:
                    random_depth_idx += 1
                elif ablation_type == "ZERO" and zero_depth_idx < target_depth_count:
                    zero_depth_idx += 1
                elif ablation_type == "RANDOM_GT_DIST" and random_depth_gt_dist_idx < target_depth_count:
                    random_depth_gt_dist_idx += 1
                continue
            
            generated_ids_list.append(next_token_id)
            next_input_embed = next_token_embed
            text_steps += 1
            
            if (not in_depth_mode) and (next_token_id in eos_token_id):
                print(f"[DEPTH DEBUG] Stopping: Hit EOS token {next_token_id}")
                break
            
            if (not in_depth_mode) and (text_steps >= max_new_tokens):
                print(f"[DEPTH DEBUG] Stopping: Hit max_new_tokens limit ({text_steps} >= {max_new_tokens})")
                break
        print(f"[DEPTH DEBUG] Generation complete. Total generated_ids: {len(generated_ids_list)}, Total depth embeds: {len(depth_embeds_list)}")
        print(f"[DEPTH DEBUG] Generated token IDs: {generated_ids_list[:20]}")  # First 20 tokens
        if use_gt_depth_embeddings and gt_depth_seq is not None:
            print(f"[GT DEPTH] Summary: Used {gt_depth_idx}/{gt_depth_seq.size(0)} GT embeddings during generation")
        if use_random_depth and random_depth_seq is not None:
            print(f"[RANDOM DEPTH] Summary: Used {random_depth_idx}/{random_depth_seq.size(0)} random embeddings during generation")
        if use_zero_depth and zero_depth_seq is not None:
            print(f"[ZERO DEPTH] Summary: Used {zero_depth_idx}/{zero_depth_seq.size(0)} zero embeddings during generation")
        if use_random_depth_gt_dist and random_depth_gt_dist_seq is not None:
            print(f"[RANDOM DEPTH GT DIST] Summary: Used {random_depth_gt_dist_idx}/{random_depth_gt_dist_seq.size(0)} distribution-matched random embeddings")
        if use_model_depth:
            print(f"[MODEL DEPTH] Summary: Captured and injected model's own predictions via override mechanism")
            print(f"[MODEL DEPTH] Total injected: {model_depth_idx-1}/{target_depth_count} (first token natural, rest injected)")
            print(f"[MODEL DEPTH] This tests the injection pipeline - should closely match baseline")
        if use_first_depth_repeat:
            repeated = max(first_repeat_depth_idx - 1, 0)
            print(f"[FIRST DEPTH REPEAT] Summary: Captured first depth vector once and reused it for {repeated}/{target_depth_count} subsequent depth tokens")
        
        if depth_embeds_list:
            depth_embeds_tensor = torch.cat(depth_embeds_list, dim=0)
        else:
            depth_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]

        if output_depth:
            return output, depth_embeds_tensor
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        depth_embeddings: Optional[torch.Tensor] = None,
        depth_indices: Optional[torch.LongTensor] = None,
        use_customize_greedy: bool = True,
        output_depth: bool = False,
        raw_hidden_state: bool = False,
        use_gt_depth_embeddings: bool = False,
        gt_depth_embeddings: Optional[torch.Tensor] = None,
        gt_discrete_token_ids: Optional[List[int]] = None,
        use_random_depth: bool = False,
        use_zero_depth: bool = False,
        use_model_depth: bool = False,
        use_first_depth_repeat: bool = False,
        use_random_depth_gt_dist: bool = False,
        gt_depth_mean: Optional[float] = None,
        gt_depth_std: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate text (and optionally depth) from multimodal inputs."""
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or depth_embeddings is not None:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _, _, _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None, None,
                images,
                image_sizes=image_sizes,
                depth_embeds=depth_embeddings,
                depth_indices=depth_indices
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        gt_depth_tensor = None
        if gt_depth_embeddings is not None:
            if not isinstance(gt_depth_embeddings, torch.Tensor):
                gt_depth_tensor = torch.as_tensor(gt_depth_embeddings)
            else:
                gt_depth_tensor = gt_depth_embeddings
        if gt_depth_tensor is not None:
            if gt_depth_tensor.dim() == 2:
                gt_depth_tensor = gt_depth_tensor.unsqueeze(0)
            elif gt_depth_tensor.dim() != 3:
                raise ValueError("gt_depth_embeddings must have shape [B, T, D] or [T, D].")
            batch_size = inputs_embeds.size(0)
            if gt_depth_tensor.size(0) not in (1, batch_size):
                if gt_depth_tensor.size(0) == 1:
                    gt_depth_tensor = gt_depth_tensor.repeat(batch_size, 1, 1)
                else:
                    raise ValueError("gt_depth_embeddings batch dimension mismatch.")
            gt_depth_tensor = gt_depth_tensor.to(inputs_embeds.device)
            # Get dtype from depth_projector (handle both Linear and Sequential)
            if self.depth_projector is not None:
                if isinstance(self.depth_projector, nn.Linear):
                    target_dtype = self.depth_projector.weight.dtype
                else:
                    # For Sequential, get the first linear layer's dtype
                    target_dtype = next(self.depth_projector.parameters()).dtype
            else:
                target_dtype = inputs_embeds.dtype
            gt_depth_tensor = gt_depth_tensor.to(target_dtype)
        elif use_gt_depth_embeddings:
            raise ValueError("gt_depth_embeddings must be provided when use_gt_depth_embeddings=True.")

        depth_mode = self._infer_depth_mode()
        continuous_mode = depth_mode == "continuous"
        discrete_mode = depth_mode == "discrete" or self.use_discrete_depth_tokens
        if continuous_mode:
            vocab_size = int(self.lm_head.out_features)
            start_id = getattr(self.config, "depth_start_id", None)
            end_id = getattr(self.config, "depth_end_id", None)
            token_id = getattr(self.config, "depth_token_id", None)
            start_ok = isinstance(start_id, int) and 0 <= start_id < vocab_size
            end_ok = isinstance(end_id, int) and 0 <= end_id < vocab_size
            token_ok = isinstance(token_id, int) and 0 <= token_id < vocab_size
            if not (start_ok and end_ok and token_ok):
                raise ValueError(
                    "Continuous depth mode requires valid depth token ids within vocab range. "
                    f"Got start={start_id}, end={end_id}, token={token_id}, vocab_size={vocab_size}."
                )

        if discrete_mode and (use_gt_depth_embeddings or gt_depth_tensor is not None):
            raise ValueError("Ground truth depth embeddings are only supported for continuous depth tokens.")
        if discrete_mode and use_first_depth_repeat:
            raise ValueError("First-depth-repeat ablation is only supported for continuous depth tokens.")

        if gt_depth_tensor is not None and not use_customize_greedy:
            raise ValueError("Ground truth depth embeddings require use_customize_greedy=True.")
        
        # Validate ablation flags - only one can be active
        ablation_flags = [
            use_gt_depth_embeddings or (gt_depth_tensor is not None),
            use_random_depth,
            use_zero_depth,
            use_model_depth,
            use_first_depth_repeat,
            use_random_depth_gt_dist,
        ]
        if sum(ablation_flags) > 1:
            raise ValueError("Only one ablation mode can be active: GT depth, random depth, zero depth, model depth, first-depth-repeat, or random-depth-gt-dist.")

        gt_depth_flag = gt_depth_tensor is not None
        if gt_depth_flag:
            print(f"[GT DEPTH] generate() called with GT embeddings: shape={gt_depth_tensor.shape}, dtype={gt_depth_tensor.dtype}")
        
        if use_random_depth:
            print(f"[RANDOM DEPTH] generate() called with random depth ablation enabled")
        
        if use_zero_depth:
            print(f"[ZERO DEPTH] generate() called with zero depth ablation enabled")
        
        if use_model_depth:
            print(f"[MODEL DEPTH] generate() called with model depth ablation enabled (sanity check)")

        if use_first_depth_repeat:
            print(f"[FIRST DEPTH REPEAT] generate() called with first-depth-repeat ablation enabled")

        if use_random_depth_gt_dist:
            print(f"[RANDOM DEPTH GT DIST] generate() called with distribution-matched random depth ablation enabled (mean={gt_depth_mean}, std={gt_depth_std})")

        depth_ablation_requested = any([
            gt_depth_flag,
            use_random_depth,
            use_zero_depth,
            use_model_depth,
            use_first_depth_repeat,
            use_random_depth_gt_dist,
            output_depth,
        ])
        if not continuous_mode and not discrete_mode and depth_ablation_requested:
            print(
                f"[WARN] Depth generation flags requested while depth_mode={depth_mode}. "
                "Falling back to standard text generation and ignoring depth controls."
            )
            output_depth = False
            gt_depth_flag = False
            use_random_depth = False
            use_zero_depth = False
            use_model_depth = False
            use_first_depth_repeat = False
            use_random_depth_gt_dist = False

        # For discrete depth tokens, use standard generation
        if discrete_mode:
            # If GT discrete tokens are provided, inject them using a LogitsProcessor
            if gt_discrete_token_ids is not None and len(gt_discrete_token_ids) > 0:
                print(f"[GT DEPTH DISCRETE] Injecting {len(gt_discrete_token_ids)} GT tokens during generation")
                
                # Get depth token IDs from config
                depth_start_id = getattr(self.config, 'depth_start_id', None)
                depth_end_id = getattr(self.config, 'depth_end_id', None)
                eos_token_id = kwargs.get('eos_token_id', self.config.eos_token_id)
                
                if depth_start_id is None or depth_end_id is None:
                    print("[WARNING] depth_start_id or depth_end_id not found in config; cannot inject GT discrete tokens")
                else:
                    # Create LogitsProcessor to force GT tokens
                    gt_processor = GTDiscreteDepthLogitsProcessor(
                        gt_token_ids=gt_discrete_token_ids,
                        depth_start_id=depth_start_id,
                        depth_end_id=depth_end_id,
                        eos_token_id=eos_token_id,
                    )
                    
                    # Add to logits_processor list
                    from transformers.generation import LogitsProcessorList
                    logits_processor = kwargs.get('logits_processor', LogitsProcessorList())
                    if not isinstance(logits_processor, LogitsProcessorList):
                        logits_processor = LogitsProcessorList([logits_processor])
                    logits_processor.append(gt_processor)
                    kwargs['logits_processor'] = logits_processor
            
            # If random depth is requested, inject random tokens using a LogitsProcessor
            elif use_random_depth:
                print(f"[RANDOM DEPTH DISCRETE] Injecting random tokens during generation")
                
                # Get depth token IDs and configuration from config
                depth_start_id = getattr(self.config, 'depth_start_id', None)
                depth_end_id = getattr(self.config, 'depth_end_id', None)
                discrete_depth_token_ids = getattr(self.config, 'discrete_depth_token_ids', None)
                eos_token_id = kwargs.get('eos_token_id', self.config.eos_token_id)
                
                # Calculate target number of tokens (usually grid_size^2)
                target_num_tokens = int(self.num_depth_tokens) if hasattr(self, 'num_depth_tokens') else 256
                
                if depth_start_id is None or depth_end_id is None:
                    print("[WARNING] depth_start_id or depth_end_id not found in config; cannot inject random discrete tokens")
                elif discrete_depth_token_ids is None or len(discrete_depth_token_ids) == 0:
                    print("[WARNING] discrete_depth_token_ids not found in config; cannot inject random discrete tokens")
                else:
                    # Create LogitsProcessor to force random tokens
                    random_processor = RandomDepthLogitsProcessor(
                        discrete_depth_token_ids=discrete_depth_token_ids,
                        target_num_tokens=target_num_tokens,
                        depth_start_id=depth_start_id,
                        depth_end_id=depth_end_id,
                        eos_token_id=eos_token_id,
                    )
                    
                    # Add to logits_processor list
                    from transformers.generation import LogitsProcessorList
                    logits_processor = kwargs.get('logits_processor', LogitsProcessorList())
                    if not isinstance(logits_processor, LogitsProcessorList):
                        logits_processor = LogitsProcessorList([logits_processor])
                    logits_processor.append(random_processor)
                    kwargs['logits_processor'] = logits_processor
            
            # If zero depth is requested, inject DEPTH_0 tokens using a LogitsProcessor
            elif use_zero_depth:
                print(f"[ZERO DEPTH DISCRETE] Injecting all DEPTH_0 tokens during generation")
                
                # Get depth token IDs and configuration from config
                depth_start_id = getattr(self.config, 'depth_start_id', None)
                depth_end_id = getattr(self.config, 'depth_end_id', None)
                discrete_depth_token_ids = getattr(self.config, 'discrete_depth_token_ids', None)
                eos_token_id = kwargs.get('eos_token_id', self.config.eos_token_id)
                
                # Calculate target number of tokens (usually grid_size^2)
                target_num_tokens = int(self.num_depth_tokens) if hasattr(self, 'num_depth_tokens') else 256
                
                if depth_start_id is None or depth_end_id is None:
                    print("[WARNING] depth_start_id or depth_end_id not found in config; cannot inject zero discrete tokens")
                elif discrete_depth_token_ids is None or len(discrete_depth_token_ids) == 0:
                    print("[WARNING] discrete_depth_token_ids not found in config; cannot inject zero discrete tokens")
                else:
                    # Get DEPTH_0 token ID (first token in the list)
                    depth_zero_token_id = discrete_depth_token_ids[0]
                    
                    # Create LogitsProcessor to force DEPTH_0 tokens
                    zero_processor = ZeroDepthLogitsProcessor(
                        depth_zero_token_id=depth_zero_token_id,
                        target_num_tokens=target_num_tokens,
                        depth_start_id=depth_start_id,
                        depth_end_id=depth_end_id,
                        eos_token_id=eos_token_id,
                    )
                    
                    # Add to logits_processor list
                    from transformers.generation import LogitsProcessorList
                    logits_processor = kwargs.get('logits_processor', LogitsProcessorList())
                    if not isinstance(logits_processor, LogitsProcessorList):
                        logits_processor = LogitsProcessorList([logits_processor])
                    logits_processor.append(zero_processor)
                    kwargs['logits_processor'] = logits_processor
            
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        elif use_customize_greedy and continuous_mode:
            return self.greedy_decode(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_depth=output_depth,
                raw_hidden_state=raw_hidden_state,
                use_gt_depth_embeddings=gt_depth_flag,
                gt_depth_embeddings=gt_depth_tensor,
                use_random_depth=use_random_depth,
                use_zero_depth=use_zero_depth,
                use_model_depth=use_model_depth,
                use_first_depth_repeat=use_first_depth_repeat,
                use_random_depth_gt_dist=use_random_depth_gt_dist,
                gt_depth_mean=gt_depth_mean,
                gt_depth_std=gt_depth_std,
                **kwargs
            )
        else:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        depth_embeddings = kwargs.pop("depth_embeddings", None)
        
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if depth_embeddings is not None:
            inputs['depth_embeddings'] = depth_embeddings
            
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
