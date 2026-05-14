# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/model/llava_arch.py

"""LLaVA Architecture Components with vision-language integration and depth processing."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:
    """Base meta-model class for LLaVA with vision tower integration."""
    
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """Remove padding from a padded and resized image tensor."""
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
    
    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    """Abstract base class for LLaVA causal LM with multimodal capabilities."""
    
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, debug=False):
        if debug:
            print(f"[DEBUG encode_images] Input images shape: {images.shape}")
        image_features = self.get_model().get_vision_tower()(images)
        if debug:
            print(f"[DEBUG encode_images] After vision tower: {image_features.shape}")
        image_features = self.get_model().mm_projector(image_features)
        if debug:
            print(f"[DEBUG encode_images] After mm_projector: {image_features.shape}")
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, depth_embeds=None, depth_indices=None
    ):
        """Prepare multimodal inputs by fusing text, vision, and depth embeddings."""
        vision_tower = self.get_vision_tower()

        # NEW: Aurora resolves depth placeholder IDs here so image patches and
        # optional depth vectors are merged into one decoder input stream.
        DEPTH_START_ID = getattr(self.config, "depth_start_id", 32000)
        DEPTH_END_ID = getattr(self.config, "depth_end_id", 32001)
        # In discrete mode there is no <DEPTH_TOKEN>; avoid a hardcoded fallback
        # that can alias a valid discrete token id (e.g. <DEPTH_0>).
        DEPTH_TOKEN_ID = getattr(self.config, "depth_token_id", None)
        use_discrete_depth_tokens = getattr(self.config, "use_discrete_depth_tokens", False)
        discrete_depth_token_ids = getattr(self.config, "discrete_depth_token_ids", None)

        depth_mode = getattr(self.config, "depth_mode", "original")
        if (not use_discrete_depth_tokens) and (DEPTH_TOKEN_ID is None) and (depth_mode == "continuous"):
            raise ValueError(
                "Continuous depth mode requires config.depth_token_id to be set. "
                "Got None; check tokenizer/config initialization."
            )
        
        depth_target_features = None
        
        # Quick return for non-multimodal case
        if vision_tower is None or (images is None and depth_embeds is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        
        # Process images
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                
                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx], 
                                    self.config.image_grid_pinpoints, 
                                    self.get_vision_tower().config.image_size
                                )
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(images)
        else:
            image_features = None
        
        # NEW: Continuous depth supervision is projected into LM hidden space
        # before token assembly so the decoder sees depth slots as normal tokens.
        depth_features = None
        if depth_embeds is not None and not use_discrete_depth_tokens:
            batch_size_depth, num_tokens, depth_dim = depth_embeds.shape
            depth_embeds_flat = depth_embeds.reshape(-1, depth_dim)
            
            if hasattr(self, 'depth_projector'):
                depth_features_flat = self.depth_projector(depth_embeds_flat)
                base_model = self.get_model()
                if hasattr(base_model, 'depth_projector_freeze_forward') and base_model.depth_projector_freeze_forward:
                    depth_features_flat = depth_features_flat.detach()
                depth_features = depth_features_flat.reshape(batch_size_depth, num_tokens, -1)
            else:
                hidden_size = self.get_model().embed_tokens.embedding_dim
                if depth_dim != hidden_size:
                    raise ValueError(f"depth_projector is required when depth_dim ({depth_dim}) != hidden_size ({hidden_size})")
                depth_features = depth_embeds
            
            depth_features = depth_features.to(self.device)
            depth_target_features = depth_embeds.detach().clone()
        
        # NEW: depth_indices preserves batch-to-depth alignment after the
        # collator compacts only the samples that actually carry depth targets.
        if depth_indices is not None:
            batch_to_depth_row = {}
            for depth_row, batch_idx in enumerate(depth_indices.tolist()):
                batch_to_depth_row[batch_idx] = depth_row
        else:
            # Audit Bug #2: Always require depth_indices when depth_embeds is provided
            if depth_embeds is not None:
                raise ValueError("depth_indices must be provided when depth_embeds is not None for proper batch alignment")
            batch_to_depth_row = None
            depth_embed_idx = 0
        
        # Prepare labels and attention mask
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # Remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        new_labels = []
        new_depth_positions = []
        cur_image_idx = 0
        depth_placeholder = []
        
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # Determine depth row for this sample
            if batch_to_depth_row is not None:
                cur_depth_row = batch_to_depth_row.get(batch_idx, None)
            else:
                cur_depth_row = depth_embed_idx if depth_features is not None and depth_embed_idx < depth_features.shape[0] else None
            
            if num_images == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_depth_positions = torch.zeros_like(cur_input_ids)
                
                # NEW: Continuous depth placeholders can expand from one marker
                # to a full sequence of projected depth tokens even without image
                # patches in the sample.
                if (not use_discrete_depth_tokens) and (DEPTH_TOKEN_ID is not None) and (cur_input_ids == DEPTH_TOKEN_ID).any():
                    depth_token_positions = torch.where(cur_input_ids == DEPTH_TOKEN_ID)[0]
                    num_placeholders = len(depth_token_positions)
                    # Depth placeholders should never contribute to CE loss, even when
                    # depth embeddings are unavailable (e.g. truncated/dropped supervision).
                    labels[batch_idx][depth_token_positions] = IGNORE_INDEX
                    
                    if num_placeholders > 0 and cur_depth_row is not None and depth_features is not None:
                        cur_depth_feature = depth_features[cur_depth_row]
                        num_depth_tokens = cur_depth_feature.shape[0]
                        
                        if num_placeholders == num_depth_tokens:
                            for j, pos in enumerate(depth_token_positions):
                                cur_input_embeds[pos] = cur_depth_feature[j]
                                cur_depth_positions[pos] = 1
                                # Always mask depth token labels in continuous mode to avoid CE loss on them
                                labels[batch_idx][pos] = IGNORE_INDEX
                            if batch_to_depth_row is None:
                                depth_embed_idx += 1
                        elif num_placeholders == 1:
                            # Correct single-token expansion case
                            depth_pos = depth_token_positions[0].item()
                            before_depth = cur_input_embeds[:depth_pos]
                            after_depth = cur_input_embeds[depth_pos+1:]
                            labels_before = labels[batch_idx][:depth_pos]
                            labels_after = labels[batch_idx][depth_pos+1:]
                            depth_pos_before = cur_depth_positions[:depth_pos]
                            depth_pos_after = cur_depth_positions[depth_pos+1:]
                            
                            depth_labels = torch.full((num_depth_tokens,), IGNORE_INDEX, device=labels[batch_idx].device, dtype=labels[batch_idx].dtype)
                            depth_position_markers = torch.ones((num_depth_tokens,), device=cur_depth_positions.device, dtype=cur_depth_positions.dtype)
                            
                            cur_input_embeds = torch.cat([before_depth, cur_depth_feature, after_depth], dim=0)
                            labels[batch_idx] = torch.cat([labels_before, depth_labels, labels_after], dim=0)
                            cur_depth_positions = torch.cat([depth_pos_before, depth_position_markers, depth_pos_after], dim=0)
                            if batch_to_depth_row is None:
                                depth_embed_idx += 1
                        else:
                            # ERROR CASE: Mismatch in depth tokens
                            raise ValueError(
                                f"Mismatch in depth tokens for batch index {batch_idx}! "
                                f"Config expects {num_depth_tokens}, but found {num_placeholders} in input. "
                                "Check train.py expansion logic vs encoder config."
                            )
                
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_depth_positions.append(cur_depth_positions)
                continue
            
            # Handle samples with images
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_depth_positions_noim = []
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_depth_positions_noim.append(torch.zeros((cur_labels[image_token_indices[i]+1:image_token_indices[i+1]].shape[0],), dtype=cur_labels.dtype, device=cur_labels.device))
            
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_depth_positions = []
            
            need_to_stop = False
            for i in range(num_images + 1):
                if need_to_stop:
                    break
                    
                segment_embeds = cur_input_embeds_no_im[i].clone()
                segment_input_ids = cur_input_ids_noim[i]
                segment_depth_positions = cur_depth_positions_noim[i].clone()
                
                # NEW: Replace or expand depth placeholders inside each text
                # segment before image features are interleaved back in.
                if (not use_discrete_depth_tokens) and (DEPTH_TOKEN_ID is not None) and (segment_input_ids == DEPTH_TOKEN_ID).any():
                    depth_token_positions = torch.where(segment_input_ids == DEPTH_TOKEN_ID)[0]
                    num_placeholders = len(depth_token_positions)
                    # Keep placeholders masked for language loss regardless of whether
                    # we can replace them with depth embeddings in this batch.
                    cur_labels_noim[i][depth_token_positions] = IGNORE_INDEX
                    
                    if num_placeholders > 0 and cur_depth_row is not None and depth_features is not None and cur_depth_row < depth_features.shape[0]:
                        cur_depth_feature = depth_features[cur_depth_row]
                        num_depth_tokens = cur_depth_feature.shape[0]
                        
                        if num_placeholders == num_depth_tokens:
                            for j, pos in enumerate(depth_token_positions):
                                segment_embeds[pos] = cur_depth_feature[j]
                                segment_depth_positions[pos] = 1
                                # Always mask depth token labels in continuous mode to avoid CE loss on them
                                cur_labels_noim[i][pos] = IGNORE_INDEX
                            if batch_to_depth_row is None:
                                depth_embed_idx += 1
                        elif num_placeholders == 1:
                            # Correct single-token expansion case
                            depth_pos = depth_token_positions[0].item()
                            before_depth = segment_embeds[:depth_pos]
                            after_depth = segment_embeds[depth_pos+1:]
                            labels_before = cur_labels_noim[i][:depth_pos]
                            labels_after = cur_labels_noim[i][depth_pos+1:]
                            depth_pos_before = segment_depth_positions[:depth_pos]
                            depth_pos_after = segment_depth_positions[depth_pos+1:]
                            
                            depth_labels = torch.full((num_depth_tokens,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                            depth_position_markers = torch.ones((num_depth_tokens,), device=segment_depth_positions.device, dtype=segment_depth_positions.dtype)
                            
                            segment_embeds = torch.cat([before_depth, cur_depth_feature, after_depth], dim=0)
                            cur_labels_noim[i] = torch.cat([labels_before, depth_labels, labels_after], dim=0)
                            segment_depth_positions = torch.cat([depth_pos_before, depth_position_markers, depth_pos_after], dim=0)
                            if batch_to_depth_row is None:
                                depth_embed_idx += 1
                        else:
                            # ERROR CASE: Mismatch in depth tokens
                            raise ValueError(
                                f"Mismatch in depth tokens for segment in batch index {batch_idx}! "
                                f"Config expects {num_depth_tokens}, but found {num_placeholders} in input. "
                                "Check train.py expansion logic vs encoder config."
                            )
                
                cur_new_input_embeds.append(segment_embeds)
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_depth_positions.append(segment_depth_positions)
                
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    image_len = cur_image_features.shape[0]
                    
                    # Check if adding image would exceed max length
                    if tokenizer_model_max_length is not None and len(torch.cat(cur_new_input_embeds)) + image_len > tokenizer_model_max_length:
                        need_to_stop = True
                        depth_placeholder.append(cur_image_idx)
                    else:
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_depth_positions.append(torch.zeros((cur_image_features.shape[0],), dtype=cur_labels.dtype, device=cur_labels.device))
                    
                    cur_image_idx += 1
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_depth_positions = torch.cat(cur_new_depth_positions)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_depth_positions.append(cur_new_depth_positions)
        
        # Simple truncation
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_depth_positions = [x[:tokenizer_model_max_length] for x in new_depth_positions]
        
        # Pad to same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        depth_positions_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=new_depth_positions[0].device)
        
        for i, (cur_new_embed, cur_new_labels, cur_new_depth_pos) in enumerate(zip(new_input_embeds, new_labels, new_depth_positions)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    depth_positions_padded[i, -cur_len:] = cur_new_depth_pos
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    depth_positions_padded[i, :cur_len] = cur_new_depth_pos
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        if _position_ids is None:
            position_ids = None
        
        # NEW: These invariants catch truncation or batch-alignment bugs before
        # they silently corrupt depth supervision.
        if depth_target_features is not None:
            # Check alignment between depth_positions mask and target features
            # depth_positions_padded is (B, L), depth_target_features is (B_depth, N, D)
            expected_total = depth_target_features.shape[0] * depth_target_features.shape[1]
            actual_total = depth_positions_padded.sum().item()
            if actual_total != expected_total:
                raise ValueError(f"Global depth mask count {actual_total} != total expected tokens {expected_total}")

            # Per-sample invariant check (Audit Bug #1)
            if batch_to_depth_row is not None:
                for b_idx, d_row in batch_to_depth_row.items():
                    expected_per_sample = depth_target_features.shape[1]
                    actual_per_sample = depth_positions_padded[b_idx].sum().item()
                    if actual_per_sample != expected_per_sample:
                        raise ValueError(
                            f"Per-sample depth mismatch at batch index {b_idx}: "
                            f"found {actual_per_sample} masks, expected {expected_per_sample}. "
                            "Likely cause: truncation in collator or tokenization error."
                        )

        if new_labels is not None and (not use_discrete_depth_tokens) and (DEPTH_TOKEN_ID is not None):
            # Ensure no depth tokens leaked into labels
            # Note: DEPTH_TOKEN_ID is a special token that should always be masked
            masked_labels = new_labels[new_labels != IGNORE_INDEX]
            if (masked_labels == DEPTH_TOKEN_ID).any():
                raise ValueError("CRITICAL: Found unmasked DEPTH_TOKEN_ID in labels!")
        # ----------------------------------------
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, depth_positions_padded, depth_target_features

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        Initialize tokenizer with vision-related special tokens.
        
        Args:
            model_args: Model arguments containing tokenizer configuration
            tokenizer: Tokenizer to add special tokens to
        """
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
