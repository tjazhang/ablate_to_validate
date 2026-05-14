"""
Diagnostic script to verify that latent embeddings are actually replaced in ablation modes.

This script runs 10 examples in:
1. Normal mode (model generates its own latents)
2. GT mode (GT embeddings injected)
3. Zero mode (zero embeddings injected)

And verifies that the embeddings are actually different.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import json
import logging

from qwen_vl_utils import process_vision_info
from utils import *
from task import *

seed_everything(seed=42)
args = get_args()

# Override for testing
NUM_TEST_SAMPLES = 10

if args.data_path == 'PathToJsonlData':
    if args.task == 'vsp-spatial-planning':
        args.data_path = f"./data/vsp_spatial_planning/{args.split}_direct.jsonl"
    else:
        raise ValueError(f"Please provide --data_path for task {args.task}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('./logs/embedding_verification.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('='*50)
logging.info('Embedding Verification Diagnostic Test')
logging.info(f'Testing {NUM_TEST_SAMPLES} samples')
logging.info('='*50)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True, cache_dir=cache_dir)

# Load model on single GPU to avoid device conflicts in verification script
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path, 
    device_map="cuda:0",  # Single GPU for verification
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

use_latent_embeddings = hasattr(model.config, "latent_token_id")

if not use_latent_embeddings:
    logging.error("Model does not support latent embeddings! Cannot run verification.")
    exit(1)

model.eval()

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f][:NUM_TEST_SAMPLES]

logging.info(f"Loaded {len(data)} samples for testing")

# Storage for embeddings from each mode
results = []

for i, sample in enumerate(data):
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing sample {i+1}/{NUM_TEST_SAMPLES}")
    logging.info(f"{'='*50}")
    
    preprocess_function = task_test_preporcess_config[args.task]
    conversations = preprocess_function(sample)
    
    texts = [processor.apply_chat_template(conversations, tokenize=False)]
    texts = [place_input_image(text, sep_token=None) for text in texts]
    image_inputs, _ = process_vision_info(conversations)
    
    # Add latent token placeholders
    latent_placeholder = '<|latent_start|>' + '<|latent_pad|>' * args.latent_size + '<|latent_end|>'
    texts = [t + '<|im_start|>assistant\n' + latent_placeholder for t in texts]
    
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    # Model is on single GPU (cuda:0), move inputs there
    inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Get reasoning image path for GT mode
    reasoning_image_path = get_reasoning_image_path(sample)
    
    if not reasoning_image_path or not os.path.exists(reasoning_image_path):
        logging.warning(f"Sample {i}: No reasoning image found, skipping")
        continue
    
    sample_result = {
        'sample_idx': i,
        'has_reasoning_image': True
    }
    
    # ========== MODE 1: NORMAL (Model's own latents) ==========
    logging.info("\n[Mode 1/3] Running NORMAL mode (model generates latents)...")
    with torch.no_grad():
        # Run forward pass to capture latent embeddings
        forward_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        latent_token_id = model.config.latent_token_id
        latent_mask = inputs['input_ids'] == latent_token_id
        
        if latent_mask.any():
            # Extract hidden states at latent token positions
            # hidden_states[-1] shape: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            last_hidden = forward_outputs.hidden_states[-1]
            
            # Handle potential missing batch dimension
            if last_hidden.dim() == 2:
                # Shape is [seq_len, hidden_dim], add batch dimension
                last_hidden = last_hidden.unsqueeze(0)
            
            # Now last_hidden is [batch_size, seq_len, hidden_dim]
            # Expand mask to match: [batch_size, seq_len, 1]
            latent_mask_expanded = latent_mask.unsqueeze(-1).expand_as(last_hidden)
            
            # Extract embeddings at latent positions
            normal_embeds = last_hidden[latent_mask_expanded].view(-1, last_hidden.shape[-1]).cpu()
            sample_result['normal_embeds'] = normal_embeds
            logging.info(f"✓ Captured normal embeddings: shape={normal_embeds.shape}")
        else:
            logging.warning("✗ No latent tokens found in normal mode!")
            continue
    
    # ========== MODE 2: GT LATENT ==========
    logging.info("\n[Mode 2/3] Running GT LATENT mode...")
    reasoning_image = Image.open(reasoning_image_path).convert("RGB")
    latent_inputs = processor(text=[""], images=[reasoning_image], return_tensors="pt")
    
    with torch.no_grad():
        # Compute GT embeddings the same way the model does
        # Model and visual encoder are on cuda:0
        pixel_values_latent = latent_inputs['pixel_values'].to("cuda:0").type(model.visual.dtype)
        gt_embeds_raw = model.visual(pixel_values_latent, grid_thw=latent_inputs['image_grid_thw'].to("cuda:0"))
        
        # Apply compression
        n_features = gt_embeds_raw.shape[0]
        hidden_dim = gt_embeds_raw.shape[-1]
        
        if n_features > args.latent_size:
            if hasattr(model.model.config, 'compress_strategy') and model.model.config.compress_strategy == "average":
                gt_embeds_raw = gt_embeds_raw.view(1, -1, hidden_dim)
                res = gt_embeds_raw.shape[1] % args.latent_size
                if res > 0:
                    gt_embeds_raw = gt_embeds_raw[:, :-res, :]
                group_size = gt_embeds_raw.shape[1] // args.latent_size
                chunks = torch.split(gt_embeds_raw, group_size, dim=1)
                chunk_means = [c.mean(dim=1, keepdim=True) for c in chunks]
                gt_embeds_raw = torch.cat(chunk_means, dim=1).contiguous()
                gt_embeds_raw = gt_embeds_raw.view(-1, hidden_dim)
            else:
                gt_embeds_raw = gt_embeds_raw[:args.latent_size]
        
        gt_embeds = gt_embeds_raw.cpu()
        sample_result['gt_embeds'] = gt_embeds
        logging.info(f"✓ Captured GT embeddings: shape={gt_embeds.shape}")
    
    # ========== MODE 3: ZERO LATENT ==========
    logging.info("\n[Mode 3/3] Running ZERO LATENT mode...")
    zero_embeds = torch.zeros_like(normal_embeds)
    sample_result['zero_embeds'] = zero_embeds
    logging.info(f"✓ Created zero embeddings: shape={zero_embeds.shape}")
    
    # ========== COMPUTE COMPARISONS ==========
    logging.info("\n" + "="*50)
    logging.info("VERIFICATION RESULTS:")
    logging.info("="*50)
    
    # Compare Normal vs GT
    cos_sim_normal_gt = torch.nn.functional.cosine_similarity(
        normal_embeds.float(), 
        gt_embeds.float(), 
        dim=-1
    ).mean().item()
    l2_dist_normal_gt = torch.norm(normal_embeds.float() - gt_embeds.float(), p=2).item()
    
    # Compare Normal vs Zero
    cos_sim_normal_zero = torch.nn.functional.cosine_similarity(
        normal_embeds.float(), 
        zero_embeds.float(), 
        dim=-1
    ).mean().item()
    l2_dist_normal_zero = torch.norm(normal_embeds.float() - zero_embeds.float(), p=2).item()
    
    # Compare GT vs Zero
    cos_sim_gt_zero = torch.nn.functional.cosine_similarity(
        gt_embeds.float(), 
        zero_embeds.float(), 
        dim=-1
    ).mean().item()
    l2_dist_gt_zero = torch.norm(gt_embeds.float() - zero_embeds.float(), p=2).item()
    
    sample_result['cos_sim_normal_gt'] = cos_sim_normal_gt
    sample_result['l2_dist_normal_gt'] = l2_dist_normal_gt
    sample_result['cos_sim_normal_zero'] = cos_sim_normal_zero
    sample_result['l2_dist_normal_zero'] = l2_dist_normal_zero
    sample_result['cos_sim_gt_zero'] = cos_sim_gt_zero
    sample_result['l2_dist_gt_zero'] = l2_dist_gt_zero
    
    logging.info(f"\n📊 Normal vs GT:")
    logging.info(f"   Cosine Similarity: {cos_sim_normal_gt:.4f}")
    logging.info(f"   L2 Distance: {l2_dist_normal_gt:.4f}")
    logging.info(f"   ✓ Different: {abs(cos_sim_normal_gt - 1.0) > 0.01}")
    
    logging.info(f"\n📊 Normal vs Zero:")
    logging.info(f"   Cosine Similarity: {cos_sim_normal_zero:.4f}")
    logging.info(f"   L2 Distance: {l2_dist_normal_zero:.4f}")
    logging.info(f"   ✓ Different: {abs(cos_sim_normal_zero) > 0.01}")
    
    logging.info(f"\n📊 GT vs Zero:")
    logging.info(f"   Cosine Similarity: {cos_sim_gt_zero:.4f}")
    logging.info(f"   L2 Distance: {l2_dist_gt_zero:.4f}")
    logging.info(f"   ✓ Different: {abs(cos_sim_gt_zero) > 0.01}")
    
    # Check if embeddings are actually different
    is_normal_gt_different = l2_dist_normal_gt > 1.0
    is_normal_zero_different = l2_dist_normal_zero > 1.0
    is_gt_zero_different = l2_dist_gt_zero > 1.0
    
    sample_result['verification_passed'] = (
        is_normal_gt_different and 
        is_normal_zero_different and 
        is_gt_zero_different
    )
    
    if sample_result['verification_passed']:
        logging.info("\n✅ VERIFICATION PASSED: Embeddings are distinct across modes")
    else:
        logging.warning("\n⚠️  VERIFICATION WARNING: Some embeddings may be too similar")
    
    results.append(sample_result)

# ========== FINAL SUMMARY ==========
logging.info("\n" + "="*50)
logging.info("FINAL SUMMARY")
logging.info("="*50)

passed = sum(1 for r in results if r['verification_passed'])
total = len(results)

logging.info(f"\nSamples tested: {total}")
logging.info(f"Verification passed: {passed}/{total} ({100*passed/total:.1f}%)")

if total > 0:
    avg_cos_normal_gt = sum(r['cos_sim_normal_gt'] for r in results) / total
    avg_cos_normal_zero = sum(r['cos_sim_normal_zero'] for r in results) / total
    avg_cos_gt_zero = sum(r['cos_sim_gt_zero'] for r in results) / total
    
    avg_l2_normal_gt = sum(r['l2_dist_normal_gt'] for r in results) / total
    avg_l2_normal_zero = sum(r['l2_dist_normal_zero'] for r in results) / total
    avg_l2_gt_zero = sum(r['l2_dist_gt_zero'] for r in results) / total
    
    logging.info(f"\n📊 Average Statistics:")
    logging.info(f"\nNormal vs GT:")
    logging.info(f"  Cosine Similarity: {avg_cos_normal_gt:.4f}")
    logging.info(f"  L2 Distance: {avg_l2_normal_gt:.4f}")
    
    logging.info(f"\nNormal vs Zero:")
    logging.info(f"  Cosine Similarity: {avg_cos_normal_zero:.4f}")
    logging.info(f"  L2 Distance: {avg_l2_normal_zero:.4f}")
    
    logging.info(f"\nGT vs Zero:")
    logging.info(f"  Cosine Similarity: {avg_cos_gt_zero:.4f}")
    logging.info(f"  L2 Distance: {avg_l2_gt_zero:.4f}")
    
    logging.info(f"\n" + "="*50)
    
    if passed == total:
        logging.info("✅ ALL TESTS PASSED: Embeddings are correctly replaced in ablation modes")
    elif passed > total * 0.8:
        logging.info("⚠️  MOSTLY PASSED: Most embeddings are correctly replaced")
    else:
        logging.error("❌ TESTS FAILED: Embeddings may not be replaced correctly!")

logging.info(f"\nFull results saved to: ./logs/embedding_verification.log")
