"""
Test script for evaluating models with latent token reasoning.

Architecture:
- GT embeddings: visual_encoder(reasoning_image) + compression
- Normal mode: Compare model predictions vs GT embeddings
- Ablation modes (random/zero/GT): Handled by transformers library during generate()
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import json
import logging
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content

from utils_new import *
from task_new import *

seed_everything(seed=42)
args=get_args()

if args.data_path == 'PathToJsonlData':
    if args.task == 'vsp-spatial-planning':
        args.data_path = f"./data/vsp_spatial_planning/{args.split}_direct.jsonl"
    else:
        raise ValueError(f"Please provide --data_path for task {args.task}")

logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Check for incompatible ablation flags
if sum([args.use_random_latent, args.use_zero_latent, args.use_gt_latent, args.use_gt_latent_count_matched, args.use_model_latent, args.use_random_latent_gt_dist, args.use_random_latent_model_dist, args.use_first_latent_repeat]) > 1:
    raise ValueError("Cannot enable more than one of --use-random-latent, --use-zero-latent, --use-gt-latent, --use-gt-latent-count-matched, --use-model-latent, --use-random-latent-gt-dist, --use-random-latent-model-dist, and --use-first-latent-repeat simultaneously.")
    
# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True, cache_dir=cache_dir)
# Use single GPU to avoid device conflicts
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

# Ensure latent_size is set in both config and model.config
if not hasattr(model.config, 'latent_size'):
    model.config.latent_size = args.latent_size
    logging.info(f"Set model.config.latent_size = {args.latent_size}")
if not hasattr(model.model.config, 'latent_size'):
    model.model.config.latent_size = args.latent_size
    logging.info(f"Set model.model.config.latent_size = {args.latent_size}")

# Check if model supports latent embeddings
use_latent_embeddings = hasattr(model.config, "latent_token_id")

if (args.use_random_latent or args.use_zero_latent or args.use_gt_latent or args.use_gt_latent_count_matched or args.use_model_latent or args.use_random_latent_gt_dist or args.use_random_latent_model_dist or args.use_first_latent_repeat) and not use_latent_embeddings:
    logging.warning("[WARNING] Latent ablations/GT requested but model does not use latent embeddings; options will be ignored.")

if args.use_random_latent:
    logging.info("[INFO] Random latent ablation enabled - replacing latent hidden states with random embeddings (sampled from standard normal N(0,1)).")
if args.use_zero_latent:
    logging.info("[INFO] Zero-latent ablation enabled - replacing latent hidden states with zero embeddings.")
if args.use_gt_latent:
    logging.info("[INFO] GT latent enabled (Mode 1) - using GT embeddings with fixed count from config (same as training).")
if args.use_gt_latent_count_matched:
    logging.info("[INFO] GT latent count-matched enabled (Mode 2) - two-pass generation with GT compressed to actual model count.")
if args.use_model_latent:
    logging.info("[INFO] Model latent enabled - reusing model output (sanity check, should match baseline).")
if args.use_first_latent_repeat:
    logging.info("[INFO] First-latent-repeat ablation enabled - first model latent is reused for all remaining latent steps.")
if args.use_random_latent_gt_dist:
    logging.info("[INFO] Random latent (GT distribution) ablation enabled - replacing latent hidden states with random embeddings using GT distribution.")
if args.use_random_latent_model_dist:
    logging.info("[INFO] Random latent (model distribution) ablation enabled - replacing latent hidden states with random embeddings using model's own generation distribution.")

model.eval()

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

    correct, invalid = 0, 0
    latent_cosine_losses = []  # Track cosine similarity losses for latent tokens (normal mode only)
    results_to_save = [] # NEW: for saving to json
    
    # Track distribution statistics for all ablations
    gt_means = []
    gt_stds = []
    model_output_means = []  # Model's original output (what gets replaced in ablations)
    model_output_stds = []
    model_means = []  # For random_latent_model_dist: model distribution from first pass
    model_stds = []
    used_means = []  # For the actual embeddings used (random, zero, GT, or model)
    used_stds = []
    
    # Initialize model tracking attributes
    if not hasattr(model, '_ablation_stats'):
        model._ablation_stats = {
            'model_output_means': [],
            'model_output_stds': []
        }
    
    for i, sample in tqdm(enumerate(data)):

        preprocess_function = task_test_preporcess_config[args.task]
        conversations = preprocess_function(sample)

        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        texts = [place_input_image(text, sep_token=None) for text in texts]
        image_inputs, _ = process_vision_info(conversations)

        # Let model generate latent tokens itself (don't include in prompt)
        texts = [t + '<|im_start|>assistant' for t in texts]
        
        inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        # Model is on single GPU (cuda:0), move inputs there
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Prepare GT latent embeddings for comparison (needed for normal mode and GT mode)
        gt_latent_embeds = None
        latent_inputs = None
        reasoning_image_path = get_reasoning_image_path(sample)
        
        if use_latent_embeddings and reasoning_image_path:
            from PIL import Image
            reasoning_image = Image.open(reasoning_image_path).convert("RGB")
            latent_inputs = processor(text=[""], images=[reasoning_image], return_tensors="pt")
            
            # Compute GT embeddings for loss calculation (normal mode) or injection (GT mode)
            with torch.no_grad():
                pixel_values_latent = latent_inputs['pixel_values'].to("cuda:0").type(model.visual.dtype)
                gt_embeds = model.visual(pixel_values_latent, grid_thw=latent_inputs['image_grid_thw'].to("cuda:0"))
                
                # Apply compression if needed
                n_features = gt_embeds.shape[0]
                hidden_dim = gt_embeds.shape[-1]
                if n_features > args.latent_size:
                    if hasattr(model.model.config, 'compress_strategy') and model.model.config.compress_strategy == "average":
                        gt_embeds = gt_embeds.view(1, -1, hidden_dim)
                        res = gt_embeds.shape[1] % args.latent_size
                        if res > 0:
                            gt_embeds = gt_embeds[:, :-res, :]
                        group_size = gt_embeds.shape[1] // args.latent_size
                        chunks = torch.split(gt_embeds, group_size, dim=1)
                        chunk_means = [c.mean(dim=1, keepdim=True) for c in chunks]
                        gt_embeds = torch.cat(chunk_means, dim=1).contiguous()
                        gt_embeds = gt_embeds.view(-1, hidden_dim)
                    else:
                        gt_embeds = gt_embeds[:args.latent_size]
                
                gt_latent_embeds = gt_embeds  # Shape: [latent_size, hidden_dim]
                
        elif use_latent_embeddings and not reasoning_image_path:
            if args.use_gt_latent:
                logging.warning(f"[Sample {i}] Reasoning image not found for GT mode. Please run generate_gt_images.py first.")

        with torch.no_grad():
            # Prepare generation inputs
            generate_inputs = inputs.copy()
            
            # Initialize gt_latent_count_matched early (will be set later for Mode 2)
            gt_latent_count_matched = None
            
            # For GT Mode 1: add pixel_values_latent for injection during generate()
            if args.use_gt_latent and latent_inputs is not None:
                generate_inputs['pixel_values_latent'] = latent_inputs['pixel_values'].to("cuda:0")
                generate_inputs['image_grid_thw_latent'] = latent_inputs['image_grid_thw'].to("cuda:0")
            
            # Track GT distribution for all ablation modes (for comparison)
            gt_latent_mean = None
            gt_latent_std = None
            if gt_latent_embeds is not None:
                gt_latent_mean = gt_latent_embeds.mean().item()
                gt_latent_std = gt_latent_embeds.std().item()
                gt_means.append(gt_latent_mean)
                gt_stds.append(gt_latent_std)
                
                # Track what embeddings will be used based on ablation mode (for final summary)
                if args.use_zero_latent:
                    used_means.append(0.0)
                    used_stds.append(0.0)
                elif args.use_gt_latent:
                    used_means.append(gt_latent_mean)
                    used_stds.append(gt_latent_std)
                elif args.use_random_latent_gt_dist:
                    # Random with GT distribution - will match GT stats on average
                    used_means.append(gt_latent_mean)
                    used_stds.append(gt_latent_std)
            
            # For GT latent count-matched (Mode 2): Two-pass generation
            # Pass 1: Generate normally to get actual token count
            # Pass 2: Compress GT to that count and use
            if args.use_gt_latent_count_matched and use_latent_embeddings and latent_inputs is not None:
                # First pass: normal generation to get token count
                first_pass_output = model.generate(
                    **generate_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    tokenizer=processor.tokenizer,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                
                # Count latent tokens in first pass
                latent_token_id = model.config.latent_token_id
                latent_mask = first_pass_output.sequences == latent_token_id
                latent_count = latent_mask.sum().item()
                
                if latent_count > 0:
                    gt_latent_count_matched = latent_count
                    if i < 5 or (i+1) % 50 == 0:
                        logging.info(f"[Sample {i}] Mode 2: First pass generated {latent_count} latent tokens")
                        logging.info(f"[Sample {i}] Mode 2: Will compress GT from {gt_latent_embeds.shape[0] if gt_latent_embeds is not None else 'unknown'} to {latent_count} tokens")
                    
                    # Now add the Mode 2 inputs for the second pass
                    generate_inputs['pixel_values_latent'] = latent_inputs['pixel_values'].to("cuda:0")
                    generate_inputs['image_grid_thw_latent'] = latent_inputs['image_grid_thw'].to("cuda:0")
                    generate_inputs['gt_target_count'] = gt_latent_count_matched
                else:
                    logging.warning(f"[Sample {i}] Mode 2: No latent tokens in first pass, skipping GT injection")
            
            # For random latent with model's own generation distribution: 
            # First do a normal generation to get the distribution, then regenerate with random
            model_latent_mean = None
            model_latent_std = None
            if args.use_random_latent_model_dist and use_latent_embeddings:
                # First pass: normal generation to collect model's latent embeddings
                first_pass_output = model.generate(
                    **generate_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    tokenizer=processor.tokenizer,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                
                # Extract latent embeddings from first pass
                latent_token_id = model.config.latent_token_id
                latent_mask = first_pass_output.sequences == latent_token_id
                
                if latent_mask.any() and hasattr(first_pass_output, 'hidden_states') and first_pass_output.hidden_states:
                    try:
                        generated_latent_embeds = []
                        latent_positions = torch.where(latent_mask[0])[0]
                        
                        for pos in latent_positions:
                            gen_step = pos - generate_inputs['input_ids'].shape[1]
                            if gen_step >= 0 and gen_step < len(first_pass_output.hidden_states):
                                step_hidden = first_pass_output.hidden_states[gen_step][-1]
                                if step_hidden.dim() == 3:
                                    generated_latent_embeds.append(step_hidden[0, -1, :])
                                elif step_hidden.dim() == 2:
                                    generated_latent_embeds.append(step_hidden[-1, :])
                        
                        if len(generated_latent_embeds) > 0:
                            generated_latent_embeds = torch.stack(generated_latent_embeds)
                            model_latent_mean = generated_latent_embeds.mean().item()
                            model_latent_std = generated_latent_embeds.std().item()
                            model_means.append(model_latent_mean)
                            model_stds.append(model_latent_std)
                            if i < 5 or (i+1) % 50 == 0:
                                logging.info(f"[Sample {i}] Model latent distribution - Mean: {model_latent_mean:.6f}, Std: {model_latent_std:.6f}")
                                logging.info(f"[Sample {i}] Using Random with Model dist - Mean: {model_latent_mean:.6f}, Std: {model_latent_std:.6f}")
                            used_means.append(model_latent_mean)
                            used_stds.append(model_latent_std)
                    except Exception as e:
                        logging.warning(f"[Sample {i}] Failed to extract model latent distribution: {e}")
            
            # Ablation modes (random/zero/GT/model) are handled by transformers library during generate()
            # For normal mode, also request output_hidden_states to compute latent loss
            output = model.generate(
                **generate_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
                use_random_latent=args.use_random_latent and use_latent_embeddings,
                use_zero_latent=args.use_zero_latent and use_latent_embeddings,
                use_gt_latent=args.use_gt_latent and use_latent_embeddings,
                use_gt_latent_count_matched=args.use_gt_latent_count_matched and use_latent_embeddings,
                use_model_latent=args.use_model_latent and use_latent_embeddings,
                use_first_latent_repeat=args.use_first_latent_repeat and use_latent_embeddings,
                use_random_latent_gt_dist=args.use_random_latent_gt_dist and use_latent_embeddings,
                use_random_latent_model_dist=args.use_random_latent_model_dist and use_latent_embeddings,
                gt_latent_mean=gt_latent_mean,
                gt_latent_std=gt_latent_std,
                model_latent_mean=model_latent_mean,
                model_latent_std=model_latent_std,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            output_ids = output.sequences
            
            # Extract model output statistics from this generation (if available)
            if hasattr(model, '_ablation_stats') and model._ablation_stats:
                if model._ablation_stats['model_output_means']:
                    # Take the most recent statistic (from this sample)
                    model_output_means.append(model._ablation_stats['model_output_means'][-1])
                    model_output_stds.append(model._ablation_stats['model_output_stds'][-1])
            
            # Compute latent loss for normal mode (no ablations) or model mode (sanity check)
            if not (args.use_random_latent or args.use_zero_latent or args.use_gt_latent or args.use_gt_latent_count_matched or args.use_random_latent_gt_dist or args.use_random_latent_model_dist or args.use_first_latent_repeat) and gt_latent_embeds is not None and use_latent_embeddings:
                # Find latent token positions in generated output
                latent_token_id = model.config.latent_token_id
                latent_mask = output_ids == latent_token_id
                
                # DEBUG: Check if latent tokens are present
                n_latents_found = latent_mask.sum().item()
                if i < 5: # Only print for first few samples
                    logging.info(f"[DEBUG Sample {i}] Latent token ID: {latent_token_id}")
                    logging.info(f"[DEBUG Sample {i}] Latents found in output: {n_latents_found}")
                    logging.info(f"[DEBUG Sample {i}] Expected latent_size: {args.latent_size}")

                if latent_mask.any() and hasattr(output, 'hidden_states') and output.hidden_states:
                    # Extract hidden states at latent token positions from the generated sequence
                    # hidden_states is a tuple of tuples: (step1_layers, step2_layers, ...)
                    # We need the last layer of each generation step
                    try:
                        # Collect hidden states for latent token positions
                        generated_latent_embeds = []
                        
                        # Get the positions of latent tokens in the sequence
                        latent_positions = torch.where(latent_mask[0])[0]  # [0] for batch dim
                        
                        # For each latent position, get the corresponding hidden state
                        for pos in latent_positions:
                            # Position in generation steps (subtract input length)
                            gen_step = pos - inputs['input_ids'].shape[1]
                            if gen_step >= 0 and gen_step < len(output.hidden_states):
                                # Get last layer hidden state for this step
                                step_hidden = output.hidden_states[gen_step][-1]  # Last layer
                                if step_hidden.dim() == 3:  # [batch, seq, hidden]
                                    generated_latent_embeds.append(step_hidden[0, -1, :])  # Last token of this step
                                elif step_hidden.dim() == 2:  # [seq, hidden]
                                    generated_latent_embeds.append(step_hidden[-1, :])
                        
                        if len(generated_latent_embeds) > 0:
                            generated_latent_embeds = torch.stack(generated_latent_embeds)  # [num_latents, hidden_dim]
                            
                            # Compare with GT
                            if generated_latent_embeds.shape[0] == gt_latent_embeds.shape[0]:
                                cosine_sim = torch.nn.functional.cosine_similarity(
                                    gt_latent_embeds.to(generated_latent_embeds.dtype),
                                    generated_latent_embeds,
                                    dim=-1
                                ).mean()
                                latent_loss = 1.0 - cosine_sim
                                latent_cosine_losses.append(latent_loss.item())
                            else:
                                if i < 5:
                                    logging.warning(f"[Sample {i}] Latent count mismatch: Generated {generated_latent_embeds.shape[0]}, GT {gt_latent_embeds.shape[0]}. Skipping loss calculation.")
                    except Exception as e:
                        # Silently skip if we can't extract hidden states
                        pass
        
        decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        answer = decoded_output.split('<|im_start|>assistant')[-1]
        
        # Extract latent tokens and content for logging
        try:
            latent_part = answer.split('<|latent_start|>')[-1].split('<|latent_end|>')[0]
            n_latents = latent_part.count('<|latent_pad|>')
        except:
            n_latents = 0

        map_desc = sample.get("map_desc", [])
        path_str = extract_boxed_content(answer)

        result = simulate_vsp(map_desc, path_str)

        if result['success']: correct += 1
        elif result['invalid']: invalid += 1

        # Save details to results list
        sample_result = {
            "index": i,
            "sample_id": sample.get("id", i),
            "map_id": sample.get("map_id", ""),
            "answer": answer.strip(),
            "n_latents": n_latents,
            "success": result['success'],
            "invalid": result['invalid'],
            "status": result.get('status', "")
        }
        results_to_save.append(sample_result)

        if (i+1) % 20 == 0:
            log_msg = f"[{i+1}] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})"
            if latent_cosine_losses:
                avg_latent_loss = sum(latent_cosine_losses) / len(latent_cosine_losses)
                log_msg += f", Avg Latent Loss: {avg_latent_loss:.4f}"
            logging.info(log_msg)
        
        # Log detailed response and latent info for every 50th sample or if requested
        if (i+1) % 50 == 0:
            logging.info(f"--- Sample {i+1} Detailed Output ---")
            logging.info(f"Latent count found: {n_latents} (Target: {args.latent_size})")
            logging.info(f"Generated Answer: {answer.strip()}")
            logging.info(f"------------------------------------")

    # Calculate final statistics
    total_samples = i + 1
    accuracy = correct / total_samples
    invalid_rate = invalid / total_samples
    
    # Log final results with clear formatting
    logging.info("="*80)
    logging.info("FINAL RESULTS - AVERAGE ACROSS ALL IMAGES")
    logging.info("="*80)
    logging.info(f"Total Images Evaluated: {total_samples}")
    logging.info(f"Correct: {correct} ({accuracy:.3f})")
    logging.info(f"Invalid: {invalid} ({invalid_rate:.3f})")
    if latent_cosine_losses:
        avg_latent_loss = sum(latent_cosine_losses) / len(latent_cosine_losses)
        logging.info(f"Average Latent Cosine Loss: {avg_latent_loss:.4f}")
    
    # Display distribution statistics
    logging.info("")
    logging.info("=" * 80)
    logging.info("OVERALL DISTRIBUTION STATISTICS (AVERAGED ACROSS ALL SAMPLES)")
    logging.info("=" * 80)
    
    if gt_means:
        avg_gt_mean = sum(gt_means) / len(gt_means)
        avg_gt_std = sum(gt_stds) / len(gt_stds)
        std_of_gt_means = (sum((m - avg_gt_mean) ** 2 for m in gt_means) / len(gt_means)) ** 0.5
        std_of_gt_stds = (sum((s - avg_gt_std) ** 2 for s in gt_stds) / len(gt_stds)) ** 0.5
        logging.info("")
        logging.info("GT Latent Distribution (from reasoning images):")
        logging.info(f"  Average Mean: {avg_gt_mean:.6f} ± {std_of_gt_means:.6f}")
        logging.info(f"  Average Std:  {avg_gt_std:.6f} ± {std_of_gt_stds:.6f}")
        logging.info(f"  (computed across {len(gt_means)} samples)")
    
    if model_output_means:
        avg_mo_mean = sum(model_output_means) / len(model_output_means)
        avg_mo_std = sum(model_output_stds) / len(model_output_stds)
        std_of_mo_means = (sum((m - avg_mo_mean) ** 2 for m in model_output_means) / len(model_output_means)) ** 0.5
        std_of_mo_stds = (sum((s - avg_mo_std) ** 2 for s in model_output_stds) / len(model_output_stds)) ** 0.5
        logging.info("")
        logging.info("Model Original Output Distribution (what gets replaced in ablations):")
        logging.info(f"  Average Mean: {avg_mo_mean:.6f} ± {std_of_mo_means:.6f}")
        logging.info(f"  Average Std:  {avg_mo_std:.6f} ± {std_of_mo_stds:.6f}")
        logging.info(f"  (computed across {len(model_output_means)} samples)")
    
    if used_means:
        avg_used_mean = sum(used_means) / len(used_means)
        avg_used_std = sum(used_stds) / len(used_stds)
        std_of_used_means = (sum((m - avg_used_mean) ** 2 for m in used_means) / len(used_means)) ** 0.5
        std_of_used_stds = (sum((s - avg_used_std) ** 2 for s in used_stds) / len(used_stds)) ** 0.5
        
        mode_name = "Used Embeddings"
        if args.use_random_latent:
            mode_name = "Random N(0,1) Embeddings (USED in model)"
        elif args.use_zero_latent:
            mode_name = "Zero Embeddings (USED in model)"
        elif args.use_gt_latent:
            mode_name = "GT Embeddings (USED in model)"
        elif args.use_random_latent_gt_dist:
            mode_name = "Random with GT Distribution (USED in model)"
        elif args.use_random_latent_model_dist:
            mode_name = "Random with Model Distribution (USED in model)"
        elif args.use_model_latent:
            mode_name = "Model-Generated Embeddings (USED in model)"
            
        logging.info("")
        logging.info(f"{mode_name}:")
        logging.info(f"  Average Mean: {avg_used_mean:.6f} ± {std_of_used_means:.6f}")
        logging.info(f"  Average Std:  {avg_used_std:.6f} ± {std_of_used_stds:.6f}")
        logging.info(f"  (computed across {len(used_means)} samples)")
    
    if model_means:
        avg_model_mean = sum(model_means) / len(model_means)
        avg_model_std = sum(model_stds) / len(model_stds)
        std_of_model_means = (sum((m - avg_model_mean) ** 2 for m in model_means) / len(model_means)) ** 0.5
        std_of_model_stds = (sum((s - avg_model_std) ** 2 for s in model_stds) / len(model_stds)) ** 0.5
        logging.info("")
        logging.info("Model-Generated Latent Distribution (from first pass):")
        logging.info(f"  Average Mean: {avg_model_mean:.6f} ± {std_of_model_means:.6f}")
        logging.info(f"  Average Std:  {avg_model_std:.6f} ± {std_of_model_stds:.6f}")
        logging.info(f"  (computed across {len(model_means)} samples)")
    
    # Show comparisons
    if gt_means and model_output_means:
        logging.info("")
        logging.info("Distribution Comparison (GT vs Model Original Output):")
        mean_diff = abs(avg_gt_mean - avg_mo_mean)
        std_diff = abs(avg_gt_std - avg_mo_std)
        logging.info(f"  Mean difference: {mean_diff:.6f}")
        logging.info(f"  Std difference:  {std_diff:.6f}")
        mismatch_pct = (mean_diff / abs(avg_gt_mean) * 100) if avg_gt_mean != 0 else 0
        logging.info(f"  Mean mismatch:   {mismatch_pct:.2f}%")
    
    if gt_means and used_means:
        logging.info("")
        logging.info("Distribution Comparison (GT vs Used Embeddings):")
        mean_diff = abs(avg_gt_mean - avg_used_mean)
        std_diff = abs(avg_gt_std - avg_used_std)
        logging.info(f"  Mean difference: {mean_diff:.6f}")
        logging.info(f"  Std difference:  {std_diff:.6f}")
        if not args.use_gt_latent:  # Only show mismatch info if not using GT
            mismatch_pct = (mean_diff / abs(avg_gt_mean) * 100) if avg_gt_mean != 0 else 0
            logging.info(f"  Mean mismatch:   {mismatch_pct:.2f}%")
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("NOTE: Per-sample distributions (GT, Model Original Output, and Used)")
    logging.info("are logged during generation. See logs above for detailed per-sample stats.")
    logging.info("=" * 80)

    # Save to JSON in results/ directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, os.path.basename(args.log_file).replace(".txt", ".json"))
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": {
                "accuracy": correct / (i + 1),
                "invalid": invalid / (i + 1),
                "total": i + 1,
                "correct": correct,
                "avg_latent_cosine_loss": sum(latent_cosine_losses) / len(latent_cosine_losses) if latent_cosine_losses else None
            },
            "results": results_to_save
        }, f, indent=4)
    logging.info(f"Results saved to {results_path}")
