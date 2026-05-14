"""
Test script for evaluating Mirage depth reasoning models on the HardBlink dataset.

Reads hardblink question files (question_id, image, text, category) and runs
Mirage-style inference with latent token ablation modes. Outputs answers in
the hardblink answer format for use with eval_answers.py.

Ablation modes:
  --use-gt-latent           : Replace latent with GT depth image embeddings
  --use-random-latent       : Replace latent with random N(0,1) embeddings
  --use-zero-latent         : Replace latent with zero embeddings
  --use-model-latent        : Use model's own latent (sanity check)
  --use-first-latent-repeat : Use first model latent and repeat for remaining latent steps
  --use-random-latent-gt-dist   : Random with GT distribution
  --use-random-latent-model-dist: Random with model's own distribution
  --use-gt-latent-count-matched : Two-pass GT with count matching
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import json
import logging
import argparse
from tqdm import tqdm

from qwen_vl_utils import process_vision_info

from utils_new import *
from task_new import *

seed_everything(seed=42)


def resolve_depth_image_path(depth_folder: str, image_file: str):
    """Resolve GT depth image path with flexible naming conventions."""
    if not depth_folder:
        return None
    base = os.path.splitext(os.path.basename(image_file))[0]
    candidates = [
        os.path.join(depth_folder, image_file),            # e.g. 0.png
        os.path.join(depth_folder, f"{base}_depth.png"),  # e.g. 0_depth.png
        os.path.join(depth_folder, f"{base}_depth.jpg"),  # fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_processor_with_fallback(load_model_path: str, base_model_name: str, cache_dir: str):
    """
    Load processor from checkpoint; fall back to base model if checkpoint is missing
    processor files (e.g., preprocessor_config.json/chat_template.json).
    """
    try:
        return AutoProcessor.from_pretrained(load_model_path, trust_remote_code=True, cache_dir=cache_dir)
    except OSError as exc:
        logging.warning(f"Failed to load processor from checkpoint '{load_model_path}': {exc}")
        logging.warning(f"Falling back to base processor '{base_model_name}'")
        return AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_dir)


def get_depth_args():
    """Parse arguments for depth reasoning evaluation on hardblink."""
    parser = argparse.ArgumentParser(description="Mirage depth reasoning evaluation on HardBlink")

    # Model
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--load_model_path", type=str, required=True,
                        help="Path to trained Mirage checkpoint")
    parser.add_argument("--cache_dir", type=str, default='./cache')
    parser.add_argument("--latent_size", type=int, default=4)

    # Data
    parser.add_argument("--question_file", type=str, required=True,
                        help="Path to hardblink question JSONL file")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing evaluation images")
    parser.add_argument("--answers_file", type=str, required=True,
                        help="Path to save generated answers JSONL file")

    # Optional: depth map folder for GT latent ablation
    parser.add_argument("--depth_image_folder", type=str, default=None,
                        help="Path to folder with depth map images (for GT latent ablation). "
                             "If not provided, GT latent modes that require depth images will be skipped.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    # Logging
    parser.add_argument("--log_file", type=str, default='./logs/log_test_depth.txt')
    parser.add_argument("--verbose", action="store_true")

    # Ablation arguments (Mirage-style)
    parser.add_argument("--use-random-latent", action="store_true",
                        help="Use random latent embeddings (ablation mode)")
    parser.add_argument("--use-zero-latent", action="store_true",
                        help="Use all zero latent embeddings (ablation mode)")
    parser.add_argument("--use-gt-latent", action="store_true",
                        help="Use GT latent embeddings with fixed count from config")
    parser.add_argument("--use-gt-latent-count-matched", action="store_true",
                        help="Two-pass: generate to get count N, then use GT compressed to N tokens")
    parser.add_argument("--use-model-latent", action="store_true",
                        help="Use model-generated latent (sanity check)")
    parser.add_argument("--use-first-latent-repeat", action="store_true",
                        help="Use first model-generated latent and repeat it for all remaining latent steps")
    parser.add_argument("--use-random-latent-gt-dist", action="store_true",
                        help="Random latent with GT distribution")
    parser.add_argument("--use-random-latent-model-dist", action="store_true",
                        help="Random latent with model's own distribution")

    return parser.parse_args()


args = get_depth_args()

# Set up logging
os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('==' * 20)
logging.info(args)
logging.info('==' * 20)

# Check for incompatible ablation flags
ablation_count = sum([
    args.use_random_latent, args.use_zero_latent, args.use_gt_latent,
    args.use_gt_latent_count_matched, args.use_model_latent,
    args.use_random_latent_gt_dist, args.use_random_latent_model_dist,
    args.use_first_latent_repeat
])
if ablation_count > 1:
    raise ValueError("Cannot enable more than one ablation mode simultaneously.")

# Determine ablation mode string for metadata
if args.use_random_latent:
    ablation_mode = "random"
elif args.use_zero_latent:
    ablation_mode = "zero"
elif args.use_gt_latent:
    ablation_mode = "gt"
elif args.use_gt_latent_count_matched:
    ablation_mode = "gt_count_matched"
elif args.use_model_latent:
    ablation_mode = "model"
elif args.use_first_latent_repeat:
    ablation_mode = "first_latent_repeat"
elif args.use_random_latent_gt_dist:
    ablation_mode = "random_gt_dist"
elif args.use_random_latent_model_dist:
    ablation_mode = "random_model_dist"
else:
    ablation_mode = "baseline"

logging.info(f"Ablation mode: {ablation_mode}")

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = load_processor_with_fallback(args.load_model_path, args.model, cache_dir)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

# Ensure latent_size is set in config
if not hasattr(model.config, 'latent_size'):
    model.config.latent_size = args.latent_size
    logging.info(f"Set model.config.latent_size = {args.latent_size}")
if not hasattr(model.model.config, 'latent_size'):
    model.model.config.latent_size = args.latent_size
    logging.info(f"Set model.model.config.latent_size = {args.latent_size}")

use_latent_embeddings = hasattr(model.config, "latent_token_id")

if ablation_count > 0 and not use_latent_embeddings:
    logging.warning("[WARNING] Latent ablations requested but model does not use latent embeddings; options will be ignored.")

model.eval()

# Load questions
logging.info(f"Loading questions from: {args.question_file}")
with open(args.question_file, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]
logging.info(f"Loaded {len(questions)} questions")

# Create output directory
os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

# Distribution tracking
gt_means, gt_stds = [], []
model_output_means, model_output_stds = [], []
model_means, model_stds = [], []
used_means, used_stds = [], []

if not hasattr(model, '_ablation_stats'):
    model._ablation_stats = {
        'model_output_means': [],
        'model_output_stds': []
    }

answers = []

for i, question in tqdm(enumerate(questions), total=len(questions)):
    question_id = question['question_id']
    image_file = question['image']
    prompt = question['text']
    category = question.get('category', 'unknown')

    # Construct full image path
    image_path = os.path.join(args.image_folder, image_file)

    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}, skipping.")
        continue

    # Build Mirage-style conversation from hardblink question format
    image = Image.open(image_path).convert("RGB")
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    texts = [processor.apply_chat_template(conversations, tokenize=False)]
    texts = [place_input_image(text, sep_token=None) for text in texts]
    image_inputs, _ = process_vision_info(conversations)

    texts = [t + '<|im_start|>assistant' for t in texts]

    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Prepare GT latent embeddings if depth images are available
    gt_latent_embeds = None
    latent_inputs = None

    if use_latent_embeddings and args.depth_image_folder:
        # Try to find a matching depth image.
        # Supports both:
        #   - same filename: 0.png
        #   - depth suffix:  0_depth.png
        depth_image_path = resolve_depth_image_path(args.depth_image_folder, image_file)
        if depth_image_path is not None:
            reasoning_image = Image.open(depth_image_path).convert("RGB")
            latent_inputs = processor(text=[""], images=[reasoning_image], return_tensors="pt")

            with torch.no_grad():
                pixel_values_latent = latent_inputs['pixel_values'].to("cuda:0").type(model.visual.dtype)
                gt_embeds = model.visual(pixel_values_latent, grid_thw=latent_inputs['image_grid_thw'].to("cuda:0"))

                # Apply compression
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

                gt_latent_embeds = gt_embeds
        else:
            if args.use_gt_latent and i < 5:
                logging.warning(
                    f"[Sample {i}] Depth image not found in {args.depth_image_folder} for image {image_file} "
                    f"(tried {image_file} and {os.path.splitext(os.path.basename(image_file))[0]}_depth.png)"
                )

    with torch.no_grad():
        generate_inputs = inputs.copy()

        # For GT Mode 1: add pixel_values_latent for injection
        if args.use_gt_latent and latent_inputs is not None:
            generate_inputs['pixel_values_latent'] = latent_inputs['pixel_values'].to("cuda:0")
            generate_inputs['image_grid_thw_latent'] = latent_inputs['image_grid_thw'].to("cuda:0")

        # Track GT distribution
        gt_latent_mean = None
        gt_latent_std = None
        if gt_latent_embeds is not None:
            gt_latent_mean = gt_latent_embeds.mean().item()
            gt_latent_std = gt_latent_embeds.std().item()
            gt_means.append(gt_latent_mean)
            gt_stds.append(gt_latent_std)

            if args.use_zero_latent:
                used_means.append(0.0)
                used_stds.append(0.0)
            elif args.use_gt_latent:
                used_means.append(gt_latent_mean)
                used_stds.append(gt_latent_std)
            elif args.use_random_latent_gt_dist:
                used_means.append(gt_latent_mean)
                used_stds.append(gt_latent_std)

        # GT latent count-matched: two-pass
        gt_latent_count_matched = None
        if args.use_gt_latent_count_matched and use_latent_embeddings and latent_inputs is not None:
            first_pass_output = model.generate(
                **generate_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            latent_token_id = model.config.latent_token_id
            latent_mask = first_pass_output.sequences == latent_token_id
            latent_count = latent_mask.sum().item()
            if latent_count > 0:
                gt_latent_count_matched = latent_count
                if i < 5 or (i + 1) % 50 == 0:
                    logging.info(f"[Sample {i}] Count-matched: first pass generated {latent_count} latent tokens")

        # For GT count-matched mode: add the info to generate_inputs
        if args.use_gt_latent_count_matched and latent_inputs is not None and gt_latent_count_matched is not None:
            generate_inputs['pixel_values_latent'] = latent_inputs['pixel_values'].to("cuda:0")
            generate_inputs['image_grid_thw_latent'] = latent_inputs['image_grid_thw'].to("cuda:0")
            generate_inputs['gt_target_count'] = gt_latent_count_matched

        # Random latent with model distribution: two-pass
        model_latent_mean = None
        model_latent_std = None
        if args.use_random_latent_model_dist and use_latent_embeddings:
            first_pass_output = model.generate(
                **generate_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
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
                        used_means.append(model_latent_mean)
                        used_stds.append(model_latent_std)
                except Exception as e:
                    logging.warning(f"[Sample {i}] Failed to extract model latent distribution: {e}")

        # Main generation
        output = model.generate(
            **generate_inputs,
            max_new_tokens=args.max_new_tokens,
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

    decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = decoded_output.split('<|im_start|>assistant')[-1]

    # Clean up special tokens from the answer for readability
    # Remove end tokens but preserve the actual content
    answer_clean = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

    # Extract latent token count for logging
    try:
        latent_part = answer.split('<|latent_start|>')[-1].split('<|latent_end|>')[0]
        n_latents = latent_part.count('<|latent_pad|>')
    except:
        n_latents = 0

    # Store answer in hardblink format
    answer_entry = {
        'question_id': question_id,
        'text': answer_clean,
        'category': category,
        'image': image_file,
    }
    if ablation_mode != "baseline":
        answer_entry['ablation_mode'] = ablation_mode

    answers.append(answer_entry)

    if args.verbose or (i + 1) % 20 == 0:
        logging.info(f"[{i + 1}/{len(questions)}] Q{question_id}: latents={n_latents}, answer={answer_clean[:150]}...")

# Save answers
logging.info(f"Saving {len(answers)} answers to: {args.answers_file}")
with open(args.answers_file, 'w') as f:
    for answer in answers:
        f.write(json.dumps(answer) + '\n')

# Log distribution statistics
logging.info("=" * 80)
logging.info("EVALUATION COMPLETE")
logging.info("=" * 80)
logging.info(f"Total questions: {len(questions)}")
logging.info(f"Total answers generated: {len(answers)}")
logging.info(f"Ablation mode: {ablation_mode}")

if gt_means:
    avg_gt_mean = sum(gt_means) / len(gt_means)
    avg_gt_std = sum(gt_stds) / len(gt_stds)
    logging.info(f"GT Latent Distribution - Avg Mean: {avg_gt_mean:.6f}, Avg Std: {avg_gt_std:.6f} ({len(gt_means)} samples)")

if used_means:
    avg_used_mean = sum(used_means) / len(used_means)
    avg_used_std = sum(used_stds) / len(used_stds)
    logging.info(f"Used Embeddings Distribution - Avg Mean: {avg_used_mean:.6f}, Avg Std: {avg_used_std:.6f} ({len(used_means)} samples)")

logging.info(f"Results saved to {args.answers_file}")
