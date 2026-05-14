# pylint: disable=all
"""Eval script for Video-R1 with Latent Token Ablation Support (no vLLM, pure transformers)."""
from collections import defaultdict
import argparse
import importlib
import json
import math
import os
import pdb
import re
import sys
from pathlib import Path
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

from aurora_eval_config import add_dataset_override_args, resolve_dataset_entries

try:
    import wandb
except ImportError:
    class _WandbStub:
        """Keep eval runnable when wandb is not installed in the target env."""

        @staticmethod
        def init(*args, **kwargs):
            print("wandb not installed; continuing without telemetry.")
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()

# Import custom model with ablation support
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))
from mmlatent_qwen_vl_sample_imonly import Qwen2_5_VLForConditionalGeneration


def main():
  BSZ = 4  # batch size (reduced from 64 to avoid OOM)
  NUM_LATENTS = 20  # Number of latent tokens to use (as per README)
  VSI_MCA_QUESTION_TYPES = {
      "object_rel_direction_easy",
      "object_rel_direction_medium",
      "object_rel_direction_hard",
      "object_rel_distance",
      "route_planning",
      "obj_appearance_order",
  }
  VSI_NA_QUESTION_TYPES = {
      "object_abs_distance",
      "object_counting",
      "object_size_estimation",
      "room_size_estimation",
  }

  parser = argparse.ArgumentParser(description="Evaluation benchmark with ablation support")
  parser.add_argument(
      "--model_path", type=str, required=True, help="Path to the model"
  )
  parser.add_argument(
      "--file_name", type=str, required=True, help="Name of the file"
  )
  parser.add_argument(
      "--dataset_names", type=str, nargs="+", required=True, help="Names of the datasets"
  )
  # Aurora overlay: keep dataset mounts configurable instead of baking in one machine layout.
  add_dataset_override_args(parser)
  
  # Ablation arguments
  parser.add_argument(
      "--use-zero-latent", action="store_true", 
      help="Replace latent embeddings with zeros (ABLATION)"
  )
  parser.add_argument(
      "--use-random-latent", action="store_true",
      help="Replace latent embeddings with random values using predicted distribution (ABLATION)"
  )
  parser.add_argument(
      "--use-same-latent", action="store_true",
      help="Replace latent embeddings with the same embeddings (identity ablation sanity check)"
  )
  parser.add_argument(
      "--use-first-latent-repeat", action="store_true",
      help="Use first model-predicted latent embedding and repeat it for all remaining latent steps"
  )
  parser.add_argument(
      "--use-random-latent-same-dist", action="store_true",
      help="Replace latent embeddings with random values matched to each replaced vector distribution"
  )
  parser.add_argument(
      "--use-random-latent-gt-dist", action="store_true",
      help="Replace latent embeddings with random values using GT distribution (ABLATION)"
  )
  parser.add_argument(
      "--use-random-latent-model-dist", action="store_true",
      help="Replace latent embeddings with random values using model's own distribution (ABLATION)"
  )
  parser.add_argument(
      "--use-gt-latent", action="store_true",
      help="Replace latent embeddings with ground truth from auxiliary images (ABLATION)"
  )
  parser.add_argument(
      "--gt-image-dir", type=str, default=None,
      help="Directory containing ground truth auxiliary images (for GT latent mode)"
  )
  parser.add_argument(
      "--gt-max-images", type=int, default=64,
      help="Maximum number of GT images to load from --gt-image-dir"
  )
  
  args = parser.parse_args()
  
  # Check for incompatible ablation flags
  ablation_flags = [
      args.use_zero_latent,
      args.use_random_latent,
      args.use_same_latent,
      args.use_first_latent_repeat,
      args.use_random_latent_same_dist,
      args.use_random_latent_gt_dist,
      args.use_random_latent_model_dist,
      args.use_gt_latent,
  ]
  if sum(ablation_flags) > 1:
      raise ValueError("Cannot enable more than one ablation mode simultaneously.")
  
  # Check if any ablation is enabled
  ablation_enabled = any(ablation_flags)
  if ablation_enabled:
      print("=" * 80)
      print("ABLATION MODE ENABLED")
      if args.use_zero_latent:
          print("MODE: Zero Latent - Replacing all latent embeddings with zeros")
      elif args.use_random_latent:
          print("MODE: Random Latent (Predicted Dist) - Using model's predicted distribution")
      elif args.use_same_latent:
          print("MODE: Same Latent - Identity replacement through ablation path")
      elif args.use_first_latent_repeat:
          print("MODE: First Latent Repeat - Reuse first model latent embedding for all remaining latent steps")
      elif args.use_random_latent_same_dist:
          print("MODE: Random Latent (Same Dist) - Matched to each replaced latent vector")
      elif args.use_random_latent_gt_dist:
          print("MODE: Random Latent (GT Dist) - Using ground truth distribution")
      elif args.use_random_latent_model_dist:
          print("MODE: Random Latent (Model Dist) - Using model's generation distribution")
      elif args.use_gt_latent:
          print("MODE: GT Latent - Using ground truth auxiliary image embeddings")
          if not args.gt_image_dir:
              raise ValueError("--gt-image-dir required for GT latent mode")
          if not os.path.isdir(args.gt_image_dir):
              raise ValueError(f"--gt-image-dir does not exist or is not a directory: {args.gt_image_dir}")
      print("=" * 80)

  print(args.dataset_names)
  dataset_entries = resolve_dataset_entries(args.dataset_names, args)

  MODEL_PATH = args.model_path
  file_name = args.file_name

  # Load processor and tokenizer
  processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  tokenizer.padding_side = "left"
  
  # CRITICAL: Add Mull special tokens (as per lmms-eval implementation)
  print("\nAdding Mull special tokens to tokenizer...")
  num_added = 0
  num_added += tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
  num_added += tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
  num_added += tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
  num_added += tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
  print(f"Added {num_added} new special tokens")
  
  # Update processor tokenizer
  processor.tokenizer = tokenizer

  # Load model with transformers (no vLLM)
  print(f"Loading model from {MODEL_PATH} with native transformers...")
  print(f"DEBUG: Using custom Qwen2_5_VLForConditionalGeneration with ablation support")
  print(f"DEBUG: Model class location: {Qwen2_5_VLForConditionalGeneration.__module__}")
  
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      MODEL_PATH,
      torch_dtype=torch.bfloat16,
      device_map="auto",
      trust_remote_code=True,
  )
  model.eval()
  
  # CRITICAL: Configure Mull token IDs in model config (as per lmms-eval lines 238-245)
  print("\nConfiguring Mull token IDs in model config...")
  latent_token_idx = tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
  latent_start_idx = tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
  latent_end_idx = tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
  imagelatent_idx = tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
  
  model.config.latent_token_id = int(latent_token_idx)
  model.config.latent_start_id = int(latent_start_idx)
  model.config.latent_end_id = int(latent_end_idx)
  model.config.imagelatent_token_id = int(imagelatent_idx)
  
  print(f"  latent_token_id (<|latent_pad|>): {model.config.latent_token_id}")
  print(f"  latent_start_id (<|latent_start|>): {model.config.latent_start_id}")
  print(f"  latent_end_id (<|latent_end|>): {model.config.latent_end_id}")
  print(f"  imagelatent_token_id (<|latent_image|>): {model.config.imagelatent_token_id}")
  
  # Resize embeddings if new tokens were added
  if num_added > 0:
      print(f"\nResizing model embeddings to accommodate {num_added} new tokens...")
      model.resize_token_embeddings(len(tokenizer))
      print(f"New vocabulary size: {len(tokenizer)}")
  
  # Set default latent_sample_temperature if not already set
  if not hasattr(model.config, 'latent_sample_temperature'):
      model.config.latent_sample_temperature = 0
      print(f"Set latent_sample_temperature to 0 (default)")
  else:
      print(f"latent_sample_temperature already set to: {model.config.latent_sample_temperature}")
  
  # Verify the model has ablation support methods
  if hasattr(model, '_sample_latent_embedding'):
      print("DEBUG: ✓ Model has _sample_latent_embedding method (ablation support confirmed)")
  else:
      print("DEBUG: ✗ WARNING: Model does NOT have _sample_latent_embedding method!")
      print("DEBUG:   This means ablation will NOT work!")

  # Set ablation flags in config if needed
  if ablation_enabled:
      print("Setting ablation flags in model config...")
      model.config.use_zero_latent = args.use_zero_latent
      model.config.use_random_latent = args.use_random_latent
      model.config.use_same_latent = args.use_same_latent
      model.config.use_first_latent_repeat = args.use_first_latent_repeat
      model.config.use_random_latent_same_dist = args.use_random_latent_same_dist
      model.config.use_random_latent_gt_dist = args.use_random_latent_gt_dist
      model.config.use_random_latent_model_dist = args.use_random_latent_model_dist
      model.config.use_gt_latent = args.use_gt_latent
      
      # DEBUG: Verify config was set correctly
      print("\nDEBUG: Verifying model config ablation flags:")
      print(f"  - use_zero_latent: {getattr(model.config, 'use_zero_latent', 'NOT SET')}")
      print(f"  - use_random_latent: {getattr(model.config, 'use_random_latent', 'NOT SET')}")
      print(f"  - use_same_latent: {getattr(model.config, 'use_same_latent', 'NOT SET')}")
      print(f"  - use_first_latent_repeat: {getattr(model.config, 'use_first_latent_repeat', 'NOT SET')}")
      print(f"  - use_random_latent_same_dist: {getattr(model.config, 'use_random_latent_same_dist', 'NOT SET')}")
      print(f"  - use_random_latent_gt_dist: {getattr(model.config, 'use_random_latent_gt_dist', 'NOT SET')}")
      print(f"  - use_random_latent_model_dist: {getattr(model.config, 'use_random_latent_model_dist', 'NOT SET')}")
      print(f"  - use_gt_latent: {getattr(model.config, 'use_gt_latent', 'NOT SET')}")
      print("=" * 80)
  else:
      print("\nDEBUG: No ablation mode - running in BASELINE mode")
      print("=" * 80)

  # Load GT latent embeddings once for GT modes (used during generation in model.forward/generate)
  def _pool_to_num_latents(feats: torch.Tensor, num_latents: int) -> torch.Tensor:
      if feats.shape[0] == num_latents:
          return feats
      if feats.shape[0] > num_latents:
          chunks = torch.tensor_split(feats, num_latents, dim=0)
          pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
          return pooled
      repeat_factor = (num_latents + feats.shape[0] - 1) // feats.shape[0]
      tiled = feats.repeat(repeat_factor, 1)
      return tiled[:num_latents]

  if args.use_gt_latent or args.use_random_latent_gt_dist:
      if not args.gt_image_dir or not os.path.isdir(args.gt_image_dir):
          raise ValueError(f"GT modes require a valid --gt-image-dir. Got: {args.gt_image_dir}")

      image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
      gt_paths = [
          p for p in sorted(Path(args.gt_image_dir).rglob("*"))
          if p.is_file() and p.suffix.lower() in image_exts
      ]
      if not gt_paths:
          raise ValueError(f"No GT images found in --gt-image-dir: {args.gt_image_dir}")

      if args.gt_max_images > 0 and len(gt_paths) > args.gt_max_images:
          gt_paths = gt_paths[: args.gt_max_images]

      print(f"\nLoading {len(gt_paths)} GT images from: {args.gt_image_dir}")
      gt_images = [Image.open(p).convert("RGB") for p in gt_paths]

      gt_inputs = processor.image_processor(images=gt_images, return_tensors="pt")
      pixel_values_latent = gt_inputs["pixel_values"]
      image_grid_thw_latent = gt_inputs["image_grid_thw"]

      visual_device = next(model.visual.parameters()).device
      pixel_values_latent = pixel_values_latent.to(visual_device, dtype=model.visual.dtype)
      image_grid_thw_latent = image_grid_thw_latent.to(visual_device)

      with torch.no_grad():
          gt_feats = model.visual(pixel_values_latent, grid_thw=image_grid_thw_latent)
      gt_feats = gt_feats.to(dtype=model.dtype)
      gt_pooled = _pool_to_num_latents(gt_feats, NUM_LATENTS)

      # Used by GT latent replacement path in generate()
      model.gt_latent_embeds = gt_pooled
      # Used by random_latent_gt_dist path
      model.config.gt_latent_mean = gt_pooled.mean().item()
      model.config.gt_latent_std = gt_pooled.std(unbiased=False).item()

      print(
          f"Prepared GT latent embeddings: shape={tuple(gt_pooled.shape)}, "
          f"mean={model.config.gt_latent_mean:.6f}, std={model.config.gt_latent_std:.6f}"
      )

  print("Model loaded successfully!")

  wandb.init(
      project=file_name,
      mode="offline",
      config=vars(args)
  )

  for dataset_entry in dataset_entries:
    # ['mvbench','tempcompass','videomme','videommmu','mmvu']
    dataset_name = dataset_entry["dataset_name"]
    
    # Add ablation mode to filename
    ablation_suffix = ""
    if args.use_zero_latent:
        ablation_suffix = "_zero_latent"
    elif args.use_random_latent:
        ablation_suffix = "_random_latent"
    elif args.use_same_latent:
        ablation_suffix = "_same_latent"
    elif args.use_first_latent_repeat:
        ablation_suffix = "_first_latent_repeat"
    elif args.use_random_latent_same_dist:
        ablation_suffix = "_random_latent_same_dist"
    elif args.use_random_latent_gt_dist:
        ablation_suffix = "_random_latent_gt_dist"
    elif args.use_random_latent_model_dist:
        ablation_suffix = "_random_latent_model_dist"
    elif args.use_gt_latent:
        ablation_suffix = "_gt_latent"
    
    OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}{ablation_suffix}_greedy_output.json"
    
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Ablation mode: {ablation_suffix if ablation_suffix else 'BASELINE (no ablation)'}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"{'='*80}\n")

    sys.path.append("dataloaders/")
    EvalDataset = importlib.import_module("custom_datasets").EvalDataset
    data = EvalDataset(dataset_entry)

    # Prompt templates aligned with lmms-eval "latents" mode
    QUESTION_TEMPLATE_LATENT_DEFAULT = (
        "{Question}\n"
        "Please think about this question deeply. "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your final answer between the <answer> </answer> tags."
    )
    # SAT "latents" template (note: no explicit final answer sentence in lmms-eval)
    QUESTION_TEMPLATE_LATENT_SAT = (
        "{Question}\n"
        "Please think about this question deeply. "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
    )

    TYPE_TEMPLATE = {
        "multiple choice": (
            " Please provide only the single option letter (e.g., A, B, C, D,"
            " etc.) within the <answer> </answer> tags."
        ),
        "numerical": (
            " Please provide the numerical value (e.g., 42 or 3.14) within the"
            " <answer> </answer> tags."
        ),
        "OCR": (
            " Please transcribe text from the image/video clearly and provide"
            " your text answer within the <answer> </answer> tags."
        ),
        "free-form": (
            " Please provide your text answer within the <answer> </answer>"
            " tags."
        ),
        "regression": (
            " Please provide the numerical value (e.g., 42 or 3.14) within the"
            " <answer> </answer> tags."
        ),
    }

    def _format_options_with_letters(options):
      letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      formatted = []
      for i, opt in enumerate(options or []):
        letter = letters[i] if i < len(letters) else letters[-1]
        text = opt
        m = re.match(r"^\s*([A-Z])[\.\)]\s*(.*)$", opt)
        if m:
          letter = m.group(1)
          text = m.group(2).strip()
        formatted.append(f"({letter}) {text}".strip())
      return "\n".join(formatted).strip()

    def _build_prompt(sample, dataset_name):
      if dataset_name == "mmvu":
        question = sample.get("question", sample.get("problem", ""))
        q_type = sample.get("question_type", "")
        if q_type == "multiple-choice" or sample.get("problem_type") == "multiple choice":
          choices = sample.get("choices", {})
          if not choices and sample.get("options"):
            options = sample.get("options", [])
            letters = ["A", "B", "C", "D", "E"]
            choices = {}
            for i, op in enumerate(options[:5]):
              clean_op = re.sub(r"^\s*[A-Z][\.\)]\s*", "", op).strip()
              choices[letters[i]] = clean_op
          return (
              f"Question:{question}\n"
              f"A: {choices.get('A', '')}\n"
              f"B: {choices.get('B', '')}\n"
              f"C: {choices.get('C', '')}\n"
              f"D: {choices.get('D', '')}\n"
              f"E: {choices.get('E', '')}\n"
              "Visual Information: processed video\n"
              "Do not generate any intermediate reasoning process. Answer directly with the option letter from the\n"
              "given choices."
          )
        return (
            f"Question:{question}\n"
            "Visual Information: processed video\n"
            "Do not generate any intermediate reasoning process. Directly output the final answer."
        )

      question = sample.get("problem", "")
      if sample.get("problem_type") == "multiple choice":
        if dataset_name == "sat":
          option_block = _format_options_with_letters(sample.get("options", []))
          if option_block:
            question = f"{question} Choose from the following options: \n{option_block}"
        elif dataset_name == "blink":
          # lmms-eval uses the prompt directly for BLINK
          question = question
        else:
          options = sample.get("options", [])
          if options:
            question = question + "Options:\n" + "\n".join(options)

      if dataset_name == "sat":
        template = QUESTION_TEMPLATE_LATENT_SAT
      else:
        template = QUESTION_TEMPLATE_LATENT_DEFAULT

      if dataset_name == "vsibench":
        q_name = sample.get("question_type", "")
        if q_name in VSI_MCA_QUESTION_TYPES:
          q_type = "multiple choice"
        elif q_name in VSI_NA_QUESTION_TYPES:
          q_type = "numerical"
        else:
          q_type = sample.get("problem_type", "free-form")
      else:
        q_type = sample.get("problem_type", "free-form")

      return template.format(Question=question) + TYPE_TEMPLATE[q_type]

    messages = []
    for idx, x in enumerate(data):

      content_multimedia = []
      for im_vid_entry in x["multimedia"]:
        # DEBUG: Print multimedia paths for first few samples
        if idx < 3:
          print(f"\nDEBUG Sample {idx}: Multimedia entry")
          print(f"  data_type: {im_vid_entry['data_type']}")
          print(f"  path type: {type(im_vid_entry['path'])}")
          path_display = im_vid_entry['path'] if isinstance(im_vid_entry['path'], str) else f"<{type(im_vid_entry['path']).__name__}>"
          print(f"  path value: {path_display}")
          if hasattr(im_vid_entry['path'], 'filename'):
            print(f"  filename attr: {im_vid_entry['path'].filename}")
        
        content_multimedia.append({
            "type": im_vid_entry["data_type"],
            im_vid_entry["data_type"]: im_vid_entry["path"],
            # "max_pixels": 360*420,
            # "fps": 1.0
        })
      # CRITICAL: Mull-Tokens requires latent thinking tokens in assistant message
      # Aligned with README.md and lmms-eval (no <|latent_start|> in prompt)
      msg = [
          {
              "role": "user",
              "content": (
                  content_multimedia
                  + [
                      {
                          "type": "text",
                          "text": (
                              _build_prompt(x, dataset_name)
                          ),
                      },
                  ]
              ),
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": "<think>" + "<|latent_pad|>" * NUM_LATENTS + "</think>\n"
                  }
              ],
          },
      ]
      messages.append(msg)

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
      try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
          existing = json.load(f)
          final_output = existing.get("results", [])
          start_idx = len(final_output)
          print(f"Resuming from sample index {start_idx}")
      except Exception as e:
        print(f"Error reading existing output file: {e}")

    def extract_think(output_str):
      pattern = r"<think>\s*(.*?)\s*</think>"
      match = re.search(pattern, output_str, re.DOTALL)
      if match:
        return match.group(1).strip()
      return ""

    def extract_answer(text):
      pattern = r"<answer>\s*(.*?)\s*</answer>"
      match = re.search(pattern, text, re.DOTALL)
      if match:
        return match.group(1).strip()
      return ""

    def normalize_number(num_str):
      try:
        num_str = num_str.replace(",", "")
        return float(num_str)
      except Exception as e:
        return None

    def mean_relative_accuracy(
        pred, target, start=0.5, end=0.95, interval=0.05
    ):

      if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
      if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

      epsilon = 1e-8
      rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

      thresholds = torch.arange(
          start, end + interval / 2, interval, dtype=torch.float32
      )

      conditions = rel_error < (1 - thresholds)
      mra = conditions.float().mean()
      return mra.item()

    def _extract_mc_answer(text, dataset_name):
      if dataset_name == "blink":
        patterns = [
            r"(?:answer is|the correct answer is|the count is|final answer:|final conlusion:|answer must be|correct option must be|correct option is|final answer is)\s+\(?([a-zA-Z0-9]+)\)?",
            r"I counted a total of\s+(\d+)",
            r"<answer>\s*(.*?)\s*</answer>",
        ]
      else:  # sat / default
        patterns = [
            r"(?:the answer is|the correct answer is|the count is|final answer:|answer must be|correct option must be|correct option is|final answer is)\s+([a-zA-Z0-9]+)",
            r"I counted a total of\s+(\d+)",
            r"<answer>\s*(.*?)\s*</answer>",
        ]

      pred_letter = ""
      for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
          pred_letter = match.group(1).strip()
          matches = re.findall(r"\b[A-Z]\b", pred_letter)
          if matches:
            pred_letter = matches[-1]
          break
      if pred_letter == "":
        matches = re.findall(r"\b[A-Z]\b", text)
        if matches:
          pred_letter = matches[-1]
      return pred_letter

    def reward_fn(sample, model_output, question_type, dataset_name):
      try:
        gt_ans = extract_answer(sample.get("solution", ""))
        if question_type == "multiple choice":
          pred_letter = _extract_mc_answer(model_output, dataset_name)
          return 1.0 if pred_letter.strip().lower() == gt_ans.strip().lower() else 0.0
        elif question_type == "numerical":
          output_ans = extract_answer(model_output)
          if output_ans == "":
            output_ans = model_output
          gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
          out_has_decimal = ("." in output_ans) or ("," in output_ans)
          if gt_has_decimal != out_has_decimal:
            return 0.0
          gt_number = normalize_number(gt_ans)
          out_number = normalize_number(output_ans)
          if gt_number is None or out_number is None:
            return 0.0
          return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
          gt_number = normalize_number(gt_ans)
          out_number = normalize_number(output_ans)
          if gt_number is None or out_number is None:
            return 0.0
          mra = mean_relative_accuracy(out_number, gt_number)
          return mra
        else:
          return 0.0
      except Exception as e:
        return 0.0

    mean_acc = []
    mean_mra = []
    mean_acc_bytype = defaultdict(list)
    mean_mra_bytype = defaultdict(list)

    def _is_valid_metric_value(value):
      return (
          isinstance(value, (int, float))
          and not (isinstance(value, float) and math.isnan(value))
      )

    def _recompute_metrics_from_output(samples):
      recomputed_mean_acc = []
      recomputed_mean_mra = []
      recomputed_mean_acc_bytype = defaultdict(list)
      recomputed_mean_mra_bytype = defaultdict(list)

      for sample in samples:
        if not isinstance(sample, dict):
          continue
        reward = sample.get("reward")
        if not _is_valid_metric_value(reward):
          continue

        reward = float(reward)
        qa_type = sample.get("original_question_type", "")
        if sample.get("problem_type", "") != "regression":
          recomputed_mean_acc.append(reward)
          recomputed_mean_acc_bytype[qa_type].append(reward)
        else:
          recomputed_mean_mra.append(reward)
          recomputed_mean_mra_bytype[qa_type].append(reward)

      return (
          recomputed_mean_acc,
          recomputed_mean_mra,
          recomputed_mean_acc_bytype,
          recomputed_mean_mra_bytype,
      )

    def _get_generation_kwargs(dataset_name):
      if dataset_name in {"sat", "blink"}:
        return {
            "max_new_tokens": 1024,
            "temperature": 0,
            "do_sample": False,
        }
      if dataset_name == "vsibench":
        return {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.001,
            "do_sample": True,
        }
      if dataset_name == "mmvu":
        # mmvu task config uses do_sample=False; keep deterministic decoding.
        return {
            "max_new_tokens": 1024,
            "temperature": 0,
            "do_sample": False,
        }
      return {
          "max_new_tokens": 1024,
          "temperature": 0,
          "do_sample": False,
      }

    for i in tqdm(
        range(start_idx, len(messages), BSZ), desc="Processing batches"
    ):
      batch_messages = messages[i : i + BSZ]

      # IMPORTANT: Process prompts for Mull-Tokens format
      # Remove end token so model continues generating (README line 163)
      prompts = [
          processor.apply_chat_template(
              msg, tokenize=False, add_generation_prompt=False
          ).replace("<|im_end|>\n", "")  # Remove end token
          for msg in batch_messages
      ]
      
      # DEBUG: Print first prompt to verify format
      if i == start_idx:
          print("\n" + "="*80)
          print("DEBUG: First prompt (check for <|latent_pad|> tokens):")
          print("="*80)
          print(prompts[0][:500] if len(prompts[0]) > 500 else prompts[0])
          print("...")
          print(prompts[0][-500:] if len(prompts[0]) > 500 else "")
          print("="*80)
          print(f"Prompt contains '<|latent_pad|>': {'<|latent_pad|>' in prompts[0]}")
          print(f"Number of '<|latent_pad|>' tokens: {prompts[0].count('<|latent_pad|>')}")
          print("="*80 + "\n")

      # pdb.set_trace()
      image_inputs, video_inputs, video_kwargs = process_vision_info(
          batch_messages, return_video_kwargs=True
      )
      # pdb.set_trace()

      # Build inputs using the processor (native transformers approach)
      # Collect all images and videos for this batch
      batch_images = image_inputs if image_inputs else None
      batch_videos = video_inputs if video_inputs else None

      inputs = processor(
          text=prompts,
          images=batch_images,
          videos=batch_videos,
          padding=True,
          return_tensors="pt",
      )
      inputs = inputs.to(model.device)

      # Generate with native transformers
      gen_kwargs = _get_generation_kwargs(dataset_name)
      with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )
      
      # DEBUG: Print first batch info
      if i == start_idx:
        print(f"\nDEBUG: First batch generation completed")
        print(f"  Batch size: {len(batch_messages)}")
        print(f"  Generated IDs shape: {generated_ids.shape}")
        print(f"  Ablation mode active: {ablation_enabled}")
        if ablation_enabled:
            print(f"  Config flags in model:")
            print(f"    - use_zero_latent: {getattr(model.config, 'use_zero_latent', False)}")
            print(f"    - use_random_latent: {getattr(model.config, 'use_random_latent', False)}")
            print(f"    - use_same_latent: {getattr(model.config, 'use_same_latent', False)}")
            print(f"    - use_first_latent_repeat: {getattr(model.config, 'use_first_latent_repeat', False)}")
            print(f"    - use_random_latent_same_dist: {getattr(model.config, 'use_random_latent_same_dist', False)}")
        print()

      # Trim prompt tokens from generated output
      prompt_len = inputs.input_ids.shape[1]
      generated_ids_trimmed = generated_ids[:, prompt_len:]
      batch_output_text = processor.batch_decode(
          generated_ids_trimmed, skip_special_tokens=True
      )

      samples = []
      for data_i in range(i, i + BSZ):
          if data_i < len(data):
            samples.append(data[data_i])
      for j, (sample, model_output) in enumerate(
          zip(samples, batch_output_text), start=i
      ):
        # pdb.set_trace()
        think_chain = extract_think(model_output)
        q_type = sample.get("problem_type", "")
        if q_type == "multiple choice":
          final_ans = _extract_mc_answer(model_output, dataset_name)
        else:
          final_ans = extract_answer(model_output)
          if final_ans == "":
            final_ans = model_output
        sample["output"] = model_output
        sample["prediction"] = final_ans
        sample["reward"] = reward_fn(sample, model_output, q_type, dataset_name)
        sample["correct"] = True if sample["reward"] == 1.0 else False
        qa_type = sample.get("original_question_type", "")
        if sample["problem_type"] != "regression":
          mean_acc.append(sample["reward"])
          mean_acc_bytype[qa_type].append(sample["reward"])
        else:
          mean_mra.append(sample["reward"])
          mean_mra_bytype[qa_type].append(sample["reward"])
        if think_chain:
          sample["process"] = f"<think>{think_chain}</think>"
        
        # Create a JSON-serializable copy of the sample
        sample_copy = sample.copy()
        # Convert multimedia entries to serializable format
        if "multimedia" in sample_copy:
          serializable_multimedia = []
          for mm_entry in sample_copy["multimedia"]:
            mm_copy = mm_entry.copy()
            # Check if 'path' is a PIL Image object and convert to string
            if "path" in mm_copy:
              path_val = mm_copy["path"]
              # DEBUG: Show what type of object we're dealing with
              if j == start_idx:
                print(f"\nDEBUG: Processing multimedia path for sample {j}")
                print(f"  path_val type: {type(path_val)}")
                print(f"  hasattr filename: {hasattr(path_val, 'filename')}")
                if hasattr(path_val, '__class__'):
                  print(f"  class name: {path_val.__class__.__name__}")
              
              if hasattr(path_val, "filename"):
                # PIL Image object - use filename if available
                filename_val = getattr(path_val, "filename", None)
                if filename_val:
                  mm_copy["path"] = str(filename_val)
                else:
                  # PIL Image without filename - describe it
                  mm_copy["path"] = f"PIL_Image_{type(path_val).__name__}"
                  if j == start_idx:
                    print(f"  WARNING: PIL Image has no filename, saved as: {mm_copy['path']}")
              elif not isinstance(path_val, str):
                # Some other non-string object
                mm_copy["path"] = f"<{type(path_val).__name__}_object>"
                if j == start_idx:
                  print(f"  WARNING: Non-string path saved as: {mm_copy['path']}")
              
              # Final debug output
              if j == start_idx:
                print(f"  Final saved path: {mm_copy['path']}")
              # DEBUG: Log what we're saving for the first sample
              if j == start_idx:
                print(f"\nDEBUG: Sample {j} multimedia path being saved:")
                print(f"  Original type: {type(path_val)}")
                print(f"  Saved value: {mm_copy['path']}")
            # Remove any PIL Image objects stored with other keys
            if "image" in mm_copy and not isinstance(mm_copy["image"], str):
              mm_copy["image"] = str(type(mm_copy["image"]))
            serializable_multimedia.append(mm_copy)
          sample_copy["multimedia"] = serializable_multimedia
        
        final_output.append(sample_copy)
        # pdb.set_trace()
      try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
          json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        print(
            f"Processed batch {(i - start_idx)//BSZ + 1}, saved"
            f" {len(final_output)} samples."
        )
      except Exception as e:
        print(f"Error writing to output file: {e}")

    # Recompute metrics from full output so resume mode includes prior samples.
    mean_acc, mean_mra, mean_acc_bytype, mean_mra_bytype = _recompute_metrics_from_output(final_output)

    final_acc = {"mean_acc": float("nan"), "mean_mra": 0.0}
    if mean_acc:
      final_acc["mean_acc"] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
      final_acc["mean_mra"] = torch.tensor(mean_mra).mean().item()
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS for {dataset_name}")
    print(f"Ablation mode: {ablation_suffix if ablation_suffix else 'BASELINE'}")
    print(f"Mean Accuracy: {final_acc['mean_acc']:.4f}")
    if mean_mra:
      print(f"Mean MRA: {final_acc['mean_mra']:.4f}")
    print(f"Total samples: {len(mean_acc)}")
    print(f"{'='*80}\n")

    # by type
    final_acc_bytype = {}
    final_acc_bytype["mean_acc"] = {}
    final_acc_bytype["mean_mra"] = {}
    for qa_type in mean_acc_bytype:
      final_acc_bytype["mean_acc"][qa_type] = torch.tensor(mean_acc_bytype[qa_type]).mean().item()
      if mean_mra_bytype[qa_type] != []:
        final_acc_bytype["mean_mra"][qa_type] = torch.tensor(mean_mra_bytype[qa_type]).mean().item()
    print(f"Final accuracy by type: {final_acc_bytype}")

    # log to wandb
    # mean for the dataset name
    if mean_acc:
      wandb.log(
          {f"{dataset_name}_mean_acc": final_acc["mean_acc"]}
      )
    # mean mra
    if mean_mra != []:
      wandb.log(
          {f"{dataset_name}_mean_mra": final_acc["mean_mra"]}
      )

    # by type
    for qa_type in final_acc_bytype["mean_acc"]:
        wandb.log(
            {f"{dataset_name}_{qa_type}_mean_acc": final_acc_bytype["mean_acc"][qa_type]}
        )
    for qa_type in final_acc_bytype["mean_mra"]:
        wandb.log(
            {f"{dataset_name}_{qa_type}_mean_mra": final_acc_bytype["mean_mra"][qa_type]}
        )


    try:
      with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": final_output, 
                "final_acc": [final_acc],
                "ablation_mode": {
                    "use_zero_latent": args.use_zero_latent,
                    "use_random_latent": args.use_random_latent,
                    "use_same_latent": args.use_same_latent,
                    "use_first_latent_repeat": args.use_first_latent_repeat,
                    "use_random_latent_same_dist": args.use_random_latent_same_dist,
                    "use_random_latent_gt_dist": args.use_random_latent_gt_dist,
                    "use_random_latent_model_dist": args.use_random_latent_model_dist,
                    "use_gt_latent": args.use_gt_latent,
                }
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
      print(f"Final accuracy saved to {OUTPUT_PATH}")
      print(f"DEBUG: File size: {os.path.getsize(OUTPUT_PATH)} bytes")
      print(f"DEBUG: Ablation settings saved in JSON:")
      print(f"  - use_zero_latent: {args.use_zero_latent}")
      print(f"  - use_random_latent: {args.use_random_latent}")
      print(f"  - use_same_latent: {args.use_same_latent}")
      print(f"  - use_first_latent_repeat: {args.use_first_latent_repeat}")
      print(f"  - use_random_latent_same_dist: {args.use_random_latent_same_dist}")
      print(f"  - use_random_latent_gt_dist: {args.use_random_latent_gt_dist}")
      print(f"  - use_random_latent_model_dist: {args.use_random_latent_model_dist}")
      print(f"  - use_gt_latent: {args.use_gt_latent}")
    except Exception as e:
      print(f"Error writing final accuracy to output file: {e}")

    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
