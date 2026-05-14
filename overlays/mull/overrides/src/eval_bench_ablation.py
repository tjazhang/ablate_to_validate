# pylint: disable=all
"""Eval script for Video-R1 with Latent Token Ablation Support."""
from collections import defaultdict
import argparse
import importlib
import json
import os
import pdb
import re
import sys
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams
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


def main():
  BSZ = 1  # batch size (reduced from 64 to avoid OOM)
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
          raise ValueError(
              "GT latent is currently unsupported in vLLM eval path (`eval_bench_ablation.py`). "
              "Use `eval_bench_ablation_novllm.py` / `eval_bench_ablation_novllm.sh` instead."
          )
      print("=" * 80)

  print(args.dataset_names)
  dataset_entries = resolve_dataset_entries(args.dataset_names, args)

  MODEL_PATH = args.model_path
  file_name = args.file_name

  # Modify config if ablation is enabled
  if ablation_enabled:
      print(f"Loading model with ablation config from {MODEL_PATH}...")
      
      # Load model with transformers to modify config
      processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
      tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
      tokenizer.padding_side = "left"
      
      # CRITICAL: Add Mull special tokens (as per lmms-eval and README)
      print("\nAdding Mull special tokens to tokenizer...")
      num_added = 0
      num_added += tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
      print(f"Added {num_added} new special tokens")
      
      processor.tokenizer = tokenizer
      
      # Load model for config modification
      model_for_config = Qwen2_5_VLForConditionalGeneration.from_pretrained(
          MODEL_PATH,
          torch_dtype=torch.bfloat16,
          device_map="auto",
          trust_remote_code=True
      )
      
      # CRITICAL: Configure Mull token IDs in model config (as per lmms-eval)
      print("\nConfiguring Mull token IDs in model config...")
      latent_token_idx = tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
      latent_start_idx = tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
      latent_end_idx = tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
      imagelatent_idx = tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
      
      model_for_config.config.latent_token_id = int(latent_token_idx)
      model_for_config.config.latent_start_id = int(latent_start_idx)
      model_for_config.config.latent_end_id = int(latent_end_idx)
      model_for_config.config.imagelatent_token_id = int(imagelatent_idx)
      
      print(f"  latent_token_id (<|latent_pad|>): {model_for_config.config.latent_token_id}")
      print(f"  latent_start_id (<|latent_start|>): {model_for_config.config.latent_start_id}")
      print(f"  imagelatent_token_id (<|latent_image|>): {model_for_config.config.imagelatent_token_id}")
      
      # Resize embeddings if new tokens were added
      if num_added > 0:
          print(f"\nResizing model embeddings to accommodate {num_added} new tokens...")
          model_for_config.resize_token_embeddings(len(tokenizer))
          print(f"New vocabulary size: {len(tokenizer)}")
      
      # Set default latent_sample_temperature if not already set
      if not hasattr(model_for_config.config, 'latent_sample_temperature'):
          model_for_config.config.latent_sample_temperature = 0
          print(f"Set latent_sample_temperature to 0 (default)")
      
      # Set ablation flags in config
      model_for_config.config.use_zero_latent = args.use_zero_latent
      model_for_config.config.use_random_latent = args.use_random_latent
      model_for_config.config.use_same_latent = args.use_same_latent
      model_for_config.config.use_first_latent_repeat = args.use_first_latent_repeat
      model_for_config.config.use_random_latent_same_dist = args.use_random_latent_same_dist
      model_for_config.config.use_random_latent_gt_dist = args.use_random_latent_gt_dist
      model_for_config.config.use_random_latent_model_dist = args.use_random_latent_model_dist
      model_for_config.config.use_gt_latent = args.use_gt_latent
      
      # Fix vLLM compatibility: ensure text_config has required attributes
      # vLLM's get_hf_text_config checks hasattr(config, 'text_config') and if found,
      # asserts text_config.num_attention_heads exists. If text_config is None, this
      # assertion fails. Deleting the attribute makes vLLM fall through to using the
      # main config directly (which already has num_attention_heads and other text fields).
      if hasattr(model_for_config.config, 'text_config'):
          if model_for_config.config.text_config is None:
              print("Note: text_config is None, removing it so vLLM uses main config (which has all text fields)")
              delattr(model_for_config.config, 'text_config')
          else:
              text_config = model_for_config.config.text_config
              # Copy missing attributes from main config if needed
              if not hasattr(text_config, 'num_attention_heads'):
                  if hasattr(model_for_config.config, 'num_attention_heads'):
                      text_config.num_attention_heads = model_for_config.config.num_attention_heads
                  else:
                      print("Warning: num_attention_heads not found in main config either")
      
      # Save modified config to temp directory
      import tempfile
      temp_dir = tempfile.mkdtemp(prefix="mull_ablation_")
      model_for_config.save_pretrained(temp_dir)
      processor.save_pretrained(temp_dir)
      
      print(f"Saved modified model to temp directory: {temp_dir}")
      MODEL_PATH_FOR_VLLM = temp_dir
      
      # CRITICAL: Delete the transformers model and free GPU memory before loading vLLM
      print("Freeing GPU memory from transformers model...")
      del model_for_config
      torch.cuda.empty_cache()
      print("GPU memory cleared successfully")
  else:
      MODEL_PATH_FOR_VLLM = MODEL_PATH
      processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
      tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
      tokenizer.padding_side = "left"
      
      # CRITICAL: Add Mull special tokens even in baseline mode
      print("\nAdding Mull special tokens to tokenizer...")
      num_added = 0
      num_added += tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
      num_added += tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
      print(f"Added {num_added} new special tokens")
      
      processor.tokenizer = tokenizer

  # Patch vLLM's get_hf_text_config to handle text_config=None.
  # The model's config.json has "text_config": null, which causes vLLM to crash with
  # AssertionError at `assert hasattr(config.text_config, "num_attention_heads")`.
  # When text_config is None, deleting the attr makes vLLM fall through to using the
  # main config directly (which already has num_attention_heads and all text fields).
  import vllm.transformers_utils.config as _vllm_config_module
  _original_get_hf_text_config = _vllm_config_module.get_hf_text_config
  def _patched_get_hf_text_config(config):
      if hasattr(config, 'text_config') and config.text_config is None:
          delattr(config, 'text_config')
      return _original_get_hf_text_config(config)
  _vllm_config_module.get_hf_text_config = _patched_get_hf_text_config
  # Also patch the reference in vllm.config since it may have been imported directly
  import vllm.config as _vllm_config
  _vllm_config.get_hf_text_config = _patched_get_hf_text_config

  wandb.init(
      project=file_name,
      mode="offline",
      config=vars(args)
  )

  llm = LLM(
      model=MODEL_PATH_FOR_VLLM,
      tensor_parallel_size=torch.cuda.device_count(),
      max_model_len=8192 * 2,  # Reduced from 16384
      gpu_memory_utilization=0.7,  # Reduced from 0.8
      limit_mm_per_prompt={"image": 4, "video": 2},
      trust_remote_code=True,
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

    sys.path.append("dataloaders/")
    EvalDataset = importlib.import_module("custom_datasets").EvalDataset
    data = EvalDataset(dataset_entry)

    QUESTION_TEMPLATE_LATENT = (
        "{Question}\n"
        "Please think about this question deeply. "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your final answer between the <answer> </answer> tags."
    )
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
      problem_type = sample.get("problem_type", "free-form")

      if dataset_name == "sat":
        option_block = _format_options_with_letters(sample.get("options", []))
        if option_block:
          question = f"{question} Choose from the following options: \n{option_block}"
        return QUESTION_TEMPLATE_LATENT_SAT.format(Question=question) + TYPE_TEMPLATE["multiple choice"]

      if dataset_name == "blink":
        return QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE["multiple choice"]

      if dataset_name == "vsibench":
        q_type = "free-form"
        q_name = sample.get("question_type", "")
        if q_name in VSI_MCA_QUESTION_TYPES:
          q_type = "multiple choice"
        elif q_name in VSI_NA_QUESTION_TYPES:
          q_type = "numerical"
        elif problem_type in TYPE_TEMPLATE:
          q_type = problem_type
        return QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE[q_type]

      return QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])

    def _get_sampling_params(dataset_name):
      if dataset_name in {"sat", "blink"}:
        return SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=[],
        )
      if dataset_name == "vsibench":
        return SamplingParams(
            temperature=0.1,
            top_p=0.001,
            max_tokens=1024,
            stop_token_ids=[],
        )
      if dataset_name == "mmvu":
        return SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=[],
        )
      return SamplingParams(
          temperature=0.0,
          max_tokens=1024,
          stop_token_ids=[],
      )

    messages = []
    for x in data:
      content_multimedia = []
      for im_vid_entry in x["multimedia"]:
        content_multimedia.append({
            "type": im_vid_entry["data_type"],
            im_vid_entry["data_type"]: im_vid_entry["path"],
            # "max_pixels": 360*420,
            # "fps": 1.0
        })
      # CRITICAL: Mull-Tokens requires latent thinking tokens in assistant message
      # As per README.md lines 148-159
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
      else:
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
        output_ans = extract_answer(model_output)
        if output_ans == "":
          output_ans = model_output
        gt_ans = extract_answer(sample.get("solution", ""))
        if question_type == "multiple choice":
          pred_letter = _extract_mc_answer(model_output, dataset_name)
          return 1.0 if pred_letter.strip().lower() == gt_ans.strip().lower() else 0.0
        elif question_type == "numerical":
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
      image_idx = 0
      video_idx = 0

      llm_inputs = []

      for idx, prompt in enumerate(prompts):
        mm_type = batch_messages[idx][0]["content"][0]["type"]
        # calc number of images based on number of times 'type': 'image' in present in batch_messages['content']
        num_images = 0
        num_videos = 0
        for content in batch_messages[idx][0]["content"]:
              if content["type"] == "image":
                  num_images += 1
              if content["type"] == "video":
                  num_videos += 1
        sample_mm_data = {}
        sample_video_kw = {}
        if mm_type == "image":
          sample_mm_data["image"] = image_inputs[image_idx:image_idx + num_images]
          image_idx += num_images
        elif mm_type == "video":# only supports one video input for now.
          sample_mm_data["video"] = video_inputs[video_idx]
          for key, value in video_kwargs.items():
            sample_video_kw[key] = value[video_idx]
          video_idx += 1

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": sample_mm_data,
            "mm_processor_kwargs": sample_video_kw,
        })

      sampling_params = _get_sampling_params(dataset_name)
      outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
      batch_output_text = [out.outputs[0].text for out in outputs]

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
              if isinstance(path_val, Image.Image):
                # PIL Image object - try to extract filename
                if hasattr(path_val, "filename") and path_val.filename:
                  mm_copy["path"] = str(path_val.filename)
                else:
                  # No filename available, use a placeholder
                  mm_copy["path"] = f"PIL_Image_{path_val.format if path_val.format else 'Unknown'}"
              elif not isinstance(path_val, str):
                # Some other non-string object
                mm_copy["path"] = str(path_val)
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

    final_acc = {"mean_acc": 0.0, "mean_mra": 0.0}
    final_acc["mean_acc"] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
      final_acc["mean_mra"] = torch.tensor(mean_mra).mean().item()
    print(f"Final accuracy: {final_acc}")

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
    except Exception as e:
      print(f"Error writing final accuracy to output file: {e}")

    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
