# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/eval/model_vqa_depth_discrete.py

import argparse
import json
import math
import os
import re
from typing import List, Optional

import PIL
import numpy as np
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEFAULT_GT_DEPTH_CODEBOOK = os.environ.get("GT_DEPTH_CODEBOOK")


class DiscreteGroundTruthDepthProvider:
    """Provide GT discrete depth token IDs from a serialized codebook."""

    def __init__(self, codebook_path: str, discrete_depth_token_ids: List[int]):
        if not os.path.isfile(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        if not discrete_depth_token_ids:
            raise ValueError("discrete_depth_token_ids must be provided for GT depth injection.")

        self.codebook_path = codebook_path
        self.codebook = np.load(codebook_path, allow_pickle=True).item()
        self.discrete_depth_token_ids = discrete_depth_token_ids
        print(f"[GT DEPTH DISCRETE] Loaded codebook '{codebook_path}' with {len(self.codebook)} entries.")

    def _parse_token_sequence(self, token_string: str) -> List[int]:
        matches = re.findall(r"<DEPTH_(\\d+)>", token_string)
        return [int(m) for m in matches]

    def _image_key(self, image_filename: str) -> str:
        base = os.path.splitext(os.path.basename(image_filename))[0]
        return f"{base}_depth.png"

    def get_token_ids(self, image_filename: str) -> List[int]:
        key = self._image_key(image_filename)
        if key not in self.codebook:
            raise KeyError(f"Image key '{key}' missing in codebook {self.codebook_path}")
        depth_levels = self._parse_token_sequence(self.codebook[key])
        if not depth_levels:
            raise ValueError(f"No discrete depth levels found for '{key}'")
        if max(depth_levels) >= len(self.discrete_depth_token_ids):
            raise ValueError(
                f"Depth level {max(depth_levels)} exceeds token mapping ({len(self.discrete_depth_token_ids)} tokens)."
            )
        return [self.discrete_depth_token_ids[level] for level in depth_levels]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # Check if mm_projector.bin exists, if not, try to extract it from non_lora_trainables.bin
    mm_projector_path = os.path.join(model_path, 'mm_projector.bin')
    if not os.path.exists(mm_projector_path):
        print(f"mm_projector.bin not found at {mm_projector_path}")
        non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_path):
            print("Attempting to extract mm_projector weights from non_lora_trainables.bin...")
            try:
                weights = torch.load(non_lora_path, map_location='cpu')
                mm_keys = [k for k in weights.keys() if 'mm_projector' in k]
                if mm_keys:
                    print(f"Found {len(mm_keys)} mm_projector keys, extracting...")
                    mm_weights = {k: weights[k] for k in mm_keys}
                    torch.save(mm_weights, mm_projector_path)
                    print(f"Successfully created mm_projector.bin with {len(mm_weights)} keys")
                else:
                    print("No mm_projector keys found in non_lora_trainables.bin")
                    raise FileNotFoundError("Cannot find mm_projector weights")
            except Exception as e:
                print(f"Error extracting mm_projector weights: {e}")
                raise
        else:
            raise FileNotFoundError(f"Neither mm_projector.bin nor non_lora_trainables.bin found in {model_path}")
    
    if args.use_gt_depth_embeddings and args.use_random_depth:
        raise ValueError("Cannot enable both --use-gt-depth-embeddings and --use-random-depth simultaneously.")
    if args.use_gt_depth_embeddings and args.use_zero_depth:
        raise ValueError("Cannot enable both --use-gt-depth-embeddings and --use-zero-depth simultaneously.")
    if args.use_random_depth and args.use_zero_depth:
        raise ValueError("Cannot enable both --use-random-depth and --use-zero-depth simultaneously.")

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    depth_mode = getattr(model.config, "depth_mode", None)
    if depth_mode not in {"original", "continuous", "discrete"}:
        if getattr(model.config, "use_discrete_depth_tokens", False):
            depth_mode = "discrete"
        elif getattr(model.config, "depth_token_id", None) is not None:
            depth_mode = "continuous"
        else:
            depth_mode = "original"
    is_original_mode = depth_mode == "original"

    if is_original_mode and (args.use_gt_depth_embeddings or args.use_random_depth or args.use_zero_depth):
        print("[WARNING] Depth ablation flags provided for original-mode checkpoint; flags will be ignored.")
        args.use_gt_depth_embeddings = False
        args.use_random_depth = False
        args.use_zero_depth = False

    discrete_gt_provider: Optional[DiscreteGroundTruthDepthProvider] = None
    if args.use_gt_depth_embeddings:
        discrete_ids = getattr(model.config, "discrete_depth_token_ids", None)
        if not getattr(model.config, "use_discrete_depth_tokens", False):
            print("[WARNING] --use-gt-depth-embeddings is only supported for discrete depth models.")
        elif not discrete_ids:
            print("[WARNING] Model config lacks discrete_depth_token_ids; cannot use GT depth tokens.")
        else:
            try:
                codebook_path = args.gt_depth_codebook or DEFAULT_GT_DEPTH_CODEBOOK
                if not codebook_path:
                    raise ValueError("No discrete GT depth codebook provided. Pass --gt-depth-codebook or set GT_DEPTH_CODEBOOK.")
                discrete_gt_provider = DiscreteGroundTruthDepthProvider(
                    codebook_path=os.path.expanduser(codebook_path),
                    discrete_depth_token_ids=discrete_ids,
                )
            except Exception as exc:
                print(f"[WARNING] Failed to initialize GT depth provider: {exc}")
                discrete_gt_provider = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # for line in tqdm(questions):
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        ###### NEW #######
        image = image.resize((336,336), resample = PIL.Image.NEAREST)
        ###### NEW #######
        image_tensor = process_images([image], image_processor, model.config)[0]
        # Ensure tensors live on the same device as the model (supports multi-GPU via HF `device_map="auto"`).
        device = next(model.parameters()).device
        gt_discrete_tokens: Optional[List[int]] = None
        with torch.inference_mode():
            print(f"Generating for question: {qs[:10]}...")
            
            # Check if model uses discrete depth tokens
            use_discrete_depth_tokens = getattr(model.config, 'use_discrete_depth_tokens', False)
            start_depth_token_id = getattr(model.config, 'depth_start_id', None)
            end_depth_token_id = getattr(model.config, 'depth_end_id', None)
            
            print(f"[DEBUG] Model uses discrete depth tokens: {use_discrete_depth_tokens}")
            print(f"[DEBUG] Start depth token ID: {start_depth_token_id}")
            print(f"[DEBUG] End depth token ID: {end_depth_token_id}")
            
            if use_discrete_depth_tokens and discrete_gt_provider is not None:
                try:
                    gt_discrete_tokens = discrete_gt_provider.get_token_ids(image_file)
                    print(f"[GT DEPTH DISCRETE] Loaded {len(gt_discrete_tokens)} GT tokens for image {image_file}")
                except Exception as exc:
                    print(f"[WARNING] Failed to fetch GT depth tokens for {image_file}: {exc}")
                    gt_discrete_tokens = None

            if is_original_mode:
                print("=== Generation for original mode (depth disabled) ===")
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    use_customize_greedy=False,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            elif use_discrete_depth_tokens:
                # For discrete tokens
                print("=== Generation with discrete depth tokens ===")
                if args.use_random_depth:
                    print("[INFO] Random depth ablation enabled for discrete tokens.")
                if args.use_zero_depth:
                    print("[INFO] Zero-depth ablation enabled (forcing depth_0 tokens).")
                if gt_discrete_tokens:
                    print("[INFO] Injecting GT discrete depth tokens.")

                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    gt_discrete_token_ids=gt_discrete_tokens,
                    use_random_depth=args.use_random_depth,
                    use_zero_depth=args.use_zero_depth,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            else:
                if args.use_random_depth:
                    print("[WARNING] --use-random-depth is only supported for discrete depth models (ignored).")
                if args.use_zero_depth:
                    print("[WARNING] --use-zero-depth is only supported for discrete depth models (ignored).")
                if args.use_gt_depth_embeddings and discrete_gt_provider is not None:
                    print("[WARNING] GT depth tokens requested but model is not discrete; skipping.")
                # For continuous/non-depth tokens
                print("=== Generation without discrete depth tokens ===")
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    use_customize_greedy=False,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            # Process generation output - this model doesn't return depth embeddings separately
            output_ids = generate_out
            depth_embeddings = torch.empty(0)  # Empty tensor as model doesn't output depth embeddings
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"Generated text: {outputs[:200]}...")
            print(f"Depth embeddings shape: {depth_embeddings.shape}")
        
        # Save depth embeddings as numpy array only if not empty
        if depth_embeddings.numel() > 0:
            depth_embeddings_np = depth_embeddings.cpu().numpy()
            
            # Create embeddings directory based on answer file name
            answer_file_basename = os.path.splitext(os.path.basename(answers_file))[0]
            embeddings_dir = os.path.join(os.path.dirname(answers_file), f"{answer_file_basename}_embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            
            depth_embeddings_path = os.path.join(embeddings_dir, f"depth_embeddings_{idx}.npy")
            np.save(depth_embeddings_path, depth_embeddings_np)
        else:
            depth_embeddings_path = None
        
        ans_id = shortuuid.uuid()
        ans_data = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "use_random_depth": args.use_random_depth and use_discrete_depth_tokens,
                "use_zero_depth": args.use_zero_depth and use_discrete_depth_tokens,
                "use_gt_depth_tokens": bool(gt_discrete_tokens),
            },
        }
        
        # Only add depth_embeddings_path if it's not None
        if depth_embeddings_path is not None:
            ans_data["depth_embeddings_path"] = depth_embeddings_path
        
        ans_file.write(json.dumps(ans_data) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--use-gt-depth-embeddings",
        action="store_true",
        help="Inject ground truth discrete depth tokens (discrete models only).",
    )
    parser.add_argument(
        "--use-random-depth",
        action="store_true",
        help="Replace model-generated depth tokens with random tokens (discrete models only).",
    )
    parser.add_argument(
        "--use-zero-depth",
        action="store_true",
        help="Replace model-generated depth tokens with the depth_0 token (discrete models only).",
    )
    parser.add_argument(
        "--gt-depth-codebook",
        type=str,
        default=None,
        help="Path to the discrete depth token codebook (.npy).",
    )
    args = parser.parse_args()

    eval_model(args)

  
