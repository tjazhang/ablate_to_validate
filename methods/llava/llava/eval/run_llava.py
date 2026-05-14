# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/eval/run_llava.py

# run_llava.py

import argparse
import json
import torch
import re
import requests

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from io import BytesIO

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def eval_model(args):
    """
    The original single-example inference path, left as-is.
    Uses --image-file and --query from the CLI to do a single generation.
    """
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Format user prompt (with or without <image> placeholders, etc.)
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Infer conversation style
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] Auto-inferred conv mode is {conv_mode}, but --conv-mode is {args.conv_mode}. Using {args.conv_mode} instead.")
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)  # GPT slot
    prompt = conv.get_prompt()

    # Load the single image
    image = load_image(args.image_file)
    images = [image]
    image_sizes = [image.size]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("=== Model Output ===")
    print(outputs)
    print("====================")


def debug_generate(args):
    """
    A new function that loads a small JSON file in 'train' style format,
    then does generate() on each sample. This is for debugging the model's
    image+text generation path using the same fields as the training data.
    """
    # Load your "mini-train" style JSON: it should be a list of dicts,
    # each with fields like "id", "image", "conversations", "embedding" (optional), etc.
    with open(args.debug_json, "r") as f:
        data = json.load(f)

    # Load model once
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Decide conversation style
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] Auto-inferred conv mode is {conv_mode}, but --conv-mode is {args.conv_mode}. Using {args.conv_mode} instead.")
        conv_mode = args.conv_mode

    # Run inference on each sample
    for idx, sample in enumerate(data):
        print(f"\n----- Sample {idx} ({sample.get('id','no-id')}) -----")

        # The training data has "conversations" as a list of turns
        # We'll gather them to produce a single prompt. We’ll skip the GPT turn
        # if it’s just the “ground truth.” Instead, we only use the user/human turn(s)
        # and then append a GPT turn for the new answer.
        conv = conv_templates[conv_mode].copy()
        for turn in sample["conversations"]:
            from_role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(from_role, turn["value"])
        # Append GPT’s next turn (our new answer)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load image
        image_path = sample["image"]
        image = load_image(image_path)

        # Preprocess image
        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        # Tokenize prompt (with <Image> tokens if needed)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Print the result
        print("Prompt:")
        print(prompt)
        print("Model Output:")
        print(outputs)
        print("----- End Sample -----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None,
                        help="Used in single-sample eval_model() mode")
    parser.add_argument("--query", type=str, default=None,
                        help="Used in single-sample eval_model() mode")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # New argument for JSON-based debugging
    parser.add_argument("--debug-json", type=str, default=None,
                        help="Path to a small JSON file in 'train' format. If provided, runs debug_generate().")

    args = parser.parse_args()

    # If --debug-json is provided, do the multi-sample debug.
    if args.debug_json:
        debug_generate(args)
    else:
        # Otherwise do the original single-image + single-query path
        if not args.image_file or not args.query:
            raise ValueError("Must provide --image-file and --query, or else use --debug-json.")
        eval_model(args)
