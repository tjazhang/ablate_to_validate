import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import json
import logging
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content

from utils import *
from task import *

seed_everything(seed=42)
args=get_args()

if args.data_path == 'PathToJsonlData':
    if args.task == 'vsp-spatial-planning':
        args.data_path = f"./data/vsp_spatial_planning/{args.split}_direct.jsonl"
    else:
        raise ValueError(f"Please provide --data_path for task {args.task}")

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

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
    
# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True, cache_dir=cache_dir)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.load_model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

model.eval()

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

    correct, invalid = 0, 0
    results_to_save = []
    for i, sample in tqdm(enumerate(data)):
        
        preprocess_function = task_test_preporcess_config[args.task]
        conversations = preprocess_function(sample)

        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        texts = [place_input_image(text, sep_token=None) for text in texts]
        image_inputs, _ = process_vision_info(conversations)

        inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
            )
        
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
            logging.info(f"[{i+1}] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")

    logging.info(f"[Final] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")

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
            },
            "results": results_to_save
        }, f, indent=4)
    logging.info(f"Results saved to {results_path}")
