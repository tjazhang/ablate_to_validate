# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/add_depth_token.py

import argparse
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "ADE20K" / "train_annealing_data"
DEFAULT_INPUT_FILE = DATA_ROOT / "train_annealing_data.json"
DEFAULT_OUTPUT_FILE = DATA_ROOT / "train_annealing_data_test_depth.json"

DEPTH_TAG = "<DEPTH_START><DEPTH_TOKEN><DEPTH_END>"

def update_gpt_responses(data):
    for item in data:
        for convo in item.get("conversations", []):
            if convo.get("from") == "gpt":
                if DEPTH_TAG not in convo["value"]:
                    convo["value"] += f" {DEPTH_TAG}"
    return data

def main():
    parser = argparse.ArgumentParser(description="Append the Aurora depth placeholder to GPT responses.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = update_gpt_responses(data)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print("Done. Updated file written to:", args.output_file)

if __name__ == "__main__":
    main()
