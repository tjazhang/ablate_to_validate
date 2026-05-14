#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: eval_summary.py

import argparse
import csv
import json
import os
import sys
from pathlib import Path

def shorten_run_name(run_name: str):
    """Remove date/time prefix from run name."""
    parts = run_name.split("_")
    # Check if first two parts look like date and time (YYYYMMDD_HHMMSS)
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit() and len(parts[1]) == 6:
        return "_".join(parts[2:])
    return run_name

def parse_run_info(run_dir: str):
    run_name = os.path.basename(run_dir.rstrip("/"))
    parts = run_name.split("_")
    tokens = parts[2:] if len(parts) > 2 else parts  # drop date/time if present

    info = {"run": run_name, "short_run": shorten_run_name(run_name)}
    info["mode"] = "continuous" if "continuous" in tokens else ("discrete" if "discrete" in tokens else "")

    info["task"] = "depth_long" if ("depth" in tokens and "long" in tokens) else "_".join(tokens)

    if "enc" in tokens:
        enc_parts = []
        for t in tokens[tokens.index("enc") + 1:]:
            if t.startswith("interploate"):
                break
            enc_parts.append(t)
        info["encoder"] = "_".join(enc_parts)
    else:
        info["encoder"] = ""

    if "interploate" in tokens:
        idx = tokens.index("interploate")
        info["interpolation"] = tokens[idx + 1] if idx + 1 < len(tokens) else ""
    else:
        info["interpolation"] = ""

    return info

def find_jsons(root: str):
    found = []
    for dirpath, _, files in os.walk(root):
        if "evaluation_summary.json" in files:
            found.append(os.path.join(dirpath, "evaluation_summary.json"))
    return found

def format_table_output(records):
    """Format records as readable tables grouped by dataset."""
    from collections import defaultdict
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for rec in records:
        by_dataset[rec["dataset"]].append(rec)
    
    output_lines = []
    output_lines.append("=" * 120)
    output_lines.append("EVALUATION SUMMARY")
    output_lines.append("=" * 120)
    output_lines.append("")
    
    for dataset, recs in sorted(by_dataset.items()):
        output_lines.append("")
        output_lines.append(f"DATASET: {dataset}")
        output_lines.append("-" * 120)
        
        # Header
        header = f"{'Run Name':<60} | {'3-pt Acc':>9} | {'4-pt Acc':>9} | {'5-pt Acc':>9} | {'Overall':>9}"
        output_lines.append(header)
        output_lines.append("-" * 120)
        
        # Rows
        for rec in recs:
            short_name = rec.get("short_run", rec.get("run", ""))
            summary = rec.get("summary", {})
            
            # Extract accuracies
            overall_acc = summary.get("overall", {}).get("accuracy", 0.0) * 100
            per_file = summary.get("per_file", {})
            
            acc_3pt = per_file.get("blink_3pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0) * 100
            acc_4pt = per_file.get("blink_4pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0) * 100
            acc_5pt = per_file.get("blink_5pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0) * 100
            
            row = f"{short_name:<60} | {acc_3pt:>8.2f}% | {acc_4pt:>8.2f}% | {acc_5pt:>8.2f}% | {overall_acc:>8.2f}%"
            output_lines.append(row)
        
        output_lines.append("")
    
    output_lines.append("=" * 120)
    return "\n".join(output_lines)

def main():
    default_out_dir = Path(__file__).resolve().parent / "data" / "evals" / "hardblink" / "answers"
    parser = argparse.ArgumentParser(description="Collect evaluation_summary.json files.")
    parser.add_argument("roots", nargs="+", help="Root directories to search.")
    parser.add_argument(
        "--txt-out",
        default=str(default_out_dir / "eval_summary.txt"),
        help="Path to write the aggregated TXT (one JSON per line).",
    )
    parser.add_argument(
        "--csv-out",
        default=str(default_out_dir / "eval_summary.csv"),
        help="Path to write the aggregated CSV.",
    )
    args = parser.parse_args()

    records = []
    for root in args.roots:
        for json_path in find_jsons(root):
            run_dir = os.path.dirname(json_path)
            dataset = os.path.basename(os.path.dirname(run_dir))
            meta = parse_run_info(run_dir)
            try:
                with open(json_path, "r") as f:
                    summary = json.load(f)
            except Exception as exc:
                summary = {"error": str(exc)}

            record = {
                "dataset": dataset,
                **meta,
                "path": json_path,
                "summary": summary,
            }
            records.append(record)

    if not records:
        print("No evaluation_summary.json files found.")
        return

    os.makedirs(os.path.dirname(args.txt_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    # Write formatted table output to txt file
    formatted_output = format_table_output(records)
    with open(args.txt_out, "w") as txt_file:
        txt_file.write(formatted_output)
    
    print(formatted_output)

    # Write CSV
    fieldnames = ["dataset", "run", "short_run", "mode", "task", "encoder", "interpolation", "acc_3pt", "acc_4pt", "acc_5pt", "overall_acc", "path"]
    with open(args.csv_out, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            summary = rec.get("summary", {})
            overall_acc = summary.get("overall", {}).get("accuracy", 0.0)
            per_file = summary.get("per_file", {})
            
            acc_3pt = per_file.get("blink_3pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0)
            acc_4pt = per_file.get("blink_4pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0)
            acc_5pt = per_file.get("blink_5pointscenter_answers_long.jsonl", {}).get("accuracy", 0.0)
            
            writer.writerow({
                "dataset": rec.get("dataset", ""),
                "run": rec.get("run", ""),
                "short_run": rec.get("short_run", ""),
                "mode": rec.get("mode", ""),
                "task": rec.get("task", ""),
                "encoder": rec.get("encoder", ""),
                "interpolation": rec.get("interpolation", ""),
                "acc_3pt": f"{acc_3pt:.4f}",
                "acc_4pt": f"{acc_4pt:.4f}",
                "acc_5pt": f"{acc_5pt:.4f}",
                "overall_acc": f"{overall_acc:.4f}",
                "path": rec.get("path", ""),
            })

if __name__ == "__main__":
    main()
