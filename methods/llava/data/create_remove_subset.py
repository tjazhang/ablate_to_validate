#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/create_remove_subset.py

import json
import os
from pathlib import Path

DEFAULT_INPUT_FILE = Path(__file__).resolve().parent / "ADE20K" / "train_annealing_data" / "train_annealing_data.json"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "ADE20K" / "train_annealing_data" / "train_annealing_data_filtered.json"


def create_filtered_dataset(input_file: Path = DEFAULT_INPUT_FILE, output_file: Path = DEFAULT_OUTPUT_FILE):
    # File paths
    input_file = str(input_file)
    output_file = str(output_file)
    
    # Ranges to remove (format: [start_index, end_index] - both inclusive)
    # You can easily add/remove/modify ranges here
    ranges_to_remove = [
        [37760, 38400],  # First range
        [60160, 61440],  # Second range
    ]
    
    print(f"Loading data from: {input_file}")
    
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)}")
    print(f"Ranges to remove: {ranges_to_remove}")
    
    # Create a set of indices to remove for efficient lookup
    indices_to_remove = set()
    
    for start_idx, end_idx in ranges_to_remove:
        print(f"  - Removing indices {start_idx} to {end_idx} (inclusive)")
        for i in range(start_idx, end_idx + 1):
            indices_to_remove.add(i)
    
    print(f"Total indices to remove: {len(indices_to_remove)}")
    
    # Create filtered dataset by keeping only indices not in the removal set
    filtered_data = []
    
    for i, item in enumerate(data):
        if i not in indices_to_remove:
            filtered_data.append(item)
    
    print(f"Filtered dataset size: {len(filtered_data)}")
    print(f"Removed {len(data) - len(filtered_data)} items")
    
    # Save the filtered dataset
    print(f"Saving filtered dataset to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print("Done!")
    
    # Print detailed summary
    print(f"\nSummary:")
    print(f"- Original size: {len(data)} items")
    print(f"- Ranges removed:")
    total_removed = 0
    for start_idx, end_idx in ranges_to_remove:
        range_size = end_idx - start_idx + 1
        total_removed += range_size
        print(f"  * [{start_idx}, {end_idx}]: {range_size} items")
    print(f"- Total removed: {total_removed} items")
    print(f"- Final size: {len(filtered_data)} items")
    print(f"- Output file: {output_file}")
    
    # Verify our calculation
    expected_final_size = len(data) - total_removed
    if len(filtered_data) == expected_final_size:
        print(f"✓ Verification passed: {len(filtered_data)} == {expected_final_size}")
    else:
        print(f"⚠ Verification failed: {len(filtered_data)} != {expected_final_size}")

if __name__ == "__main__":
    create_filtered_dataset()
