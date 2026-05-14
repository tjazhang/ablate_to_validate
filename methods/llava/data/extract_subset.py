#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/extract_subset.py

import json
import os
from pathlib import Path

def extract_dataset_subset():
    """Extract a subset of the dataset based on problematic indices."""
    
    # Define the problematic indices
    # problematic_indices = set(range(39152, 39152 + 128))  # Start at 39152, include 128 items
    problematic_indices = set(range(20000))  # Start at 39152, include 128 items
    print(f"🎯 Will print detailed info for original_indices: {sorted(list(problematic_indices))}")
    
    # Input and output paths
    input_file = "./ADE20K/train_annealing_data/train_annealing_data.json"
    output_file = "./ADE20K/train_annealing_data/train_annealing_data_test.json"
    
    print(f"📂 Loading dataset from: {input_file}")
    
    # Load the original dataset
    try:
        with open(input_file, 'r') as f:
            dataset = json.load(f)
        print(f"✅ Loaded dataset with {len(dataset)} total entries")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return
    
    # Extract subset based on indices
    subset = []
    found_indices = []
    
    for idx in problematic_indices:
        if idx < len(dataset):
            entry = dataset[idx]
            subset.append(entry)
            found_indices.append(idx)
            
            # Print detailed information for this entry
            print(f"\n📋 Index {idx}:")
            print(f"   ID: {entry.get('id', 'N/A')}")
            print(f"   Image: {entry.get('image', 'N/A')}")
            print(f"   Embedding: {entry.get('embedding', 'N/A')}")
            
            # Print conversation details
            conversations = entry.get('conversations', [])
            print(f"   Conversations ({len(conversations)} turns):")
            for i, conv in enumerate(conversations):
                from_field = conv.get('from', 'N/A')
                value = conv.get('value', 'N/A')
                # Truncate long values for readability
                display_value = value if len(value) <= 100 else value[:100] + "..."
                print(f"     {i+1}. {from_field}: {display_value}")
        else:
            print(f"⚠️  Index {idx} is out of range (dataset has {len(dataset)} entries)")
    
    print(f"\n📊 Summary:")
    print(f"   Requested indices: {len(problematic_indices)}")
    print(f"   Found indices: {len(found_indices)}")
    print(f"   Missing indices: {len(problematic_indices) - len(found_indices)}")
    
    if subset:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the subset
        with open(output_file, 'w') as f:
            json.dump(subset, f, indent=2)
        
        print(f"💾 Saved subset with {len(subset)} entries to: {output_file}")
        
        # Show file sizes
        original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        subset_size = os.path.getsize(output_file) / 1024  # KB
        print(f"📏 File sizes:")
        print(f"   Original: {original_size:.1f} MB")
        print(f"   Subset: {subset_size:.1f} KB")
    else:
        print("❌ No valid entries found in the specified range")

if __name__ == "__main__":
    extract_dataset_subset() 