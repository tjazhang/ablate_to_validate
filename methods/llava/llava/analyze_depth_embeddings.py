#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/analyze_depth_embeddings.py

"""
Simplified depth embedding analyzer:
1. Check embeddings exist for data with depth placeholders
2. Check shapes match
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import re
from tqdm import tqdm

METHOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = METHOD_ROOT / "data" / "ADE20K" / "mixed_depth" / "mixed_depth_long.json"
DEFAULT_CONFIG_PATH = METHOD_ROOT / "data" / "encoder_config.json"


def check_depth_tokens_in_conversations(conversations):
    """Check if conversations contain any depth tokens."""
    depth_tokens = ['<DEPTH_TOKEN>', '<DEPTH_START>', '<DEPTH_END>']
    for conv in conversations:
        value = conv.get('value', '')
        for token in depth_tokens:
            if token in value:
                return True
    return False


def load_encoder_config(config_path, encoder_name=None):
    """Load encoder config and derive the encoder stub."""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as cf:
        shared_cfg = json.load(cf)
    
    # Determine encoder to use
    depth_encoder_name = encoder_name or shared_cfg.get("default_encoder")
    
    # Handle interpolate suffix
    interpolate_match = re.match(r"(.+)_interploate_(\d+)$", depth_encoder_name)
    if interpolate_match is not None:
        base_encoder_name = interpolate_match.group(1)
        lookup_encoder_name = base_encoder_name
    else:
        lookup_encoder_name = depth_encoder_name
    
    if lookup_encoder_name not in shared_cfg.get("models", {}):
        raise ValueError(f"Encoder '{lookup_encoder_name}' not found in config models list.")
    
    # Derive stub
    depth_encoder_stub = depth_encoder_name.replace("/", "_").replace("-", "_")
    
    print(f"Using encoder stub: {depth_encoder_stub}\n")
    
    return depth_encoder_stub


def analyze_dataset(dataset_path, encoder_stub):
    """Analyze dataset for two simple checks."""
    
    print(f"Analyzing: {dataset_path}\n")
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file not found: {dataset_path}")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Replace <encoder_stub> placeholder
    for sample in dataset:
        if 'embedding' in sample and '<encoder_stub>' in sample['embedding']:
            sample['embedding'] = sample['embedding'].replace('<encoder_stub>', encoder_stub)
    
    # Stats
    depth_samples = 0
    missing_embeddings = []
    embedding_shapes = defaultdict(int)
    
    # Check each sample with progress bar
    print("Checking samples...")
    for idx, sample in enumerate(tqdm(dataset, desc="Progress")):
        # Check for depth tokens
        has_depth = check_depth_tokens_in_conversations(sample['conversations'])
        
        if has_depth:
            depth_samples += 1
            
            # Check 1: Embedding exists
            if 'embedding' not in sample:
                missing_embeddings.append({
                    'index': idx,
                    'reason': 'No embedding key'
                })
                continue
            
            embedding_path = sample['embedding']
            
            if not os.path.exists(embedding_path):
                missing_embeddings.append({
                    'index': idx,
                    'reason': 'File not found',
                    'path': embedding_path
                })
                continue
            
            # Check 2: Load and check shape
            try:
                embedding = np.load(embedding_path)
                shape_str = str(embedding.shape)
                embedding_shapes[shape_str] += 1
            except Exception as e:
                missing_embeddings.append({
                    'index': idx,
                    'reason': f'Load error: {e}',
                    'path': embedding_path
                })
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Total samples:                  {len(dataset)}")
    print(f"Samples with depth tokens:      {depth_samples}")
    print()
    
    # Check 1: Embeddings exist
    print("CHECK 1: Embeddings exist for depth samples")
    if missing_embeddings:
        print(f"  ✗ FAILED - {len(missing_embeddings)} missing embeddings")
        print()
        for info in missing_embeddings[:5]:
            print(f"  Sample {info['index']}: {info['reason']}")
            if 'path' in info:
                print(f"    Path: {info['path']}")
        if len(missing_embeddings) > 5:
            print(f"  ... and {len(missing_embeddings) - 5} more")
    else:
        print(f"  ✓ PASSED - All depth samples have embeddings")
    print()
    
    # Check 2: Shapes match
    print("CHECK 2: Embedding shapes match")
    if len(embedding_shapes) == 0:
        print(f"  ✗ FAILED - No embeddings loaded")
    elif len(embedding_shapes) == 1:
        shape = list(embedding_shapes.keys())[0]
        count = list(embedding_shapes.values())[0]
        print(f"  ✓ PASSED - All embeddings have shape {shape} ({count} samples)")
    else:
        print(f"  ✗ FAILED - Multiple shapes found:")
        for shape, count in sorted(embedding_shapes.items()):
            print(f"    {shape}: {count} samples")
    print()
    
    # Final verdict
    print(f"{'='*60}")
    if not missing_embeddings and len(embedding_shapes) == 1:
        print("✓ ALL CHECKS PASSED - Dataset ready for training")
    else:
        print("✗ VALIDATION FAILED - Fix issues before training")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Simplified depth embedding analyzer'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help='Path to dataset JSON file or directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help='Path to encoder config JSON file'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default=None,
        help='Depth encoder name (if not specified, uses default from config)'
    )
    
    args = parser.parse_args()
    
    # Load encoder config
    try:
        encoder_stub = load_encoder_config(args.config, args.encoder)
    except Exception as e:
        print(f"ERROR: Failed to load encoder config: {e}")
        return
    
    # Handle directory or single file
    if os.path.isdir(args.dataset):
        json_files = list(Path(args.dataset).glob("*.json"))
        
        if not json_files:
            print(f"ERROR: No JSON files found in {args.dataset}")
            return
        
        print(f"Found {len(json_files)} JSON file(s)\n")
        
        # Analyze each file
        for json_file in sorted(json_files):
            analyze_dataset(str(json_file), encoder_stub)
            print()
    else:
        analyze_dataset(args.dataset, encoder_stub)


if __name__ == "__main__":
    main()
