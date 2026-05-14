#!/bin/bash

# Script to verify that embeddings are actually replaced in GT and Zero modes

cache_dir="${MIRAGE_CACHE_DIR:-$(pwd)/checkpoints}"

source ~/.bashrc

mkdir -p logs

echo "=========================================="
echo "Embedding Verification Test"
echo "Testing 10 samples to verify embeddings are replaced"
echo "Note: Using single GPU (cuda:0) to avoid device conflicts"
echo "=========================================="

# Use single GPU for verification
export CUDA_VISIBLE_DEVICES=0

python src/test_embedding_verification.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --task vsp-spatial-planning \
    --split test \
    --load_model_path Miiche/vsp_spatial_planning_direct_sft \
    --cache_dir ${cache_dir}

echo ""
echo "=========================================="
echo "Verification complete!"
echo "Check ./logs/embedding_verification.log for detailed results"
echo "=========================================="
