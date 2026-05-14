#!/bin/bash
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: visualization/visual.sh


# Copyright (c) Meta Platforms, Inc. and affiliates.

# Set up Weight & Biases API key
# Replace with your actual keys
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load gcc/11.2.0
source ~/.bashrc

export WANDB_PROJECT='llavametamorph' # Set your W&B project name (e.g., 'metamorph-debug-pretrain')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${LLAVA_DATA_ROOT:-${METHOD_ROOT}/data}"
VISUAL_OUTPUT_ROOT="${LLAVA_VISUAL_OUTPUT_ROOT:-${METHOD_ROOT}/checkpoints/visual}"

# Define the data subdirectory as a variable
DATA_NAME="ADE20K/train_annealing_data_10"
DATA_DIR="${DATA_ROOT}/${DATA_NAME}_wds/shard_list.json"
EXP_NAME="$(date +%Y%m%d_%H%M%S)" 

encoder_type="dinov2" # Set the encoder type

OUTPUT_DIR="${VISUAL_OUTPUT_ROOT}/${DATA_NAME}/${encoder_type}_${EXP_NAME}"
mkdir -p $OUTPUT_DIR # Use -p to create parent directories if they don't exist

NUM_GPUS=1
NNODES=1 # Adjust to the number of nodes you are using
MASTER_ADDR="localhost" # For a single node, localhost is usually fine
MASTER_PORT=9901 # Choose an available port

# Run the training script with the current set of parameters
torchrun \
  --nproc_per_node $NUM_GPUS \
  --nnodes $NNODES \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
  --project_name $WANDB_PROJECT \
  --run_name $EXP_NAME \
  --output_dir $OUTPUT_DIR \
  --num_layers 2 \
  --mode "mlp" \
  --hidden_dim 2048 \
  --lr 1e-5 \
  --save_step 1000 \
  --batch_size 24 \
  --num_tokens 64 \
  --resolution 320 \
  --cfg_prob 0.8 \
  --image_jsons ${DATA_DIR} \
  --unfreeze_unet \
  --encoder_type ${encoder_type}
