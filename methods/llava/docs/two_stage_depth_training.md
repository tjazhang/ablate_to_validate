<!--
NEW: Aurora-only file; not present in upstream LLAVA.
NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
NEW: Aurora path: docs/two_stage_depth_training.md
-->

# Two-Stage Depth Training for LLaVA

This document describes the two-stage training approach for LLaVA with depth tokens, similar to the MetaMorph training strategy.

## Overview

The two-stage training approach separates the learning of different model components to improve stability and performance:

1. **Stage 1**: Train only the `depth_projector` (freeze `lm_head` and `depth_head`)
2. **Stage 2**: Train `lm_head` and `depth_head` (freeze `depth_projector`)

This approach is fully compatible with:
- Multi-GPU training via DeepSpeed [[memory:4968535]]
- LoRA fine-tuning (in Stage 2)
- Mixed precision training (bf16/fp16)
- Gradient checkpointing

## Training Stages

### Stage 1: Depth Projector Training

In this stage, we focus on learning the projection from depth embeddings to the language model's hidden space.

**What gets trained:**
- `depth_projector`: Linear layer projecting from depth embedding dimension (e.g., 768) to hidden size (e.g., 4096)
- `embed_tokens`: Token embeddings (if new depth tokens are added)

**What stays frozen:**
- `lm_head`: Language modeling head
- `depth_head`: Depth prediction head
- `vision_tower`: Vision encoder (optional)
- All other model parameters

**Key parameters:**
```bash
--depth_training_stage 1
--depth_projector_lr 2e-3  # Optional: higher LR for faster convergence
--freeze_vision_for_depth True
```

### Stage 2: Head Fine-tuning

In this stage, we fine-tune the prediction heads while keeping the learned projection fixed.

**What gets trained:**
- `lm_head`: Language modeling head
- `depth_head`: Depth prediction head
- `embed_tokens`: Token embeddings
- (Optional) LoRA adapters on language model layers

**What stays frozen:**
- `depth_projector`: Projection layer (learned in Stage 1)
- `vision_tower`: Vision encoder (optional)

**Key parameters:**
```bash
--depth_training_stage 2
--depth_head_lr 1e-4  # Optional: separate LR for depth head
--freeze_vision_for_depth True
--lora_enable True  # Optional: use LoRA for efficient fine-tuning
```

## Usage Examples

### Basic Stage 1 Training

```bash
python LLaVA/llava/train/train.py \
    --model_name_or_path llava-v1.5-7b \
    --data_path /path/to/data.json \
    --depth_data True \
    --depth_training_stage 1 \
    --output_dir checkpoints/stage1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-4 \
    --depth_projector_lr 2e-3
```

### Basic Stage 2 Training

```bash
python LLaVA/llava/train/train.py \
    --model_name_or_path checkpoints/stage1 \
    --data_path /path/to/data.json \
    --depth_data True \
    --depth_training_stage 2 \
    --output_dir checkpoints/stage2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --depth_head_lr 1e-4
```

### Stage 2 with LoRA

```bash
python LLaVA/llava/train/train.py \
    --model_name_or_path checkpoints/stage1 \
    --data_path /path/to/data.json \
    --depth_data True \
    --depth_training_stage 2 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --output_dir checkpoints/stage2_lora \
    --num_train_epochs 3
```

## Multi-GPU Training with DeepSpeed

Both stages support distributed training with DeepSpeed:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    LLaVA/llava/train/train.py \
    --depth_training_stage 1 \
    --deepspeed scripts/zero2.json \
    # ... other arguments
```

## Implementation Details

### Parameter Groups

The implementation automatically creates parameter groups with appropriate learning rates:

1. **Stage 1 Optimizer Groups:**
   - Depth projector parameters (with optional `depth_projector_lr`)
   - Other trainable parameters (with base `learning_rate`)

2. **Stage 2 Optimizer Groups:**
   - Depth head parameters (with optional `depth_head_lr`)
   - Language model parameters (with base `learning_rate`)
   - LoRA parameters (if enabled)

### LoRA Compatibility

- **Stage 1**: LoRA is automatically disabled (only projector training)
- **Stage 2**: LoRA can be enabled for efficient fine-tuning
- Depth-related modules (`depth_head`, `depth_projector`, `depth_context_gate`) are automatically excluded from LoRA targets

### Checkpoint Management

The training stage information is saved in the model config:
```python
model.config.depth_training_stage = training_args.depth_training_stage
```

This allows you to:
- Track which stage a checkpoint came from
- Resume training from the correct stage
- Load Stage 1 checkpoints for Stage 2 training

## Recommended Training Recipe

1. **Prepare your data** with depth embeddings
2. **Stage 1**: Train depth_projector for 1 epoch with higher learning rate
3. **Evaluate** Stage 1 checkpoint to ensure proper projection learning
4. **Stage 2**: Fine-tune heads for 3-5 epochs with lower learning rate
5. **Optional**: Further fine-tune with LoRA for specific tasks

## Hyperparameter Recommendations

### Stage 1
- Learning rate: 2e-4 to 5e-4
- Depth projector LR: 2e-3 to 5e-3 (10x base LR)
- Epochs: 1
- Batch size: As large as GPU memory allows
- Warmup ratio: 0.03

### Stage 2
- Learning rate: 2e-5 to 5e-5
- Depth head LR: 1e-4 to 2e-4 (5x base LR)
- Epochs: 3-5
- Batch size: Match Stage 1 or smaller if using LoRA
- Warmup ratio: 0.03
- LoRA rank: 64-256 (if using LoRA)

## Monitoring Training

Watch for these metrics:
- **Stage 1**: `loss_depth_ar` should decrease steadily
- **Stage 2**: Both `loss_language` and `loss_depth_ar` should improve
- Check gradient norms to ensure stable training
- Monitor GPU memory usage for optimal batch size

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce batch size or enable gradient checkpointing
2. **Loss Spikes**: Lower learning rates or increase warmup
3. **Slow Convergence**: Increase stage-specific learning rates
4. **LoRA Not Applied**: Ensure Stage 2 is set when using LoRA

### Debugging

Enable debug mode to see which parameters are being trained:
```python
# The training script will print:
# [STAGE 1] Unfreezing: depth_projector, shape: torch.Size([768, 4096])
# [STAGE 2] Unfreezing: lm_head, shape: torch.Size([32000, 4096])
```

## Advanced Configuration

### Custom Learning Rate Schedules

You can use different schedulers per stage:
```bash
# Stage 1: Constant LR
--lr_scheduler_type constant

# Stage 2: Cosine annealing
--lr_scheduler_type cosine
```

### Mixed Precision Training

Both stages support bf16/fp16:
```bash
--bf16 True  # Recommended for A100/H100
--fp16 True  # For older GPUs
--tf32 True  # Enable TF32 for additional speedup
```

### Gradient Accumulation

For effective larger batch sizes:
```bash
--gradient_accumulation_steps 4
--per_device_train_batch_size 4
# Effective batch size = 4 * 4 * num_gpus
``` 