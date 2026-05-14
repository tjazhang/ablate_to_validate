# NEW: Aurora modification relative to upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Upstream-tracked path: qwen-vl-finetune/qwenvl/train/argument.py

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_embeddings: bool = field(default=False, metadata={"help": "Whether to make input embeddings trainable"})
    tune_mm_vision: bool = field(default=False)
    new_tokens_file: Optional[str] = field(default=None, metadata={"help": "Path to file containing new tokens to add"})
    reinitialization_method: Optional[str] = field(default="random", metadata={"help": "Reinitialization method for new tokens"})
    # <NEW>
    depth_input_dim: int = field(default=768)
    continuous_K: int = field(default=64)
    # </NEW>

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_sequential: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    # <NEW>
    # Aurora keeps encoder metadata in a shared registry so the training script can
    # derive the depth feature dimension and token count from an encoder name alone.
    config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to methods/llava/data/encoder_config.json. "
                    "If omitted, train_qwen.py auto-detects the vendored repo default."
        }
    )
    # Name of encoder to use (must exist in config['models']). If None, fallback to config['default_encoder']
    depth_encoder_name: Optional[str] = field(default=None)
    # Populated at runtime after loading config. Holds encoder stub string
    depth_encoder_stub: Optional[str] = field(default=None, init=False)
    # Flag to indicate discrete vs continuous mode (copied from TrainingArguments)
    use_discrete_depth_tokens: bool = field(default=False, init=False)
    # </NEW>


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # <NEW>
    lambda_depth: float = field(default=1.0)
    use_discrete_depth_tokens: bool = field(default=False)
    depth_head_type: str = field(default="linear", metadata={"help": "Depth head type: linear, mlp, mlp2x_gelu"})
    depth_projector_type: str = field(default="linear", metadata={"help": "Depth projector type: linear, mlp, mlp2x_gelu"})
    depth_normalize: bool = field(default=False, metadata={"help": "Apply L2 normalization to depth vectors"})
    depth_apply_softmax: bool = field(default=False, metadata={"help": "Apply softmax to depth vectors (requires normalize)"})
    depth_temperature: float = field(default=1.0, metadata={"help": "Temperature for depth softmax"})
    depth_loss_type: str = field(default="mse", metadata={"help": "Depth loss type: mse, cosine, softmax"})
    depth_loss_layer: int = field(
        default=-1,
        metadata={"help": "Decoder layer index for depth loss (0-based). Use -1 for final hidden state."}
    )
    use_depth_embed_ar: bool = field(default=False, metadata={"help": "Train with depth embedding-space autoregression"})
    depth_embed_ar_gt_ratio: float = field(default=1.0, metadata={
        "help": "Scheduled sampling: probability of using GT depth embeddings vs model predictions "
                "during embed-AR training. 1.0 = always GT, 0.0 = always model. "
                "Use curriculum: start at 1.0 and decay to 0.0 over training."
    })
    # </NEW>
