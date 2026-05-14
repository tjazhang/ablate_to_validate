# NEW: Aurora-only file; not present in upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Aurora path: qwen-vl-finetune/qwenvl/__init__.py

# Qwen2.5-VL custom module with depth additions

from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from .modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
)

__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen2_5_VLCausalLMOutputWithPast",
]
