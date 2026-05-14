"""
CoVT-LLaVA evaluation wrapper for VLMEvalKit.

Loads a CoVT-finetuned LLaVA-v1.5 model (e.g., Wakals/CoVT-LLaVA-13B-depth)
and runs inference.  At eval time the anchor projection heads are **not** used;
the model simply generates text that may contain visual-thinking tokens, and
we extract the final answer with the same <answer> regex used by CoVTChat.
"""
from __future__ import annotations

import os
import re
import warnings

import torch
from PIL import Image

from ..base import BaseModel
from ...smp import *

ANCHOR_PAD_TOKENS = [
    "<|sam_pad|>",
    "<|dino_pad|>",
    "<|depth_pad|>",
    "<|sd_pad|>",
    "<|intern_pad|>",
    "<|pidinet_pad|>",
    "<|siglip_pad|>",
    "<|metaclip_pad|>",
]

SUPPORTED_ABLATION_MODES = ("zero", "random", "same", "random_dist", "first_repeat")


class _AblationStats:
    """Tracks cumulative ablation replacements across forward calls."""

    def __init__(self):
        self.total_replacements = 0
        self.total_forward_calls = 0
        self.calls_with_replacement = 0
        self._logged_first = False
        self.first_anchor_embed = None

    def reset(self):
        prev = self.total_replacements
        self.total_replacements = 0
        self.total_forward_calls = 0
        self.calls_with_replacement = 0
        self._logged_first = False
        self.first_anchor_embed = None
        return prev


def _install_ablation(model, anchor_token_ids, mode):
    """Monkey-patch model.forward to replace anchor-token embeddings.

    The outer model first computes inputs_embeds from input_ids, then passes
    them to the inner transformer.  We wrap forward so that right after
    inputs_embeds is computed, anchor positions are overridden.
    """
    anchor_id_set = set(anchor_token_ids)
    _original_forward = model.forward
    stats = _AblationStats()
    model._ablation_stats = stats

    def _ablation_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds", None)

        stats.total_forward_calls += 1

        if input_ids is not None and inputs_embeds is None:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in anchor_id_set:
                mask |= (input_ids == tid)

            if mask.any():
                n = mask.sum().item()
                stats.calls_with_replacement += 1
                stats.total_replacements += n

                with torch.no_grad():
                    inputs_embeds = model.get_input_embeddings()(input_ids)
                    anchor_embeds = inputs_embeds[mask].float()
                    orig_norm = anchor_embeds.norm(dim=-1).mean().item()

                    if mode == "zero":
                        replacement = torch.zeros_like(anchor_embeds)
                    elif mode == "random":
                        replacement = torch.randn_like(anchor_embeds)
                    elif mode == "same":
                        # Identity replacement to verify the hook path is active.
                        replacement = anchor_embeds.clone()
                    elif mode == "random_dist":
                        # Sample from a normal with per-token mean/std matching the original vector.
                        mu = anchor_embeds.mean(dim=-1, keepdim=True)
                        sigma = anchor_embeds.std(dim=-1, unbiased=False, keepdim=True)
                        replacement = torch.randn_like(anchor_embeds) * sigma + mu
                    elif mode == "first_repeat":
                        if stats.first_anchor_embed is None:
                            stats.first_anchor_embed = anchor_embeds[:1].clone()
                        replacement = stats.first_anchor_embed.expand_as(anchor_embeds)
                    else:
                        replacement = anchor_embeds

                    inputs_embeds[mask] = replacement.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )

                    new_norm = inputs_embeds[mask].float().norm(dim=-1).mean().item()

                if not stats._logged_first:
                    print(f"[ABLATION-SANITY] First replacement: n_positions={n}, "
                          f"orig_embed_norm={orig_norm:.4f}, new_embed_norm={new_norm:.4f}, "
                          f"mode={mode}")
                    stats._logged_first = True

                kwargs["inputs_embeds"] = inputs_embeds
                kwargs["input_ids"] = None
                if len(args) > 0:
                    args = (None,) + args[1:]

        return _original_forward(*args, **kwargs)

    model.forward = _ablation_forward
    return stats


class CoVTLLaVAChat(BaseModel):

    INSTALL_REQ = True          # needs the llava package
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "Wakals/CoVT-LLaVA-13B-depth",
        model_base: str | None = None,
        use_custom_prompt: bool = True,
        verbose: bool = False,
        ablation_mode: str = "none",
        **kwargs,
    ):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as err:
            logging.critical(
                "Please install llava from https://github.com/haotian-liu/LLaVA"
            )
            raise err

        self.model_path = model_path
        self.verbose = verbose
        self._use_custom_prompt = use_custom_prompt
        self.ablation_mode = ablation_mode

        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=model_base,
                model_name=model_name,
                device_map="cpu",
            )
        )
        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"

        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        # --- Resolve anchor token IDs and install ablation ---
        self._anchor_token_ids = []
        self._anchor_token_map = {}
        for tok_str in ANCHOR_PAD_TOKENS:
            ids = self.tokenizer(tok_str, add_special_tokens=False).input_ids
            if len(ids) == 1:
                tid = ids[0]
                decoded = self.tokenizer.decode([tid])
                self._anchor_token_ids.append(tid)
                self._anchor_token_map[tid] = tok_str
                print(f"[ABLATION-INIT] Token '{tok_str}' -> id={tid}, "
                      f"round-trip='{decoded}', match={decoded.strip() == tok_str}")
            else:
                print(f"[ABLATION-INIT] WARNING: '{tok_str}' tokenizes to {len(ids)} ids "
                      f"({ids}), skipping — not a single token in this vocab")

        self._ablation_stats = None
        if self.ablation_mode in SUPPORTED_ABLATION_MODES:
            if not self._anchor_token_ids:
                print("[ABLATION-INIT] WARNING: No anchor tokens found in vocab! "
                      "Ablation will have NO effect.")
            else:
                hidden_size = self.model.config.hidden_size
                self._ablation_stats = _install_ablation(
                    self.model, self._anchor_token_ids, self.ablation_mode
                )
                print(f"[ABLATION-INIT] CoVT-LLaVA ablation installed: mode={self.ablation_mode}, "
                      f"n_anchor_ids={len(self._anchor_token_ids)}, hidden_size={hidden_size}")
        elif self.ablation_mode != "none":
            print(f"[ABLATION-INIT] WARNING: Unsupported ablation_mode='{self.ablation_mode}'. "
                  f"Supported modes are: {SUPPORTED_ABLATION_MODES}.")

    # ------------------------------------------------------------------
    # Custom prompt (same logic used in CoVTChat / Qwen2VLPrompt)
    # ------------------------------------------------------------------
    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE

        if not self._use_custom_prompt:
            return False
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset in {"MMMU_DEV_VAL", "MMMU_TEST"}:
            return True
        if dataset_type == "MCQ":
            if dataset is not None and "LEGO" in dataset:
                return False
            return True
        if dataset_type == "Y/N" and dataset in {"HallusionBench", "POPE"}:
            return True
        if dataset_type == "VQA" and dataset not in {
            "MMVet", "ChartQAPro", "ChartQAPro_CoT", "ChartQAPro_PoT", "ChartMuseum"
        }:
            return True
        return False

    def build_prompt(self, line, dataset: str):
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {"MMMU_DEV_VAL", "MMMU_TEST"}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == "MCQ":
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == "Y/N":
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == "VQA":
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f"Unsupported dataset: {dataset}")

    # ---- prompt builders (mirrors CoVTPrompt / Qwen2VLPrompt) ----
    def _build_mmmu_prompt(self, line, dataset):
        import string
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += "Please select the correct answer from the options above. \n"
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset):
        MCQ_CN_PROMPT = "请直接回答选项字母。"
        MCQ_EN_PROMPT = "Please select the correct answer from the options above."

        import string
        import pandas as pd

        def cn_string(s):
            import re as _re
            return bool(_re.search("[\u4e00-\u9fff]", s))

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset):
        YORN_PROMPT = " Please answer yes or no."
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=question))
        assert msgs[-1]["type"] == "text"
        msgs[-1]["value"] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset):
        VQA_PROMPT = "\nPlease try to answer the question with short words or phrases if possible."
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=question))
        assert msgs[-1]["type"] == "text"
        msgs[-1]["value"] += VQA_PROMPT
        return msgs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _concat_tilist(message):
        """Concatenate text/image items into a single prompt string + list of image paths."""
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        content, images = self._concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        from abc import abstractproperty
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        if self._ablation_stats is not None:
            self._ablation_stats.reset()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        if self._ablation_stats is not None:
            s = self._ablation_stats
            print(f"[ABLATION-CHECK] sample done: "
                  f"forward_calls={s.total_forward_calls}, "
                  f"calls_with_replacement={s.calls_with_replacement}, "
                  f"total_positions_replaced={s.total_replacements}")
            if s.total_replacements == 0:
                print("[ABLATION-CHECK] WARNING: zero replacements this sample — "
                      "model may not have generated any anchor tokens")

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        if self.verbose:
            print(f"\033[31m{response}\033[0m")

        # ---- Extract answer using the same regex as CoVTChat ----
        ANSWER_REGEX = re.compile(
            r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL
        )
        m = ANSWER_REGEX.search(response)
        if not m:
            response = response.strip()
        else:
            response = m.group(1).strip()

        return response
