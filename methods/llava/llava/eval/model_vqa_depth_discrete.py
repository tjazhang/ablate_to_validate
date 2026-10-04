# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/eval/model_vqa_depth_discrete.py

import argparse
import hashlib
import json
import math
import os
import re
from typing import List, Optional

import PIL
import numpy as np
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
# Slot-permutation helpers for the slot-shuffle arm (same module as the Qwen driver's).
from llava import gt_spatial as _gts

DEFAULT_GT_DEPTH_CODEBOOK = os.environ.get("GT_DEPTH_CODEBOOK")


class DiscreteGroundTruthDepthProvider:
    """Provide GT discrete depth token IDs from a serialized codebook."""

    def __init__(self, codebook_path: str, discrete_depth_token_ids: List[int]):
        if not os.path.isfile(codebook_path):
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        if not discrete_depth_token_ids:
            raise ValueError("discrete_depth_token_ids must be provided for GT depth injection.")

        self.codebook_path = codebook_path
        self.codebook = np.load(codebook_path, allow_pickle=True).item()
        self.discrete_depth_token_ids = discrete_depth_token_ids
        print(f"[GT DEPTH DISCRETE] Loaded codebook '{codebook_path}' with {len(self.codebook)} entries.")

    def _parse_token_sequence(self, token_string: str) -> List[int]:
        matches = re.findall(r"<DEPTH_(\d+)>", token_string)
        return [int(m) for m in matches]

    def _image_key(self, image_filename: str) -> str:
        base = os.path.splitext(os.path.basename(image_filename))[0]
        return f"{base}_depth.png"

    def get_token_ids(self, image_filename: str) -> List[int]:
        key = self._image_key(image_filename)
        if key not in self.codebook:
            raise KeyError(f"Image key '{key}' missing in codebook {self.codebook_path}")
        depth_levels = self._parse_token_sequence(self.codebook[key])
        if not depth_levels:
            raise ValueError(f"No discrete depth levels found for '{key}'")
        if max(depth_levels) >= len(self.discrete_depth_token_ids):
            raise ValueError(
                f"Depth level {max(depth_levels)} exceeds token mapping ({len(self.discrete_depth_token_ids)} tokens)."
            )
        return [self.discrete_depth_token_ids[level] for level in depth_levels]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # Check if mm_projector.bin exists, if not, try to extract it from non_lora_trainables.bin
    mm_projector_path = os.path.join(model_path, 'mm_projector.bin')
    if not os.path.exists(mm_projector_path):
        print(f"mm_projector.bin not found at {mm_projector_path}")
        non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_path):
            print("Attempting to extract mm_projector weights from non_lora_trainables.bin...")
            try:
                weights = torch.load(non_lora_path, map_location='cpu')
                mm_keys = [k for k in weights.keys() if 'mm_projector' in k]
                if mm_keys:
                    print(f"Found {len(mm_keys)} mm_projector keys, extracting...")
                    mm_weights = {k: weights[k] for k in mm_keys}
                    torch.save(mm_weights, mm_projector_path)
                    print(f"Successfully created mm_projector.bin with {len(mm_weights)} keys")
                else:
                    print("No mm_projector keys found in non_lora_trainables.bin")
                    raise FileNotFoundError("Cannot find mm_projector weights")
            except Exception as e:
                print(f"Error extracting mm_projector weights: {e}")
                raise
        else:
            raise FileNotFoundError(f"Neither mm_projector.bin nor non_lora_trainables.bin found in {model_path}")
    
    if args.use_gt_depth_embeddings and args.use_random_depth:
        raise ValueError("Cannot enable both --use-gt-depth-embeddings and --use-random-depth simultaneously.")
    if args.use_gt_depth_embeddings and args.use_zero_depth:
        raise ValueError("Cannot enable both --use-gt-depth-embeddings and --use-zero-depth simultaneously.")
    if args.use_random_depth and args.use_zero_depth:
        raise ValueError("Cannot enable both --use-random-depth and --use-zero-depth simultaneously.")
    # Slot-shuffle arm: the oracle arm plus one flag. It needs the oracle's own operand
    # (--use-gt-depth-embeddings + codebook) and reorders it.
    if args.use_gt_depth_permuted_discrete and not args.use_gt_depth_embeddings:
        raise ValueError(
            "--use-gt-depth-permuted-discrete requires --use-gt-depth-embeddings: the shuffled "
            "span is the oracle's GT code sequence reordered, so the oracle arguments plus this "
            "flag are the only valid invocation.")
    # Span length of the discrete random/zero arms: --discrete-span-length N, else this
    # image's GT code count from the codebook (the sequence the oracle arm forces). Without
    # either, the forcing processors in llava_llama.py would use the checkpoint's
    # num_depth_tokens, which train.py sets from the depth encoder's patch grid (256 for the
    # default encoder), not the span length identity emits. Checked before the model loads.
    span_arm = "random" if args.use_random_depth else ("zero" if args.use_zero_depth else None)
    span_codebook = args.gt_depth_codebook or DEFAULT_GT_DEPTH_CODEBOOK
    if args.discrete_span_length is not None and args.discrete_span_length < 1:
        raise SystemExit(f"FATAL: --discrete-span-length must be a positive number of codes, "
                         f"got {args.discrete_span_length}.")
    if span_arm and args.discrete_span_length is None and not span_codebook:
        raise SystemExit(
            f"FATAL: the discrete {span_arm} arm needs a span length. Pass "
            "--discrete-span-length N (100 for the paper's models) or --gt-depth-codebook PATH "
            "to force each image's GT code count.")

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # --resync-discrete-depth-ids (opt-in). On checkpoints whose config.json carries no
    # depth fields, builder.py resolves depth_start_id / depth_end_id /
    # discrete_depth_token_ids before the depth tokens are added to the tokenizer, so every
    # id is 0 (<unk>) and no forcing processor ever engages. This rebuilds them from the
    # returned tokenizer in the training order (llava/train/train.py adds <DEPTH_START>,
    # <DEPTH_END>, <DEPTH_0..N-1>) and writes them where the forcing path reads them
    # (model.config) and to the matching runtime attributes.
    if args.resync_discrete_depth_ids:
        _n_levels = int(getattr(model.config, "num_discrete_depth_levels", None) or 128)
        _embed_rows = int(model.get_input_embeddings().weight.shape[0])
        _start = tokenizer.convert_tokens_to_ids("<DEPTH_START>")
        _end = tokenizer.convert_tokens_to_ids("<DEPTH_END>")
        _levels = [tokenizer.convert_tokens_to_ids(f"<DEPTH_{i}>") for i in range(_n_levels)]
        _all = [_start, _end] + _levels
        _unk = tokenizer.unk_token_id
        _base = int(tokenizer.vocab_size)   # base SentencePiece vocab, before added tokens
        _problems = []
        if any(not isinstance(t, int) for t in _all):
            _problems.append("non-int id")
        if len(set(_all)) != len(_all):
            _problems.append(f"ids not distinct ({len(set(_all))} unique of {len(_all)})")
        if any(t in (0, _unk) for t in _all):
            _problems.append(f"an id is 0 or unk ({_unk})")
        if any(not (_base <= t < _embed_rows) for t in _all):
            _problems.append(f"an id is outside the added-token range [{_base}, {_embed_rows})")
        if len(tokenizer) != _embed_rows:
            _problems.append(f"len(tokenizer)={len(tokenizer)} != embedding rows {_embed_rows}")
        if _problems:
            raise SystemExit(f"FATAL --resync-discrete-depth-ids: {_problems}; start={_start} "
                             f"end={_end} levels[:3]={_levels[:3]} levels[-1]={_levels[-1]}")
        _before = (getattr(model.config, "depth_start_id", None), getattr(model.config, "depth_end_id", None),
                   list(getattr(model.config, "discrete_depth_token_ids", None) or [])[:3])
        model.config.depth_start_id = _start
        model.config.depth_end_id = _end
        model.config.discrete_depth_token_ids = list(_levels)
        for _obj in (model, getattr(model, "get_model", lambda: None)()):
            if _obj is None:
                continue
            for _attr, _val in (("depth_start_id", _start), ("depth_end_id", _end),
                                ("discrete_depth_token_ids", list(_levels))):
                if hasattr(_obj, _attr):
                    setattr(_obj, _attr, _val)
        print(f"[RESYNC DEPTH IDS] before (start, end, levels[:3]) = {_before}")
        print(f"[RESYNC DEPTH IDS] after  start={_start} end={_end} levels[0..2]={_levels[:3]} "
              f"levels[-1]={_levels[-1]} n_levels={_n_levels} embed_rows={_embed_rows} "
              f"len(tokenizer)={len(tokenizer)} base_vocab={_base}")

    depth_mode = getattr(model.config, "depth_mode", None)
    if depth_mode not in {"original", "continuous", "discrete"}:
        if getattr(model.config, "use_discrete_depth_tokens", False):
            depth_mode = "discrete"
        elif getattr(model.config, "depth_token_id", None) is not None:
            depth_mode = "continuous"
        else:
            depth_mode = "original"
    is_original_mode = depth_mode == "original"

    if is_original_mode and (args.use_gt_depth_embeddings or args.use_random_depth or args.use_zero_depth):
        print("[WARNING] Depth ablation flags provided for original-mode checkpoint; flags will be ignored.")
        args.use_gt_depth_embeddings = False
        args.use_random_depth = False
        args.use_zero_depth = False

    discrete_gt_provider: Optional[DiscreteGroundTruthDepthProvider] = None
    if args.use_gt_depth_embeddings:
        discrete_ids = getattr(model.config, "discrete_depth_token_ids", None)
        if not getattr(model.config, "use_discrete_depth_tokens", False):
            print("[WARNING] --use-gt-depth-embeddings is only supported for discrete depth models.")
        elif not discrete_ids:
            print("[WARNING] Model config lacks discrete_depth_token_ids; cannot use GT depth tokens.")
        else:
            try:
                codebook_path = args.gt_depth_codebook or DEFAULT_GT_DEPTH_CODEBOOK
                if not codebook_path:
                    raise ValueError("No discrete GT depth codebook provided. Pass --gt-depth-codebook or set GT_DEPTH_CODEBOOK.")
                discrete_gt_provider = DiscreteGroundTruthDepthProvider(
                    codebook_path=os.path.expanduser(codebook_path),
                    discrete_depth_token_ids=discrete_ids,
                )
            except Exception as exc:
                print(f"[WARNING] Failed to initialize GT depth provider: {exc}")
                discrete_gt_provider = None

    # Hard stop: a forcing flag with unusable depth ids is a silent identity run (the
    # processors wait for depth_start_id and force by id). Checked on model.config, the
    # object the forcing path reads, after the original-mode reset above.
    _forcing = [f for f in ("use_gt_depth_embeddings", "use_gt_depth_permuted_discrete",
                            "use_random_depth", "use_zero_depth") if getattr(args, f, False)]
    if _forcing:
        _cs = getattr(model.config, "depth_start_id", None)
        _ce = getattr(model.config, "depth_end_id", None)
        _cl = list(getattr(model.config, "discrete_depth_token_ids", None) or [])
        _ids = [_cs, _ce] + _cl
        print(f"[DEPTH IDS] forcing flags {_forcing}: depth_start_id={_cs} depth_end_id={_ce} "
              f"n_levels={len(_cl)} levels[:3]={_cl[:3]} levels[-1:]={_cl[-1:]}")
        if (not _cl or any(not isinstance(t, int) or t == 0 for t in _ids)
                or len(set(_ids)) != len(_ids)):
            raise SystemExit(
                f"FATAL: forcing flag(s) {_forcing} set but the depth ids are unusable "
                f"(start={_cs}, end={_ce}, {len(_cl)} level ids, {len(set(_cl))} distinct, "
                f"zeros={sum(1 for t in _ids if t == 0)}). No span would be forced. Pass "
                "--resync-discrete-depth-ids if the checkpoint config lacks depth fields.")

    # The oracle arm and the slot shuffle (which reorders the oracle's operand) both need
    # the GT discrete provider. Without it every row would be an identity pass recorded
    # under the arm's name, so a provider that was not built (the warning above says why)
    # stops the run here.
    if (args.use_gt_depth_embeddings or args.use_gt_depth_permuted_discrete) and discrete_gt_provider is None:
        raise SystemExit(
            "FATAL: --use-gt-depth-embeddings needs the GT discrete provider (built from "
            "--gt-depth-codebook or GT_DEPTH_CODEBOOK) and it was not built. Without it every "
            "row would be an identity pass recorded as an oracle or shuffled one.")

    # The random/zero forcing processors read model.num_depth_tokens at generate time, so
    # each row's span length is written there before its generate call (below).
    forces_span = bool((args.use_random_depth or args.use_zero_depth)
                       and getattr(model.config, "use_discrete_depth_tokens", False))
    span_provider: Optional[DiscreteGroundTruthDepthProvider] = None
    if forces_span and args.discrete_span_length is None:
        try:
            span_provider = DiscreteGroundTruthDepthProvider(
                codebook_path=os.path.expanduser(span_codebook),
                discrete_depth_token_ids=list(getattr(model.config, "discrete_depth_token_ids", None) or []),
            )
        except Exception as exc:
            raise SystemExit(
                f"FATAL: the discrete {span_arm} arm takes each image's span length from the GT "
                f"codebook, which could not be loaded: {exc}. Pass --discrete-span-length N to "
                "set it directly.") from exc
    if forces_span:
        print(f"[DISCRETE SPAN] {span_arm} arm forces "
              + (f"{args.discrete_span_length} codes per row (--discrete-span-length)"
                 if span_provider is None
                 else f"each image's GT code count from {span_provider.codebook_path}"))

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Slot-shuffle arm (discrete): the one fixed slot permutation over the 10x10 VQ-VAE
    # code grid. Same helper, seed and acceptance bar as the Qwen driver's
    # gt_permuted_discrete arm, and its sha is asserted equal to that arm's, so the two
    # shuffled arms are one operator. Built once, written beside the answers, carried in
    # every row.
    discrete_perm = None
    gt_permutation_record = None
    tid2level = None
    if args.use_gt_depth_permuted_discrete:
        _grid = 10                      # VQ-VAE 10x10 code grid
        _k = _grid * _grid
        _bar = int(round(_gts.PERM_MIN_MOVED / _gts.K_DEFAULT * _k))   # 94, as in the Qwen driver
        discrete_perm = _gts.build_slot_permutation(
            k=_k, seed=_gts.PERM_SEED, grid=_grid,
            min_moved=_bar, min_cheb=_gts.PERM_MIN_CHEBYSHEV)
        _facts = _gts.permutation_facts(discrete_perm, _grid)
        _sha = _gts.permutation_sha(discrete_perm)
        # The Qwen driver's gt_permuted_discrete arm builds this same permutation.
        _expected_sha = "123a0b34303ebada3dd580795a3621b2a47d2d98c4313ae5761b8c889eab1b33"
        if _sha != _expected_sha:
            raise SystemExit(f"FATAL: discrete permutation sha {_sha} != the Qwen arm's "
                             f"{_expected_sha}; the two shuffled arms would not be one operator.")
        if not (_facts["is_permutation"] and _facts["is_derangement"]):
            raise SystemExit(f"FATAL: discrete slot permutation is not a derangement: {_facts}")
        gt_permutation_record = {
            "mode": "gt_permuted_discrete",
            "perm_seed": _gts.PERM_SEED,
            "perm": list(discrete_perm),
            "perm_sha256": _sha,
            "grid": _grid,
            "K": _k,
            "builder": ("gt_spatial.build_slot_permutation"
                        "(k=K, seed=perm_seed, grid=grid, min_moved=min_moved_cheb, "
                        "min_cheb=min_chebyshev)"),
            "properties": _facts,
            "min_moved_cheb": _bar,
            "min_chebyshev": _gts.PERM_MIN_CHEBYSHEV,
            "operand": "vqvae_code_ids",
            "codebook": discrete_gt_provider.codebook_path,
        }
        _pj = os.path.join(os.path.dirname(answers_file), "_permutation.json")
        if os.path.exists(_pj):
            # Chunked runs of one arm write into one dir: compare, never overwrite.
            _prev = json.load(open(_pj))
            if (_prev.get("perm_sha256") != _sha
                    or _prev.get("perm_seed") != gt_permutation_record["perm_seed"]):
                raise SystemExit(f"FATAL: {_pj} records sha {_prev.get('perm_sha256')}, this run "
                                 f"would inject {_sha}. One arm, one permutation.")
        else:
            with open(_pj, "w") as _fh:
                json.dump(gt_permutation_record, _fh, indent=1)
        tid2level = {int(t): i for i, t in enumerate(discrete_gt_provider.discrete_depth_token_ids)}

        def _codes_sha(levels):
            return hashlib.sha256(",".join(str(int(x)) for x in levels).encode("utf-8")).hexdigest()
        print(f"[GT PERMUTED DISCRETE] {_grid}x{_grid} code grid, slot s <- GT[perm[s]]; seed "
              f"{_gts.PERM_SEED}, sha {_sha[:12]}, derangement, {_facts['n_moved_cheb_ge']}/{_k} "
              f"slots moved >= {_gts.PERM_MIN_CHEBYSHEV} cells (bar {_bar}) -> {_pj}")

    ans_file = open(answers_file, "w")
    # for line in tqdm(questions):
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        ###### NEW #######
        image = image.resize((336,336), resample = PIL.Image.NEAREST)
        ###### NEW #######
        image_tensor = process_images([image], image_processor, model.config)[0]
        # Ensure tensors live on the same device as the model (supports multi-GPU via HF `device_map="auto"`).
        device = next(model.parameters()).device
        gt_discrete_tokens: Optional[List[int]] = None
        row_perm_meta = None
        # This row's span length for the random/zero arms. A codebook without this image
        # raises here and stops the run rather than forcing a span of another length.
        row_span = None
        if forces_span:
            row_span = (int(args.discrete_span_length) if span_provider is None
                        else len(span_provider.get_token_ids(image_file)))
            model.num_depth_tokens = row_span
            print(f"[DISCRETE SPAN] qid={idx} {span_arm}: forcing {row_span} codes")
        with torch.inference_mode():
            print(f"Generating for question: {qs[:10]}...")
            
            # Check if model uses discrete depth tokens
            use_discrete_depth_tokens = getattr(model.config, 'use_discrete_depth_tokens', False)
            start_depth_token_id = getattr(model.config, 'depth_start_id', None)
            end_depth_token_id = getattr(model.config, 'depth_end_id', None)
            
            print(f"[DEBUG] Model uses discrete depth tokens: {use_discrete_depth_tokens}")
            print(f"[DEBUG] Start depth token ID: {start_depth_token_id}")
            print(f"[DEBUG] End depth token ID: {end_depth_token_id}")
            
            if use_discrete_depth_tokens and discrete_gt_provider is not None:
                try:
                    gt_discrete_tokens = discrete_gt_provider.get_token_ids(image_file)
                    print(f"[GT DEPTH DISCRETE] Loaded {len(gt_discrete_tokens)} GT tokens for image {image_file}")
                except Exception as exc:
                    print(f"[WARNING] Failed to fetch GT depth tokens for {image_file}: {exc}")
                    gt_discrete_tokens = None

            # The oracle arms. Outside the fetch's try/except on purpose: there a missing GT
            # sequence degrades to a free span, which would be an identity row recorded as
            # an oracle or shuffled one, so here it stops the run instead.
            if discrete_gt_provider is not None and (not use_discrete_depth_tokens
                                                     or not gt_discrete_tokens):
                raise RuntimeError(
                    f"--use-gt-depth-embeddings: no GT code sequence for {image_file} "
                    f"(use_discrete_depth_tokens={use_discrete_depth_tokens}); refusing to "
                    "record a free span as an oracle or shuffled row.")
            # The shuffle.
            if discrete_perm is not None:
                if len(gt_discrete_tokens) != len(discrete_perm):
                    raise RuntimeError(
                        f"--use-gt-depth-permuted-discrete: {image_file} has "
                        f"{len(gt_discrete_tokens)} GT codes but the permutation is over "
                        f"{len(discrete_perm)} slots.")
                _oracle_tokens = list(gt_discrete_tokens)
                # slot s receives GT[perm[s]], on code ids.
                gt_discrete_tokens = [_oracle_tokens[discrete_perm[s]]
                                      for s in range(len(discrete_perm))]
                _o_lv = [tid2level[t] for t in _oracle_tokens]
                _f_lv = [tid2level[t] for t in gt_discrete_tokens]
                _multiset_equal = sorted(_o_lv) == sorted(_f_lv)
                if not _multiset_equal:
                    raise RuntimeError(f"shuffled code multiset != oracle's for {image_file}")
                row_perm_meta = {
                    "oracle_codes_sha256": _codes_sha(_o_lv),
                    "forced_codes_sha256": _codes_sha(_f_lv),
                    "multiset_equal": _multiset_equal,
                    "n_index_moved": sum(1 for s in range(len(discrete_perm)) if discrete_perm[s] != s),
                    "n_code_changed": sum(1 for a, b in zip(_o_lv, _f_lv) if a != b),
                    "forced_codes": _f_lv,
                }
                print(f"[GT PERMUTED DISCRETE] qid={idx} image={image_file} oracle_sha="
                      f"{row_perm_meta['oracle_codes_sha256'][:12]} forced_sha="
                      f"{row_perm_meta['forced_codes_sha256'][:12]} multiset_equal=True "
                      f"index_moved={row_perm_meta['n_index_moved']}/{len(discrete_perm)} "
                      f"code_changed={row_perm_meta['n_code_changed']}/{len(discrete_perm)}")

            if is_original_mode:
                print("=== Generation for original mode (depth disabled) ===")
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    use_customize_greedy=False,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            elif use_discrete_depth_tokens:
                # For discrete tokens
                print("=== Generation with discrete depth tokens ===")
                if args.use_random_depth:
                    print("[INFO] Random depth ablation enabled for discrete tokens.")
                if args.use_zero_depth:
                    print("[INFO] Zero-depth ablation enabled (forcing depth_0 tokens).")
                if gt_discrete_tokens:
                    print("[INFO] Injecting GT discrete depth tokens.")

                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    gt_discrete_token_ids=gt_discrete_tokens,
                    use_random_depth=args.use_random_depth,
                    use_zero_depth=args.use_zero_depth,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            else:
                if args.use_random_depth:
                    print("[WARNING] --use-random-depth is only supported for discrete depth models (ignored).")
                if args.use_zero_depth:
                    print("[WARNING] --use-zero-depth is only supported for discrete depth models (ignored).")
                if args.use_gt_depth_embeddings and discrete_gt_provider is not None:
                    print("[WARNING] GT depth tokens requested but model is not discrete; skipping.")
                # For continuous/non-depth tokens
                print("=== Generation without discrete depth tokens ===")
                generate_out = model.generate(
                    inputs=input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    image_sizes=[image.size],
                    use_customize_greedy=False,
                    # Sampling / decoding knobs
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            # Process generation output - this model doesn't return depth embeddings separately
            output_ids = generate_out
            depth_embeddings = torch.empty(0)  # Empty tensor as model doesn't output depth embeddings
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"Generated text: {outputs[:200]}...")
            print(f"Depth embeddings shape: {depth_embeddings.shape}")
        
        # Save depth embeddings as numpy array only if not empty
        if depth_embeddings.numel() > 0:
            depth_embeddings_np = depth_embeddings.cpu().numpy()
            
            # Create embeddings directory based on answer file name
            answer_file_basename = os.path.splitext(os.path.basename(answers_file))[0]
            embeddings_dir = os.path.join(os.path.dirname(answers_file), f"{answer_file_basename}_embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            
            depth_embeddings_path = os.path.join(embeddings_dir, f"depth_embeddings_{idx}.npy")
            np.save(depth_embeddings_path, depth_embeddings_np)
        else:
            depth_embeddings_path = None
        
        ans_id = shortuuid.uuid()
        ans_data = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "use_random_depth": args.use_random_depth and use_discrete_depth_tokens,
                "use_zero_depth": args.use_zero_depth and use_discrete_depth_tokens,
                "use_gt_depth_tokens": bool(gt_discrete_tokens),
            },
        }
        
        # Slot-shuffle provenance, added only when the flag is active so answers produced
        # without it keep their format (the full perm list is in `_permutation.json`).
        if gt_permutation_record is not None:
            ans_data["metadata"]["use_gt_depth_permuted_discrete"] = True
            ans_data["metadata"]["gt_permutation"] = {
                k: gt_permutation_record[k]
                for k in ("mode", "perm_seed", "perm_sha256", "grid", "K",
                          "min_moved_cheb", "min_chebyshev")}
            ans_data["metadata"]["gt_forced"] = row_perm_meta
        # The arm and the span length it forced, on every row of the random/zero arms.
        if row_span is not None:
            ans_data["metadata"]["ablation_mode"] = span_arm
            ans_data["metadata"]["discrete_span_length"] = row_span

        # Only add depth_embeddings_path if it's not None
        if depth_embeddings_path is not None:
            ans_data["depth_embeddings_path"] = depth_embeddings_path
        
        ans_file.write(json.dumps(ans_data) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--use-gt-depth-embeddings",
        action="store_true",
        help="Inject ground truth discrete depth tokens (discrete models only).",
    )
    parser.add_argument(
        "--use-random-depth",
        action="store_true",
        help="Replace model-generated depth tokens with random tokens (discrete models only).",
    )
    parser.add_argument(
        "--use-zero-depth",
        action="store_true",
        help="Replace model-generated depth tokens with the depth_0 token (discrete models only).",
    )
    parser.add_argument(
        "--gt-depth-codebook",
        type=str,
        default=None,
        help="Path to the discrete depth token codebook (.npy).",
    )
    parser.add_argument(
        "--use-gt-depth-permuted-discrete",
        action="store_true",
        help="Slot shuffle: force this image's GT code sequence (the --use-gt-depth-embeddings "
        "operand, which it requires) with slot s <- GT[perm[s]] for one fixed derangement "
        "(gt_spatial.build_slot_permutation, seed 20260902, 10x10 grid; the same permutation "
        "as the Qwen gt_permuted_discrete arm). Same code multiset, same forcing path.",
    )
    parser.add_argument(
        "--discrete-span-length",
        type=int,
        default=None,
        help="Number of depth codes the random and zero arms force between <DEPTH_START> and "
        "<DEPTH_END>. Defaults to this image's GT code count from --gt-depth-codebook, which "
        "keeps those arms length-matched to identity and the oracle; with neither, the two arms "
        "stop. Does not affect the other arms.",
    )
    parser.add_argument(
        "--resync-discrete-depth-ids",
        action="store_true",
        help="After load, rebuild depth_start_id / depth_end_id / the level ids from the "
        "returned tokenizer, assert they are distinct, non-zero, in the added-token range and "
        "that len(tokenizer) == embedding rows, and write them to model.config and the runtime "
        "attributes. Needed when the checkpoint config has no depth fields (the ids otherwise "
        "load as 0).",
    )
    args = parser.parse_args()

    eval_model(args)

  
