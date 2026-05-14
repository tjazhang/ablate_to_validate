# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/feature_interpolation.py

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def interpolate_features_linear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate a sequence of patch features to ``target_len`` using 1-D linear interpolation.

    Args:
        features: Tensor of shape [N, D].
        target_len: Desired number of patch tokens.

    Returns:
        Tensor of shape [target_len, D].
    """
    # [N, D] ➔ [1, D, N] so that interpolate works on the sequence dimension
    features_1d = features.unsqueeze(0).permute(0, 2, 1)
    interpolated = F.interpolate(features_1d, size=target_len, mode="linear", align_corners=False)
    return interpolated.squeeze(0).permute(1, 0)  # ➔ [target_len, D]


def interpolate_features_bilinear(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate patch features on a 2-D spatial grid using bilinear mode.

    The input tokens are assumed to form a square grid (H × W). The function reshapes the
    sequence back to the grid, applies ``F.interpolate`` with ``mode='bilinear'``, and then
    flattens it back to a sequence.
    """
    num_tokens, dim = features.shape

    h = w = int(np.sqrt(num_tokens))
    if h * w != num_tokens:
        raise ValueError(
            f"Input token length {num_tokens} is not a perfect square – cannot use bilinear interpolation.")

    target_h = target_w = int(np.sqrt(target_len))
    if target_h * target_w != target_len:
        raise ValueError(
            f"Target token length {target_len} is not a perfect square – cannot use bilinear interpolation.")

    # [N, D] ➔ [H, W, D] ➔ [1, D, H, W]
    features_2d = features.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)

    # Interpolate on the spatial dimensions (H, W)
    interpolated = F.interpolate(
        features_2d.to(torch.float32),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(features.dtype)

    # Back to [target_len, D]
    interpolated = interpolated.squeeze(0).permute(1, 2, 0).contiguous().view(target_len, dim)
    return interpolated


def process_embeddings(input_dir: str, output_dir: str, target_len: int, mode: str = "linear") -> None:
    """Iterate over all .npy embedding files, interpolate, and save.

    Args:
        input_dir: Directory containing original *.npy embeddings.
        output_dir: Directory to write interpolated embeddings (created if absent).
        target_len: Desired number of tokens after interpolation.
        mode: 'linear' or 'bilinear'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".npy"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load numpy array ➔ torch tensor (retain dtype)
        features_np = np.load(input_path)
        features = torch.from_numpy(features_np)

        if features.dim() != 2:
            raise ValueError(
                f"Expected 2-D tensor in {filename} of shape [N, D], got shape {features.shape}.")

        if mode == "linear":
            interpolated = interpolate_features_linear(features, target_len)
        elif mode == "bilinear":
            interpolated = interpolate_features_bilinear(features, target_len)
        else:
            raise ValueError("Unknown interpolation mode: {mode}")

        # Save output as numpy (same dtype)
        np.save(output_path, interpolated.cpu().numpy())

        print(f"{filename}: {tuple(features.shape)} ➔ {tuple(interpolated.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate SigLIP patch embeddings.")
    parser.add_argument("--target_num_patches", type=int, required=True, help="Desired number of patch tokens.")
    parser.add_argument(
        "--mode",
        type=str,
        default="bilinear",
        choices=["linear", "bilinear"],
        help="Interpolation mode to use.",
    )
    parser.add_argument("--encoder_name", type=str, default="google_siglip2_large_patch16_256", help="Name of the encoder to interpolate.")
    args = parser.parse_args()

    ade20k_root = Path(os.environ.get("LLAVA_ADE20K_ROOT", str(Path(__file__).resolve().parent / "ADE20K")))
    INPUT_DIR = str(ade20k_root / args.encoder_name)
    OUTPUT_DIR = str(ade20k_root / f"{args.encoder_name}_interploate_{args.target_num_patches}")

    process_embeddings(INPUT_DIR, OUTPUT_DIR, args.target_num_patches, args.mode)
