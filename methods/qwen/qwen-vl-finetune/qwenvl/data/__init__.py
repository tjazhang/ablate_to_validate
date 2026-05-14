# NEW: Aurora modification relative to upstream QWEN.
# NEW: Baseline https://github.com/QwenLM/Qwen2.5-VL.git @ HEAD (96588727e44c78b25ba03ea03b8e12f7e64fd0da).
# NEW: Upstream-tracked path: qwen-vl-finetune/qwenvl/data/__init__.py

import re
import os

# Release note: the public training examples pass dataset JSON files directly via
# --dataset_use. These named aliases are kept only for backwards compatibility,
# so they resolve through environment variables instead of private absolute paths.
AURORA_DATA_ROOT = os.environ.get("AURORA_DATA_ROOT", "PATH_TO_AURORA_DATA_ROOT")
AURORA_BLIP3O_ROOT = os.environ.get("AURORA_BLIP3O_ROOT", "PATH_TO_AURORA_BLIP3O_ROOT")
AURORA_LLAVA_DATA_ROOT = os.environ.get("AURORA_LLAVA_DATA_ROOT", "PATH_TO_AURORA_LLAVA_DATA_ROOT")
AURORA_QWEN_DATA_ROOT = os.environ.get("AURORA_QWEN_DATA_ROOT", "PATH_TO_AURORA_QWEN_DATA_ROOT")


def _dataset(annotation_path):
    return {
        "annotation_path": annotation_path,
        "data_path": "",
    }


PERSPECTIVE_IMAGINEDEPTH_ONLY_VQGAN1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_imagine_onlydepth_perspective_vqgan1024.json")
)
PERSPECTIVE_IMAGINEDEPTH_ONLY_VQVAE = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_imagine_onlydepth_perspective_vqvae.json")
)
PERSPECTIVE_DEPTH_ONLY_VQGAN1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_onlydepth_perspective_vqgan1024.json")
)
PATHTARCING_GRAY_RECONSTRUCT = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_reconstruct_gray.json")
)
HABITAT_DEPTH_ONLY_VQVAE = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_onlydepth_habitat_vqvae.json")
)
ADE_DEPTH_ONLY_VQVAE = _dataset(
    os.path.join(AURORA_BLIP3O_ROOT, "just_depth_1x_vqvae.json")
)
PERSPECTIVE_DEPTH_ONLY_VQVAE = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_onlydepth_perspective_vqvae.json")
)
PERSPECTIVE_IMAGE_16384 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_image_perspective_16384.json")
)
PERSPECTIVE_DEPTH_IMAGE_VQVAE = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_depthmap_image_perspective_vqvae.json")
)
PERSPECTIVE_GRAYSCALE_IMAGE_1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_grayscale_image_perspective_1024.json")
)
PERSPECTIVE_IMAGE_1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_image_perspective_1024.json")
)
PERSPECTIVE_DIRECT = _dataset(
    os.path.join(AURORA_DATA_ROOT, "train_direct_perspective.json")
)
PATHTRACING_2POINTS_SKETCHTHOUGHT_16k = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_sketch_vqvae.json")
)
PATHTRACING_WITHHINT_2POINTS_16k = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_withhint.json")
)
PATHTRACING_2POINTS_GRAYTHOUGHT_16k_VQGAN = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_grays_vqgan.json")
)
PATHTRACING_2POINTS_GRAYTHOUGHT_16k = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_grays.json")
)
PATHTRACING_2POINTS_NEWDEPTHTHOUGHT_16k = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_newdepthmaps.json")
)
PATHTRACING_2POINTS_DEPTHTHOUGHT_16k = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_16k_depthmaps.json")
)
PATHTRACING_2POINTS_MMTHOUGHT_16384 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_vqgan16384_mmthought.json")
)
PATHTRACING_2POINTS_MMTHOUGHT_1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_vqgan1024_mmthought.json")
)
PATHTRACING_2POINTS_TEXTTHOUGHT = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_textthought.json")
)
PATHTRACING_2POINTS_IMAGETHOUGHT_16384 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_imagethought_16384.json")
)
PATHTRACING_2POINTS_IMAGETHOUGHT_1024 = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_imagethought_1024.json")
)
PATHTRACING_2POINTS_DIRECT = _dataset(
    os.path.join(AURORA_DATA_ROOT, "path_tracing_2point_direct.json")
)
DEPTH_SHORT_AURORA = _dataset(
    os.path.join(AURORA_BLIP3O_ROOT, "only_interleaved_depth_1x_justanswer.json")
)
DEPTH_LONGTEXT_AURORA = _dataset(
    os.path.join(AURORA_BLIP3O_ROOT, "only_interleaved_depth_1x_justtext.json")
)
DEPTH_VQVAE_LONG_AURORA = _dataset(
    os.path.join(AURORA_BLIP3O_ROOT, "only_interleaved_depth_1x_vqvae.json")
)
DEPTH_VQGAN_LONG_AURORA = _dataset(
    os.path.join(AURORA_BLIP3O_ROOT, "only_interleaved_depth_1x_vqgan.json")
)
DEPTH_AURORA = _dataset(
    os.path.join(AURORA_LLAVA_DATA_ROOT, "train_annealing_data.json")
)
TEXT_AURORA = _dataset(
    os.path.join(AURORA_LLAVA_DATA_ROOT, "train_baselinewithcoords90k_annealing_data.json")
)
JUST_DEPTH = _dataset(
    os.path.join(AURORA_QWEN_DATA_ROOT, "outputs", "train_depth_20k_noquestion.json")
)

CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "pathtacing_withhint_2points_16k": PATHTRACING_WITHHINT_2POINTS_16k,
    "pathtracing_gray_reconstruct": PATHTARCING_GRAY_RECONSTRUCT,
    "habitat_depth_only_vqvae": HABITAT_DEPTH_ONLY_VQVAE,
    "ade_depth_only_vqvae": ADE_DEPTH_ONLY_VQVAE,
    "perspective_imaginedepth_vqgan1024": PERSPECTIVE_IMAGINEDEPTH_ONLY_VQGAN1024,
    "perspective_depth_only_vqgan1024": PERSPECTIVE_DEPTH_ONLY_VQGAN1024,
    "perspective_imaginedepth_vqvae": PERSPECTIVE_IMAGINEDEPTH_ONLY_VQVAE,
    "perspective_depth_only_vqvae": PERSPECTIVE_DEPTH_ONLY_VQVAE,
    "perspective_depth_image_vqvae": PERSPECTIVE_DEPTH_IMAGE_VQVAE,
    "perspective_grayscale_image_1024": PERSPECTIVE_GRAYSCALE_IMAGE_1024,
    "perspective_image_16384": PERSPECTIVE_IMAGE_16384,
    "perspective_image_1024": PERSPECTIVE_IMAGE_1024,
    "perspective_direct": PERSPECTIVE_DIRECT,
    "pathtracing_2points_sketchthought_16k": PATHTRACING_2POINTS_SKETCHTHOUGHT_16k,
    "pathtracing_2points_depththought_16k": PATHTRACING_2POINTS_DEPTHTHOUGHT_16k,
    "pathtracing_2points_newdepththought_16k": PATHTRACING_2POINTS_NEWDEPTHTHOUGHT_16k,
    "pathtracing_2points_graythought_16k": PATHTRACING_2POINTS_GRAYTHOUGHT_16k,
    "pathtracing_2points_graythought_16k_vqgan": PATHTRACING_2POINTS_GRAYTHOUGHT_16k_VQGAN,
    "pathtracing_2points_textthought": PATHTRACING_2POINTS_TEXTTHOUGHT,
    "pathtracing_2points_imagethought_1024": PATHTRACING_2POINTS_IMAGETHOUGHT_1024,
    "pathtracing_2points_direct": PATHTRACING_2POINTS_DIRECT,
    "pathtracing_2points_imagethought_16384": PATHTRACING_2POINTS_IMAGETHOUGHT_16384,
    "pathtracing_2points_mmthought_1024": PATHTRACING_2POINTS_MMTHOUGHT_1024,
    "pathtracing_2points_mmthought_16384": PATHTRACING_2POINTS_MMTHOUGHT_16384,
    "depth_vqgan_long_aurora": DEPTH_VQGAN_LONG_AURORA,
    "depth_vqvae_long_aurora": DEPTH_VQVAE_LONG_AURORA,
    "depth_short_aurora": DEPTH_SHORT_AURORA,
    "depth_longtext_aurora": DEPTH_LONGTEXT_AURORA,
    "text_aurora": TEXT_AURORA,
    "depth_aurora": DEPTH_AURORA,
    "just_depth": JUST_DEPTH,
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        elif os.path.exists(dataset_name):
            # <NEW>
            # Support direct path to JSON file
            config = {
                "annotation_path": dataset_name,
                "data_path": "",
                "sampling_rate": sampling_rate
            }
            config_list.append(config)
            # </NEW>
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
