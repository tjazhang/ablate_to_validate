# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/constants.py

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEPTH_TOKEN = "<DEPTH_TOKEN>"
DEPTH_START_TOKEN = "<DEPTH_START>"
DEPTH_END_TOKEN = "<DEPTH_END>"
DEPTH_TOKEN_ID = 32002
DEPTH_START_ID = 32000
DEPTH_END_ID = 32001