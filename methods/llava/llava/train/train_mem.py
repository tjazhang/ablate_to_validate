# NEW: Aurora modification relative to upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Upstream-tracked path: llava/train/train_mem.py

import os
import sys

# Make `llava` importable regardless of launch working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
