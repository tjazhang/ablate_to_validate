<!--
NEW: Aurora modification relative to upstream LLAVA.
NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
NEW: Upstream-tracked path: docs/Windows.md
-->

# Run LLaVA on Windows

*NOTE: LLaVA on Windows is not fully supported. Currently we only support 16-bit inference. For a more complete support, please use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for now. More functionalities on Windows is to be added soon, stay tuned.*

## Installation

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install -e .
pip uninstall bitsandbytes
```

## Run demo

See instructions [here](https://github.com/haotian-liu/LLaVA#demo).

Note that quantization (4-bit, 8-bit) is *NOT* supported on Windows. Stay tuned for the 4-bit support on Windows!
