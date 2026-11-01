#!/bin/bash
python3.12 -m venv .venv
source .venv/bin/activate

# pinned dependencies (CUDA 13.0 / NCCL 2.28.9)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130 --no-deps

# general dependencies
sudo apt update && sudo apt install -y ffmpeg pdsh python3.12-dev

# models and weights (vansh's HF key from .env)
source .env
hf auth login --token "$VANSH_HF_TOKEN"

hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./wan22-5b-weights

# Pose Tokenizer (only with: bash setup.sh --all)
if [ "$1" = "--all" ]; then
    pip install git+https://[REDACTED_CREDENTIAL]@github.com/canopy-labs-internal/pose-tokenizer.git
fi

if [ -d /mnt/somfs/hands/pose-checkpoints ] && [ ! -L ./pose-checkpoints ]; then
    ln -s /mnt/somfs/hands/pose-checkpoints ./pose-checkpoints
    echo "Symlinked /mnt/somfs/hands/pose-checkpoints to ./pose-checkpoints"
fi