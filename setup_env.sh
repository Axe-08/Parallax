#!/usr/bin/env bash
# ==============================================================================
# Parallax - Amazon ML Challenge 2026 Environment Bootstrap Script
# ==============================================================================
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh [--gpu | --cpu]
# ==============================================================================

set -euo pipefail

MODE="${1:---gpu}"

echo "=========================================================="
echo "  Parallax: Setting up Amazon ML Challenge Environment"
echo "  Mode: ${MODE}"
echo "=========================================================="

# 1. System packages
echo "==> [1/5] Installing essential system packages..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        git-lfs \
        tmux \
        htop \
        nvtop \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0
fi

# 2. Install uv if not present
echo "==> [2/5] Ensuring uv package manager is installed..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

# 3. Create virtual environment
echo "==> [3/5] Creating Python virtual environment via uv..."
uv venv .venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Install PyTorch with appropriate CUDA backend
echo "==> [4/5] Installing PyTorch..."
if [ "${MODE}" = "--gpu" ]; then
    echo "Installing PyTorch with CUDA 12.4 support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing CPU-only PyTorch..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 5. Install project dependencies & ML packages
echo "==> [5/5] Installing core ML stack & Parallax dependencies..."
uv pip install \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    timm \
    albumentations \
    opencv-python-headless \
    pillow \
    polars \
    pyarrow \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    aiohttp \
    boto3 \
    pydantic \
    tqdm \
    pytest \
    ruff \
    mypy

# Install parallax in editable mode
uv pip install -e .

echo "=========================================================="
echo "  🎉 Parallax environment ready!"
echo "  Activate with: source .venv/bin/activate"
echo "=========================================================="
