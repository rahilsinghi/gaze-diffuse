#!/bin/bash
# ============================================================
# GazeDiffuse — NYU Torch HPC Environment Setup
# Run this ONCE on Torch to create the conda environment
# and clone all submodules.
#
# Usage:
#   ssh rs9174@login.torch.hpc.nyu.edu   # (or: ssh torch)
#   cd $SCRATCH
#   git clone <your-repo-url> gaze-diffuse
#   cd gaze-diffuse
#   bash scripts/setup_hpc.sh
# ============================================================
set -euo pipefail

echo "=== GazeDiffuse HPC Setup ==="

# --- Check we're on Torch ---
if [[ ! -d "/scratch" ]] && [[ -z "${SCRATCH:-}" ]]; then
    echo "WARNING: SCRATCH not set. Are you on Torch HPC?"
    echo "Set SCRATCH manually or run on Torch."
    exit 1
fi

SCRATCH="${SCRATCH:-/scratch/$USER}"
ENV_DIR="$SCRATCH/envs/gazediffuse"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Project dir: $PROJECT_DIR"
echo "Env dir: $ENV_DIR"

# --- Load modules ---
module purge
module load anaconda3/2024.02
echo "Loaded anaconda3"

# --- Create conda environment in $SCRATCH (NOT $HOME — 50GB quota!) ---
if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR, activating..."
else
    echo "Creating conda environment at $ENV_DIR..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
fi

conda activate "$ENV_DIR"
echo "Python: $(python --version)"
echo "Location: $(which python)"

# --- Install PyTorch with CUDA 12.1 ---
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# --- Install core dependencies ---
echo "Installing core dependencies..."
pip install lightning==2.2.0 \
    transformers==4.38.2 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    wandb>=0.16.0 \
    scipy>=1.12.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0

# --- Install evaluation dependencies ---
echo "Installing evaluation dependencies..."
pip install textstat>=0.7.0 \
    mauve-text>=0.3.0

# --- Install dev tools ---
echo "Installing dev tools..."
pip install pytest>=8.0.0 \
    pytest-cov>=4.1.0 \
    black>=24.0.0 \
    ruff>=0.3.0 \
    isort>=5.13.0

# --- Install flash-attention (optional, speeds up DiT attention) ---
echo "Installing flash-attention (may take a few minutes)..."
pip install flash-attn==2.5.6 || echo "flash-attn install failed (optional, continuing)"

# --- Init and clone submodules ---
echo "Initializing git submodules..."
cd "$PROJECT_DIR"
git submodule update --init --recursive

# --- Install MDLM requirements ---
if [ -f "submodules/mdlm/requirements.txt" ]; then
    echo "Installing MDLM requirements..."
    pip install -r submodules/mdlm/requirements.txt || echo "Some MDLM deps may have conflicts (non-fatal)"
fi

# --- Create required directories ---
mkdir -p data/geco data/ucl checkpoints logs results

# --- Verify GPU access ---
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import transformers
print(f'Transformers: {transformers.__version__}')
import textstat
print(f'textstat: {textstat.__version__}')
print('All imports OK!')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate $ENV_DIR"
echo "Next steps:"
echo "  1. bash scripts/download_data.sh     # Download GECO corpus"
echo "  2. bash scripts/download_checkpoints.sh  # Download model checkpoints"
echo "  3. pytest tests/ -m unit             # Run unit tests"
