#!/bin/bash
# ============================================================
# Download pretrained model checkpoints
#
# Models:
#   1. MDLM-OWT (~1.2 GB) — Primary model for Experiments 1 + 4
#   2. LLaDA-8B-Instruct (~16 GB) — Experiment 5 (needs A100)
#
# Usage:
#   bash scripts/download_checkpoints.sh          # Download MDLM only
#   bash scripts/download_checkpoints.sh --llada   # Download both
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

DOWNLOAD_LLADA=false
for arg in "$@"; do
    if [ "$arg" = "--llada" ]; then
        DOWNLOAD_LLADA=true
    fi
done

echo "=== Downloading Model Checkpoints ==="

# --- MDLM-OWT (~1.2 GB) ---
MDLM_DIR="$CKPT_DIR/mdlm-owt"
if [ -d "$MDLM_DIR" ] && [ "$(ls -A "$MDLM_DIR" 2>/dev/null)" ]; then
    echo "MDLM-OWT already downloaded at $MDLM_DIR"
else
    echo "Downloading MDLM-OWT from HuggingFace (~1.2 GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'kuleshov-group/mdlm-owt',
    local_dir='$MDLM_DIR',
    local_dir_use_symlinks=False,
)
print('MDLM-OWT download complete!')
"
fi

# --- LLaDA-8B-Instruct (~16 GB) ---
if [ "$DOWNLOAD_LLADA" = true ]; then
    LLADA_DIR="$CKPT_DIR/llada-8b"
    if [ -d "$LLADA_DIR" ] && [ "$(ls -A "$LLADA_DIR" 2>/dev/null)" ]; then
        echo "LLaDA-8B already downloaded at $LLADA_DIR"
    else
        echo "Downloading LLaDA-8B-Instruct from HuggingFace (~16 GB)..."
        echo "This will take a while. Ensure sufficient disk space."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'GSAI-ML/LLaDA-8B-Instruct',
    local_dir='$LLADA_DIR',
    local_dir_use_symlinks=False,
)
print('LLaDA-8B download complete!')
"
    fi
else
    echo "Skipping LLaDA-8B (use --llada flag to download)"
fi

# --- Verify ---
echo ""
echo "=== Checkpoint Status ==="
if [ -d "$MDLM_DIR" ]; then
    echo "MDLM-OWT: $(du -sh "$MDLM_DIR" | cut -f1)"
else
    echo "MDLM-OWT: NOT FOUND"
fi

LLADA_DIR="$CKPT_DIR/llada-8b"
if [ -d "$LLADA_DIR" ]; then
    echo "LLaDA-8B: $(du -sh "$LLADA_DIR" | cut -f1)"
else
    echo "LLaDA-8B: Not downloaded (use --llada to download)"
fi
