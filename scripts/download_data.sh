#!/bin/bash
# ============================================================
# Download eye-tracking datasets for gaze predictor training
#
# Datasets:
#   1. GECO: English novel, 14 participants, 5,031 sentences
#   2. UCL Corpus: 350 English sentences, 43 participants
#
# Usage:
#   bash scripts/download_data.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

echo "=== Downloading Eye-Tracking Datasets ==="

# --- GECO Corpus ---
GECO_DIR="$DATA_DIR/geco"
mkdir -p "$GECO_DIR"

if [ -f "$GECO_DIR/MonolingualReadingData.xlsx" ]; then
    echo "GECO already downloaded at $GECO_DIR"
else
    echo "Downloading GECO corpus..."
    echo ""
    echo "IMPORTANT: GECO requires manual download."
    echo "1. Go to: https://expsy.ugent.be/downloads/geco/"
    echo "2. Download 'MonolingualReadingData.xlsx'"
    echo "3. Place it in: $GECO_DIR/"
    echo ""
    echo "Alternative: If you have access to the direct download URL:"
    echo "  wget -O $GECO_DIR/MonolingualReadingData.xlsx <URL>"
    echo ""

    # Try automated download (may not work if behind auth)
    GECO_URL="https://expsy.ugent.be/downloads/geco/MonolingualReadingData.xlsx"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$GECO_DIR/MonolingualReadingData.xlsx" "$GECO_URL" 2>/dev/null || \
            echo "Automated download failed. Please download manually (see above)."
    elif command -v curl &>/dev/null; then
        curl -fsSL -o "$GECO_DIR/MonolingualReadingData.xlsx" "$GECO_URL" 2>/dev/null || \
            echo "Automated download failed. Please download manually (see above)."
    fi

    if [ -f "$GECO_DIR/MonolingualReadingData.xlsx" ]; then
        echo "GECO downloaded successfully!"
        ls -lh "$GECO_DIR/MonolingualReadingData.xlsx"
    fi
fi

# --- UCL Corpus ---
UCL_DIR="$DATA_DIR/ucl"
mkdir -p "$UCL_DIR"

if [ -f "$UCL_DIR/ucl_corpus.csv" ] || [ -f "$UCL_DIR/all_data.csv" ]; then
    echo "UCL corpus already downloaded at $UCL_DIR"
else
    echo ""
    echo "UCL Corpus: Manual download required."
    echo "1. Search for 'UCL corpus reading eye-tracking Frank et al 2013'"
    echo "2. The corpus is available from the authors' website"
    echo "3. Place data files in: $UCL_DIR/"
    echo ""
    echo "Reference: Frank, S. L., Monsalve, I. F., Thompson, R. L., & Vigliocco, G. (2013)."
    echo "Reading time data for evaluating broad-coverage models of English sentence processing."
    echo "Behavior Research Methods, 45(4), 1182-1190."
fi

# --- Verify ---
echo ""
echo "=== Dataset Status ==="
echo "GECO dir: $GECO_DIR"
if [ -f "$GECO_DIR/MonolingualReadingData.xlsx" ]; then
    echo "  MonolingualReadingData.xlsx: $(ls -lh "$GECO_DIR/MonolingualReadingData.xlsx" | awk '{print $5}')"
else
    echo "  MonolingualReadingData.xlsx: NOT FOUND (download manually)"
fi

echo "UCL dir: $UCL_DIR"
if ls "$UCL_DIR"/*.csv 1>/dev/null 2>&1; then
    echo "  CSV files found"
else
    echo "  No data files (download manually)"
fi
