#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Helper Script for Deep Clinic RAG & GAR Project
# ==============================================================================

set -e

ENV_NAME="deep_clinic_rag"

echo "=== Deep Clinic Environment Initialization ==="

# Try loading environment modules if available on HPC cluster (KISSKI / GWDG)
if command -v module &> /dev/null; then
    echo "[+] HPC module system detected. Loading Python/Conda modules..."
    module load Anaconda3 2>/dev/null || module load python/3.10 2>/dev/null || true
fi

# Source user Conda if present in standard locations
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda &> /dev/null; then
    echo "[+] Conda detected. Creating/updating Conda environment '${ENV_NAME}'..."
    conda env create -f environment.yml --overwrite || conda env update -f environment.yml --prune
    echo "[+] To activate the environment run:"
    echo "    conda activate ${ENV_NAME}"
elif command -v python3 &> /dev/null; then
    echo "[+] Conda not found. Creating Python virtual environment '.venv' with Python 3..."
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
    echo "[+] Virtual environment created. Activated via: source .venv/bin/activate"
else
    echo "[-] Error: Neither Conda nor Python 3 was found in PATH."
    exit 1
fi

echo "=== Environment Setup Complete ==="
