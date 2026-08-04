#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Helper Script for Deep Clinic RAG & GAR Project
# ==============================================================================

set -e

ENV_NAME="deep_clinic_rag"

echo "=== Deep Clinic Environment Initialization ==="

if command -v conda &> /dev/null; then
    echo "[+] Conda detected. Creating/updating Conda environment '${ENV_NAME}'..."
    conda env create -f environment.yml --overwrite || conda env update -f environment.yml --prune
    echo "[+] To activate the environment run:"
    echo "    conda activate ${ENV_NAME}"
elif command -v python3 &> /dev/null; then
    echo "[+] Conda not found. Creating Python virtual environment '.venv'..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[+] Virtual environment created. Activated via: source .venv/bin/activate"
else
    echo "[-] Error: Neither Conda nor Python 3 was found in PATH."
    exit 1
fi

echo "=== Environment Setup Complete ==="
