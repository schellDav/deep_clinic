#!/usr/bin/env bash
#SBATCH --job-name=rag_04_full_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=kisski-arbscg
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --constraint=80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ==============================================================================
# Step 4: Full 62k Corpus Benchmarking & Ablation Study
# ==============================================================================

set -e

# Enable HuggingFace offline mode for compute nodes without internet access
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Starting Background Job: 62k Full Corpus Ablation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

# Load Conda or Virtual Environment Python Interpreter
if [ -x "$HOME/.conda/envs/deep_clinic_rag/bin/python" ]; then
    PYTHON_EXEC="$HOME/.conda/envs/deep_clinic_rag/bin/python"
elif [ -x "$HOME/miniconda3/envs/deep_clinic_rag/bin/python" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/deep_clinic_rag/bin/python"
elif [ -x "./.venv/bin/python" ]; then
    PYTHON_EXEC="./.venv/bin/python"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate 2>/dev/null || true
    PYTHON_EXEC=$(which python)
else
    PYTHON_EXEC="python3"
fi

echo "[+] Using Python Interpreter: $PYTHON_EXEC"
$PYTHON_EXEC -c "import sys, numpy; print('[+] Environment Verified:', sys.version, '| NumPy:', numpy.__version__)"

mkdir -p logs outputs data/full_62k cache/full_62k

echo "[+] Step 1: Building Full 62k Corpus Graph (62,249 abstracts)..."
$PYTHON_EXEC -m src.build_graph --config config/default_config.yaml --max_passages 62249

echo "[+] Step 2: Running 62k Full Corpus Stage 1 Initial Retrieval (BM25s & ModernColBERT)..."
$PYTHON_EXEC -m src.retrieve_and_rerank --config config/default_config.yaml

echo "=== Completed Background Job: 62k Full Corpus Ablation ==="
