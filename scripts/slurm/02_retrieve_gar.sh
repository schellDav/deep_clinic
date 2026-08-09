#!/usr/bin/env bash
#SBATCH --job-name=rag_02_retrieve_gar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=kisski-arbscg
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --constraint=80gb_vram
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ==============================================================================
# Step 2: Dense/Lexical Retrieval, Graph-Adaptive Candidate Expansion (GAR),
#         and Cross-Encoder Re-Ranking
# ==============================================================================

set -e

echo "=== Starting Slurm Job: Retrieval, GAR & Re-Ranking ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Load KISSKI Environment Modules & Activate Python Virtual Environment
module load gcc/13.2.0 python/3.11.9 2>/dev/null || true

# Load Conda or Virtual Environment Python Interpreter
if [ -x "$HOME/miniconda3/envs/deep_clinic_rag/bin/python" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/deep_clinic_rag/bin/python"
elif [ -x "./.venv/bin/python" ]; then
    PYTHON_EXEC="./.venv/bin/python"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate 2>/dev/null || true
    PYTHON_EXEC=$(which python)
else
    PYTHON_EXEC="python3"
fi

mkdir -p logs outputs

echo "[+] Executing candidate retrieval, GAR, and cross-encoder re-ranking..."
$PYTHON_EXEC -m src.retrieve_and_rerank --config config/default_config.yaml

echo "=== Completed Job: Retrieval, GAR & Re-Ranking ==="
