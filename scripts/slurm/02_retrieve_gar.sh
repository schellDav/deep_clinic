#!/usr/bin/env bash
#SBATCH --job-name=rag_02_retrieve_gar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
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

# Activate Python Environment
if [ -d "$HOME/.conda/envs/deep_clinic_rag" ]; then
    source activate deep_clinic_rag
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

mkdir -p logs outputs

echo "[+] Executing candidate retrieval, GAR, and cross-encoder re-ranking..."
python3 -m src.retrieve_and_rerank --config config/default_config.yaml

echo "=== Completed Job: Retrieval, GAR & Re-Ranking ==="
