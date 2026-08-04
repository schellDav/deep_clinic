#!/usr/bin/env bash
#SBATCH --job-name=rag_03_eval_ragas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ==============================================================================
# Step 3: LLM Generation (Llama-3B-Instruct) & End-to-End RAGAS Evaluation
# ==============================================================================

set -e

echo "=== Starting Slurm Job: LLM Generation & RAGAS Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Activate Python Environment
if [ -d "$HOME/.conda/envs/deep_clinic_rag" ]; then
    source activate deep_clinic_rag
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

mkdir -p logs outputs

echo "[+] Executing end-to-end LLM generation and RAGAS evaluation..."
python3 -m src.evaluate_ragas --config config/default_config.yaml

echo "=== Completed Job: LLM Generation & RAGAS Evaluation ==="
