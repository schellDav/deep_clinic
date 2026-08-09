#!/usr/bin/env bash
#SBATCH --job-name=rag_03_eval_ragas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=kisski-arbscg
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --constraint=80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# ==============================================================================
# Step 3: End-to-End LLM Response Generation & RAGAS Evaluation
# Generates medical QA answers for Stage 1, Stage 2 (Cross-Encoder), and Stage 3 (GAR)
# contexts, and evaluates Faithfulness & Answer Relevance via RAGAS.
# Output saved to outputs/stage4_ragas_results_1k.json & outputs/stage4_ragas_results_62k.json.
# ==============================================================================

set -e

echo "=== Starting Slurm Job: LLM Generation & RAGAS Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

# Activate Conda Environment
source ~/.bashrc 2>/dev/null || true
conda activate deep_clinic_rag 2>/dev/null || source activate deep_clinic_rag 2>/dev/null || true

mkdir -p logs outputs

echo "[+] Executing Stage 4 LLM Generation & RAGAS Analysis..."
python -m src.run_generation_and_eval --config config/default_config.yaml

echo "=== Completed Job: LLM Generation & RAGAS Evaluation ==="
