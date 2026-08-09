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

# Source Conda Environment
for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "/opt/conda" "$HOME/mambaforge"; do
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        source "$conda_path/etc/profile.d/conda.sh"
        break
    fi
done

if command -v conda &> /dev/null; then
    conda activate deep_clinic_rag 2>/dev/null || source activate deep_clinic_rag 2>/dev/null || true
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

PYTHON_BIN=$(which python 2>/dev/null || which python3 2>/dev/null)
echo "[+] Using Python interpreter: $PYTHON_BIN"

mkdir -p logs outputs

echo "[+] Executing Stage 4 LLM Generation & RAGAS Analysis..."
$PYTHON_BIN -m src.run_generation_and_eval --config config/default_config.yaml

echo "=== Completed Job: LLM Generation & RAGAS Evaluation ==="
