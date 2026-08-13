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

# Enable HuggingFace offline mode for compute nodes without internet access
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Starting Slurm Job: LLM Generation & RAGAS Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

# Load KISSKI Environment Modules & Activate Python Virtual Environment
module load gcc/13.2.0 python/3.11.9 2>/dev/null || true

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

mkdir -p logs outputs

echo "[+] Executing Stage 4 LLM Generation & RAGAS Analysis..."
$PYTHON_EXEC -m src.run_generation_and_eval --config config/default_config.yaml

echo "=== Completed Job: LLM Generation & RAGAS Evaluation ==="
