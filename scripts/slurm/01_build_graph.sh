#!/usr/bin/env bash
#SBATCH --job-name=rag_01_build_graph
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
# Step 1: Corpus Ingestion & Corpus-Graph Construction
# ==============================================================================

set -e

echo "=== Starting Slurm Job: Corpus Ingestion & Graph Construction ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

# Load environment modules (uncomment/modify according to HPC cluster setup)
# module load cuda/12.1
# module load python/3.10

# Source User Profile & Conda Environment
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
fi

for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "/opt/conda" "$HOME/mambaforge"; do
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        source "$conda_path/etc/profile.d/conda.sh" 2>/dev/null || true
        break
    fi
done

PYTHON_BIN=""
for candidate in \
    "$HOME/.conda/envs/deep_clinic_rag/bin/python" \
    "$HOME/miniconda3/envs/deep_clinic_rag/bin/python" \
    "$HOME/anaconda3/envs/deep_clinic_rag/bin/python" \
    "$CONDA_PREFIX/bin/python"; do
    if [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    conda activate deep_clinic_rag 2>/dev/null || source activate deep_clinic_rag 2>/dev/null || true
    PYTHON_BIN=$(which python 2>/dev/null || which python3)
fi

echo "[+] Active Python Interpreter: $PYTHON_BIN"

mkdir -p logs data outputs cache

echo "[+] Executing corpus graph building script..."
$PYTHON_BIN -m src.build_graph --config config/default_config.yaml

echo "=== Completed Job: Corpus Graph Construction ==="
