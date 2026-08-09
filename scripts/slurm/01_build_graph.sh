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

# Load KISSKI Environment Modules & Activate Python Virtual Environment
module load gcc/13.2.0 python/3.11.9 2>/dev/null || true

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p logs data outputs cache

echo "[+] Executing corpus graph building script..."
python -m src.build_graph --config config/default_config.yaml

echo "=== Completed Job: Corpus Graph Construction ==="
