#!/usr/bin/env bash
#SBATCH --job-name=rag_04_full_corpus_ablation
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
# Background Ablation Experiment: Full 62k Corpus Scaling Benchmark
# Evaluates 1,000 expert QA queries against 62,249 expanded PubMed abstracts
# (pqa_labeled + pqa_unlabeled) across Stage 1, Stage 2, and Stage 3 GAR.
# ==============================================================================

set -e

echo "=== Starting Background Slurm Job: 62k Full Corpus Ablation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

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

mkdir -p logs outputs data/full_62k cache/full_62k

echo "[+] Step 1: Building Full 62k Corpus Graph (62,249 abstracts)..."
$PYTHON_BIN -m src.build_graph --config config/default_config.yaml --max_passages 62249

echo "[+] Step 2: Running 62k Full Corpus Stage 1 Initial Retrieval (BM25s & ModernColBERT)..."
$PYTHON_BIN -m src.retrieve_and_rerank --config config/default_config.yaml

echo "=== Completed Background Job: 62k Full Corpus Ablation ==="
