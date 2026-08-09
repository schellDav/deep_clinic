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

# Load KISSKI Environment Modules & Activate Python Virtual Environment
module load gcc/13.2.0 python/3.11.9 2>/dev/null || true

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p logs outputs data/full_62k cache/full_62k

echo "[+] Step 1: Building Full 62k Corpus Graph (62,249 abstracts)..."
python -m src.build_graph --config config/default_config.yaml --max_passages 62249

echo "[+] Step 2: Running 62k Full Corpus Stage 1 Initial Retrieval (BM25s & ModernColBERT)..."
python -m src.retrieve_and_rerank --config config/default_config.yaml

echo "=== Completed Background Job: 62k Full Corpus Ablation ==="
