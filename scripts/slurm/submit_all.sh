#!/usr/bin/env bash
# ==============================================================================
# Slurm Job Pipeline Submission Orchestrator
# Submits jobs in sequence using dependency chaining (afterok).
# ==============================================================================

set -e

mkdir -p logs

echo "=== Submitting RAG & GAR Pipeline to Slurm Cluster ==="

# Step 1: Build Corpus Graph
JOB1_ID=$(sbatch --parsable scripts/slurm/01_build_graph.sh)
echo "[+] Submitted Job 1 (Build Graph): ID ${JOB1_ID}"

# Step 2: Retrieval, GAR & Cross-Encoder Re-Ranking (depends on Job 1)
JOB2_ID=$(sbatch --parsable --dependency=afterok:${JOB1_ID} scripts/slurm/02_retrieve_gar.sh)
echo "[+] Submitted Job 2 (Retrieval & GAR): ID ${JOB2_ID} (depends on ${JOB1_ID})"

# Step 3: LLM Generation & RAGAS Evaluation (depends on Job 2)
JOB3_ID=$(sbatch --parsable --dependency=afterok:${JOB2_ID} scripts/slurm/03_eval_ragas.sh)
echo "[+] Submitted Job 3 (LLM & RAGAS Eval): ID ${JOB3_ID} (depends on ${JOB2_ID})"

echo "=== Full Pipeline Submitted Successfully ==="
echo "Monitor progress with: squeue -u \$USER"
