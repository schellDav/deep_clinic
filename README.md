# Advanced RAG Benchmarking: Standard, Cross-Encoder, and Graph-Adaptive Re-Ranking (GAR)

**Course:** Deep Learning in the Clinic: From Algorithms to Virtual Cells  
**Authors:** Alexey Wratschinski and David Schell  
**Dataset:** PubMedQA (Biomedical Question Answering)

---

## 1. Project Overview & Architecture

This repository evaluates biomedical Retrieval-Augmented Generation (RAG) across three distinct pipeline stages:
1. **Initial Retrieval (Stage 1):** Sparse Lexical (`BM25s`) & Dense Embedding (`Reason-ModernColBERT` / `qwen3-embed-0.6b`).
2. **Static Re-Ranking (Stage 2):** HuggingFace Cross-Encoder / `monoT5-base`.
3. **Graph-Adaptive Re-Ranking (GAR - Stage 3):** Dynamic candidate pool expansion using multi-hop graph traversal on a pre-computed passage similarity corpus graph (`networkx` & `scipy`).
4. **End-to-End LLM & RAGAS Analysis:** Final response generation using `Llama-3B-Instruct` evaluated via `RAGAS` (Faithfulness & Answer Relevance).

---

## 2. Directory Layout

```
Project/
├── config/
│   └── default_config.yaml     # Project configuration (models, batch sizes, graph parameters)
├── scripts/
│   ├── setup_env.sh            # Helper script to initialize virtual environment
│   └── slurm/                  # HPC Slurm batch execution scripts
│       ├── 01_build_graph.sh   # Build PubMedQA corpus graph
│       ├── 02_retrieve_gar.sh  # Run retrieval, GAR expansion, and re-ranking
│       ├── 03_eval_ragas.sh    # Run LLM generation and RAGAS evaluation
│       └── submit_all.sh       # Master Slurm job orchestrator (dependency chaining)
├── src/                        # Core Python source package
│   └── __init__.py
├── requirements.txt            # Python pip dependencies
├── environment.yml             # Conda environment definition
├── PROJECT_PLAN.md             # Detailed project specification & execution plan
├── REQUIREMENTS_AND_PLAN.txt   # Text plan reference
└── README.md                   # Setup & usage instructions
```

---

## 3. Environment Setup

### Option A: Local Development (NVIDIA RTX 5060)

```bash
# Using Conda / Micromamba
conda env create -f environment.yml
conda activate deep_clinic_rag

# Or using standard pip & venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Slurm Cluster Execution (KISSKI GPU Cluster)

Execute the setup helper script on the cluster head node:

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

---

## 4. Running the Benchmark on Slurm

To submit individual pipeline stages:

```bash
# Stage 1: Build Corpus Graph
sbatch scripts/slurm/01_build_graph.sh

# Stage 2: Retrieval & GAR Re-Ranking
sbatch scripts/slurm/02_retrieve_gar.sh

# Stage 3: LLM Generation & RAGAS Evaluation
sbatch scripts/slurm/03_eval_ragas.sh
```

To submit the entire automated pipeline with job dependency chaining (`afterok`):

```bash
chmod +x scripts/slurm/*.sh
./scripts/slurm/submit_all.sh
```

Monitor job status using:
```bash
squeue -u $USER
```
