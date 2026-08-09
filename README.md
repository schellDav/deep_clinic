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
4. **End-to-End LLM & RAGAS Analysis:** Final response generation using `Qwen/Qwen3-30B-A3B-Instruct-2507` (or `Qwen2.5-3B-Instruct`) evaluated via `RAGAS` (Faithfulness & Answer Relevance).

---

## 2. Directory Layout

```
Project/
├── cache/                      # Generated vector embedding caches (.npy)
│   └── embeddings.npy
├── config/
│   └── default_config.yaml     # Project configuration (models, batch sizes, graph parameters)
├── data/                       # Ingested corpus and serialized graph artifacts (.npz, .json, .parquet)
│   ├── corpus_graph.npz
│   ├── doc_ids.json
│   ├── ori_pqal.json
│   ├── passages.json
│   ├── pqa_unlabeled.parquet
│   └── qrels.json
├── outputs/                    # Evaluation benchmark results (.json)
│   ├── stage1_retrieval_results.json
│   ├── stage1_retrieval_results_1k.json
│   ├── stage1_retrieval_results_62k.json
│   ├── stage2_rerank_results_1k.json
│   ├── stage2_rerank_results_62k.json
│   ├── stage3_gar_results_1k.json
│   └── stage3_gar_results_62k.json
├── scripts/
│   ├── setup_env.sh            # Helper script to initialize virtual environment
│   └── slurm/                  # HPC Slurm batch execution scripts
│       ├── 01_build_graph.sh   # Build PubMedQA corpus graph
│       ├── 02_retrieve_gar.sh  # Run retrieval, GAR expansion, and re-ranking
│       ├── 03_eval_ragas.sh    # Run LLM generation and RAGAS evaluation
│       ├── 04_full_corpus_ablation.sh # 62k full corpus scaling ablation
│       └── submit_all.sh       # Master Slurm job orchestrator (dependency chaining)
├── src/                        # Core Python source package
│   ├── __init__.py
│   ├── build_graph.py          # Corpus ingestion, embedding & graph builder
│   ├── evaluator.py            # TREC metric evaluation functions (pytrec_eval)
│   ├── preload_models.py       # Model pre-caching utility for offline HPC worker nodes
│   ├── ragas_eval.py           # Stage 4 RAGAS evaluation module
│   ├── rerankers.py            # Stage 2 Cross-Encoder & Stage 3 GAR re-ranking
│   ├── retrievers.py           # Stage 1 BM25s lexical & ModernColBERT dense search
│   ├── retrieve_and_rerank.py  # Stage 1-3 pipeline execution entrypoint
│   └── run_generation_and_eval.py # Stage 4 end-to-end LLM & RAGAS entrypoint
├── tests/                      # Automated test suite
│   ├── test_all_phases.py      # Master test runner
│   ├── test_phase2.py          # Phase 2 graph construction unit tests
│   ├── test_phase3.py          # Phase 3 retrieval baseline unit tests
│   ├── test_phase4.py          # Phase 4 Cross-Encoder & GAR unit tests
│   └── test_phase5.py          # Phase 5 end-to-end LLM & RAGAS unit tests
├── environment.yml             # Conda environment definition
├── PROGRESS.md                 # Minimal functional progress tracker log
├── PROJECT_PLAN.md             # Detailed project specification & execution plan
├── README.md                   # Setup & usage instructions
└── requirements.txt            # Python pip dependencies
```

---

## 3. Environment Setup

Execute the setup helper script on the cluster login node (`glogin10` / KISSKI GPU Cluster):

```bash
module load gcc/13.2.0 python/3.11.9
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate
```

---

## 4. Slurm Cluster Execution (KISSKI Platform)

### Step 0: Pre-cache Models on Login Node (Internet Connection Required)
Because HPC GPU compute worker nodes do not have outbound internet access, pre-download model weights on the login node first:

```bash
python -m src.preload_models
```

### Step 1: Submit Slurm Pipeline Jobs
Submit individual pipeline stages via Slurm:

```bash
# Stage 1: Build Corpus Graph & Embeddings
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

---

## 5. How to Reproduce All Results

Anyone can reproduce the full benchmark pipeline from scratch by following these steps:

### Step 1: Clone Repository & Setup Environment
```bash
git clone https://github.com/schellDav/deep_clinic.git
cd deep_clinic
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### Step 2: Build Corpus Graph & Embeddings
Run the graph builder script to download PubMedQA, compute BM25s index + ModernColBERT dense embeddings, and generate the hybrid k-NN Corpus Graph (`.npz`):
```bash
# Standard 1k Labeled Corpus (Default):
python -m src.build_graph --config config/default_config.yaml

# Full 62k Expanded Corpus (62,249 abstracts):
python -m src.build_graph --config config/default_config.yaml --max_passages 62249

# Or on Slurm Cluster (KISSKI):
sbatch scripts/slurm/01_build_graph.sh
```

### Step 3: Execute Retrieval, GAR Expansion & Re-Ranking
Run retrieval baselines (BM25s & Reason-ModernColBERT), Static Cross-Encoder re-ranking, and Graph-Adaptive Re-Ranking (GAR):
```bash
# Run locally over 1k Labeled Corpus:
python -m src.retrieve_and_rerank --config config/default_config.yaml --max_passages 1000

# Run locally over 62k Full Expanded Corpus:
python -m src.retrieve_and_rerank --config config/default_config.yaml

# Or on Slurm Cluster (KISSKI):
sbatch scripts/slurm/02_retrieve_gar.sh
```

### Step 4: Run End-to-End LLM Generation & RAGAS Evaluation
```bash
# Run end-to-end LLM generation & RAGAS evaluation on Slurm:
sbatch scripts/slurm/03_eval_ragas.sh
```
