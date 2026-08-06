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
│   └── stage1_retrieval_results_62k.json
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
│   └── retrieve_and_rerank.py  # Stage 1 retrieval baselines & TREC evaluation
├── tests/                      # Automated test suite
│   ├── test_all_phases.py      # Master test runner
│   ├── test_phase2.py          # Phase 2 graph construction unit tests
│   └── test_phase3.py          # Phase 3 retrieval baseline unit tests
├── environment.yml             # Conda environment definition
├── PROGRESS.md                 # Minimal functional progress tracker log
├── PROJECT_PLAN.md             # Detailed project specification & execution plan
├── README.md                   # Setup & usage instructions
└── requirements.txt            # Python pip dependencies
```

---

## 3. Environment Setup

Execute the setup helper script on the cluster head node (KISSKI GPU Cluster):

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

### Step 3: Execute Retrieval, GAR Expansion & Evaluation
```bash
# Run retrieval & GAR candidate expansion:
sbatch scripts/slurm/02_retrieve_gar.sh

# Run end-to-end LLM generation & RAGAS evaluation:
sbatch scripts/slurm/03_eval_ragas.sh
```

