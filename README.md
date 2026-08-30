# Comparative Evaluation of Multi-Stage Retrieval and Re-Ranking Strategies for Clinical Question Answering

**Course:** Deep Learning in the Clinic: From Algorithms to Virtual Cells  
**Authors:** Alexey Wratschinski and David Schell  
**Dataset:** PubMedQA (Biomedical Question Answering over PubMed Abstracts)  
**Hardware Platform:** KISSKI High-Performance Computing Cluster (NVIDIA A100-SXM4-80GB VRAM)  

---

## 1. Project Overview & Architecture

This repository contains the complete implementation and empirical benchmarks for **Comparative Multi-Stage Retrieval and Re-Ranking**, comparing lexical, dense, neural cross-attention, and graph-adaptive retrieval strategies on biomedical QA:

1. **Stage 1 (Initial Hybrid Retrieval):** Sparse Lexical (`BM25s`) and Transformer Dense Vector (`lightonai/Reason-ModernColBERT`, 768-dim) retrieval.
2. **Stage 2 (Static Neural Re-Ranking):** Full token-level cross-attention re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` over the Stage 1 top-100 candidate pool.
3. **Stage 3 (Graph-Adaptive Re-Ranking / GAR):** Multi-hop dynamic candidate expansion over a pre-computed $k$-NN Corpus Graph ($1.64$ Million edges, $\tau=0.65$, depth $h=2$, decay $\alpha=0.5$) followed by Cross-Encoder neural re-scoring.
4. **Stage 4 (Generative Synthesis & LLM-as-a-Judge Evaluation):** Clinical answer generation with `google/gemma-4-12B-it` evaluated by `Qwen/Qwen3-30B-A3B-Instruct-2507` via RAGAS metrics (*Faithfulness* and *Answer Relevance*).

```
                             [ Clinical Question ]
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
[ BM25s Lexical Index ]                                  [ Reason-ModernColBERT Dense ]
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │   Stage 1: Initial Hybrid Retrieval   │
                   └───────────────────┬───────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────────────┐
│ Stage 2: Static Re-Ranking    │             │ Stage 3: Graph-Adaptive (GAR)         │
│ • ms-marco-MiniLM-L-6-v2      │             │ • Multi-Hop Graph Expansion (1.64M)   │
│ • Re-scores Top-100 Candidates│             │ • Cross-Encoder Neural Re-Scoring     │
└───────────────┬───────────────┘             └───────────────────┬───────────────────┘
                │                                                 │
                └──────────────────────┬──────────────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    Top-5 Verified Context Passages    │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Stage 4: Generative LLM & Evaluation  │
                   │ • google/gemma-4-12B-it Generator     │
                   │ • Qwen/Qwen3-30B-A3B RAGAS Judge      │
                   └───────────────────────────────────────┘
```

---

## 2. Key Empirical Findings

*All reported metrics represent the arithmetic mean across **$N=3$ independent evaluation runs**.*

### 📊 Information Retrieval Benchmarks (1k Labeled vs. 62k Expanded Corpus)

| Corpus Scale | Size | Pipeline Method | nDCG@10 | Recall@10 | Recall@100 | Latency (ms) | Throughput (QPS) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1k Labeled** | 1,000 | BM25s Lexical | 0.9687 | 0.9860 | 0.9930 | 0.16 ms | 6,348 QPS |
| **1k Labeled** | 1,000 | ModernColBERT Dense | 0.9685 | 0.9940 | 1.0000 | 3.32 ms | 301 QPS |
| **1k Labeled** | 1,000 | Cross-Encoder Re-Ranking | 0.9862 | 0.9940 | 0.9940 | 162.74 ms | 6.14 QPS |
| **1k Labeled** | 1,000 | **Graph-Adaptive Re-Ranking (GAR)** | **0.9874** | **0.9960** | **0.9960** | **61.09 ms** | **16.37 QPS** |
| | | | | | | | |
| **62k Full** | 62,249 | BM25s Lexical | 0.8433 | 0.9290 | 0.9750 | 18.38 ms | 54.42 QPS |
| **62k Full** | 62,249 | ModernColBERT Dense | 0.7376 | 0.8900 | 0.9780 | 15.81 ms | 63.25 QPS |
| **62k Full** | 62,249 | Cross-Encoder Re-Ranking | **0.9227** | **0.9640** | 0.9640 | 153.61 ms | 6.51 QPS |
| **62k Full** | 62,249 | **Graph-Adaptive Re-Ranking (GAR)** | **0.9167** | **0.9570** | 0.9570 | 170.83 ms | 5.85 QPS |

### 🔬 End-to-End LLM Generation & Clinical RAGAS Evaluation

*Generator: `google/gemma-4-12B-it` (12B BF16) | Judge: `Qwen/Qwen3-30B-A3B-Instruct-2507` ($N=3$ runs).*

| Corpus Scale | Context Pipeline Method | RAGAS Faithfulness (Factual Grounding) | RAGAS Answer Relevance | Generation Latency |
| :--- | :--- | :---: | :---: | :---: |
| **1k Labeled** | ModernColBERT Dense Baseline | 0.6368 | 0.6414 | 1h 36m (5.78s/query) |
| **1k Labeled** | Cross-Encoder Re-Ranking | 0.6474 (+1.06%) | 0.6351 | 1h 57m (7.07s/query) |
| **1k Labeled** | **Graph-Adaptive Re-Ranking (GAR)** | **0.6685** (**+3.17%**) | 0.6405 | 2h 01m (7.29s/query) |
| | | | | |
| **62k Full** | ModernColBERT Dense Baseline | 0.6577 | **0.7012** | 1h 42m (6.12s/query) |
| **62k Full** | Cross-Encoder Re-Ranking | 0.6716 (+1.39%) | 0.6881 | 2h 13m (8.03s/query) |
| **62k Full** | **Graph-Adaptive Re-Ranking (GAR)** | **0.6991** (**+4.14%**) | 0.6937 | **1h 32m (5.57s/query)** |

---

## 3. Directory Layout

```
Project/
├── config/
│   └── default_config.yaml     # Model configurations, batch sizes, and graph parameters
├── outputs/                    # Benchmark JSON logs and generated publication figures
│   ├── figures/                # Publication-grade PNG (300 DPI) and vector PDF plots
│   │   ├── fig1_retrieval_ndcg_recall.png / .pdf
│   │   ├── fig2_ragas_faithfulness_relevance.png / .pdf
│   │   ├── fig3_corpus_scaling_ablation.png / .pdf
│   │   ├── fig4_latency_throughput_tradeoff.png / .pdf
│   │   └── fig5_latency_faithfulness_tradeoff.png / .pdf
│   ├── stage1_retrieval_results_1k.json / stage1_retrieval_results_62k.json
│   ├── stage2_rerank_results_1k.json / stage2_rerank_results_62k.json
│   ├── stage3_gar_results_1k.json / stage3_gar_results_62k.json
│   └── stage4_ragas_results_1k.json / stage4_ragas_results_62k.json
├── scripts/
│   ├── setup_env.sh            # Virtual environment & Conda initializer
│   └── slurm/                  # HPC Slurm batch execution scripts
│       ├── 01_build_graph.sh   # Build 62k corpus graph & compute dense embeddings
│       ├── 02_retrieve_gar.sh  # Run Stage 1-3 retrieval and GAR re-ranking
│       ├── 03_eval_ragas.sh    # Run Stage 4 Gemma-12B generation & Qwen3-30B judge
│       ├── 04_full_corpus_ablation.sh # End-to-end 62k scaling pipeline job
│       └── submit_all.sh       # Slurm master pipeline orchestrator (dependency chaining)
├── src/                        # Core Python package
│   ├── build_graph.py          # Corpus ingestion, embedding calculation & k-NN graph builder
│   ├── evaluator.py            # TREC IR evaluation engine (pytrec_eval)
│   ├── generator.py            # LLM answer generation module (Gemma-12B)
│   ├── preload_models.py       # Offline model & dataset pre-caching utility
│   ├── ragas_eval.py           # RAGAS LLM-as-a-Judge evaluation engine (Qwen3-30B)
│   ├── rerankers.py            # Cross-Encoder & Graph-Adaptive Re-Ranking (GAR)
│   ├── retrievers.py           # BM25s lexical & ModernColBERT dense search
│   ├── retrieve_and_rerank.py  # Stage 1-3 benchmarking entrypoint
│   ├── run_generation_and_eval.py # Stage 4 end-to-end LLM & RAGAS entrypoint
│   └── visualize_results.py    # Publication-grade plotting script
├── tests/                      # Automated test suite (26 unit & integration tests)
├── README.md                   # Project documentation & quickstart
└── requirements.txt            # Python dependencies
```

---

## 4. Quickstart & Reproducibility Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/schellDav/deep_clinic.git
cd deep_clinic

# Option A: Automated setup script
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate

# Option B: Conda environment setup
conda env create -f environment.yml
conda activate deep_clinic_rag

# Option C: Standard pip install
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Pre-cache Models for Offline HPC Worker Nodes
```bash
python -m src.preload_models
```

### Step 3: Run Retrieval, Graph Construction & Re-Ranking (Stages 1–3)
```bash
# 1k Labeled Corpus (Fast baseline):
python -m src.build_graph --config config/default_config.yaml --max_passages 1000
python -m src.retrieve_and_rerank --config config/default_config.yaml --max_passages 1000

# Full 62k Expanded Corpus (with 61,249 Distractors):
python -m src.build_graph --config config/default_config.yaml --max_passages 62249
python -m src.retrieve_and_rerank --config config/default_config.yaml
```

### Step 4: Run End-to-End LLM Generation & RAGAS Evaluation (Stage 4)
```bash
# Interactive / Direct Python execution:
python -m src.run_generation_and_eval --config config/default_config.yaml

# On Slurm HPC Cluster (KISSKI A100 GPU):
sbatch scripts/slurm/03_eval_ragas.sh

# Or Full 62k Pipeline Job (Chained Stages 1–4):
sbatch scripts/slurm/04_full_corpus_ablation.sh

# Or Master Orchestrator (Submits all jobs with automatic dependencies):
bash scripts/slurm/submit_all.sh
```

### Step 5: Generate Publication Figures
```bash
python -m src.visualize_results
```
All figures (`fig1`–`fig5`) will be saved to `outputs/figures/` in both 300 DPI PNG and vector PDF formats.

### Step 6: Run Automated Test Suite
```bash
pytest -v
# Or
python -m tests.test_all_phases
```
