# Project Requirements & Execution Plan

**Course:** Deep Learning in the Clinic: From Algorithms to Virtual Cells  
**Project Title:** Comparative Evaluation of Multi-Stage Retrieval and Re-Ranking Strategies for Clinical Question Answering
**Authors:** Alexey Wratschinski and David Schell  
**Dataset:** PubMedQA (Biomedical Question Answering)

---

## 1. Project Requirements & Content Breakdown

### A. Data & Corpus Infrastructure
- **Primary Dataset:** PubMedQA (`pqa_labeled` / `pqa_unlabeled` / `pqa_artificial`).
- **Passage Corpus:** Medical abstracts extracted into passage units for unstructured knowledge indexing.
- **Queries & Relevance Judgments (Qrels):** Question-Answer pairs mapped to document IDs for IR evaluation.

### B. Core Technical Stack & Models
1. **Initial Retrieval (Lexical & Dense):**
   - **BM25s:** Sparse lexical baseline (`bm25s` Python library).
   - **Dense Retrieval / Embeddings:** `lightonai/Reason-ModernColBERT` or `qwen3-embed-0.6b` (or `TCT-ColBERT`).
2. **Re-Ranking Models:**
   - **Static Re-Ranker:** `monoT5-base` or HuggingFace Cross-Encoder models.
3. **Corpus Graph & GAR (Graph-based Adaptive Re-ranking):**
   - **Corpus Graph Construction:** Pre-computed similarity edges between passages using `BM25s` k-NN and dense embeddings (`Reason-ModernColBERT`, `qwen3-embed-0.6b`).
   - **GAR Candidate Pool Expansion:** Multi-hop graph traversal to dynamically retrieve relevant context neighbors.
4. **End-to-End LLM Generator (Executed Last):**
   - **Generative Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (or `Qwen2.5-3B-Instruct`).

### C. Evaluation Framework & Metrics
1. **Retrieval Benchmark (Primary Target):**
   - **Evaluation Tool:** `pytrec_eval` library (or native TREC metric evaluator).
   - **Key Metrics:** `nDCG@10`, `Recall@10`, `Recall@100`, `Recall@1000`.
2. **End-to-End LLM Evaluation (Executed LAST):**
   - **Framework:** `RAGAS` (LLM-as-a-Judge framework).
   - **Key Metrics:** *Faithfulness* (hallucination detection) and *Answer Relevance*.

### D. Compute Infrastructure & Workload Management
- **Local Development:** NVIDIA RTX 5060 Ti (16 GB VRAM).
- **Cluster Resource:** **KISSKI GPU Cluster** ([KISSKI Training Platform](https://kisski.gwdg.de/en/leistungen/2-01-01_trainingsplattform/)) for corpus graph indexing and large model inference.
- **Workload Manager / Job Scheduler:** **Slurm** cluster environment for high-performance computing (`sbatch` job submission, GPU resource requests `#SBATCH --gpus=...`, log redirection, job execution pipelines).

---

## 2. Step-by-Step Execution Plan

### Phase 1: Environment, Compute & Slurm Setup
1. Submit GPU resource application for the **KISSKI GPU platform**.
2. Set up Slurm workload management infrastructure:
   - Create reusable Slurm batch scripts (`sbatch`) for memory-heavy indexing, re-ranking, and LLM inference.
   - Configure reproducible cluster environment (Conda / Virtualenv / Apptainer containers).
3. Document project requirements in `requirements.txt` and `.md` project guides.

### Phase 2: Corpus Ingestion & Corpus-Graph Construction
1. Load PubMedQA abstracts and format into passage corpus.
2. Build similarity adjacency graph (Corpus Graph) using `BM25s` and dense embeddings (`Reason-ModernColBERT` / `qwen3-embed-0.6b`).

### Phase 3: Stage 1 — Initial Retrieval Baseline
1. Index passage corpus with `BM25s` and dense retrievers.
2. Setup evaluation pipeline using `pytrec_eval` / `TrecEvaluator`.
3. Compute initial retrieval baselines (`nDCG@10`, `Recall@10`, `Recall@100`, `Recall@1000`).

### Phase 4: Stage 2 & 3 — Static Re-Ranking & GAR Integration
1. **Static Re-Ranking:** Pass Top-(N+M) retrieved passages to `monoT5-base` / Cross-Encoder.
2. **Graph-Adaptive Re-Ranking (GAR):** Expand initial seed candidates via multi-hop corpus graph traversal and re-score candidates dynamically.

### Phase 5: Quantitative Retrieval Evaluation & Comparison
1. Benchmark 3 pipeline stages: Standard Retrieval vs. Static Re-Ranking vs. GAR.
2. Compare retrieval precision, recall trade-offs, and multi-hop neighbor discovery gains.
3. **Optional Background Ablation (Full 62k Corpus Scaling):** Run `scripts/slurm/04_full_corpus_ablation.sh` on KISSKI GPU cluster to measure GAR precision gain over 62,249 expanded PubMed abstracts.

### Phase 6: End-to-End LLM & RAGAS Analysis (PRIORITIZED LAST)
1. Feed retrieved Top-N context passages into `Llama-3B-Instruct`.
2. Run `RAGAS` evaluation for Faithfulness and Answer Relevance.

### Phase 7: Synthesis, Visualization & Report Writing
1. Generate comparative benchmark charts (nDCG@10 vs Recall@K, RAGAS radar, scaling ablation, Pareto frontier).
2. Finalize project presentation and final write-up with publication-grade rigor.

---

## 5. Strategic Report & Presentation Blueprint (Evaluation Core Points)

The final report (`FINAL_REPORT.md`) and slide deck (`PRESENTATION.md`) will center on these five core pillars:

### 1. Clinical Motivation & Problem Statement
* **The Clinical Bottleneck:** LLMs in medicine hallucinate when context is noisy or incomplete. Standard single-vector Dense Retrieval (e.g. ModernColBERT) performs well on small closed corpora (nDCG=0.968 on 1k) but degrades drastically when scaling to large, real-world collections with 61,000+ distractor abstracts (nDCG drops from 0.9685 down to 0.7376).
* **The Graph Hypothesis:** Medical knowledge is naturally relational. By constructing a hybrid k-NN corpus graph over PubMed abstracts, Graph-Adaptive Re-Ranking (GAR) captures multi-hop contextual pathways that standard vector search misses.

### 2. Mathematical Graph Formulation
* **Corpus Graph:** Hybrid Sparse-Dense k-NN adjacency matrix $A \in \mathbb{R}^{N \times N}$ with $1,645,128$ edges, filtered by similarity threshold $\tau = 0.65$.
* **Multi-Hop Dynamic Rescoring:**
  $$S_{GAR}^{(h)} = S_{seed} + \sum_{k=1}^h \alpha^k A^k S_{seed}$$
  where $\alpha = 0.5$ provides exponential multi-hop decay, expanding seed candidates into a rich, coherent context pool.

### 3. Dual Evaluation Framework (IR + Generative)
* **Information Retrieval (IR) Metrics:** Evaluated over 1,000 test queries via `pytrec_eval` (nDCG@10, Recall@10, Recall@100, QPS Throughput).
* **Generative RAGAS Metrics:** Local LLM-as-a-Judge pipeline using `google/gemma-4-12B-it` generator and `Qwen/Qwen3-30B-A3B-Instruct-2507` judge on NVIDIA A100 GPUs (Faithfulness & Answer Relevance).

### 4. Key Empirical Findings (1k vs. 62k Scaling Ablation)
1. **Faithfulness Amplification:** GAR improves factual faithfulness over baseline by **+3.17%** on the 1k corpus (`0.6368` $\rightarrow$ `0.6685`) and **+4.14%** on the 62k corpus (`0.6577` $\rightarrow$ `0.6991`), showing a **+30.6% stronger advantage under real-world scaling**.
2. **Distractor Recovery:** Re-ranking recovers nDCG@10 from `0.7376` (Dense on 62k) back up to `0.9167` (GAR) and `0.9227` (Cross-Encoder), regaining +18.5% precision.
3. **Relevance Consistency:** Answer Relevance remains consistently high across all stages (~64% on 1k, ~69–70% on 62k), verifying that the factual faithfulness gain is achieved without losing query focus.
4. **Computational Efficiency:** GAR on 62k achieves the fastest LLM generation time (1h 32m, 0.18 QPS) because graph-traversed contexts are concise, cohesive, and redundancy-free.

### 5. Infrastructure & Reproducibility
* High-Performance Cluster Orchestration on KISSKI (A100 80GB GPUs).
* 100% offline cluster compatibility with automated pre-caching (`src/preload_models.py`).
* Comprehensive test suite (`tests/test_all_phases.py` with 13 automated tests).
