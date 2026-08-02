# Project Requirements & Execution Plan

**Course:** Deep Learning in the Clinic: From Algorithms to Virtual Cells  
**Project Title:** Comparative Benchmarking of Standard RAG vs. Cross-Encoder Re-Ranking & Graph-Adaptive Re-Ranking (GAR)  
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
   - **Generative Model:** Small LLM (e.g. `Llama-3B-Instruct` or `Qwen2.5-3B-Instruct`).

### C. Evaluation Framework & Metrics
1. **Retrieval Benchmark (Primary Target):**
   - **Evaluation Tool:** `pytrec_eval` library (or native TREC metric evaluator).
   - **Key Metrics:** `nDCG@10`, `Recall@10`, `Recall@100`, `Recall@1000`.
2. **End-to-End LLM Evaluation (Executed LAST):**
   - **Framework:** `RAGAS` (LLM-as-a-Judge framework).
   - **Key Metrics:** *Faithfulness* (hallucination detection) and *Answer Relevance*.

### D. Compute Infrastructure
- **Local Development:** NVIDIA RTX 5060 (16 GB VRAM).
- **Cluster Resource:** **KISSKI GPU Cluster** ([KISSKI Training Platform](https://kisski.gwdg.de/en/leistungen/2-01-01_trainingsplattform/)) for corpus graph indexing and large model inference.

---

## 2. Step-by-Step Execution Plan

### Phase 1: Environment & Compute Allocation
1. Submit GPU resource application for the **KISSKI GPU platform**.
2. Document project requirements in `requirements.txt` and `.md` project guides.

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

### Phase 6: End-to-End LLM & RAGAS Analysis (PRIORITIZED LAST)
1. Feed retrieved Top-N context passages into `Llama-3B-Instruct`.
2. Run `RAGAS` evaluation for Faithfulness and Answer Relevance.

### Phase 7: Synthesis, Visualization & Report Writing
1. Generate comparative benchmark charts (nDCG@10 vs Recall@K).
2. Finalize project presentation and final write-up.
