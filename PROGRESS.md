Project Progress Tracker

Project: Advanced RAG Benchmarking: Standard, Cross-Encoder, and GAR
Dataset: PubMedQA (pqa_labeled & pqa_unlabeled full corpus)

================================================================================
STATUS OVERVIEW
================================================================================
Phase 1: Environment, Slurm & Config Setup    [COMPLETED]
Phase 2: Corpus Ingestion & Graph Building     [COMPLETED]
Phase 3: Initial Retrieval Baselines           [COMPLETED]
Phase 4: Static Re-Ranking & GAR Integration   [IN PROGRESS]
Phase 5: Quantitative IR Evaluation            [PENDING]
Phase 6: End-to-End LLM & RAGAS Analysis       [PENDING]
Phase 7: Visualization & Report Synthesis      [PENDING]

================================================================================
COMPLETED WORK
================================================================================

Phase 1: Environment, Infrastructure & Configuration
- Created requirements.txt and environment.yml for local/cluster environments.
- Configured PyTorch Blackwell sm_120 CUDA 12.8 nightly support for RTX 50-series GPUs.
- Created config/default_config.yaml with model and pipeline settings:
  * Dense retriever: lightonai/Reason-ModernColBERT
  * Generator model: Qwen/Qwen3-30B-A3B-Instruct-2507
  * RAGAS judge: openai/gpt-oss-120b
  * GAR expansion: seed N=20, 2-hop depth, alpha=0.5 decay, pool size K=100
- Created Slurm batch scripts in scripts/slurm/ with exact KISSKI cluster headers:
  * account: kisski-arbscg
  * partition: kisski
  * gpus: A100:1
  * constraint: 80gb_vram

Phase 2: Corpus Ingestion & Corpus Graph Construction
- Created src/build_graph.py module supporting both 1k labeled and 62k full corpus.
- Downloaded and parsed PubMedQA dataset (1,000 expert QA abstracts + 61,249 unlabeled distractor abstracts).
- Computed dense passage embeddings (Reason-ModernColBERT, shape 62249x768).
- Built BM25s lexical index and hybrid k-NN similarity adjacency matrix:
  * 1k Corpus Graph: 1,000 nodes, 25,556 edges
  * 62k Corpus Graph: 62,249 nodes, 1,645,128 edges
- Generated serialized artifacts in data/ and cache/.

Phase 3: Stage 1 — Initial Retrieval Baselines (BM25s & Dense)
- Created src/retrieve_and_rerank.py module for Stage 1 initial retrieval baselines.
- Executed BM25s sparse lexical retrieval and Reason-ModernColBERT dense vector similarity search across both 1k and 62k corpus scales.
- Evaluated metrics using pytrec_eval and native Python fallback.
- Saved benchmark metrics side-by-side in outputs/:
  * outputs/stage1_retrieval_results_1k.json
  * outputs/stage1_retrieval_results_62k.json
  * outputs/stage1_retrieval_results.json (Master output)
- Created automated test suite (tests/test_phase2.py, tests/test_phase3.py, tests/test_all_phases.py).

================================================================================
STAGE 1 RETRIEVAL BENCHMARK SUMMARY TABLE
================================================================================
Corpus Scale | Size   | Retriever Method           | nDCG@10 | Recall@10 | Recall@100 | Recall@1000
-----------------------------------------------------------------------------------------------------
1k Labeled   | 1,000  | BM25s Lexical              | 0.9687  | 0.9860    | 0.9930     | 0.9930
1k Labeled   | 1,000  | Reason-ModernColBERT Dense | 0.9685  | 0.9940    | 1.0000     | 1.0000
62k Full     | 62,249 | BM25s Lexical              | 0.8387  | 0.9164    | 0.9670     | 0.9670
62k Full     | 62,249 | Reason-ModernColBERT Dense | 0.7865  | 0.9058    | 0.9775     | 0.9775
================================================================================================-----

================================================================================
CURRENT STEP
================================================================================

Phase 4: Stage 2 & 3 — Static Re-Ranking & GAR Integration
- Extend src/retrieve_and_rerank.py to add Static Re-Ranking (Cross-Encoder).
- Implement Graph-Adaptive Re-Ranking (GAR) candidate pool expansion via multi-hop traversal on corpus_graph.npz.
- Compare candidate precision recovery across Stage 1, Stage 2, and Stage 3 GAR.
