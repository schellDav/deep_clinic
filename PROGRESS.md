Project Progress Tracker

Project: Advanced RAG Benchmarking: Standard, Cross-Encoder, and GAR
Dataset: PubMedQA (pqa_labeled & pqa_unlabeled full corpus)

================================================================================
STATUS OVERVIEW
================================================================================
Phase 1: Environment, Slurm & Config Setup    [COMPLETED]
Phase 2: Corpus Ingestion & Graph Building     [COMPLETED]
Phase 3: Initial Retrieval Baselines           [COMPLETED]
Phase 4: Static Re-Ranking & GAR Integration   [COMPLETED]
Phase 5: Quantitative IR Evaluation            [COMPLETED]
Phase 6: End-to-End LLM & RAGAS Analysis       [PENDING]
Phase 7: Visualization & Report Synthesis      [PENDING]

================================================================================
COMPLETED WORK
================================================================================

Phase 1: Environment, Infrastructure & Configuration
- Created requirements.txt and environment.yml for local/cluster environments.
- Configured PyTorch Blackwell sm_120 CUDA 12.8 nightly support for RTX 50-series GPUs.
- Created config/default_config.yaml with model and pipeline settings.
- Created Slurm batch scripts in scripts/slurm/ for KISSKI GPU cluster execution.

Phase 2: Corpus Ingestion & Corpus Graph Construction
- Created src/build_graph.py module supporting both 1k labeled and 62k full corpus.
- Downloaded and parsed PubMedQA dataset (1,000 expert QA abstracts + 61,249 unlabeled distractor abstracts).
- Computed dense passage embeddings (Reason-ModernColBERT, shape 62249x768).
- Built BM25s lexical index and hybrid k-NN similarity adjacency matrix (1,645,128 graph edges).

Phase 3 & Phase 4: Comparative IR Benchmarking (Stage 1 vs Stage 2 vs Stage 3 GAR)
- Created src/retrieve_and_rerank.py implementing Stage 1 baselines, Stage 2 Static Cross-Encoder re-ranking, and Stage 3 Graph-Adaptive Re-Ranking (GAR).
- Evaluated metrics (nDCG@10, Recall@10, Recall@100) across both 1k Labeled and 62k Full Expanded Corpus scales.
- Saved benchmark metrics side-by-side in outputs/:
  * outputs/stage1_retrieval_results_1k.json / outputs/stage1_retrieval_results_62k.json
  * outputs/stage2_rerank_results_1k.json / outputs/stage2_rerank_results_62k.json
  * outputs/stage3_gar_results_1k.json / outputs/stage3_gar_results_62k.json
- Created automated test suite (tests/test_phase2.py, tests/test_phase3.py, tests/test_phase4.py, tests/test_all_phases.py).

================================================================================
COMPLETE IR BENCHMARK SUMMARY TABLE
================================================================================
Corpus Scale | Size   | Pipeline Stage                      | nDCG@10 | Recall@10 | Recall@100
------------------------------------------------------------------------------------------------
1k Labeled   | 1,000  | Stage 1: BM25s Lexical              | 0.9687  | 0.9860    | 0.9930
1k Labeled   | 1,000  | Stage 1: ModernColBERT Dense        | 0.9685  | 0.9940    | 1.0000
1k Labeled   | 1,000  | Stage 2: Static Cross-Encoder       | 0.9862  | 0.9940    | 0.9940
1k Labeled   | 1,000  | Stage 3: Graph-Adaptive (GAR)       | 0.9874  | 0.9960    | 0.9960
------------------------------------------------------------------------------------------------
62k Full     | 62,249 | Stage 1: BM25s Lexical              | 0.8433  | 0.9290    | 0.9750
62k Full     | 62,249 | Stage 1: ModernColBERT Dense        | 0.7376  | 0.8900    | 0.9780
62k Full     | 62,249 | Stage 2: Static Cross-Encoder       | 0.9227  | 0.9640    | 0.9640
62k Full     | 62,249 | Stage 3: Graph-Adaptive (GAR)       | 0.9167  | 0.9570    | 0.9570
================================================================================----------------

================================================================================
NEXT STEP
================================================================================

Phase 6: End-to-End LLM Generation & RAGAS Analysis
- Integrate generator LLM (Qwen/Qwen3-30B-A3B-Instruct or Llama-3B-Instruct) for final medical QA synthesis.
- Evaluate Faithfulness and Answer Relevance using RAGAS judge framework.
