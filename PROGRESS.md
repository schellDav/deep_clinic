Project Progress Tracker

Project: Advanced RAG Benchmarking: Standard, Cross-Encoder, and GAR
Dataset: PubMedQA (pqa_labeled)

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
Ablation: Full 62k Corpus Scaling Experiment   [OPTIONAL SLURM BACKGROUND JOB]

================================================================================
COMPLETED WORK
================================================================================

Phase 1: Environment, Infrastructure & Configuration
- Created requirements.txt and environment.yml for local/cluster environments.
- Created config/default_config.yaml with model and pipeline settings:
  * Dense retriever: lightonai/Reason-ModernColBERT
  * Generator model: Qwen/Qwen3-30B-A3B-Instruct-2507
  * RAGAS judge: openai/gpt-oss-120b
  * GAR expansion: seed N=20, 2-hop depth, alpha=0.5 decay, pool size K=100
- Created Slurm batch scripts in scripts/slurm/ (01_build_graph.sh, 02_retrieve_gar.sh, 03_eval_ragas.sh, submit_all.sh).
- Created scripts/setup_env.sh and README.md.

Phase 2: Corpus Ingestion & Corpus Graph Construction
- Created src/build_graph.py module for dataset loading and graph building.
- Downloaded and parsed official PubMedQA labeled corpus (1,000 expert QA abstracts).
- Computed dense passage embeddings (Reason-ModernColBERT, shape 1000x768).
- Built BM25s lexical index and hybrid k-NN similarity adjacency matrix (25,556 edges).
- Generated serialized artifacts in data/ and cache/.

Phase 3: Stage 1 — Initial Retrieval Baselines (BM25s & Dense)
- Created src/retrieve_and_rerank.py module for Stage 1 initial retrieval baselines.
- Executed BM25s sparse lexical retrieval for 1,000 queries (nDCG@10: 0.9687, Recall@10: 0.9860, Recall@100: 0.9930).
- Executed Reason-ModernColBERT dense vector similarity search (nDCG@10: 0.9685, Recall@10: 0.9940, Recall@100: 1.0000).
- Evaluated metrics using pytrec_eval and native Python fallback.
- Saved benchmark metrics to outputs/stage1_retrieval_results.json.
- Created automated test suite (tests/test_phase2.py, tests/test_phase3.py, tests/test_all_phases.py).

================================================================================
CURRENT STEP
================================================================================

Phase 4: Stage 2 & 3 — Static Re-Ranking & GAR Integration
- Extend src/retrieve_and_rerank.py to add Static Re-Ranking (Cross-Encoder / monoT5).
- Implement Graph-Adaptive Re-Ranking (GAR) candidate pool expansion via multi-hop traversal on corpus_graph.npz.
- Compare candidates and re-ranking accuracy across all 3 retrieval stages.

================================================================================
BACKGROUND / SLURM ABLATION STUDY TRACKER
================================================================================
- Optional 62k Full Corpus Scaling Job:
  * Script: scripts/slurm/04_full_corpus_ablation.sh
  * Slurm execution: sbatch scripts/slurm/04_full_corpus_ablation.sh
  * Objective: Evaluates 1,000 queries against full 62,249 expanded abstracts (pqa_labeled + pqa_unlabeled) in background on KISSKI GPU cluster.
