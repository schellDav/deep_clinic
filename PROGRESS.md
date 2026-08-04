Project Progress Tracker

Project: Advanced RAG Benchmarking: Standard, Cross-Encoder, and GAR
Dataset: PubMedQA (pqa_labeled)

================================================================================
STATUS OVERVIEW
================================================================================
Phase 1: Environment, Slurm & Config Setup    [COMPLETED]
Phase 2: Corpus Ingestion & Graph Building     [IN PROGRESS]
Phase 3: Initial Retrieval Baselines           [PENDING]
Phase 4: Static Re-Ranking & GAR Integration   [PENDING]
Phase 5: Quantitative IR Evaluation            [PENDING]
Phase 6: End-to-End LLM & RAGAS Analysis       [PENDING]
Phase 7: Visualization & Report Synthesis      [PENDING]

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

================================================================================
CURRENT STEP
================================================================================

Phase 2: Corpus Ingestion & Corpus Graph Construction
- Implementing src/build_graph.py
- Processing PubMedQA (pqa_labeled) into document nodes
- Building BM25s index and dense passage embeddings
- Generating similarity matrix (.npz) and ID mappings (.json)
