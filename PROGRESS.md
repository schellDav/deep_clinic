Project Progress Tracker

Project: Advanced RAG Benchmarking: Standard, Cross-Encoder, and GAR
Dataset: PubMedQA (pqa_labeled)

================================================================================
STATUS OVERVIEW
================================================================================
Phase 1: Environment, Slurm & Config Setup    [COMPLETED]
Phase 2: Corpus Ingestion & Graph Building     [COMPLETED]
Phase 3: Initial Retrieval Baselines           [IN PROGRESS]
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

Phase 2: Corpus Ingestion & Corpus Graph Construction
- Created src/build_graph.py module for dataset loading and graph building.
- Downloaded and parsed official PubMedQA labeled corpus (1,000 expert QA abstracts).
- Computed dense passage embeddings (Reason-ModernColBERT, shape 1000x768).
- Built BM25s lexical index and hybrid k-NN similarity adjacency matrix (25,556 edges).
- Generated serialized artifacts in data/ and cache/:
  * data/corpus_graph.npz (SciPy CSR Matrix)
  * data/doc_ids.json (Document ID map)
  * data/passages.json (Parsed passage corpus)
  * data/qrels.json (Relevance ground-truth)
  * cache/embeddings.npy (Dense vectors)

================================================================================
CURRENT STEP
================================================================================

Phase 3: Stage 1 — Initial Retrieval Baselines (BM25s & Dense)
- Implement baseline retrieval pipeline in src/retrieve_and_rerank.py
- Compute Stage 1 Lexical (BM25s) and Dense (Reason-ModernColBERT) retrieval results.
- Evaluate initial retrieval baselines using pytrec_eval (nDCG@10, Recall@10, Recall@100, Recall@1000).
