"""
================================================================================
Deep Clinic Biomedical RAG & Graph-Adaptive Re-Ranking (GAR) Package
================================================================================

This package contains core modules for evaluating biomedical Question Answering
using standard RAG, Cross-Encoder re-ranking, and Graph-Adaptive Re-ranking (GAR).

Modules:
    - build_graph: Ingests PubMedQA, computes dense/lexical representations, and builds Corpus Graph.
    - retrieve_and_rerank: Implements Stage 1 retrieval, Stage 2 cross-encoder, and Stage 3 GAR candidate expansion.
    - evaluate_ragas: Executes LLM generation (Qwen3-30B) and RAGAS judging (gpt-oss-120b).
"""

__version__ = "0.1.0"
__authors__ = ["Alexey Wratschinski", "David Schell"]
