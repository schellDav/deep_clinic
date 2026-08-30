"""
================================================================================
Stage 1-3 Retrieval & Re-Ranking Master Execution Pipeline (retrieve_and_rerank.py)
================================================================================

This module orchestrates Stage 1 (Sparse/Dense), Stage 2 (Cross-Encoder), and 
Stage 3 (GAR) retrieval and re-ranking baselines.
================================================================================
"""

import os
import sys
import json
import argparse
import yaml
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.sparse as sp

from src.retrievers import run_bm25_retrieval, run_dense_retrieval
from src.rerankers import run_cross_encoder_rerank, run_gar_expansion
from src.evaluator import evaluate_with_pytrec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval and re-ranking execution."""
    parser = argparse.ArgumentParser(description="Retrieval and Re-Ranking Benchmark Execution Pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="Optional maximum number of passages to evaluate (e.g. 1000 for 1k corpus)."
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_artifacts(data_dir: str = "data", cache_dir: str = "cache") -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], np.ndarray, sp.csr_matrix]:
    """Load dataset, qrels, document IDs, embeddings, and corpus graph artifacts."""
    print(f"[+] Loading dataset artifacts from '{data_dir}' and '{cache_dir}'...")

    passages_path = os.path.join(data_dir, "passages.json")
    qrels_path = os.path.join(data_dir, "qrels.json")
    doc_ids_path = os.path.join(data_dir, "doc_ids.json")
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")
    graph_path = os.path.join(data_dir, "corpus_graph.npz")

    for path in [passages_path, qrels_path, doc_ids_path, embeddings_path, graph_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact '{path}' missing. Please run 'python -m src.build_graph' first.")

    with open(passages_path, "r", encoding="utf-8") as f:
        passages = json.load(f)

    with open(qrels_path, "r", encoding="utf-8") as f:
        raw_qrels = json.load(f)

    # Filter qrels to exact expert QA test questions from ori_pqal.json (1,000 items)
    local_pqal_path = os.path.join(data_dir, "ori_pqal.json")
    if os.path.exists(local_pqal_path):
        with open(local_pqal_path, "r", encoding="utf-8") as f:
            pqal_dict = json.load(f)
        pqal_qids = set(str(k) for k in pqal_dict.keys())
        qrels = {qid: data for qid, data in raw_qrels.items() if str(qid) in pqal_qids}
    else:
        qrels = {qid: data for qid, data in raw_qrels.items() if data.get("question", "").strip() and "relevant_doc_ids" in data}

    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    passage_embeddings = np.load(embeddings_path)
    corpus_graph = sp.load_npz(graph_path)

    print(f"[+] Successfully loaded {len(passages)} passages and {len(qrels)} test queries.")
    return passages, qrels, doc_ids, passage_embeddings, corpus_graph


def main() -> None:
    """Main execution pipeline for retrieval baselines, re-ranking, and evaluation."""
    args = parse_args()
    config = load_config(args.config)

    data_dir = config["project"]["data_dir"]
    cache_dir = config["project"]["cache_dir"]
    output_dir = config["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load artifacts
    passages, qrels, doc_ids, passage_embeddings, corpus_graph = load_artifacts(data_dir, cache_dir)

    max_p = args.max_passages or config["dataset"].get("max_passages")
    if max_p is not None and max_p < len(passages):
        print(f"[+] Slicing dataset to top {max_p} passages for 1k corpus benchmark...")
        passages = passages[:max_p]
        doc_ids = doc_ids[:max_p]
        passage_embeddings = passage_embeddings[:max_p]
        corpus_graph = corpus_graph[:max_p, :max_p]
        valid_doc_set = set(doc_ids)
        qrels = {qid: d for qid, d in qrels.items() if qid in valid_doc_set}

    # 2. Stage 1 Lexical Retrieval (BM25s)
    bm25_top_k = config["retrieval"]["lexical"]["top_k"]
    bm25_run, bm25_latency = run_bm25_retrieval(passages, qrels, top_k=bm25_top_k)

    # 3. Stage 1 Dense Retrieval (Reason-ModernColBERT)
    dense_model = config["retrieval"]["dense"]["model_name"]
    dense_top_k = config["retrieval"]["dense"]["top_k"]
    dense_batch_size = config["retrieval"]["dense"]["batch_size"]
    dense_run, dense_latency = run_dense_retrieval(
        passages,
        qrels,
        passage_embeddings,
        model_name=dense_model,
        top_k=dense_top_k,
        batch_size=dense_batch_size
    )

    # 4. Stage 2 Static Cross-Encoder Re-Ranking
    ce_model_name = config["cross_encoder_reranking"]["model_name"]
    ce_batch_size = config["cross_encoder_reranking"]["batch_size"]
    ce_final_top_k = config["cross_encoder_reranking"]["final_top_k"]

    stage2_rerank_run, stage2_latency = run_cross_encoder_rerank(
        passages=passages,
        qrels=qrels,
        candidate_run=dense_run,
        model_name=ce_model_name,
        top_k=ce_final_top_k,
        batch_size=ce_batch_size
    )

    # 5. Stage 3 Graph-Adaptive Re-Ranking (GAR) Candidate Expansion & Re-Ranking
    gar_seed_k = config["graph_adaptive_reranking"]["seed_top_k"]
    gar_depth = config["graph_adaptive_reranking"]["max_hops"]
    gar_alpha = config["graph_adaptive_reranking"]["alpha_decay"]
    gar_pool_k = config["graph_adaptive_reranking"]["candidate_pool_size"]

    seed_run = {qid: dict(list(scores.items())[:gar_seed_k]) for qid, scores in dense_run.items()}
    gar_expanded_pools, gar_expansion_latency = run_gar_expansion(
        seed_run=seed_run,
        qrels=qrels,
        corpus_graph_matrix=corpus_graph,
        doc_ids_list=doc_ids,
        depth=gar_depth,
        alpha=gar_alpha,
        expanded_k=gar_pool_k
    )

    gar_candidate_run = {}
    for qid, cand_list in gar_expanded_pools.items():
        gar_candidate_run[qid] = {did: float(1.0 / (idx + 1)) for idx, did in enumerate(cand_list)}

    stage3_gar_run, stage3_ce_latency = run_cross_encoder_rerank(
        passages=passages,
        qrels=qrels,
        candidate_run=gar_candidate_run,
        model_name=ce_model_name,
        top_k=ce_final_top_k,
        batch_size=ce_batch_size
    )

    num_queries = max(len(stage3_gar_run), 1)
    gar_total_sec = gar_expansion_latency["graph_expansion_time_sec"] + stage3_ce_latency["total_time_sec"]
    gar_mean_ms = (gar_total_sec / num_queries) * 1000.0
    gar_qps = num_queries / max(gar_total_sec, 1e-6)

    stage3_latency = {
        "graph_expansion_time_sec": gar_expansion_latency["graph_expansion_time_sec"],
        "cross_encoder_time_sec": stage3_ce_latency["total_time_sec"],
        "total_time_sec": round(gar_total_sec, 4),
        "mean_latency_ms": round(gar_mean_ms, 3),
        "throughput_qps": round(gar_qps, 2)
    }

    # 6. Evaluate all 3 stages using TREC evaluator
    metrics_to_eval = config["evaluation"]["metrics"]
    bm25_eval = evaluate_with_pytrec(qrels, bm25_run, metric_names=metrics_to_eval)
    dense_eval = evaluate_with_pytrec(qrels, dense_run, metric_names=metrics_to_eval)
    stage2_eval = evaluate_with_pytrec(qrels, stage2_rerank_run, metric_names=metrics_to_eval)
    stage3_eval = evaluate_with_pytrec(qrels, stage3_gar_run, metric_names=metrics_to_eval)

    # 7. Summary Table
    print("\n====================================================================================================")
    print("COMPARATIVE RAG vs CROSS-ENCODER & GAR BENCHMARK RESULTS (PRECISION & LATENCY)")
    print("====================================================================================================")
    print(f"{'Pipeline Stage':<35} | {'nDCG@10':<10} | {'Recall@100':<10} | {'Latency (ms/query)':<20} | {'QPS':<10}")
    print("-" * 100)
    print(f"{'Stage 1: BM25s Lexical':<35} | {bm25_eval.get('ndcg_cut_10', 0):<10.4f} | {bm25_eval.get('recall_100', 0):<10.4f} | {bm25_latency['mean_latency_ms']:<20.2f} | {bm25_latency['throughput_qps']:<10.1f}")
    print(f"{'Stage 1: ModernColBERT Dense':<35} | {dense_eval.get('ndcg_cut_10', 0):<10.4f} | {dense_eval.get('recall_100', 0):<10.4f} | {dense_latency['mean_latency_ms']:<20.2f} | {dense_latency['throughput_qps']:<10.1f}")
    print(f"{'Stage 2: Static Cross-Encoder':<35} | {stage2_eval.get('ndcg_cut_10', 0):<10.4f} | {stage2_eval.get('recall_100', 0):<10.4f} | {stage2_latency['mean_latency_ms']:<20.2f} | {stage2_latency['throughput_qps']:<10.1f}")
    print(f"{'Stage 3: Graph-Adaptive (GAR)':<35} | {stage3_eval.get('ndcg_cut_10', 0):<10.4f} | {stage3_eval.get('recall_100', 0):<10.4f} | {stage3_latency['mean_latency_ms']:<20.2f} | {stage3_latency['throughput_qps']:<10.1f}")
    print("====================================================================================================")

    # 8. Save Output Files
    corpus_size = len(passages)
    suffix = "62k" if corpus_size > 1000 else "1k"
    
    specific_stage1_file = os.path.join(output_dir, f"stage1_retrieval_results_{suffix}.json")
    stage2_file = os.path.join(output_dir, f"stage2_rerank_results_{suffix}.json")
    stage3_file = os.path.join(output_dir, f"stage3_gar_results_{suffix}.json")

    stage1_data = {
        "stage": "Stage 1 - Initial Retrieval Baselines",
        "dataset": config["dataset"]["subset"],
        "num_queries": len(qrels),
        "corpus_size": corpus_size,
        "metrics": {"BM25s_Lexical": bm25_eval, "ModernColBERT_Dense": dense_eval},
        "latency": {"BM25s_Lexical": bm25_latency, "ModernColBERT_Dense": dense_latency}
    }
    stage2_data = {
        "stage": "Stage 2 - Static Cross-Encoder Re-Ranking",
        "model": ce_model_name,
        "corpus_size": corpus_size,
        "metrics": stage2_eval,
        "latency": stage2_latency
    }
    stage3_data = {
        "stage": "Stage 3 - Graph-Adaptive Re-Ranking (GAR)",
        "model": ce_model_name,
        "corpus_size": corpus_size,
        "metrics": stage3_eval,
        "latency": stage3_latency
    }

    with open(specific_stage1_file, "w", encoding="utf-8") as f:
        json.dump(stage1_data, f, indent=2)
    with open(stage2_file, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, indent=2)
    with open(stage3_file, "w", encoding="utf-8") as f:
        json.dump(stage3_data, f, indent=2)

    print(f"\n[+] Stage 1 results updated at '{specific_stage1_file}'")
    print(f"[+] Stage 2 results saved to '{stage2_file}'")
    print(f"[+] Stage 3 GAR results saved to '{stage3_file}'")


if __name__ == "__main__":
    main()
