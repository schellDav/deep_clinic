"""
================================================================================
Re-Rankers Module (Static Cross-Encoder & Graph-Adaptive Re-Ranking / GAR)
================================================================================
"""

import time
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def run_cross_encoder_rerank(
    passages: List[Dict[str, Any]],
    qrels: Dict[str, Any],
    candidate_run: Dict[str, Dict[str, float]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 10,
    batch_size: int = 16
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Execute Stage 2 Static Cross-Encoder Re-Ranking over candidate passage pools and track latency.

    Args:
        passages (List[Dict[str, Any]]): Passage corpus list.
        qrels (Dict[str, Any]): Ground-truth QA dictionary containing questions.
        candidate_run (Dict[str, Dict[str, float]]): Stage 1 initial candidate run dictionary.
        model_name (str): Hugging Face Cross-Encoder model identifier.
        top_k (int): Number of top re-ranked passages to return per query.
        batch_size (int): Mini-batch size for transformer cross-attention scoring.

    Returns:
        Tuple containing re-ranked run dictionary {query_id: {doc_id: score}} and latency metrics.
    """
    print(f"[+] Executing Cross-Encoder Re-Ranking with model '{model_name}'...")
    t0 = time.perf_counter()
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            _ = torch.arange(2, device="cuda")
        except Exception:
            print("[!] CUDA sm_120 compatibility error detected. Falling back to CPU for Cross-Encoder...")
            device = "cpu"

    model = CrossEncoder(model_name, device=device)

    # Fast lookup map doc_id -> passage text
    doc_map = {str(p["doc_id"]): p["text"] for p in passages}

    reranked_run = {}
    for qid, candidate_scores in tqdm(candidate_run.items(), desc="Cross-Encoder Re-Ranking"):
        question = qrels.get(qid, {}).get("question", "").strip()
        if not question:
            continue

        cand_doc_ids = list(candidate_scores.keys())
        pairs = [(question, doc_map.get(doc_id, "")) for doc_id in cand_doc_ids]
        if not pairs:
            continue

        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Sort candidate doc_ids by Cross-Encoder score descending
        sorted_pairs = sorted(zip(cand_doc_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]

        query_rerank = {doc_id: float(score) for doc_id, score in sorted_pairs}
        reranked_run[qid] = query_rerank

    total_time_sec = time.perf_counter() - t0
    num_queries = max(len(reranked_run), 1)
    mean_latency_ms = (total_time_sec / num_queries) * 1000.0
    qps = num_queries / max(total_time_sec, 1e-6)

    latency_metrics = {
        "total_time_sec": round(total_time_sec, 4),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "throughput_qps": round(qps, 2)
    }

    print(f"[+] Cross-Encoder Re-Ranking finished for {num_queries} queries in {total_time_sec:.2f}s ({mean_latency_ms:.2f}ms/query, {qps:.1f} QPS).")
    return reranked_run, latency_metrics


def run_gar_expansion(
    seed_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Any],
    corpus_graph_matrix: sp.csr_matrix,
    doc_ids_list: List[str],
    depth: int = 2,
    alpha: float = 0.5,
    expanded_k: int = 100
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Execute Stage 3 Graph-Adaptive Re-Ranking (GAR) multi-hop candidate pool expansion and track traversal latency.

    Args:
        seed_run (Dict[str, Dict[str, float]]): Top N=20 seed candidates from Stage 1.
        qrels (Dict[str, Any]): Ground-truth QA dictionary containing questions.
        corpus_graph_matrix (sp.csr_matrix): Pre-computed hybrid similarity Corpus Graph (SciPy CSR).
        doc_ids_list (List[str]): Ordered document ID mapping matching corpus_graph_matrix rows/cols.
        depth (int): Multi-hop graph traversal depth (default: 2 hops).
        alpha (float): Score decay factor per graph hop (default: 0.5).
        expanded_k (int): Target expanded candidate pool size per query.

    Returns:
        Tuple containing expanded candidate document ID lists and graph expansion latency metrics.
    """
    print(f"[+] Executing GAR Candidate Expansion (depth={depth}, alpha={alpha}, target_k={expanded_k})...")
    t0 = time.perf_counter()
    doc_id_to_idx = {str(did): idx for idx, did in enumerate(doc_ids_list)}
    idx_to_doc_id = {idx: str(did) for idx, did in enumerate(doc_ids_list)}

    expanded_candidates = {}

    for qid, seed_scores in seed_run.items():
        question = qrels.get(qid, {}).get("question", "").strip()
        if not question:
            continue

        # Map seed candidate doc_ids to matrix node indices
        seed_indices = [doc_id_to_idx[did] for did in seed_scores.keys() if did in doc_id_to_idx]
        if not seed_indices:
            expanded_candidates[qid] = list(seed_scores.keys())
            continue

        # Initial seed score vector v0
        num_nodes = corpus_graph_matrix.shape[0]
        v_current = np.zeros(num_nodes, dtype=np.float32)
        for did, score in seed_scores.items():
            if did in doc_id_to_idx:
                v_current[doc_id_to_idx[did]] = max(score, 1e-4)

        v_accumulated = v_current.copy()
        current_weight = 1.0

        # Multi-hop matrix graph traversal: v_{t+1} = v_t * A
        for hop in range(1, depth + 1):
            current_weight *= alpha
            v_current = corpus_graph_matrix.dot(v_current)
            v_accumulated += current_weight * v_current

        # Extract top expanded_k node indices with highest accumulated graph scores
        top_k_indices = np.argsort(v_accumulated)[-expanded_k:][::-1]
        expanded_doc_ids = [idx_to_doc_id[idx] for idx in top_k_indices if v_accumulated[idx] > 0]

        expanded_candidates[qid] = expanded_doc_ids

    total_time_sec = time.perf_counter() - t0
    num_queries = max(len(expanded_candidates), 1)
    mean_latency_ms = (total_time_sec / num_queries) * 1000.0
    qps = num_queries / max(total_time_sec, 1e-6)

    latency_metrics = {
        "graph_expansion_time_sec": round(total_time_sec, 4),
        "graph_expansion_mean_ms": round(mean_latency_ms, 3),
        "graph_expansion_qps": round(qps, 2)
    }

    print(f"[+] GAR Expansion finished for {num_queries} queries in {total_time_sec:.4f}s ({mean_latency_ms:.3f}ms/query, {qps:.1f} QPS).")
    return expanded_candidates, latency_metrics
