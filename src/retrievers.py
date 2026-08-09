"""
================================================================================
Stage 1 Retrievers Module (Sparse Lexical & Dense Vector Retrieval)
================================================================================
"""

import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
import bm25s
from transformers import AutoTokenizer, AutoModel


def run_bm25_retrieval(
    passages: List[Dict[str, Any]],
    qrels: Dict[str, Any],
    top_k: int = 100
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Execute Stage 1 BM25s sparse lexical retrieval for all queries and track latency.

    Args:
        passages (List[Dict[str, Any]]): Passage corpus list.
        qrels (Dict[str, Any]): Ground-truth QA dictionary containing questions.
        top_k (int): Number of top documents to retrieve per query.

    Returns:
        Tuple containing run dictionary {query_id: {doc_id: score}} and latency metrics.
    """
    print(f"[+] Running Stage 1 BM25s Lexical Retrieval (Top-{top_k})...")
    t0 = time.perf_counter()
    
    passage_texts = [p["text"] for p in passages]
    corpus_tokens = bm25s.tokenize(passage_texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    query_ids = list(qrels.keys())
    query_texts = [qrels[qid]["question"] for qid in query_ids]
    query_tokens = bm25s.tokenize(query_texts, stopwords="en")

    results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(passages)))

    run_dict = {}
    for i, qid in enumerate(query_ids):
        query_run = {}
        for target_idx, score in zip(results[i], scores[i]):
            target_idx = int(target_idx)
            target_doc_id = passages[target_idx]["doc_id"]
            query_run[target_doc_id] = float(score)
        run_dict[qid] = query_run

    total_time_sec = time.perf_counter() - t0
    num_queries = max(len(run_dict), 1)
    mean_latency_ms = (total_time_sec / num_queries) * 1000.0
    qps = num_queries / max(total_time_sec, 1e-6)

    latency_metrics = {
        "total_time_sec": round(total_time_sec, 4),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "throughput_qps": round(qps, 2)
    }

    print(f"[+] BM25s Lexical Retrieval finished for {num_queries} queries in {total_time_sec:.2f}s ({mean_latency_ms:.2f}ms/query, {qps:.1f} QPS).")
    return run_dict, latency_metrics


def run_dense_retrieval(
    passages: List[Dict[str, Any]],
    qrels: Dict[str, Any],
    passage_embeddings: np.ndarray,
    model_name: str = "lightonai/Reason-ModernColBERT",
    top_k: int = 100,
    batch_size: int = 32
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Execute Stage 1 Dense Retrieval using PyTorch CUDA query encoding and matrix dot-product similarity.

    Args:
        passages (List[Dict[str, Any]]): Passage corpus records.
        qrels (Dict[str, Any]): Ground-truth QA dictionary.
        passage_embeddings (np.ndarray): L2-normalized dense passage vector matrix [N_passages, D].
        model_name (str): Transformer embedding model identifier.
        top_k (int): Number of top candidate documents to retrieve per query.
        batch_size (int): Mini-batch size for query encoding.

    Returns:
        Tuple containing run dictionary {query_id: {doc_id: score}} and latency metrics.
    """
    print(f"[+] Running Stage 1 Dense Embedding Retrieval using '{model_name}' (Top-{top_k})...")
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            # Test CUDA kernel execution
            test_tensor = torch.zeros(1, device="cuda") + 1
        except Exception as err:
            print(f"[!] Warning: CUDA device test failed ({err}). Falling back to CPU for local execution.")
            device = "cpu"

    print(f"[+] Encoding queries on device: {device.upper()}")

    # Load model and tokenizer (strict loading)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    query_ids = list(qrels.keys())
    query_texts = [qrels[qid]["question"] for qid in query_ids]
    query_embeddings_list = []

    # Encode queries in mini-batches
    with torch.no_grad():
        for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding Queries"):
            batch_texts = query_texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)

            if hasattr(outputs, "sentence_embedding"):
                batch_emb = outputs.sentence_embedding
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_emb = outputs.pooler_output
            else:
                input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_emb = sum_embeddings / sum_mask

            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            query_embeddings_list.append(batch_emb.cpu().numpy())

    query_embeddings = np.vstack(query_embeddings_list)

    # Calculate matrix dot-product cosine similarity: [N_queries, D] @ [D, N_passages] -> [N_queries, N_passages]
    print("[+] Calculating vector cosine similarity scores...")
    similarity_matrix = np.dot(query_embeddings, passage_embeddings.T)

    run_dict = {}
    for i, qid in enumerate(query_ids):
        scores = similarity_matrix[i]
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        
        query_run = {}
        for target_idx in top_k_indices:
            target_doc_id = passages[target_idx]["doc_id"]
            query_run[target_doc_id] = float(scores[target_idx])
        run_dict[qid] = query_run

    total_time_sec = time.perf_counter() - t0
    num_queries = max(len(run_dict), 1)
    mean_latency_ms = (total_time_sec / num_queries) * 1000.0
    qps = num_queries / max(total_time_sec, 1e-6)

    latency_metrics = {
        "total_time_sec": round(total_time_sec, 4),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "throughput_qps": round(qps, 2)
    }

    print(f"[+] Dense Retrieval finished for {num_queries} queries in {total_time_sec:.2f}s ({mean_latency_ms:.2f}ms/query, {qps:.1f} QPS).")
    return run_dict, latency_metrics
