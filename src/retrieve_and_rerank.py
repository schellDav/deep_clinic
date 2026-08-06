"""
================================================================================
Stage 1 Retrieval & Benchmark Module (Phase 3)
================================================================================

This module implements Stage 1 sparse lexical retrieval (BM25s) and dense embedding
retrieval (LightOn AI Reason-ModernColBERT) against PubMedQA (pqa_labeled).

Main Pipeline Steps:
    1. Load pre-computed dataset and graph artifacts from data/ and cache/.
    2. Execute BM25s sparse lexical retrieval for all 1,000 query questions.
    3. Execute dense vector cosine similarity search (Reason-ModernColBERT on CUDA).
    4. Compute TREC evaluation metrics (nDCG@10, Recall@10, Recall@100, Recall@1000) using pytrec_eval.
    5. Save comparative performance metrics to outputs/stage1_retrieval_results.json.

Author: Alexey Wratschinski and David Schell
Course: Deep Learning in the Clinic
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
import torch
from tqdm import tqdm
import pytrec_eval

import bm25s
from transformers import AutoTokenizer, AutoModel


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for retrieval and re-ranking.

    Returns:
        argparse.Namespace: Parsed arguments containing configuration path.
    """
    parser = argparse.ArgumentParser(description="Stage 1 Retrieval Baselines (BM25s & ModernColBERT).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file."
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path (str): Filepath to the YAML configuration file.

    Returns:
        dict: Parsed configuration parameters as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_artifacts(data_dir: str = "data", cache_dir: str = "cache") -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], np.ndarray, sp.csr_matrix]:
    """
    Load pre-computed dataset, document IDs, embeddings, and corpus graph artifacts.

    Args:
        data_dir (str): Directory containing passage corpus, qrels, and doc IDs.
        cache_dir (str): Directory containing dense embedding matrices.

    Returns:
        Tuple containing passages, qrels, doc_ids, passage_embeddings, corpus_graph.
    """
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
        qrels = json.load(f)

    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    passage_embeddings = np.load(embeddings_path)
    corpus_graph = sp.load_npz(graph_path)

    print(f"[+] Successfully loaded {len(passages)} passages and {len(qrels)} qrels.")
    return passages, qrels, doc_ids, passage_embeddings, corpus_graph


def run_bm25_retrieval(passages: List[Dict[str, Any]], qrels: Dict[str, Any], top_k: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Execute Stage 1 BM25s sparse lexical retrieval for all queries.

    Args:
        passages (List[Dict[str, Any]]): Passage corpus list.
        qrels (Dict[str, Any]): Ground-truth QA dictionary containing questions.
        top_k (int): Number of top documents to retrieve per query.

    Returns:
        Dict[str, Dict[str, float]]: Pytrec_eval formatted run dictionary {query_id: {doc_id: score}}.
    """
    print(f"[+] Running Stage 1 BM25s Lexical Retrieval (Top-{top_k})...")
    
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

    print(f"[+] BM25s Lexical Retrieval finished for {len(run_dict)} queries.")
    return run_dict


def run_dense_retrieval(
    passages: List[Dict[str, Any]],
    qrels: Dict[str, Any],
    passage_embeddings: np.ndarray,
    model_name: str = "lightonai/Reason-ModernColBERT",
    top_k: int = 100,
    batch_size: int = 32
) -> Dict[str, Dict[str, float]]:
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
        Dict[str, Dict[str, float]]: Pytrec_eval formatted run dictionary {query_id: {doc_id: score}}.
    """
    print(f"[+] Running Stage 1 Dense Embedding Retrieval using '{model_name}' (Top-{top_k})...")
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

    print(f"[+] Dense Retrieval finished for {len(run_dict)} queries.")
    return run_dict


def evaluate_with_pytrec(
    qrels: Dict[str, Any],
    run_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate retrieval run results against ground-truth qrels using pytrec_eval or native Python fallback.

    Args:
        qrels (Dict[str, Any]): Ground-truth QA dictionary.
        run_dict (Dict[str, Dict[str, float]]): Run dictionary {query_id: {doc_id: score}}.
        metric_names (List[str], optional): List of TREC metric strings.

    Returns:
        Dict[str, float]: Mean performance scores for each metric across all queries.
    """
    if metric_names is None:
        metric_names = ["ndcg_cut_10", "recall_10", "recall_100", "recall_1000"]

    # Try C-bindings pytrec_eval if installed
    try:
        import pytrec_eval
        pytrec_qrels = {}
        for qid, data in qrels.items():
            rel_map = {}
            for rel_doc_id in data.get("relevant_doc_ids", [qid]):
                rel_map[rel_doc_id] = 1
            pytrec_qrels[qid] = rel_map

        evaluator = pytrec_eval.RelevanceEvaluator(pytrec_qrels, set(metric_names))
        results = evaluator.evaluate(run_dict)

        mean_metrics = {}
        for metric in metric_names:
            scores = [query_results[metric] for query_results in results.values()]
            mean_metrics[metric] = float(np.mean(scores))
        return mean_metrics

    except (ImportError, Exception) as e:
        print(f"[!] Info: Using native Python TREC evaluator fallback ({e}).")

    # Native Python TREC Metric Computation (100% exact match for binary qrels)
    query_metrics = {m: [] for m in metric_names}

    for qid, data in qrels.items():
        rel_set = set(data.get("relevant_doc_ids", [qid]))
        query_run = run_dict.get(qid, {})
        
        # Sort retrieved documents by score descending
        sorted_docs = sorted(query_run.keys(), key=lambda d: query_run[d], reverse=True)

        for metric in metric_names:
            if metric.startswith("recall_"):
                k = int(metric.split("_")[1])
                retrieved_top_k = set(sorted_docs[:k])
                num_rel_retrieved = len(retrieved_top_k.intersection(rel_set))
                recall_val = num_rel_retrieved / max(len(rel_set), 1)
                query_metrics[metric].append(recall_val)

            elif metric.startswith("ndcg_cut_"):
                k = int(metric.split("_")[2])
                top_k_docs = sorted_docs[:k]
                dcg = 0.0
                for rank_idx, doc_id in enumerate(top_k_docs):
                    if doc_id in rel_set:
                        dcg += 1.0 / np.log2(rank_idx + 2)  # rank is 1-indexed

                # Ideal DCG for binary relevance
                idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel_set), k)))
                ndcg_val = (dcg / idcg) if idcg > 0 else 0.0
                query_metrics[metric].append(ndcg_val)

    mean_metrics = {m: float(np.mean(query_metrics[m])) for m in metric_names}
    return mean_metrics


def main() -> None:
    """
    Main execution pipeline for Phase 3 Stage 1 retrieval baselines and evaluation.
    """
    args = parse_args()
    config = load_config(args.config)

    data_dir = config["project"]["data_dir"]
    cache_dir = config["project"]["cache_dir"]
    output_dir = config["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load artifacts
    passages, qrels, doc_ids, passage_embeddings, corpus_graph = load_artifacts(data_dir, cache_dir)

    # 2. Run Stage 1 Lexical Retrieval (BM25s)
    bm25_top_k = config["retrieval"]["lexical"]["top_k"]
    bm25_run = run_bm25_retrieval(passages, qrels, top_k=bm25_top_k)

    # 3. Run Stage 1 Dense Retrieval (Reason-ModernColBERT)
    dense_model = config["retrieval"]["dense"]["model_name"]
    dense_top_k = config["retrieval"]["dense"]["top_k"]
    dense_batch_size = config["retrieval"]["dense"]["batch_size"]
    dense_run = run_dense_retrieval(
        passages,
        qrels,
        passage_embeddings,
        model_name=dense_model,
        top_k=dense_top_k,
        batch_size=dense_batch_size
    )

    # 4. Evaluate Stage 1 baselines using pytrec_eval
    metrics_to_eval = config["evaluation"]["metrics"]
    bm25_eval = evaluate_with_pytrec(qrels, bm25_run, metric_names=metrics_to_eval)
    dense_eval = evaluate_with_pytrec(qrels, dense_run, metric_names=metrics_to_eval)

    # 5. Output benchmark results summary table
    print("\n================================================================================")
    print("STAGE 1 INITIAL RETRIEVAL BENCHMARK RESULTS")
    print("================================================================================")
    print(f"{'Retriever Method':<30} | {'nDCG@10':<10} | {'Recall@10':<10} | {'Recall@100':<10} | {'Recall@1000':<10}")
    print("-" * 80)
    print(f"{'BM25s Lexical':<30} | {bm25_eval.get('ndcg_cut_10', 0):<10.4f} | {bm25_eval.get('recall_10', 0):<10.4f} | {bm25_eval.get('recall_100', 0):<10.4f} | {bm25_eval.get('recall_1000', 0):<10.4f}")
    print(f"{'Reason-ModernColBERT Dense':<30} | {dense_eval.get('ndcg_cut_10', 0):<10.4f} | {dense_eval.get('recall_10', 0):<10.4f} | {dense_eval.get('recall_100', 0):<10.4f} | {dense_eval.get('recall_1000', 0):<10.4f}")
    print("================================================================================")

    # 6. Save JSON results file
    results_output_path = os.path.join(output_dir, "stage1_retrieval_results.json")
    results_data = {
        "stage": "Stage 1 - Initial Retrieval Baselines",
        "dataset": config["dataset"]["subset"],
        "num_queries": len(qrels),
        "metrics": {
            "BM25s_Lexical": bm25_eval,
            "ModernColBERT_Dense": dense_eval
        }
    }
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[+] Results successfully saved to '{results_output_path}'")


if __name__ == "__main__":
    main()
