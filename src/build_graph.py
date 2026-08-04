"""
================================================================================
Corpus Ingestion & Corpus-Graph Construction Module (Phase 2)
================================================================================

This module handles the data ingestion, text preprocessing, dense embedding computation,
lexical indexing, and hybrid k-NN graph construction for PubMedQA.

Main Pipeline Steps:
    1. Parse configuration parameters from YAML file.
    2. Load official PubMedQA dataset (pqa_labeled: 1,000 expert-annotated QA pairs).
    3. Extract passage text nodes and QA ground-truth relevance pairs.
    4. Compute dense passage embeddings using LightOn AI Reason-ModernColBERT (PyTorch/CUDA).
    5. Construct sparse BM25s lexical k-NN similarity adjacency matrix.
    6. Construct dense cosine similarity k-NN adjacency matrix.
    7. Merge lexical and dense graphs into a symmetric hybrid adjacency matrix.
    8. Persist graph and dataset artifacts to disk (SciPy CSR matrix, JSON, NumPy arrays).

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

import bm25s
from transformers import AutoTokenizer, AutoModel


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for graph construction.

    Returns:
        argparse.Namespace: Parsed arguments containing configuration path and max_passages override.
    """
    parser = argparse.ArgumentParser(
        description="Ingest PubMedQA, compute dense/lexical representations, and build Corpus Graph."
    )
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
        help="Optional maximum number of passages to process (useful for fast local debugging)."
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path (str): Filepath to the YAML configuration file.

    Returns:
        dict: Parsed configuration parameters as a nested dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pubmedqa_corpus(subset: str = "pqa_labeled", max_passages: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load official PubMedQA labeled dataset (ori_pqal.json) from local cache or GitHub.

    Extracts document IDs (PubMed IDs), question strings, full abstract text paragraphs,
    ground-truth long answers, and final decision labels (yes/no/maybe).

    Args:
        subset (str): Dataset subset name (default: 'pqa_labeled').
        max_passages (int, optional): Truncate dataset size for debugging.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            - passages: List of passage record dictionaries containing text and metadata.
            - qrels: Dictionary mapping document IDs to ground-truth QA relevance data.
    """
    print(f"[+] Loading PubMedQA dataset (subset: '{subset}')...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    local_json_path = os.path.join(data_dir, "ori_pqal.json")

    # Step 1: Check local disk cache or download official PubMedQA JSON dataset
    raw_data = None
    if os.path.exists(local_json_path):
        print(f"[+] Found cached PubMedQA file at '{local_json_path}'.")
        with open(local_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
        print(f"[+] Downloading official PubMedQA labeled data from '{url}'...")
        import urllib.request
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")
            raw_data = json.loads(content)
        # Write raw content to local cache
        with open(local_json_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Cached PubMedQA dataset locally to '{local_json_path}'.")

    passages = []
    qrels = {}

    # Step 2: Parse raw JSON records into structured passage objects
    for i, (pubid, item) in enumerate(raw_data.items()):
        if max_passages is not None and i >= max_passages:
            break

        doc_id = str(pubid)
        question = item.get("QUESTION", "").strip()

        # Combine abstract context paragraphs into a single unified passage text
        context_list = item.get("CONTEXTS", [])
        if isinstance(context_list, list):
            abstract_text = " ".join(context_list).strip()
        else:
            abstract_text = str(context_list).strip()

        # Fallback to question if abstract text is empty
        if not abstract_text:
            abstract_text = question

        long_answer = item.get("LONG_ANSWER", "").strip()
        final_decision = item.get("final_decision", "").strip()

        passage_entry = {
            "doc_id": doc_id,
            "index": i,
            "question": question,
            "text": abstract_text,
            "long_answer": long_answer,
            "final_decision": final_decision
        }
        passages.append(passage_entry)

        # Map query to ground-truth relevant document ID for evaluation metrics
        qrels[doc_id] = {
            "question": question,
            "relevant_doc_ids": [doc_id],
            "long_answer": long_answer,
            "final_decision": final_decision
        }

    print(f"[+] Successfully loaded {len(passages)} passages into memory.")
    return passages, qrels


def compute_dense_embeddings(passages: List[Dict[str, Any]], model_name: str = "lightonai/Reason-ModernColBERT", batch_size: int = 32) -> np.ndarray:
    """
    Compute dense vector embeddings for all passage texts using PyTorch and Hugging Face Transformers.

    Initializes PyTorch CUDA device if available, loads transformer model, processes passages
    in batches, applies mean/pooler extraction, and L2-normalizes feature vectors.

    Args:
        passages (List[Dict[str, Any]]): List of passage records.
        model_name (str): Hugging Face model identifier for dense embedding extraction.
        batch_size (int): Mini-batch size for GPU transformer forward passes.

    Returns:
        np.ndarray: Dense embedding matrix of shape [N_passages, feature_dim] normalized to unit length.
    """
    print(f"[+] Generating dense embeddings using transformer model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Computing on device: {device.upper()}")

    # Load model and tokenizer from Hugging Face Hub with fallback safety
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"[!] Warning: Could not load '{model_name}' ({e}). Falling back to 'sentence-transformers/all-MiniLM-L6-v2'.")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

    model.eval()
    texts = [p["text"] for p in passages]
    embeddings_list = []

    # Batch forward pass over passage texts
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Passages"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)

            # Pooling strategy selection: sentence_embedding > pooler_output > mean pooling
            if hasattr(outputs, "sentence_embedding"):
                batch_emb = outputs.sentence_embedding
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_emb = outputs.pooler_output
            else:
                input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_emb = sum_embeddings / sum_mask

            # L2 normalize embeddings for unit length vector dot product cosine similarity
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings_list.append(batch_emb.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    print(f"[+] Computed dense embeddings matrix shape: {embeddings.shape}")
    return embeddings


def build_bm25_similarity(passages: List[Dict[str, Any]], knn_neighbors: int = 10) -> sp.csr_matrix:
    """
    Construct sparse BM25s lexical similarity k-NN adjacency matrix.

    Tokenizes text, indexes terms via BM25s, queries each document against the corpus,
    extracts top-k scores, and normalizes scores into a sparse SciPy matrix.

    Args:
        passages (List[Dict[str, Any]]): List of passage records.
        knn_neighbors (int): Number of top lexical nearest neighbors to retrieve per document.

    Returns:
        sp.csr_matrix: Sparse lexical adjacency matrix of shape [N, N] with values in [0, 1].
    """
    print(f"[+] Building BM25s lexical index...")
    texts = [p["text"] for p in passages]
    tokens = bm25s.tokenize(texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokens)

    # Retrieve top (k+1) neighbors per document (accounting for self-match at rank 0)
    results, scores = retriever.retrieve(tokens, k=min(knn_neighbors + 1, len(texts)))

    num_nodes = len(passages)
    bm25_adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        row_scores = scores[i]
        max_score = row_scores.max() if row_scores.max() > 0 else 1.0
        for target_idx, score in zip(results[i], row_scores):
            target_idx = int(target_idx)
            # Exclude self-loop edges
            if target_idx != i and score > 0:
                norm_score = float(score / max_score)
                bm25_adj[i, target_idx] = max(bm25_adj[i, target_idx], norm_score)

    return bm25_adj.tocsr()


def build_hybrid_corpus_graph(
    passages: List[Dict[str, Any]],
    embeddings: np.ndarray,
    knn_neighbors: int = 10,
    similarity_threshold: float = 0.65
) -> sp.csr_matrix:
    """
    Build hybrid (Lexical BM25s + Dense Cosine) symmetric Corpus Graph.

    Combines sparse lexical k-NN edges with dense embedding cosine similarity k-NN edges,
    applies similarity threshold filtering, and enforces graph symmetry for bi-directional
    multi-hop candidate expansion during Graph-Adaptive Re-ranking (GAR).

    Args:
        passages (List[Dict[str, Any]]): List of passage records.
        embeddings (np.ndarray): L2-normalized dense embedding matrix [N, D].
        knn_neighbors (int): Number of nearest neighbors per node.
        similarity_threshold (float): Minimum edge weight threshold for dense similarity.

    Returns:
        sp.csr_matrix: Symmetric hybrid Corpus Graph adjacency matrix [N, N].
    """
    print(f"[+] Building Hybrid Similarity Corpus Graph (k-NN={knn_neighbors}, threshold={similarity_threshold})...")
    num_nodes = len(passages)

    # Step 1: Compute sparse BM25s lexical adjacency matrix
    bm25_sparse = build_bm25_similarity(passages, knn_neighbors=knn_neighbors)

    # Step 2: Compute dense cosine similarity matrix (dot product of L2-normalized vectors)
    dense_sim = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(dense_sim, 0.0)  # Zero out self-loop diagonal

    dense_adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        top_k_indices = np.argsort(dense_sim[i])[-knn_neighbors:]
        for j in top_k_indices:
            sim_val = float(dense_sim[i, j])
            if sim_val >= similarity_threshold:
                dense_adj[i, j] = sim_val

    dense_sparse = dense_adj.tocsr()

    # Step 3: Combine sparse matrices (Mean of normalized lexical and dense scores)
    hybrid_sparse = 0.5 * (bm25_sparse + dense_sparse)

    # Step 4: Symmetrize graph for bi-directional multi-hop graph traversal
    symmetric_graph = hybrid_sparse.maximum(hybrid_sparse.T)

    num_edges = symmetric_graph.nnz
    print(f"[+] Hybrid Corpus Graph constructed: {num_nodes} nodes, {num_edges} similarity edges.")
    return symmetric_graph.tocsr()


def main() -> None:
    """
    Main execution pipeline for Phase 2 graph building.
    """
    args = parse_args()
    config = load_config(args.config)

    data_dir = config["project"]["data_dir"]
    cache_dir = config["project"]["cache_dir"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    subset = config["dataset"]["subset"]
    max_passages = args.max_passages or config["dataset"]["max_passages"]

    # 1. Load PubMedQA corpus and qrels ground-truth
    passages, qrels = load_pubmedqa_corpus(subset=subset, max_passages=max_passages)

    # 2. Compute dense embeddings via PyTorch
    dense_model = config["retrieval"]["dense"]["model_name"]
    batch_size = config["retrieval"]["dense"]["batch_size"]
    embeddings = compute_dense_embeddings(passages, model_name=dense_model, batch_size=batch_size)

    # 3. Construct hybrid BM25s + Dense similarity Corpus Graph
    knn_neighbors = config["graph_adaptive_reranking"]["knn_neighbors"]
    similarity_threshold = config["graph_adaptive_reranking"]["similarity_threshold"]
    corpus_graph = build_hybrid_corpus_graph(
        passages,
        embeddings,
        knn_neighbors=knn_neighbors,
        similarity_threshold=similarity_threshold
    )

    # 4. Serialize dataset and graph artifacts to disk
    graph_path = os.path.join(data_dir, "corpus_graph.npz")
    doc_ids_path = os.path.join(data_dir, "doc_ids.json")
    passages_path = os.path.join(data_dir, "passages.json")
    qrels_path = os.path.join(data_dir, "qrels.json")
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")

    print(f"[+] Persisting Corpus Graph artifacts to '{data_dir}' and '{cache_dir}'...")
    sp.save_npz(graph_path, corpus_graph)

    doc_ids = [p["doc_id"] for p in passages]
    with open(doc_ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=2)

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)

    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)

    np.save(embeddings_path, embeddings)

    print("=== Phase 2 Graph Building Pipeline Complete ===")
    print(f"  - Graph Matrix:    {graph_path} (shape: {corpus_graph.shape}, nnz: {corpus_graph.nnz})")
    print(f"  - Doc IDs Map:     {doc_ids_path}")
    print(f"  - Passage Corpus:  {passages_path}")
    print(f"  - Qrels File:      {qrels_path}")
    print(f"  - Dense Vectors:   {embeddings_path}")


if __name__ == "__main__":
    main()
