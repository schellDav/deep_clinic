"""
Corpus Ingestion & Corpus Graph Construction (Phase 2).
Ingests PubMedQA (pqa_labeled), computes BM25s lexical index and dense passage embeddings,
constructs a hybrid similarity adjacency matrix, and saves serialized graph artifacts.
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from datasets import load_dataset

import bm25s
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build Corpus Graph for RAG & GAR Benchmarking.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML file.")
    parser.add_argument("--max_passages", type=int, default=None, help="Limit number of passages for debugging.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pubmedqa_corpus(subset="pqa_labeled", max_passages=None):
    print(f"[+] Loading PubMedQA dataset (subset: {subset})...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    local_json_path = os.path.join(data_dir, "ori_pqal.json")

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
        with open(local_json_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Cached PubMedQA dataset to '{local_json_path}'.")

    passages = []
    qrels = {}

    for i, (pubid, item) in enumerate(raw_data.items()):
        if max_passages is not None and i >= max_passages:
            break

        doc_id = str(pubid)
        question = item.get("QUESTION", "").strip()

        context_list = item.get("CONTEXTS", [])
        if isinstance(context_list, list):
            abstract_text = " ".join(context_list).strip()
        else:
            abstract_text = str(context_list).strip()

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

        qrels[doc_id] = {
            "question": question,
            "relevant_doc_ids": [doc_id],
            "long_answer": long_answer,
            "final_decision": final_decision
        }

    print(f"[+] Successfully loaded {len(passages)} passages into memory.")
    return passages, qrels


def compute_dense_embeddings(passages, model_name="lightonai/Reason-ModernColBERT", batch_size=32):
    print(f"[+] Generating dense embeddings using model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Computing on device: {device.upper()}")

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

            # Check if model has pooler_output or sentence_embedding, else use mean pooling
            if hasattr(outputs, "sentence_embedding"):
                batch_emb = outputs.sentence_embedding
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_emb = outputs.pooler_output
            else:
                # Mean pooling
                input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_emb = sum_embeddings / sum_mask

            # L2 normalize embeddings
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings_list.append(batch_emb.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    print(f"[+] Computed dense embeddings matrix shape: {embeddings.shape}")
    return embeddings


def build_bm25_similarity(passages, knn_neighbors=10):
    print(f"[+] Building BM25s lexical index...")
    texts = [p["text"] for p in passages]
    tokens = bm25s.tokenize(texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokens)

    # Retrieve top-(k+1) to account for self-match
    results, scores = retriever.retrieve(tokens, k=min(knn_neighbors + 1, len(texts)))
    
    num_nodes = len(passages)
    bm25_adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)

    # Normalize scores per query row
    for i in range(num_nodes):
        row_scores = scores[i]
        max_score = row_scores.max() if row_scores.max() > 0 else 1.0
        for target_idx, score in zip(results[i], row_scores):
            target_idx = int(target_idx)
            if target_idx != i and score > 0:
                norm_score = float(score / max_score)
                bm25_adj[i, target_idx] = max(bm25_adj[i, target_idx], norm_score)

    return bm25_adj.tocsr()


def build_hybrid_corpus_graph(passages, embeddings, knn_neighbors=10, similarity_threshold=0.65):
    print(f"[+] Building Hybrid Similarity Corpus Graph (k-NN={knn_neighbors}, threshold={similarity_threshold})...")
    num_nodes = len(passages)

    # 1. Lexical BM25s Graph
    bm25_sparse = build_bm25_similarity(passages, knn_neighbors=knn_neighbors)

    # 2. Dense Cosine Similarity Graph
    # Embeddings are L2 normalized, so dot product = cosine similarity
    dense_sim = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(dense_sim, 0.0)

    dense_adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        top_k_indices = np.argsort(dense_sim[i])[-knn_neighbors:]
        for j in top_k_indices:
            sim_val = float(dense_sim[i, j])
            if sim_val >= similarity_threshold:
                dense_adj[i, j] = sim_val

    dense_sparse = dense_adj.tocsr()

    # 3. Combine Sparse Matrices (Hybrid = Mean of normalized lexical and dense scores)
    hybrid_sparse = 0.5 * (bm25_sparse + dense_sparse)

    # Make graph symmetric for bidirectional multi-hop GAR expansion
    symmetric_graph = hybrid_sparse.maximum(hybrid_sparse.T)

    num_edges = symmetric_graph.nnz
    print(f"[+] Hybrid Corpus Graph constructed: {num_nodes} nodes, {num_edges} similarity edges.")
    return symmetric_graph.tocsr()


def main():
    args = parse_args()
    config = load_config(args.config)

    data_dir = config["project"]["data_dir"]
    cache_dir = config["project"]["cache_dir"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    subset = config["dataset"]["subset"]
    max_passages = args.max_passages or config["dataset"]["max_passages"]

    # 1. Load Dataset
    passages, qrels = load_pubmedqa_corpus(subset=subset, max_passages=max_passages)

    # 2. Compute Dense Embeddings
    dense_model = config["retrieval"]["dense"]["model_name"]
    batch_size = config["retrieval"]["dense"]["batch_size"]
    embeddings = compute_dense_embeddings(passages, model_name=dense_model, batch_size=batch_size)

    # 3. Build Hybrid Corpus Graph
    knn_neighbors = config["graph_adaptive_reranking"]["knn_neighbors"]
    similarity_threshold = config["graph_adaptive_reranking"]["similarity_threshold"]
    corpus_graph = build_hybrid_corpus_graph(
        passages,
        embeddings,
        knn_neighbors=knn_neighbors,
        similarity_threshold=similarity_threshold
    )

    # 4. Serialize Artifacts
    graph_path = os.path.join(data_dir, "corpus_graph.npz")
    doc_ids_path = os.path.join(data_dir, "doc_ids.json")
    passages_path = os.path.join(data_dir, "passages.json")
    qrels_path = os.path.join(data_dir, "qrels.json")
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")

    print(f"[+] Saving Corpus Graph artifacts to '{data_dir}' and '{cache_dir}'...")
    sp.save_npz(graph_path, corpus_graph)

    doc_ids = [p["doc_id"] for p in passages]
    with open(doc_ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=2)

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)

    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)

    np.save(embeddings_path, embeddings)

    print("=== Phase 2 Completed Successfully ===")
    print(f"Artifacts generated:")
    print(f"  - Graph Matrix:    {graph_path} (shape: {corpus_graph.shape}, nnz: {corpus_graph.nnz})")
    print(f"  - Doc IDs Map:     {doc_ids_path}")
    print(f"  - Passage Corpus:  {passages_path}")
    print(f"  - Qrels File:      {qrels_path}")
    print(f"  - Dense Vectors:   {embeddings_path}")


if __name__ == "__main__":
    main()
