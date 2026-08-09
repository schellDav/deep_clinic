"""
================================================================================
Evaluation Module (TREC Metrics & pytrec_eval Evaluation)
================================================================================
"""

from typing import Dict, List, Any
import numpy as np


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
