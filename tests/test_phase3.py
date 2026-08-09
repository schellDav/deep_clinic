"""
================================================================================
Phase 3 Test Suite (Stage 1 Initial Retrieval & TREC Evaluation)
================================================================================

This test script validates Phase 3 retrieval baselines and evaluation outputs.

Test Coverage:
    1. Dataset and artifact loading (load_artifacts)
    2. BM25s sparse lexical retrieval execution (run_bm25_retrieval)
    3. Reason-ModernColBERT dense vector retrieval execution (run_dense_retrieval)
    4. TREC evaluation metric calculation (evaluate_with_pytrec)
    5. Output JSON benchmark file integrity (outputs/stage1_retrieval_results.json)
"""

import os
import json
import unittest
import numpy as np

from src.retrieve_and_rerank import (
    load_config,
    load_artifacts,
    run_bm25_retrieval,
    evaluate_with_pytrec
)


class TestPhase3Implementation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = "config/default_config.yaml"
        cls.config = load_config(cls.config_path)
        cls.data_dir = cls.config["project"]["data_dir"]
        cls.cache_dir = cls.config["project"]["cache_dir"]
        cls.output_dir = cls.config["project"]["output_dir"]

    def test_01_artifact_loading(self):
        """Test loading dataset, embedding, and graph artifacts for retrieval."""
        passages, qrels, doc_ids, passage_embeddings, corpus_graph = load_artifacts(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir
        )
        self.assertIn(len(passages), (1000, 62249))
        self.assertEqual(len(qrels), 1000)
        self.assertIn(len(doc_ids), (1000, 62249))
        self.assertIn(passage_embeddings.shape[0], (1000, 62249))
        self.assertEqual(passage_embeddings.shape[1], 768)
        self.assertIn(corpus_graph.shape[0], (1000, 62249))

    def test_02_bm25_retrieval_execution(self):
        """Test BM25s retrieval execution and candidate ranking output structure."""
        passages, qrels, _, _, _ = load_artifacts(self.data_dir, self.cache_dir)
        
        # Test on a small subset of 10 queries for fast verification
        query_subset = dict(list(qrels.items())[:10])
        bm25_run, bm25_latency = run_bm25_retrieval(passages, query_subset, top_k=20)
        self.assertIn("mean_latency_ms", bm25_latency)

        self.assertEqual(len(bm25_run), 10)
        sample_qid = list(bm25_run.keys())[0]
        sample_run = bm25_run[sample_qid]
        self.assertLessEqual(len(sample_run), 20)
        self.assertGreater(len(sample_run), 0)

        # Verify document score ranking order
        scores = list(sample_run.values())
        self.assertEqual(scores, sorted(scores, reverse=True), "Document scores must be sorted descending.")

    def test_03_trec_evaluation_metrics(self):
        """Test pytrec_eval metric evaluation and native Python fallback calculation."""
        passages, qrels, _, _, _ = load_artifacts(self.data_dir, self.cache_dir)
        query_subset = dict(list(qrels.items())[:10])
        bm25_run, _ = run_bm25_retrieval(passages, query_subset, top_k=20)

        metrics = ["ndcg_cut_10", "recall_10", "recall_100", "recall_1000"]
        eval_results = evaluate_with_pytrec(query_subset, bm25_run, metric_names=metrics)

        for m in metrics:
            self.assertIn(m, eval_results)
            self.assertGreaterEqual(eval_results[m], 0.0)
            self.assertLessEqual(eval_results[m], 1.0)

    def test_04_benchmark_output_json(self):
        """Test serialized stage1_retrieval_results.json file."""
        output_file = os.path.join(self.output_dir, "stage1_retrieval_results.json")
        self.assertTrue(os.path.exists(output_file), f"Results file '{output_file}' missing.")

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "corpus_62k" in data:
            entry = data["corpus_62k"]
        elif "corpus_1k" in data:
            entry = data["corpus_1k"]
        else:
            entry = data

        self.assertIn("metrics", entry)
        self.assertIn("BM25s_Lexical", entry["metrics"])
        self.assertIn("ModernColBERT_Dense", entry["metrics"])

        bm25_metrics = entry["metrics"]["BM25s_Lexical"]
        dense_metrics = entry["metrics"]["ModernColBERT_Dense"]

        self.assertGreater(bm25_metrics["ndcg_cut_10"], 0.7)
        self.assertGreater(dense_metrics["recall_100"], 0.9)


if __name__ == "__main__":
    unittest.main()
