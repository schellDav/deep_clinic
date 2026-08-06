"""
================================================================================
Phase 4 Test Suite (Stage 2 Cross-Encoder & Stage 3 GAR Candidate Expansion)
================================================================================

This test script validates Phase 4 Cross-Encoder re-ranking and Graph-Adaptive Re-Ranking (GAR).

Test Coverage:
    1. Cross-Encoder initialization and scoring output (run_cross_encoder_rerank)
    2. GAR multi-hop matrix graph traversal & candidate expansion (run_gar_expansion)
"""

import os
import unittest
import numpy as np
import scipy.sparse as sp

from src.retrieve_and_rerank import (
    load_config,
    load_artifacts,
    run_cross_encoder_rerank,
    run_gar_expansion
)


class TestPhase4Implementation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = "config/default_config.yaml"
        cls.config = load_config(cls.config_path)
        cls.data_dir = cls.config["project"]["data_dir"]
        cls.cache_dir = cls.config["project"]["cache_dir"]

    def test_01_gar_candidate_expansion(self):
        """Test GAR multi-hop graph matrix traversal and candidate pool expansion math."""
        passages, qrels, doc_ids, _, corpus_graph = load_artifacts(self.data_dir, self.cache_dir)
        
        # Mock seed run with top 5 candidates for 3 queries
        sample_qids = list(qrels.keys())[:3]
        mock_seed_run = {}
        for qid in sample_qids:
            mock_seed_run[qid] = {doc_ids[i]: float(1.0 / (i + 1)) for i in range(5)}

        expanded = run_gar_expansion(
            seed_run=mock_seed_run,
            qrels=qrels,
            corpus_graph_matrix=corpus_graph,
            doc_ids_list=doc_ids,
            depth=2,
            alpha=0.5,
            expanded_k=20
        )

        self.assertEqual(len(expanded), 3)
        for qid in sample_qids:
            self.assertIn(qid, expanded)
            # Expanded pool must contain candidates up to expanded_k
            self.assertGreater(len(expanded[qid]), 0)
            self.assertLessEqual(len(expanded[qid]), 20)

    def test_02_cross_encoder_rescoring(self):
        """Test Cross-Encoder pair creation and candidate re-ranking output structure."""
        passages, qrels, _, _, _ = load_artifacts(self.data_dir, self.cache_dir)
        
        # Mock candidate run for 2 queries with 5 candidates each
        sample_qids = list(qrels.keys())[:2]
        mock_candidate_run = {}
        for qid in sample_qids:
            mock_candidate_run[qid] = {passages[i]["doc_id"]: float(0.8 - i * 0.1) for i in range(5)}

        reranked = run_cross_encoder_rerank(
            passages=passages,
            qrels=qrels,
            candidate_run=mock_candidate_run,
            model_name=self.config["cross_encoder_reranking"]["model_name"],
            top_k=3,
            batch_size=4
        )

        self.assertEqual(len(reranked), 2)
        sample_qid = sample_qids[0]
        self.assertLessEqual(len(reranked[sample_qid]), 3)
        
        # Verify score sorting order descending
        scores = list(reranked[sample_qid].values())
        self.assertEqual(scores, sorted(scores, reverse=True), "Cross-Encoder scores must be sorted descending.")


if __name__ == "__main__":
    unittest.main()
