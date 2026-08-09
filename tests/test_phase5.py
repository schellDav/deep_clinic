"""
================================================================================
Phase 5 / Stage 4 Test Suite (LLM Generation & RAGAS Analysis Setup)
================================================================================

This test script validates Stage 4 prompt formatting, generator output structures,
RAGAS evaluation datasets, and error traceback log dumping.

Test Coverage:
    1. Clinical prompt template formatting (MedicalQAGenerator.format_prompt)
    2. RAGAS evaluation metrics & fallback calculation (evaluate_ragas_metrics)
    3. Error log traceback dumping mechanism (dump_error_log)
"""

import os
import json
import shutil
import unittest

from src.generator import MedicalQAGenerator
from src.ragas_eval import evaluate_ragas_metrics
from src.run_generation_and_eval import dump_error_log


class TestPhase5Implementation(unittest.TestCase):

    def setUp(self):
        self.test_log_dir = "tests/scratch_logs"
        self.test_output_dir = "tests/scratch_outputs"
        os.makedirs(self.test_log_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_01_prompt_formatting(self):
        """Test clinical prompt construction logic."""
        passages = [
            {"doc_id": "101", "text": "Aspirin reduces mortality in acute coronary syndrome."},
            {"doc_id": "102", "text": "Dual antiplatelet therapy shows superior outcomes."}
        ]
        question = "Should aspirin be administered in acute coronary syndrome?"
        
        prompt = MedicalQAGenerator.format_prompt(question, passages)
        
        self.assertIn("Context:", prompt)
        self.assertIn("[1] Aspirin reduces mortality", prompt)
        self.assertIn("[2] Dual antiplatelet therapy", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("answer the question as Yes, No, or Maybe", prompt)

    def test_02_ragas_evaluation_metrics(self):
        """Test RAGAS evaluation dataset building and score calculation."""
        sample_dataset = {
            "question": ["Is aspirin effective for ACS?"],
            "contexts": [["Aspirin reduces mortality in acute coronary syndrome."]],
            "answer": ["Yes, aspirin reduces mortality in acute coronary syndrome."],
            "ground_truth": ["yes"]
        }
        
        scores = evaluate_ragas_metrics(sample_dataset, judge_model_name="openai/gpt-oss-120b")
        
        self.assertIn("faithfulness", scores)
        self.assertIn("answer_relevance", scores)
        self.assertGreaterEqual(scores["faithfulness"], 0.0)
        self.assertLessEqual(scores["faithfulness"], 1.0)
        self.assertGreaterEqual(scores["answer_relevance"], 0.0)
        self.assertLessEqual(scores["answer_relevance"], 1.0)

    def test_03_error_dump_logging(self):
        """Test error traceback log dumping to logs/stage4_error.log and outputs/stage4_error.json."""
        try:
            # Trigger intentional zero division test error
            _ = 1 / 0
        except Exception as test_err:
            dump_error_log(test_err, log_dir=self.test_log_dir, output_dir=self.test_output_dir)

        log_file = os.path.join(self.test_log_dir, "stage4_error.log")
        json_file = os.path.join(self.test_output_dir, "stage4_error.json")

        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(os.path.exists(json_file))

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["error_type"], "ZeroDivisionError")
        self.assertIn("division by zero", data["error_message"])
        self.assertIn("ZeroDivisionError: division by zero", data["traceback"])


if __name__ == "__main__":
    unittest.main()
