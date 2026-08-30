"""
================================================================================
Stage 4 Generation & RAGAS Evaluation Master Orchestrator (run_generation_and_eval.py)
================================================================================

This module orchestrates End-to-End LLM Generation & RAGAS Analysis across Stage 1,
Stage 2 (Cross-Encoder), and Stage 3 (GAR) contexts. Output JSON results are saved
to 'outputs/stage4_ragas_results_1k.json' and 'outputs/stage4_ragas_results_62k.json'.

In case of failure, a complete un-truncated error traceback is automatically dumped to
'logs/stage4_error.log' and 'outputs/stage4_error.json' for debugging.
================================================================================
"""

import os
import sys
import json
import argparse
import traceback
import yaml
from typing import Dict, List, Tuple, Any

import numpy as np

from src.retrievers import run_bm25_retrieval, run_dense_retrieval
from src.rerankers import run_cross_encoder_rerank, run_gar_expansion
from src.generator import MedicalQAGenerator
from src.ragas_eval import evaluate_ragas_metrics
from src.retrieve_and_rerank import load_artifacts, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Stage 4 LLM Generation and RAGAS Analysis."""
    parser = argparse.ArgumentParser(description="Stage 4 End-to-End LLM Generation & RAGAS Evaluation.")
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


def dump_error_log(err: Exception, log_dir: str = "logs", output_dir: str = "outputs") -> None:
    """
    Dump full un-truncated error traceback to log files for easy debugging.

    Args:
        err (Exception): Caught exception object.
        log_dir (str): Log directory path.
        output_dir (str): Output directory path.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    traceback_str = traceback.format_exc()
    error_payload = {
        "error_type": type(err).__name__,
        "error_message": str(err),
        "traceback": traceback_str
    }

    log_file = os.path.join(log_dir, "stage4_error.log")
    json_error_file = os.path.join(output_dir, "stage4_error.json")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== STAGE 4 EXECUTION ERROR LOG ===\n{traceback_str}\n")

    with open(json_error_file, "w", encoding="utf-8") as f:
        json.dump(error_payload, f, indent=2)

    print(f"\n[!] STAGE 4 EXECUTION FAILED: Error traceback dumped to '{log_file}' and '{json_error_file}'", file=sys.stderr)


def main() -> None:
    """Main execution pipeline for Stage 4 LLM Generation and RAGAS Evaluation."""
    args = parse_args()
    config = load_config(args.config)

    data_dir = config["project"]["data_dir"]
    cache_dir = config["project"]["cache_dir"]
    output_dir = config["project"]["output_dir"]
    log_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    try:
        # 1. Load artifacts (with automatic building if missing)
        try:
            passages, qrels, doc_ids, passage_embeddings, corpus_graph = load_artifacts(data_dir, cache_dir)
        except FileNotFoundError as err:
            print(f"[!] Dataset artifacts missing ({err}). Auto-generating graph & embeddings...", file=sys.stderr)
            from src.build_graph import build_corpus_and_graph
            build_corpus_and_graph(config)
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

        corpus_size = len(passages)
        suffix = "62k" if corpus_size > 1000 else "1k"

        # 2. Stage 1 Dense Retrieval
        dense_model = config["retrieval"]["dense"]["model_name"]
        dense_run, _ = run_dense_retrieval(
            passages, qrels, passage_embeddings,
            model_name=dense_model, top_k=20
        )

        # 3. Stage 2 Static Cross-Encoder Re-Ranking
        ce_model_name = config["cross_encoder_reranking"]["model_name"]
        stage2_run, _ = run_cross_encoder_rerank(
            passages=passages, qrels=qrels, candidate_run=dense_run,
            model_name=ce_model_name, top_k=10
        )

        # 4. Stage 3 GAR Candidate Expansion & Re-Ranking
        gar_seed_k = config["graph_adaptive_reranking"]["seed_top_k"]
        seed_run = {qid: dict(list(scores.items())[:gar_seed_k]) for qid, scores in dense_run.items()}
        gar_pools, _ = run_gar_expansion(
            seed_run=seed_run, qrels=qrels, corpus_graph_matrix=corpus_graph,
            doc_ids_list=doc_ids, depth=2, alpha=0.5, expanded_k=100
        )
        gar_candidate_run = {qid: {did: float(1.0 / (idx + 1)) for idx, did in enumerate(c_list)} for qid, c_list in gar_pools.items()}
        stage3_run, _ = run_cross_encoder_rerank(
            passages=passages, qrels=qrels, candidate_run=gar_candidate_run,
            model_name=ce_model_name, top_k=10
        )

        # 5. Fast doc lookup
        doc_map = {str(p["doc_id"]): p for p in passages}

        # 6. Initialize Generator Model
        gen_model_name = config["generation"]["model_name"]
        generator = MedicalQAGenerator(
            model_name=gen_model_name,
            max_new_tokens=config["generation"].get("max_new_tokens", 256),
            temperature=config["generation"].get("temperature", 0.1)
        )

        # Build query items for each stage
        stage1_items = []
        stage2_items = []
        stage3_items = []

        for qid, qdata in qrels.items():
            question = qdata.get("question", "")
            if not question:
                continue

            # Stage 1 top passages
            s1_dids = list(dense_run.get(qid, {}).keys())[:5]
            s1_passages = [doc_map[did] for did in s1_dids if did in doc_map]
            stage1_items.append({"qid": qid, "question": question, "passages": s1_passages, "ground_truth": qdata.get("long_answer", "")})

            # Stage 2 top passages
            s2_dids = list(stage2_run.get(qid, {}).keys())[:5]
            s2_passages = [doc_map[did] for did in s2_dids if did in doc_map]
            stage2_items.append({"qid": qid, "question": question, "passages": s2_passages, "ground_truth": qdata.get("long_answer", "")})

            # Stage 3 top passages
            s3_dids = list(stage3_run.get(qid, {}).keys())[:5]
            s3_passages = [doc_map[did] for did in s3_dids if did in doc_map]
            stage3_items.append({"qid": qid, "question": question, "passages": s3_passages, "ground_truth": qdata.get("long_answer", "")})

        # 7. Generate Answers & Evaluate RAGAS
        print("\n[+] Stage 4: Generating responses for Stage 1 Baseline contexts...")
        s1_answers, s1_gen_latency = generator.generate_responses(stage1_items)
        s1_ragas_dataset = {
            "question": [it["question"] for it in stage1_items],
            "contexts": [[p["text"] for p in it["passages"]] for it in stage1_items],
            "answer": s1_answers,
            "ground_truth": [it["ground_truth"] for it in stage1_items]
        }
        s1_ragas_scores = evaluate_ragas_metrics(s1_ragas_dataset, judge_model_name=config["evaluation"]["ragas"]["judge_model_name"])

        print("\n[+] Stage 4: Generating responses for Stage 2 Cross-Encoder contexts...")
        s2_answers, s2_gen_latency = generator.generate_responses(stage2_items)
        s2_ragas_dataset = {
            "question": [it["question"] for it in stage2_items],
            "contexts": [[p["text"] for p in it["passages"]] for it in stage2_items],
            "answer": s2_answers,
            "ground_truth": [it["ground_truth"] for it in stage2_items]
        }
        s2_ragas_scores = evaluate_ragas_metrics(s2_ragas_dataset, judge_model_name=config["evaluation"]["ragas"]["judge_model_name"])

        print("\n[+] Stage 4: Generating responses for Stage 3 GAR contexts...")
        s3_answers, s3_gen_latency = generator.generate_responses(stage3_items)
        s3_ragas_dataset = {
            "question": [it["question"] for it in stage3_items],
            "contexts": [[p["text"] for p in it["passages"]] for it in stage3_items],
            "answer": s3_answers,
            "ground_truth": [it["ground_truth"] for it in stage3_items]
        }
        s3_ragas_scores = evaluate_ragas_metrics(s3_ragas_dataset, judge_model_name=config["evaluation"]["ragas"]["judge_model_name"])

        # 8. Construct Stage 4 Output Matrix with model-specific filename preservation
        model_tag = gen_model_name.split("/")[-1].lower().replace("_", "-")
        specific_model_file = os.path.join(output_dir, f"stage4_ragas_results_{model_tag}_{suffix}.json")
        default_stage4_file = os.path.join(output_dir, f"stage4_ragas_results_{suffix}.json")

        stage4_data = {
            "stage": "Stage 4 - End-to-End LLM Generation & RAGAS Analysis",
            "generator_model": gen_model_name,
            "judge_model": config["evaluation"]["ragas"]["judge_model_name"],
            "corpus_size": corpus_size,
            "num_queries": len(stage1_items),
            "stage1_baseline": {
                "ragas_scores": s1_ragas_scores,
                "generation_latency": s1_gen_latency
            },
            "stage2_cross_encoder": {
                "ragas_scores": s2_ragas_scores,
                "generation_latency": s2_gen_latency
            },
            "stage3_gar": {
                "ragas_scores": s3_ragas_scores,
                "generation_latency": s3_gen_latency
            }
        }

        # Save both model-specific file and default stage4 file
        with open(specific_model_file, "w", encoding="utf-8") as f:
            json.dump(stage4_data, f, indent=2)
        with open(default_stage4_file, "w", encoding="utf-8") as f:
            json.dump(stage4_data, f, indent=2)

        print("\n====================================================================================================")
        print("STAGE 4 END-TO-END GENERATION & RAGAS EVALUATION RESULTS")
        print("====================================================================================================")
        print(f"{'Pipeline Context Stage':<35} | {'Faithfulness':<15} | {'Answer Relevance':<20} | {'Latency (ms/q)':<15}")
        print("-" * 100)
        print(f"{'Stage 1 Baseline Context':<35} | {s1_ragas_scores.get('faithfulness', 0):<15.4f} | {s1_ragas_scores.get('answer_relevance', 0):<20.4f} | {s1_gen_latency['mean_latency_ms']:<15.2f}")
        print(f"{'Stage 2 Cross-Encoder Context':<35} | {s2_ragas_scores.get('faithfulness', 0):<15.4f} | {s2_ragas_scores.get('answer_relevance', 0):<20.4f} | {s2_gen_latency['mean_latency_ms']:<15.2f}")
        print(f"{'Stage 3 GAR Context':<35} | {s3_ragas_scores.get('faithfulness', 0):<15.4f} | {s3_ragas_scores.get('answer_relevance', 0):<20.4f} | {s3_gen_latency['mean_latency_ms']:<15.2f}")
        print(f"[+] Stage 4 model-specific results saved to '{specific_model_file}'")
        print(f"[+] Stage 4 results saved to '{default_stage4_file}'")

    except Exception as err:
        dump_error_log(err, log_dir=log_dir, output_dir=output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
