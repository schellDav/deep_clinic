"""
================================================================================
Model Pre-Caching Script for HPC Clusters (Offline Worker Node Preparation)
Pre-downloads Cross-Encoder, Dense Retriever, and LLM Generator models on the login node.
================================================================================
"""

import sys
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import yaml


def main():
    print("=== Deep Clinic HPC Model Pre-Caching Utility ===")
    
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dense_model_name = config["retrieval"]["dense"]["model_name"]
    cross_encoder_name = config["cross_encoder_reranking"]["model_name"]
    generator_name = config["generation"]["model_name"]
    judge_name = config["evaluation"]["ragas"]["judge_model_name"]

    print(f"\n[1/4] Pre-downloading Dense Retriever model '{dense_model_name}'...")
    try:
        AutoTokenizer.from_pretrained(dense_model_name, trust_remote_code=True)
        snapshot_download(repo_id=dense_model_name)
        print(f"[+] Successfully cached Dense Retriever '{dense_model_name}'.")
    except Exception as e:
        print(f"[!] Error pre-downloading Dense Retriever: {e}", file=sys.stderr)

    print(f"\n[2/4] Pre-downloading Cross-Encoder model '{cross_encoder_name}'...")
    try:
        snapshot_download(repo_id=cross_encoder_name)
        print(f"[+] Successfully cached Cross-Encoder '{cross_encoder_name}'.")
    except Exception as e:
        print(f"[!] Error pre-downloading Cross-Encoder: {e}", file=sys.stderr)

    print(f"\n[3/4] Pre-downloading Generator LLM '{generator_name}'...")
    try:
        AutoTokenizer.from_pretrained(generator_name)
        snapshot_download(repo_id=generator_name)
        print(f"[+] Successfully cached Generator LLM '{generator_name}'.")
    except Exception as e:
        print(f"[!] Error pre-downloading Generator LLM: {e}", file=sys.stderr)

    print(f"\n[4/4] Pre-downloading RAGAS Judge Model '{judge_name}'...")
    try:
        AutoTokenizer.from_pretrained(judge_name)
        snapshot_download(repo_id=judge_name)
        print(f"[+] Successfully cached RAGAS Judge Model '{judge_name}'.")
    except Exception as e:
        print(f"[!] Info: RAGAS Judge pre-download notice: {e}", file=sys.stderr)

    print("\n=== Model Pre-Caching Complete ===")


if __name__ == "__main__":
    main()
