"""
================================================================================
Model Pre-Caching Script for HPC Clusters (Offline Worker Node Preparation)
Pre-downloads Cross-Encoder, Dense Retriever, and LLM Generator models on the login node.
================================================================================
"""

import sys
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml


def main():
    print("=== Deep Clinic HPC Model Pre-Caching Utility ===")
    
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cross_encoder_name = config["cross_encoder_reranking"]["model_name"]
    generator_name = config["generation"]["model_name"]

    print(f"\n[1/2] Pre-downloading Cross-Encoder model '{cross_encoder_name}'...")
    try:
        CrossEncoder(cross_encoder_name)
        print(f"[+] Successfully cached Cross-Encoder '{cross_encoder_name}'.")
    except Exception as e:
        print(f"[!] Error pre-downloading Cross-Encoder: {e}", file=sys.stderr)

    print(f"\n[2/2] Pre-downloading Generator LLM '{generator_name}'...")
    try:
        AutoTokenizer.from_pretrained(generator_name)
        AutoModelForCausalLM.from_pretrained(generator_name)
        print(f"[+] Successfully cached Generator LLM '{generator_name}'.")
    except Exception as e:
        print(f"[!] Error pre-downloading Generator LLM: {e}", file=sys.stderr)

    print("\n=== Model Pre-Caching Complete ===")


if __name__ == "__main__":
    main()
