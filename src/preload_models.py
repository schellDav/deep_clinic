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


import os
import urllib.request
import pandas as pd

def main():
    print("=== Deep Clinic HPC Model & Dataset Pre-Caching Utility ===")
    os.makedirs("data", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Pre-cache Datasets
    print("\n[+] Checking & Pre-caching PubMedQA datasets in './data'...")
    local_json_path = os.path.join("data", "ori_pqal.json")
    if not os.path.exists(local_json_path):
        url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
        print(f"[+] Downloading official PubMedQA labeled data from '{url}'...")
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")
        with open(local_json_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Successfully cached '{local_json_path}'.")
    else:
        print(f"[+] Found cached '{local_json_path}'.")

    local_unlabeled_path = os.path.join("data", "pqa_unlabeled.parquet")
    if not os.path.exists(local_unlabeled_path):
        url_u = "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/main/pqa_unlabeled/train-00000-of-00001.parquet"
        print(f"[+] Downloading official PubMedQA unlabeled distractor data (61k abstracts)...")
        df_u = pd.read_parquet(url_u)
        df_u.to_parquet(local_unlabeled_path)
        print(f"[+] Successfully cached '{local_unlabeled_path}'.")
    else:
        print(f"[+] Found cached '{local_unlabeled_path}'.")

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
