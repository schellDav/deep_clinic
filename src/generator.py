"""
================================================================================
Stage 4 Generator Module (Medical QA Prompt Formatting & LLM Inference)
================================================================================
"""

import time
import sys
import traceback
from typing import Dict, List, Tuple, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class MedicalQAGenerator:
    """
    Medical QA response generator leveraging Hugging Face transformer models.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.1):
        """
        Initialize generator model and tokenizer.

        Args:
            model_name (str): Hugging Face model identifier (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507 or Qwen2.5-3B-Instruct).
            max_new_tokens (int): Maximum new response tokens to generate.
            temperature (float): Sampling temperature for generation.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[+] Initializing MedicalQAGenerator with model '{model_name}' on device '{self.device.upper()}'...")

        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load tokenizer and causal language model with appropriate precision."""
        if self.model is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda":
                self.model = self.model.to("cpu")

            self.model.eval()
            print(f"[+] Successfully loaded generator model '{self.model_name}'.")

        except Exception as err:
            print(f"[!] Exception during generator model loading ({err}).", file=sys.stderr)
            raise err

    @staticmethod
    def format_prompt(question: str, passages: List[Dict[str, Any]], max_context_passages: int = 5) -> str:
        """
        Construct structured clinical RAG prompt template combining context passages and query.

        Args:
            question (str): Patient or clinical query question.
            passages (List[Dict[str, Any]]): List of retrieved passage dictionaries.
            max_context_passages (int): Maximum passages to include in prompt context.

        Returns:
            str: Formatted instruction prompt.
        """
        context_texts = []
        for idx, p in enumerate(passages[:max_context_passages], 1):
            text = p.get("text", "").strip()
            context_texts.append(f"[{idx}] {text}")

        formatted_context = "\n\n".join(context_texts) if context_texts else "No medical context available."

        prompt = (
            f"Context:\n{formatted_context}\n\n"
            f"Question:\n{question}\n\n"
            f"Based on the medical context provided above, answer the question as Yes, No, or Maybe, "
            f"followed by a concise clinical explanation."
        )
        return prompt

    def generate_responses(
        self,
        questions_with_passages: List[Dict[str, Any]],
        batch_size: int = 4
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Generate answers for a list of query context dicts and track throughput latency.

        Args:
            questions_with_passages (List[Dict[str, Any]]): List containing {"qid": ..., "question": ..., "passages": [...]}.
            batch_size (int): Mini-batch size for generation inference.

        Returns:
            Tuple containing list of response strings and generation latency metrics.
        """
        self.load_model()
        t0 = time.perf_counter()

        prompts = [self.format_prompt(item["question"], item["passages"]) for item in questions_with_passages]
        responses = []

        print(f"[+] Generating answers for {len(prompts)} queries (batch_size={batch_size})...")

        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM Generation"):
            batch_prompts = prompts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            for j, out_tokens in enumerate(outputs):
                prompt_len = encoded["input_ids"][j].shape[0]
                gen_tokens = out_tokens[prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                responses.append(text)

        total_time_sec = time.perf_counter() - t0
        num_queries = max(len(responses), 1)
        mean_latency_ms = (total_time_sec / num_queries) * 1000.0
        qps = num_queries / max(total_time_sec, 1e-6)

        latency_metrics = {
            "total_time_sec": round(total_time_sec, 4),
            "mean_latency_ms": round(mean_latency_ms, 3),
            "throughput_qps": round(qps, 2)
        }

        print(f"[+] Generation finished in {total_time_sec:.2f}s ({mean_latency_ms:.2f}ms/query, {qps:.1f} QPS).")
        return responses, latency_metrics
