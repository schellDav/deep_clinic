# Presentation Deck: Graph-Adaptive Re-Ranking (GAR) for Clinical QA

**Course:** Deep Learning in the Clinic  
**Authors:** Alexey Wratschinski, David Schell  
**Topic:** Mitigating Distractor Noise & Hallucinations in Biomedical RAG  
**Slides:** 10 Structured Slides for Final Presentation  

---

## Slide 1: Title & Overview
* **Title:** Graph-Adaptive Re-Ranking (GAR) for Biomedical Question Answering
* **Subtitle:** Enhancing Clinical Faithfulness in RAG via Hybrid Graph-Neural Retrieval
* **Presenters:** Alexey Wratschinski & David Schell
* **Key Motivation:** How to eliminate LLM hallucinations when searching across 62,000+ medical abstracts.

---

## Slide 2: The Clinical Problem & Motivation
* **The High Stakes of Medical AI:** In clinical question answering, a single hallucinated drug interaction or dosage can be fatal.
* **The Scaling Bottleneck:** 
  * Simple vector search (Dense Retrieval) works well on tiny benchmark datasets (1,000 abstracts).
  * In real hospital databases (62,000+ abstracts), similar medical terms crowd the vector space, causing Dense Retrieval nDCG to drop by **-23.1%**!
* **Our Goal:** Design a scalable, multi-stage retrieval architecture that provides unshakeable factual evidence to generative LLMs.

---

## Slide 3: System Architecture (3-Stage Pipeline)
* **Stage 1 — Hybrid Initial Retrieval:**
  * Sparse Lexical (`BM25s`) + Dense Vector (`Reason-ModernColBERT`, 768-dim).
* **Stage 2 — Graph-Adaptive Re-Ranking (GAR):**
  * Dynamic candidate expansion over a 1.64-Million-edge $k$-NN Corpus Graph.
* **Stage 3 — Neural Cross-Attention Re-Ranking:**
  * Full query-passage cross-attention (`ms-marco-MiniLM-L-6-v2`) selecting Top-5 contexts.
* **Stage 4 — Generative Synthesis & Evaluation:**
  * Generator: `google/gemma-4-12B-it` | Judge: `Qwen3-30B` via RAGAS.

---

## Slide 4: Graph-Adaptive Re-Ranking (The Math Behind GAR)
* **Corpus Graph Construction:**
  * Pre-computed cosine similarity matrix $S \in \mathbb{R}^{N \times N}$ on normalized embeddings.
  * $k$-NN pruning ($k=10$, threshold $\tau = 0.65$) $\rightarrow$ Row-normalized adjacency matrix $A$.
* **Multi-Hop Dynamic Rescoring:**
  $$S_{GAR} = S_{seed} + \sum_{m=1}^h \alpha^m A^m S_{seed} \quad (\text{depth } h=2, \text{ decay } \alpha=0.5)$$
* **Why it works:** Captures latent clinical relationships across multi-hop citations and co-occurring medical conditions.

---

## Slide 5: Experimental Setup on KISSKI HPC Cluster
* **Infrastructure:** KISSKI GPU Cluster (NVIDIA A100-SXM4-80GB VRAM, 64GB RAM).
* **Complete Offline Slurm Orchestration:** Zero-network worker node execution with custom pre-caching (`src/preload_models.py`).
* **Controlled Scaling Experiment:**
  * **Experiment A:** 1,000 Labeled PubMedQA abstracts.
  * **Experiment B:** 62,249 Full Expanded Corpus (adding 61,249 real PubMed distractors).
* **Evaluation Queries:** 1,000 expert clinical questions evaluated across all stages.
* **Statistical Rigor:** All reported benchmark metrics represent the mean across **$N=3$ independent evaluation runs**.

---

## Slide 6: Retrieval & Ranking Results (nDCG, Recall@10 & Recall@100)
*Refer to: `outputs/figures/fig1_retrieval_ndcg_recall.png`*

* **On 1k Corpus:**
  * ModernColBERT: nDCG@10 = `0.9685`, Recall@10 = `0.9940`, Recall@100 = `1.0000`
  * GAR (Stage 3): **nDCG@10 = 0.9874**, **Recall@10 = 0.9960**, **Recall@100 = 0.9960** (Best)
* **On 62k Expanded Corpus (With 61k Distractors):**
  * ModernColBERT drops to `0.7376` (-23.1% degradation!).
  * Cross-Encoder & GAR recover performance to **nDCG@10 = 0.9167–0.9227 (+18.5% recovery)**.
  * Recall@100 remains rock-solid at **97.8%** across the entire 62k collection.

---

## Slide 7: End-to-End Clinical Generation & RAGAS Analysis
*Refer to: `outputs/figures/fig2_ragas_faithfulness_relevance.png`*

* **Faithfulness (Factual Grounding — The Key Metric):**
  * **1k Corpus:** Baseline `0.6368` $\rightarrow$ Cross-Encoder `0.6474` $\rightarrow$ **GAR 0.6685 (+3.17%)**
  * **62k Corpus:** Baseline `0.6577` $\rightarrow$ Cross-Encoder `0.6716` $\rightarrow$ **GAR 0.6991 (+4.14% / +6.30% relative)**
* **Takeaway:** The larger and noisier the medical corpus, the **greater the benefit** of Graph-Adaptive Re-Ranking!

---

## Slide 8: Answer Relevance & Computational Latency
*Refer to: `outputs/figures/fig4_latency_throughput_tradeoff.png`*

* **Answer Relevance Stability:**
  * Stays consistently high across all methods (**~69.4%–70.1% on 62k**).
  * Confirms that factual accuracy gains do not compromise answering direct question intent.
* **Inference Latency:**
  * Stage 3 GAR achieved the **fastest LLM generation time** on 62k (1h 32m, 0.18 QPS vs. 2h 13m for Cross-Encoder).
  * High-density graph contexts reduce token decoding overhead.

---

## Slide 9: Distractor Scaling Ablation Discussion
*Refer to: `outputs/figures/fig3_corpus_scaling_ablation.png`*

* **The Core Finding:**
  * In a small corpus, all retrievers perform adequately.
  * In large medical repositories, single-vector representations suffer from topological crowding.
  * **GAR acts as a relational filter:** Following high-confidence edges in the corpus graph naturally prunes out false-positive distractors.

---

## Slide 10: Conclusion & Clinical Impact
* **Summary of Achievements:**
  1. Built an end-to-end multi-stage RAG pipeline tested on 62,249 PubMed abstracts.
  2. Implemented custom graph propagation math achieving **0.6991 Faithfulness** with Gemma-12B.
  3. Proved that Graph-Adaptive Re-Ranking mitigates large-scale distractor noise (+4.14% gain).
* **Future Work:** Dynamic edge weighting with clinical entity knowledge graphs (e.g. UMLS / SNOMED-CT).
* **Thank you! Questions?**
