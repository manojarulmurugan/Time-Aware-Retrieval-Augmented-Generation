---
title: Time-Aware RAG
emoji: ⏱
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# Time-Aware RAG: Temporal Question Answering

Fine-tuned Contriever + MRAG temporal re-ranking + GPT-4o-mini generation.
Ask questions with temporal constraints and watch the pipeline retrieve,
re-rank, and generate grounded answers.

**Full eval results (CAQA benchmark):**
- mrag_time_aware Hit@1: **59.1%**
- base_only Hit@1: 40.4%
- Relative improvement: **+46%**

**Demo corpus:** 10k passages from ChroniclingAmericaQA (newspaper archives, 1850–1963)

**Model:** Fine-tuned `facebook/contriever-msmarco` with temporal hard negatives via triplet margin loss

**Stack:** FAISS retrieval → MRAG sliding-window MaxSim + temporal decay → GPT-4o-mini answer synthesis

GitHub: https://github.com/manojarulmurugan/Time-Aware-Retrieval-Augmented-Generation
