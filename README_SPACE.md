---
title: Time-Aware RAG
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# Time-Aware RAG: Temporal Question Answering

Standard dense retrieval is temporally blind. This demo shows how a fine-tuned Contriever model and MRAG re-ranking anchor retrieval to the correct historical era, using 12,695 passages from the Chronicling America newspaper archive (1800-1963).

## What this demos

A side-by-side comparison of two retrieval systems on the same query:

- **Standard Retrieval** — `facebook/contriever-msmarco` with no temporal awareness
- **Time-Aware RAG** — Contriever fine-tuned with temporal triplet training, re-ranked by MRAG (sliding-window MaxSim × temporal decay), and answered by Llama 3.1 via Groq

## Eval results (ChroniclingAmericaQA, 12,695 passages)

| Configuration | Hit@1 | MRR@10 |
|---|---:|---:|
| Base Contriever | 40.4% | 47.7% |
| Time-Aware + MRAG | **59.1%** | **65.7%** |

## How to use

1. Select a pre-validated query from the era tabs (Founding Era through Gilded Age), or write your own in the text field below.
2. The year is auto-filled from the selected query. Adjust it if using a custom query.
3. Click **Run Comparison**. Both systems search the same corpus simultaneously.
4. Read the generated answer, then scroll down to compare which model retrieved passages from the correct era. Amber border indicates an era match, red indicates a mismatch.

## Corpus

Chronicling America (Library of Congress) — historical American newspapers, 1800-1963. Queries work best as specific factual questions: *Who, What, Which* with a named entity and a year. Broad explanatory questions do not match the newspaper excerpt format.

## Answer generation

Answer generation uses `llama-3.1-8b-instant` via the Groq API (free tier). Requires a `GROQ_API_KEY` secret configured in Space settings. Retrieval and comparison work without it.

## Stack

FAISS retrieval (dual index) → MRAG sliding-window MaxSim + temporal decay → Groq LLM grounded answer synthesis

GitHub: https://github.com/manojarulmurugan/Time-Aware-Retrieval-Augmented-Generation
