# Time-Aware RAG — Inference Pipeline & Generation Layer

## Overview

This document describes the complete inference-time system built on top of the fine-tuned Contriever retriever. It covers how a user query flows from the frontend through dual retrieval, MRAG re-ranking, and grounded answer generation, as well as how the demo is structured and deployed.

The system has three distinct concerns:

1. **Precomputation** — building the indices and window embeddings once before deployment.
2. **Retrieval (the "R" and "A" of RAG)** — dual-branch dense retrieval + MRAG re-ranking at query time.
3. **Generation (the "G" of RAG)** — grounded answer synthesis from the top-ranked passages.

---

## Architecture Diagram

```
User Query + Target Year
         │
         ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend (api/main.py)       │
│  Loaded once at startup via lifespan hook        │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  encode_texts()      │
          │  fine-tuned model    │
          │  → q_emb (1 × 768)  │
          └──────────┬──────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
 ┌───────▼──────┐       ┌────────▼─────────────────┐
 │  base.index  │       │    timeaware.index        │
 │  (baseline)  │       │  (fine-tuned retriever)   │
 │  FAISS IP    │       │  FAISS IP                 │
 └───────┬──────┘       └────────┬─────────────────┘
         │                       │
  top-20 passages          top-20 passages
  (scored, ranked)               │
         │                       ▼
         │             ┌─────────────────────────┐
         │             │      mrag_rerank_1()     │
         │             │  1. decompose_question   │
         │             │  2. dense MaxSim         │
         │             │  3. temporal scoring     │
         │             │  4. hybrid re-rank       │
         │             └─────────┬───────────────┘
         │                       │
         │                 top-10 re-ranked
         │                       │
         └────────────┬──────────┘
                      ▼
          ┌───────────────────────┐
          │    generate_answer()  │
          │  top-5 MRAG passages  │
          │  → Groq LLM call      │
          │  llama-3.1-8b-instant │
          └───────────┬───────────┘
                      │
                      ▼
          QueryResponse (JSON) → Frontend
```

---

## Part 1 — Precomputation (`scripts/precompute_demo_state.py`)

Before the server can be started, four artefacts must be built from the corpus and committed to `demo_data/`. This script is run once locally and the outputs are shipped with the deployment.

### 1.1 Corpus Loading

The corpus is the **ChroniclingAmericaQA** dataset (`Bhawna/ChroniclingAmericaQA`, validation split) — digitized American newspaper excerpts spanning 1800–1963.

```python
ds = load_dataset("Bhawna/ChroniclingAmericaQA", split="validation", revision=revision)
```

Passages are deduplicated and capped at `DEMO_PASSAGE_COUNT = 2,500` unique texts to keep CPU encoding time manageable (~36 minutes). The full evaluation numbers reported in the UI (59.1% Hit@1) were produced over 12,695 passages using the separate evaluation scripts.

### 1.2 Passage Encoding

Both the **base** Contriever (`facebook/contriever-msmarco`) and the **fine-tuned** model encode every passage:

```
encode_passages(model, tokenizer, passages)
  → mean pooling of last hidden state across non-padding tokens
  → L2 normalization
  → returns float32 numpy array of shape (N, 768)
```

Batch size is kept at 32 on CPU to avoid segfaults from tokenizer parallelism.

### 1.3 FAISS Indices

Two flat inner-product indices are built — one per model:

| File | Source model | Purpose |
|------|-------------|---------|
| `demo_data/base.index` | `facebook/contriever-msmarco` | Baseline retrieval branch |
| `demo_data/timeaware.index` | `contriever_finetuned_NEW_20k` | Time-aware retrieval branch |

Index type: `faiss.IndexIDMap2(faiss.IndexFlatIP(768))` — exhaustive exact search, integer IDs align with `passages.json` list positions.

### 1.4 Window Embeddings (`window_state.pt`)

This is the most computationally expensive precomputation step and is the key enabler of MRAG re-ranking at low latency.

**Why windows?** The MRAG re-ranker needs a fine-grained score of how well each **part** of a passage matches the query. Encoding the full passage into a single vector loses intra-passage structure. Each passage is therefore split into overlapping sentence windows and each window is encoded independently.

**Windowing parameters:**
- `WINDOW_SIZE = 3` sentences per window
- `WINDOW_STRIDE = 1` sentence step between consecutive windows

For a passage with sentences `[s0, s1, s2, s3, s4]` this produces windows:
```
[s0, s1, s2]
[s1, s2, s3]
[s2, s3, s4]
```

Sentence splitting uses `nltk.sent_tokenize`; falls back to regex on failure.

**Output structure:**

```python
{
  "window_emb_tensor": Tensor(shape=(Total_Windows, 768)),  # all windows, all passages
  "doc_window_map":    {"0": (start_idx, count), "1": (...), ...}
}
```

The `doc_window_map` maps each passage id (as string) to a `(start, count)` slice into `window_emb_tensor`. This allows O(1) lookup of a passage's window embeddings at query time without re-encoding.

### 1.5 Output Files

```
demo_data/
├── passages.json       # List of 2,500 passage strings (~1.5 MB)
├── base.index          # FAISS index, base model   (~5 MB)
├── timeaware.index     # FAISS index, fine-tuned   (~5 MB)
├── window_state.pt     # Pre-computed window embs  (~25–40 MB)
└── metadata.json       # num_passages, model path, dataset revision, timestamp
```

---

## Part 2 — Pipeline Startup (`src/pipeline.py`)

The pipeline is loaded **once** when the FastAPI server starts (via the `lifespan` context manager in `api/main.py`). It is never re-loaded or re-computed at request time.

### 2.1 Model Loading

```python
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

# Prefer local fine-tuned weights; fall back to HF Hub if missing
if (full_time_path / "model.safetensors").exists():
    model = AutoModel.from_pretrained(str(full_time_path))
else:
    model = AutoModel.from_pretrained("manojarulmurugan/time-aware-contriever")

model.eval()
```

A single model instance serves **both** the time-aware retrieval branch and the MRAG re-ranker. The base Contriever baseline uses a separately pre-encoded `base.index`; no second model is loaded at inference time.

Thread counts are capped at 1 to prevent thrashing on multi-worker setups:
```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

### 2.2 PipelineState Dataclass

```python
@dataclass
class PipelineState:
    model: Any                    # fine-tuned Contriever (eval mode)
    tokenizer: Any                # bert-base tokenizer
    base_index: Any               # FAISS index — baseline vectors
    timeaware_index: Any          # FAISS index — fine-tuned vectors
    passages: List[str]           # passage texts, indexed by position
    window_emb_tensor: Tensor     # (Total_Windows, 768)
    doc_window_map: Dict          # passage_id → (start, count)
    num_passages: int
```

All state lives in RAM. No DB, no network call, no disk read happens during query processing.

---

## Part 3 — Retrieval at Query Time (`src/pipeline.py → run_query`)

`run_query(state, query, target_year, top_k=20)` is the core inference function.

### Step 1 — Encode the Query

```python
q_emb = encode_texts(state.model, state.tokenizer, [query])  # → (1, 768)
```

Mean pooling + L2 normalization, identical to how passages were encoded during precomputation. The same embedding is reused by both retrieval branches.

### Step 2 — Baseline Branch: Base Contriever Retrieval

```python
base_scores, base_ids = state.base_index.search(q_emb, top_k)
```

Inner-product search against the base Contriever passage index. Returns top-20 passages with their inner-product scores. No re-ranking is applied. This branch exists purely for comparison in the demo UI.

**Result format per passage:**
```python
{
  "passage_id": int,
  "text": str,
  "score": float,       # raw FAISS inner-product score
  "rank": int,          # 1-based rank from FAISS
  "years_in_text": list # years extracted via YEAR_PATTERN regex
}
```

### Step 3 — Time-Aware Branch: Fine-Tuned Retriever

```python
ta_scores, ta_ids = state.timeaware_index.search(q_emb, top_k)
```

Same operation, but against the index encoded by the fine-tuned model. Retrieves top-20 candidates for MRAG re-ranking.

### Step 4 — MRAG Re-ranking (`src/mrag_integration.py → mrag_rerank_1`)

This is the core contribution on top of the fine-tuned retriever. It takes the 20 candidates from Step 3 and re-ranks them using three signals:

#### 4a. Temporal Question Decomposition

```python
mc_text, tc = decompose_question_temporal_v1(question_text)
```

Parses the query for temporal expressions using three regex patterns, in order of specificity:

| Pattern | Example | Output |
|--------|---------|--------|
| `TEMP_RANGE_RE` | "between 1850 and 1870" | `TemporalConstraintV1(type="range", start_year=1850, end_year=1870)` |
| `TEMP_POINT_RE` | "as of 1920" / "in 1865" | `TemporalConstraintV1(type="point", year=1865)` |
| `YEAR_ANY_RE` | bare "1920" in query | `TemporalConstraintV1(type="point", year=1920)` |

`mc_text` (main content text) is the query with the temporal expression stripped out. It is used as the semantic query for MaxSim scoring.

#### 4b. Dense MaxSim Score (Sub-passage Granularity)

```python
granular_scores = get_dense_maxsim_scores_fast(
    mc_text, candidate_ids, model, tokenizer,
    window_emb_tensor, doc_window_map
)
```

For each candidate passage:

1. Encode `mc_text` → `q_emb` (1 × 768).
2. Gather the pre-computed window embeddings for all 20 candidates using `doc_window_map`.
3. Compute cosine similarities: `sims = subset_embs @ q_emb.T` — one score per window.
4. Take the **maximum** similarity across all windows for each passage: `MaxSim`.

This gives a sub-passage granularity score that rewards passages where *any* window is a strong semantic match, even if the passage as a whole is long and diluted.

The operation is efficient because all window embeddings are pre-computed; only `q_emb` is computed on-the-fly.

#### 4c. Temporal Proximity Score

```python
t_score = compute_temporal_score(tc, doc_years)
```

For each candidate passage, years are extracted from the passage text via regex and cached in `DOC_YEARS_CACHE`.

The temporal score uses a **triangular decay** function with a span of 20 years:

```python
def triangular(distance):
    return max(0.0, 1.0 - (distance / 20.0))
```

| `tc.type` | Score logic |
|----------|-------------|
| `"point"` | `triangular(min(|doc_year - target_year|))` over all years in passage |
| `"range"` | `1.0` if any doc year falls in `[start, end]`; else `triangular(min distance to range boundary)` |
| `None` | `1.0` (no temporal constraint, score unchanged) |

#### 4d. Hybrid Score Combination

```python
semantic_score = (blend_weight * base_norm + (1 - blend_weight) * granular_norm)
hybrid_factor  = (temporal_weight * t_score + (1 - temporal_weight))
final_score    = semantic_score * hybrid_factor
```

In the demo, `blend_weight=0.0` and `temporal_weight=1.0`, meaning:

- `semantic_score` is purely the MaxSim score (FAISS retrieval score unused in re-ranking).
- `hybrid_factor` is purely the temporal proximity score.
- The final score is MaxSim × temporal proximity.

Both scores are min-max normalized across the 20 candidates before combining.

Passages are sorted by `final_score` descending. The top 10 are returned as `reranked`.

---

## Part 4 — Generation (`src/generator.py`)

The generation layer synthesizes a grounded natural-language answer from the top-5 MRAG re-ranked passages.

### 4.1 System Prompt

```
You are a temporal question-answering assistant. Answer using ONLY the
provided passages. Each passage lists the years mentioned in its text —
use these to reason about time. If facts changed over time, explicitly
state when each version was true, citing passage numbers. If the passages
do not contain sufficient information, say exactly:
'The retrieved passages do not contain enough information to answer this question.'
Never use knowledge outside the provided passages.
```

Key constraints enforced by the prompt:
- **Grounding**: answer exclusively from provided passages.
- **Temporal attribution**: must cite *when* each stated fact was true.
- **Graceful failure**: explicit fallback sentence if passages are insufficient.
- **Citation**: must reference passage numbers.

### 4.2 Context Construction

The top-5 passages are formatted with their year metadata prepended:

```
[Passage 1 | Years mentioned: 1861, 1862]
<passage text>

[Passage 2 | Years mentioned: 1850]
<passage text>

...
```

The full user message is:
```
Context:
<formatted passages>

Question: <query>

As of year: <target_year>

Answer (cite passage numbers):
```

### 4.3 LLM Call

```python
client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[system_msg, user_msg],
    temperature=0.1,
    max_tokens=400,
)
```

Model: **Llama 3.1 8B Instruct** via Groq's free-tier inference API.  
Temperature: 0.1 (near-deterministic, factual answers).  
Max tokens: 400 (sufficient for a paragraph, avoids runaway generation).

If `GROQ_API_KEY` is not set, `generate_answer` returns `None` and the UI falls back to showing retrieval results only — the comparison panel still works.

---

## Part 5 — FastAPI Backend (`api/main.py`)

### 5.1 Startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = load_pipeline()   # blocks until all artefacts are loaded
    yield
```

All heavy loading (model weights, FAISS indices, window tensor) happens synchronously before the server begins accepting requests. No lazy initialization.

### 5.2 `/query` Endpoint

```
POST /query
Content-Type: application/json

{
  "query":       "Who was president in 1808?",
  "target_year": 1808,
  "top_k":       20,
  "generate":    true
}
```

The handler is a **synchronous** function (not `async def`). FastAPI places sync endpoints in a thread pool, which keeps PyTorch matrix operations off the async event loop and avoids blocking other requests.

**Response schema:**
```json
{
  "query":        "...",
  "target_year":  1808,
  "base_results": [ { "passage_id", "text", "score", "rank", "years_in_text" }, ... ],
  "reranked":     [ { ... } ],
  "answer":       "...",
  "latency_ms":   312.5
}
```

`latency_ms` covers retrieval + re-ranking + generation (Groq round-trip included).

### 5.3 `/health` Endpoint

```
GET /health
→ { "status": "ok", "passages_loaded": 2500 }
```

Used by the Docker healthcheck (checked every 30s, 120s startup grace period).

### 5.4 Static Frontend

```python
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

Mounted last so it acts as a catch-all. `GET /` serves `frontend/index.html` directly from disk.

---

## Part 6 — Frontend Demo (`frontend/index.html`)

A single self-contained HTML file; zero build toolchain, no JavaScript dependencies.

### 6.1 UI Layout

```
┌─────────────────────────────────────────────────┐
│  Sticky header  │  corpus stats  │  Hit@1 badge  │
├─────────────────────────────────────────────────┤
│  About section  │  system description + steps   │
├─────────────────────────────────────────────────┤
│  Era tabs (5 historical periods)                 │
│  Pre-validated query cards (click-to-run)        │
│  ── or write your own ──                         │
│  Custom textarea                                 │
│  [As-of Year input]  [Run Comparison]            │
├─────────────────────────────────────────────────┤
│  Generated Answer card (green)                   │
├─────────────────────────────────────────────────┤
│  Standard Retrieval │ Time-Aware RAG             │
│  (base results)     │ (MRAG re-ranked)           │
└─────────────────────────────────────────────────┘
```

### 6.2 Era Tabs

Five time-period tabs with pre-validated queries:

| Tab | Period | Examples |
|-----|--------|---------|
| Founding Era | 1800–1815 | Louisiana Purchase, 1808 presidency |
| New Nation | 1816–1845 | Tariff of 1816, 1844 election |
| Antebellum | 1841–1860 | NYC population 1850, abolition |
| Civil War | 1861–1875 | Lincoln, telegraph, Homestead Act |
| Gilded Age | 1875–1900 | Railroads, Spanish-American War |

Clicking a card sets the query and year, then immediately calls `runPipeline()`.

### 6.3 Query Execution Flow

```javascript
async function runPipeline() {
  // 1. Resolve query (card selection or custom textarea)
  // 2. Show loading states: Encoding → Retrieval → Re-ranking → Generating
  // 3. POST /query
  // 4. Render passage cards with era-match highlighting
  // 5. Show generated answer
  // 6. Scroll answer into view
}
```

### 6.4 Passage Card Rendering

Each passage card shows:
- Rank and inner-product score
- Year pills extracted from passage text (amber if within ±5 years of target year)
- First 240 characters of passage text
- "Era match" (amber border) or "Era mismatch" (red border) indicator

The ±5 year tolerance on era match gives context for minor year-reference drift in newspaper text.

### 6.5 Answer Display

If a `GROQ_API_KEY` is configured, the generated answer appears above the comparison panel. The footer line shows:
```
Grounded to top-5 MRAG-re-ranked passages · <latency> ms · llama-3.1-8b-instant via Groq
```

Without the key, a note explains the retrieval comparison is still fully functional.

---

## Part 7 — Deployment

### 7.1 Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

Port 7860 is the HuggingFace Spaces standard. `--workers 1` avoids duplicate model loading.

### 7.2 HuggingFace Spaces

The fine-tuned model weights are also published to `manojarulmurugan/time-aware-contriever` on the HF Hub. The pipeline falls back to this if local weights are missing:

```python
hf_hub_id = os.getenv("MODEL_HUB_ID", "manojarulmurugan/time-aware-contriever")
if not local_weights_exist:
    model = AutoModel.from_pretrained(hf_hub_id)
```

`GROQ_API_KEY` is configured as a Space secret to enable generation.

### 7.3 Key Environment Variables

| Variable | Default | Purpose |
|---------|---------|---------|
| `GROQ_API_KEY` | *(none)* | Enables generation; retrieval works without it |
| `MODEL_HUB_ID` | `manojarulmurugan/time-aware-contriever` | Override HF Hub model source |
| `TOKENIZERS_PARALLELISM` | `false` | Suppresses tokenizer fork warnings |

---

## Part 8 — Evaluation Results

Evaluated on 500 temporal questions from ChroniclingAmericaQA (questions containing a year reference):

| System | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|--------|-------|-------|--------|--------|
| Base Contriever only | 41.2% | 57.8% | 64.6% | 48.5% |
| Time-Aware Contriever only | 46.8% | 69.4% | 74.8% | 56.5% |
| MRAG + Base Contriever | 54.2% | 72.4% | 76.2% | 61.8% |
| **MRAG + Time-Aware (full system)** | **55.4%** | **74.8%** | **79.0%** | **63.4%** |

The results show two additive gains:
1. **Fine-tuning alone** (+5.6pp Hit@1 over base): the model learns to weight temporal cues in its embeddings.
2. **MRAG re-ranking alone** (+13.0pp Hit@1 over base with base retriever): the re-ranker surfaces temporally aligned passages regardless of retriever quality.
3. **Both combined** (+14.2pp Hit@1): the fine-tuned retriever provides better candidates and MRAG re-ranks them more precisely.

---

## Summary: Data Flow for a Single Query

```
User: "Who was president in 1808?"  target_year=1808
         │
         ▼
[tokenizer]  → token ids
[fine-tuned Contriever]  → mean pool → L2 norm → q_emb (768-dim)
         │
         ├──► base.index.search(q_emb, 20)
         │         → 20 passage ids + inner-product scores
         │         → extract years from text → passage cards
         │         → return as base_results (no re-ranking)
         │
         └──► timeaware.index.search(q_emb, 20)
                   → 20 passage id candidates
                   │
                   ▼
              decompose_question("Who was president in 1808?")
                   → mc_text = "Who was president"
                   → tc = TemporalConstraintV1(type="point", year=1808)
                   │
                   ▼
              get_dense_maxsim_scores_fast(mc_text, 20 candidates)
                   → encode mc_text → q_emb
                   → gather pre-computed window embeddings for 20 passages
                   → q_emb @ window_embs.T → per-window cosine sims
                   → max over windows per passage → MaxSim[0..19]
                   │
                   ▼
              compute_temporal_score(tc, doc_years) for each candidate
                   → extract years from each passage (cached)
                   → triangular(min |doc_year - 1808|) → t_score[0..19]
                   │
                   ▼
              hybrid_score = MaxSim_norm * t_score
              sort candidates → top-10 reranked passage ids
                   │
                   ▼
              generate_answer(query, top_5_reranked, 1808)
                   → format passages with year metadata
                   → POST to Groq: llama-3.1-8b-instant, temp=0.1
                   → grounded answer with passage citations
                   │
                   ▼
              QueryResponse → JSON → Frontend
              Frontend renders:
                - Generated answer (green card)
                - base_results column (left panel)
                - reranked column (right panel, amber/red era indicators)
```
