"""
FastAPI backend for Time-Aware RAG demo.
Pipeline state is loaded once at startup and reused across all requests.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

# Prevent HuggingFace fast-tokenizer from forking child processes inside
# the uvicorn worker — avoids the deadlock/segfault under async event loops.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline import PipelineState, load_pipeline, run_query
from src.generator import generate_answer


_state: Optional[PipelineState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = load_pipeline()
    yield


app = FastAPI(
    title="Time-Aware RAG",
    description="Temporal QA with fine-tuned Contriever + MRAG re-ranking",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    target_year: int = Field(default=2020, ge=1850, le=2024)
    top_k: int = Field(default=20, ge=1, le=50)
    generate: bool = True


class QueryResponse(BaseModel):
    query: str
    target_year: int
    candidates: list
    reranked: list
    answer: Optional[str]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    # Sync def — FastAPI runs this in a thread pool, keeping torch out of the
    # async event loop (required to avoid tokenizer multiprocessing deadlock).
    start = time.time()
    result = run_query(_state, req.query, req.target_year, req.top_k)
    answer = None
    if req.generate:
        answer = generate_answer(req.query, result["reranked"], req.target_year)
    latency_ms = round((time.time() - start) * 1000, 1)
    return QueryResponse(
        query=req.query,
        target_year=req.target_year,
        candidates=result["candidates"],
        reranked=result["reranked"],
        answer=answer,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "passages_loaded": _state.num_passages if _state else 0,
    }


# Static frontend — must be mounted LAST (catch-all route)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
