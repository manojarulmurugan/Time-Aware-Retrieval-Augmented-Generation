"""
FastAPI backend for Time-Aware RAG demo.
Pipeline state is loaded once at startup and reused across all requests.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

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
    description="Temporal QA — Base Contriever vs Time-Aware Contriever + MRAG",
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
    target_year: int = Field(default=1920, ge=1800, le=1963)
    top_k: int = Field(default=20, ge=1, le=50)
    generate: bool = True


class QueryResponse(BaseModel):
    query: str
    target_year: int
    base_results: list        # top-10 from base Contriever (baseline)
    reranked: list            # top-10 from time-aware Contriever + MRAG (your system)
    answer: Optional[str]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    # Sync def — FastAPI runs in thread pool, keeping torch off the async event loop.
    start = time.time()
    result = run_query(_state, req.query, req.target_year, req.top_k)
    answer = None
    if req.generate:
        answer = generate_answer(req.query, result["reranked"], req.target_year)
    latency_ms = round((time.time() - start) * 1000, 1)
    return QueryResponse(
        query=req.query,
        target_year=req.target_year,
        base_results=result["base_results"],
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


# Static frontend — mounted LAST (catch-all route)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
