"""
Pipeline: loads precomputed demo state and runs dual retrieve + rerank.
Loads once at startup via load_pipeline(); never recomputed at request time.
"""

import json
import os
import re
import yaml
import faiss
import torch
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch as _torch
_torch.set_num_threads(1)
_torch.set_num_interop_threads(1)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoModel, AutoTokenizer

from src.mrag_integration import (
    encode_texts,
    mrag_rerank_1,
    YEAR_PATTERN,
)

_REPO_ROOT = Path(__file__).parent.parent


@dataclass
class PipelineState:
    model: Any
    tokenizer: Any
    base_index: Any          # FAISS index — base Contriever (baseline)
    timeaware_index: Any     # FAISS index — fine-tuned model (your system)
    passages: List[str]
    window_emb_tensor: torch.Tensor
    doc_window_map: Dict[str, Tuple[int, int]]
    num_passages: int


def load_pipeline() -> PipelineState:
    cfg_path = _REPO_ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_name = cfg["models"]["base_contriever"]["name"]
    time_path = cfg["models"]["time_aware_contriever"]["output_dir"]

    print(f"[pipeline] Loading tokenizer from {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)

    full_time_path = _REPO_ROOT / time_path.lstrip("./")
    hf_hub_id = os.getenv("MODEL_HUB_ID", "manojarulmurugan/time-aware-contriever")
    if full_time_path.exists() and (full_time_path / "model.safetensors").exists():
        print(f"[pipeline] Loading fine-tuned model from {full_time_path}")
        model = AutoModel.from_pretrained(str(full_time_path))
    else:
        print(f"[pipeline] Local weights not found — downloading from HF Hub: {hf_hub_id}")
        model = AutoModel.from_pretrained(hf_hub_id)
    model.eval()

    demo_dir = _REPO_ROOT / "demo_data"

    print("[pipeline] Loading passages...")
    with open(demo_dir / "passages.json") as f:
        passages = json.load(f)

    print("[pipeline] Loading FAISS indices...")
    base_index = faiss.read_index(str(demo_dir / "base.index"))
    timeaware_index = faiss.read_index(str(demo_dir / "timeaware.index"))

    print("[pipeline] Loading window embeddings...")
    state = torch.load(str(demo_dir / "window_state.pt"), weights_only=False)
    window_emb_tensor = state["window_emb_tensor"]
    doc_window_map = state["doc_window_map"]

    print(f"[pipeline] Ready — {len(passages)} passages, "
          f"base index: {base_index.ntotal} vectors, "
          f"time-aware index: {timeaware_index.ntotal} vectors.")

    return PipelineState(
        model=model,
        tokenizer=tokenizer,
        base_index=base_index,
        timeaware_index=timeaware_index,
        passages=passages,
        window_emb_tensor=window_emb_tensor,
        doc_window_map=doc_window_map,
        num_passages=len(passages),
    )


def _build_passage_cards(
    cand_ids: List[int],
    cand_scores: List[float],
    passages: List[str],
    top_n: int = 10,
) -> List[dict]:
    cards = []
    for i, (pid, score) in enumerate(zip(cand_ids[:top_n], cand_scores[:top_n])):
        text = passages[pid]
        years = sorted(set(int(m.group(1)) for m in YEAR_PATTERN.finditer(text)))
        cards.append({
            "passage_id": pid,
            "text": text,
            "score": round(score, 4),
            "rank": i + 1,
            "years_in_text": years,
        })
    return cards


def run_query(
    state: PipelineState,
    query: str,
    target_year: int,
    top_k: int = 20,
) -> dict:
    # 1. Encode query once — shared by both retrieval branches
    q_emb = encode_texts(state.model, state.tokenizer, [query])  # (1, 768)

    # 2. Base Contriever retrieval (baseline)
    base_scores, base_ids = state.base_index.search(q_emb, top_k)
    base_cand_ids = [int(x) for x in base_ids[0] if 0 <= x < len(state.passages)]
    base_cand_scores = [float(base_scores[0][i]) for i in range(len(base_cand_ids))]
    base_results = _build_passage_cards(base_cand_ids, base_cand_scores, state.passages)

    # 3. Time-aware Contriever retrieval
    ta_scores, ta_ids = state.timeaware_index.search(q_emb, top_k)
    ta_cand_ids = [int(x) for x in ta_ids[0] if 0 <= x < len(state.passages)]
    ta_cand_texts = [state.passages[i] for i in ta_cand_ids]
    ta_cand_scores = [float(ta_scores[0][i]) for i in range(len(ta_cand_ids))]

    # 4. MRAG re-rank the time-aware candidates
    ranked_ids = mrag_rerank_1(
        query,
        ta_cand_texts,
        ta_cand_ids,
        state.model,
        state.tokenizer,
        base_scores=np.array(ta_cand_scores, dtype=np.float32),
        blend_weight=0.0,
        temporal_weight=1.0,
        window_emb_tensor=state.window_emb_tensor,
        doc_window_map=state.doc_window_map,
    )

    # 5. Build MRAG-reranked result cards
    ta_id_to_score = {cid: score for cid, score in zip(ta_cand_ids, ta_cand_scores)}
    reranked = []
    for rank, pid in enumerate(ranked_ids[:10], start=1):
        text = state.passages[pid]
        years = sorted(set(int(m.group(1)) for m in YEAR_PATTERN.finditer(text)))
        reranked.append({
            "passage_id": pid,
            "text": text,
            "score": round(ta_id_to_score.get(pid, 0.0), 4),
            "rank": rank,
            "years_in_text": years,
        })

    return {
        "query": query,
        "target_year": target_year,
        "base_results": base_results,       # top-10, base Contriever order
        "reranked": reranked,               # top-10, time-aware + MRAG order
    }
