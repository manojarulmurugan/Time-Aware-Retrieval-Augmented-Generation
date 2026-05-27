"""
Pipeline: loads precomputed demo state and runs retrieve + rerank.
All heavy assets are loaded once at startup via load_pipeline().
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
    index: Any
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
        print(f"[pipeline] Local weights not found — loading from HF Hub: {hf_hub_id}")
        model = AutoModel.from_pretrained(hf_hub_id)
    model.eval()

    demo_dir = _REPO_ROOT / "demo_data"

    print("[pipeline] Loading passages...")
    with open(demo_dir / "passages.json") as f:
        passages = json.load(f)

    print("[pipeline] Loading FAISS index...")
    index = faiss.read_index(str(demo_dir / "demo.index"))

    print("[pipeline] Loading window embeddings...")
    state = torch.load(str(demo_dir / "window_state.pt"), weights_only=False)
    window_emb_tensor = state["window_emb_tensor"]
    doc_window_map = state["doc_window_map"]

    print(f"[pipeline] Ready — {len(passages)} passages loaded.")
    return PipelineState(
        model=model,
        tokenizer=tokenizer,
        index=index,
        passages=passages,
        window_emb_tensor=window_emb_tensor,
        doc_window_map=doc_window_map,
        num_passages=len(passages),
    )


def run_query(
    state: PipelineState,
    query: str,
    target_year: int,
    top_k: int = 20,
) -> dict:
    # 1. Encode query
    q_emb = encode_texts(state.model, state.tokenizer, [query])  # (1, 768)

    # 2. FAISS retrieval
    raw_scores, raw_ids = state.index.search(q_emb, top_k)

    # 3. Filter out FAISS sentinel -1 IDs
    cand_ids = [
        int(x) for x in raw_ids[0] if 0 <= x < len(state.passages)
    ]
    cand_texts = [state.passages[i] for i in cand_ids]
    cand_scores = [float(raw_scores[0][i]) for i in range(len(cand_ids))]

    # 4. MRAG rerank
    ranked_ids = mrag_rerank_1(
        query,
        cand_texts,
        cand_ids,
        state.model,
        state.tokenizer,
        base_scores=np.array(cand_scores, dtype=np.float32),
        blend_weight=0.0,
        temporal_weight=1.0,
        window_emb_tensor=state.window_emb_tensor,
        doc_window_map=state.doc_window_map,
    )

    # 5. Build raw-order candidates list
    candidates = [
        {
            "id": cand_ids[i],
            "text": cand_texts[i],
            "score": round(cand_scores[i], 4),
        }
        for i in range(len(cand_ids))
    ]

    # 6. Build reranked list (top 10)
    cand_id_to_idx = {cid: i for i, cid in enumerate(cand_ids)}
    reranked = []
    for rank, pid in enumerate(ranked_ids[:10], start=1):
        idx = cand_id_to_idx.get(pid)
        if idx is None:
            continue
        text = state.passages[pid]
        years = sorted(set(int(m.group(1)) for m in YEAR_PATTERN.finditer(text)))
        reranked.append(
            {
                "passage_id": pid,
                "text": text,
                "retrieval_score": round(cand_scores[idx], 4),
                "final_rank": rank,
                "years_in_text": years,
            }
        )

    return {
        "query": query,
        "target_year": target_year,
        "candidates": candidates,
        "reranked": reranked,
    }
