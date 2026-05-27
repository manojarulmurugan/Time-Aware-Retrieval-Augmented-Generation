"""
Precompute demo state for HuggingFace Spaces deployment.

Run this ONCE locally before pushing to HF Spaces:
    python scripts/precompute_demo_state.py

Outputs (commit all to the Space repo via Git LFS):
    demo_data/passages.json      ~1.5 MB
    demo_data/demo.index         ~5 MB
    demo_data/window_state.pt    ~25-40 MB
    demo_data/metadata.json      <1 KB

Passage count is capped at 2,500 (not 10k) so this script completes in
~40 minutes on CPU. The full eval numbers in the UI reflect the published
evaluation, not this subset size.
"""

import json
import os
import sys
import yaml
import faiss
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from src.mrag_integration import precompute_window_embeddings

# ── Config ────────────────────────────────────────────────────────────────────
DEMO_PASSAGE_COUNT = 2_500   # Keep encoding time ~36 min on CPU
ENCODE_BATCH_SIZE  = 32      # Safe on CPU; default 128 causes segfault
ENCODE_MAX_LEN     = 128     # Matches window embedding encoder
REPO_ROOT = Path(__file__).parent.parent


def load_config():
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(base_name: str, time_path: str):
    print(f"Loading tokenizer from: {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)

    full_time_path = REPO_ROOT / time_path.lstrip("./")
    if full_time_path.exists() and (full_time_path / "model.safetensors").exists():
        print(f"Loading fine-tuned model from: {full_time_path}")
        model = AutoModel.from_pretrained(str(full_time_path))
        model_path_used = str(full_time_path)
    else:
        print("=" * 60)
        print("WARNING: Fine-tuned weights not found at", full_time_path)
        print("WARNING: Falling back to base model — results will be weaker!")
        print("=" * 60)
        model = AutoModel.from_pretrained(base_name)
        model_path_used = base_name

    model.eval()
    return model, tokenizer, model_path_used


def get_dataset_revision() -> str:
    try:
        from huggingface_hub import dataset_info
        info = dataset_info("Bhawna/ChroniclingAmericaQA")
        sha = info.sha or "unknown"
        print(f"Dataset HEAD commit: {sha}")
        return sha
    except Exception as e:
        print(f"Could not fetch dataset revision ({e}), using 'latest'")
        return "latest"


def load_caqa_passages(revision: str) -> list:
    print(f"Loading ChroniclingAmericaQA validation split (revision={revision!r})...")
    kwargs = {}
    if revision and revision != "latest":
        kwargs["revision"] = revision

    ds = load_dataset("Bhawna/ChroniclingAmericaQA", split="validation", **kwargs)

    seen = set()
    passages = []
    for ex in ds:
        text = (
            ex.get("context")
            or ex.get("positive_passage")
            or ex.get("passage")
            or ""
        ).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        passages.append(text)
        if len(passages) >= DEMO_PASSAGE_COUNT:
            break

    print(f"Collected {len(passages)} unique passages (cap: {DEMO_PASSAGE_COUNT})")
    return passages


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_passages(model, tokenizer, passages: list) -> np.ndarray:
    """Encode passages with safe batch size (no segfault)."""
    print(f"Encoding {len(passages)} passages (batch={ENCODE_BATCH_SIZE}, max_len={ENCODE_MAX_LEN})...")
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(passages), ENCODE_BATCH_SIZE):
            batch = passages[i : i + ENCODE_BATCH_SIZE]
            toks = tokenizer(
                batch, padding=True, truncation=True,
                max_length=ENCODE_MAX_LEN, return_tensors="pt"
            )
            out = model(**toks)
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            all_vecs.append(pooled.cpu().numpy())

            if (i // ENCODE_BATCH_SIZE) % 20 == 0:
                pct = 100 * (i + len(batch)) / len(passages)
                print(f"  {i + len(batch)}/{len(passages)} ({pct:.0f}%)")

    embs = np.vstack(all_vecs).astype(np.float32)
    print(f"Embeddings done: shape {embs.shape}")
    return embs


def build_index(embs: np.ndarray) -> faiss.IndexIDMap2:
    dim = embs.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    ids = np.arange(len(embs), dtype=np.int64)
    index.add_with_ids(embs, ids)
    print(f"FAISS index built — ntotal={index.ntotal}")
    return index


def print_file_size(path: Path, label: str) -> float:
    mb = path.stat().st_size / (1024 ** 2)
    print(f"  {label}: {mb:.1f} MB")
    return mb


def main():
    cfg = load_config()
    base_name = cfg["models"]["base_contriever"]["name"]
    time_path = cfg["models"]["time_aware_contriever"]["output_dir"]

    model, tokenizer, model_path_used = load_model_and_tokenizer(base_name, time_path)

    revision = get_dataset_revision()
    passages = load_caqa_passages(revision)

    out_dir = REPO_ROOT / "demo_data"
    out_dir.mkdir(exist_ok=True)

    # 1. Save passages
    passages_path = out_dir / "passages.json"
    print(f"\nSaving {len(passages)} passages...")
    with open(passages_path, "w") as f:
        json.dump(passages, f)

    # 2. FAISS index
    index_path = out_dir / "demo.index"
    print(f"\nBuilding FAISS index...")
    embs = encode_passages(model, tokenizer, passages)
    index = build_index(embs)
    faiss.write_index(index, str(index_path))
    print(f"Index saved to {index_path}")

    # 3. Window embeddings
    state_path = out_dir / "window_state.pt"
    print(f"\nPrecomputing window embeddings (batch={ENCODE_BATCH_SIZE})...")
    window_emb_tensor, doc_window_map = precompute_window_embeddings(
        model, tokenizer, passages, batch_size=ENCODE_BATCH_SIZE
    )
    torch.save(
        {"window_emb_tensor": window_emb_tensor, "doc_window_map": doc_window_map},
        str(state_path),
    )
    print(f"Window state saved to {state_path}")

    # 4. Metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "num_passages": len(passages),
            "dataset_revision": revision,
            "model_path": model_path_used,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)

    # 5. Size report
    print("\n─── File size report ───────────────────────────────")
    print_file_size(passages_path, "passages.json")
    print_file_size(index_path, "demo.index")
    state_mb = print_file_size(state_path, "window_state.pt")
    print_file_size(meta_path, "metadata.json")

    if state_mb < 10:
        print(f"\nWARNING: window_state.pt is suspiciously small ({state_mb:.1f} MB). Check passage count.")
    elif state_mb > 400:
        print(f"\nWARNING: window_state.pt is very large ({state_mb:.1f} MB). Check window count.")
    else:
        print(f"\nAll sizes look good.")

    print("\nDone! Commit demo_data/ to the HF Space repo via Git LFS.")


if __name__ == "__main__":
    main()
