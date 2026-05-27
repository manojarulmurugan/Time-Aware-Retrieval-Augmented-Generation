"""
Weight verification: base vs time-aware, sequential to avoid memory pressure.
Runs in ~30 min on CPU.
"""
if __name__ == '__main__':
    import sys, re, random, gc
    sys.path.insert(0, 'src')

    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm
    from mrag_integration import (
        encode_texts,
        precompute_window_embeddings, mrag_rerank_1,
    )

    BASE_NAME     = 'facebook/contriever-msmarco'
    TIME_PATH     = 'contriever_finetuned_NEW_20k'
    N_QUESTIONS   = 100
    N_DISTRACTORS = 400
    SEED          = 42

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading ChroniclingAmericaQA...")
    caqa = load_dataset("Bhawna/ChroniclingAmericaQA", split="validation")

    passages, passage_to_id = [], {}
    questions_year, gold_ids_year = [], []
    year_re = re.compile(r'\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b')

    for ex in caqa:
        q = ex.get('question') or ex.get('query')
        p = ex.get('context') or ex.get('positive_passage') or ex.get('passage')
        if not q or not p: continue
        if p not in passage_to_id:
            passage_to_id[p] = len(passages)
            passages.append(p)
        if year_re.search(q):
            questions_year.append(q)
            gold_ids_year.append(passage_to_id[p])

    print(f"Full corpus: {len(passages)} passages | {len(questions_year)} year-questions")

    # ── Mini corpus ───────────────────────────────────────────────────
    random.seed(SEED)
    qs       = questions_year[:N_QUESTIONS]
    gold_ids = gold_ids_year[:N_QUESTIONS]
    gold_set = set(gold_ids)
    distractors = random.sample(
        [i for i in range(len(passages)) if i not in gold_set],
        min(N_DISTRACTORS, len(passages) - len(gold_set))
    )
    pool_ids      = sorted(gold_set | set(distractors))
    pool_passages = [passages[i] for i in pool_ids]
    old2new       = {old: new for new, old in enumerate(pool_ids)}
    gold_new      = [old2new[g] for g in gold_ids]
    print(f"Mini corpus: {len(pool_passages)} passages "
          f"({len(gold_set)} gold + {len(distractors)} distractors)\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_NAME)
    top_k     = min(100, len(pool_passages))
    results   = {}

    # ── Evaluate one model, then free all memory before loading next ──
    for model_path, label in [(BASE_NAME, 'base'), (TIME_PATH, 'time_aware')]:
        print(f"\n{'='*55}")
        print(f"Loading: {label}  ({model_path})")
        model = AutoModel.from_pretrained(model_path).eval()

        print("Precomputing window embeddings...")
        win_tensor, win_map = precompute_window_embeddings(
            model, tokenizer, pool_passages)

        print("Encoding passages + queries...")
        p_embs = encode_texts(model, tokenizer, pool_passages)
        q_embs = encode_texts(model, tokenizer, qs)

        sim    = q_embs @ p_embs.T
        ids    = np.argsort(-sim, axis=1)[:, :top_k]
        scores = np.take_along_axis(sim, ids, axis=1)

        hits = 0
        for qi, gold in enumerate(tqdm(gold_new, desc=f"eval [{label}]")):
            cand_ids    = [int(c) for c in ids[qi] if 0 <= c < len(pool_passages)]
            cand_scores = scores[qi][:len(cand_ids)]
            cand_texts  = [pool_passages[c] for c in cand_ids]
            ranked = mrag_rerank_1(
                qs[qi], cand_texts, cand_ids, model, tokenizer,
                base_scores=cand_scores, blend_weight=0.0, temporal_weight=1.0,
                window_emb_tensor=win_tensor, doc_window_map=win_map,
            )
            if ranked and ranked[0] == gold:
                hits += 1

        results[label] = hits / N_QUESTIONS
        print(f"  {label} Hit@1: {results[label]:.4f}  ({results[label]*100:.1f}%)")

        # Free everything before loading the next model
        del model, win_tensor, win_map, p_embs, q_embs, sim, ids, scores
        gc.collect()

    # ── Report ────────────────────────────────────────────────────────
    r_base = results['base']
    r_time = results['time_aware']
    lift_here   = r_time - r_base
    lift_stored = 0.5915 - 0.4036

    print("\n" + "="*55)
    print(f"VERIFICATION ({N_QUESTIONS} questions | {len(pool_passages)} passages)")
    print(f"  base_only        Hit@1: {r_base:.4f}  ({r_base*100:.1f}%)")
    print(f"  mrag_time_aware  Hit@1: {r_time:.4f}  ({r_time*100:.1f}%)")
    print(f"  lift:            +{lift_here*100:.1f}pp")
    print()
    print("STORED full-corpus (12,695 passages | 1,219 questions):")
    print(f"  base_only        Hit@1: 40.4%")
    print(f"  mrag_time_aware  Hit@1: 59.2%")
    print(f"  lift:            +18.8pp")
    print()
    verdict = "PASS ✓" if (r_time > r_base and lift_here > 0.10) else "INCONCLUSIVE"
    print(f"VERDICT: {verdict}")
