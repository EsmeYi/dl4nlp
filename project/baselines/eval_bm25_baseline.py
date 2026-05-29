"""
eval_bm25_baseline.py — BM25 sparse retrieval baseline for NL retrieval tasks.

Task 1 (NL→Code): query=NL description, corpus=source code tokens
Task 2 (Code→NL): query=source code tokens, corpus=NL description tokens

Uses the same EVAL_N=2000 pool as our model evaluation.
"""

import json
import re
import sys
from pathlib import Path

CACHE_DIR = Path("/cephyr/users/lirongy/Alvis/.cache/dl4nlp_crossmodal")
EVAL_N    = 2000


def tokenize(text: str):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())


def load_split(split: str):
    nl_map = {}
    with open(CACHE_DIR / f"nl_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            nl_map[r["id"]] = r["nl"]

    records = []
    with open(CACHE_DIR / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in nl_map:
                records.append({
                    "source": r["source"],
                    "nl":     nl_map[r["id"]],
                })
    return records


def recall_at_k(scores_matrix, ks=(1, 5, 10)):
    """scores_matrix: list of lists (n_queries, n_docs), diagonal is ground truth."""
    n = len(scores_matrix)
    results = {}
    for k in ks:
        hits = 0
        for i in range(n):
            row = scores_matrix[i]
            top_k = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[:k]
            if i in top_k:
                hits += 1
        results[f"recall@{k}"] = hits / n
    return results


def evaluate_bm25(records, eval_n):
    from rank_bm25 import BM25Okapi

    records = records[:eval_n]
    n = len(records)

    nl_tokens  = [tokenize(r["nl"])     for r in records]
    src_tokens = [tokenize(r["source"]) for r in records]

    # Task 1: NL→Code  (query=NL, corpus=source)
    bm25_src = BM25Okapi(src_tokens)
    scores_t1 = []
    for i, q_toks in enumerate(nl_tokens):
        scores_t1.append(bm25_src.get_scores(q_toks).tolist())
        if (i + 1) % 200 == 0:
            print(f"  T1: {i+1}/{n}", flush=True)

    # Task 2: Code→NL  (query=source, corpus=NL)
    bm25_nl = BM25Okapi(nl_tokens)
    scores_t2 = []
    for i, q_toks in enumerate(src_tokens):
        scores_t2.append(bm25_nl.get_scores(q_toks).tolist())
        if (i + 1) % 200 == 0:
            print(f"  T2: {i+1}/{n}", flush=True)

    t1 = recall_at_k(scores_t1)
    t2 = recall_at_k(scores_t2)
    return t1, t2


def main():
    for split in ("val", "test"):
        print(f"\nEvaluating {split} ...", flush=True)
        records = load_split(split)
        print(f"  Loaded {len(records)} records, using first {EVAL_N}", flush=True)
        t1, t2 = evaluate_bm25(records, EVAL_N)
        tag = split.upper()
        print(f"[{tag}] BM25 NL→Code  R@1={t1['recall@1']:.4f}  R@5={t1['recall@5']:.4f}  R@10={t1['recall@10']:.4f}")
        print(f"[{tag}] BM25 Code→NL  R@1={t2['recall@1']:.4f}  R@5={t2['recall@5']:.4f}  R@10={t2['recall@10']:.4f}")


if __name__ == "__main__":
    main()
