#!/usr/bin/env python3
"""Score every retriever over the gold set and print the table that goes on the projector.

    python eval/run_benchmark.py                 # the full set
    python eval/run_benchmark.py --skip-rerank   # skip the slow part

Retrieval here is at document level, which is what the gold set is written against. The
chunk-level version of this same measurement is what module 7 builds, and the numbers are
meant to be compared with these.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
import retrieval as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "2026-Q3"
GOLD = ROOT / "eval" / "gold_questions.jsonl"
RERANK_DEPTH = 6


def load_corpus() -> tuple[list[str], list[str], dict[str, str]]:
    paths = sorted(CORPUS.glob("*.md"))
    if not paths:
        sys.exit(f"no documents in {CORPUS}")
    ids = [p.stem for p in paths]
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return ids, texts, dict(zip(ids, texts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--embed-models", default="bge-m3,nomic-embed-text")
    args = parser.parse_args()

    ids, texts, documents = load_corpus()
    questions = metrics.load_gold(GOLD)
    kb = sum(len(t) for t in texts) / 1024
    print(f"corpus  {len(ids)} documents, {kb:.0f} KB")
    print(f"gold    {len(questions)} questions\n")

    runs: dict[str, dict[str, list[str]]] = {}

    bm25 = R.BM25(ids, texts)
    runs["BM25"] = {q["id"]: bm25.rank(q["query"]) for q in questions}

    dense_by_model = {}
    for model in [m.strip() for m in args.embed_models.split(",") if m.strip()]:
        t0 = time.time()
        try:
            dense = R.DenseRetriever(ids, texts, model=model)
        except RuntimeError as e:
            print(f"skipping {model}: {e}\n")
            continue
        rankings = {q["id"]: dense.rank(q["query"]) for q in questions}
        dense_by_model[model] = rankings
        runs[model] = rankings
        print(f"  {model} indexed and queried in {time.time() - t0:.1f}s")

    if "bge-m3" in dense_by_model:
        runs["RRF(bge-m3+BM25)"] = {
            q["id"]: R.rrf([dense_by_model["bge-m3"][q["id"]], runs["BM25"][q["id"]]])
            for q in questions
        }

    if not args.skip_rerank and "bge-m3" in dense_by_model:
        t0 = time.time()
        reranked = {}
        for n, q in enumerate(questions, 1):
            top = dense_by_model["bge-m3"][q["id"]][:RERANK_DEPTH]
            order = R.pointwise_rerank(q["query"], top, documents)
            reranked[q["id"]] = order + [d for d in dense_by_model["bge-m3"][q["id"]] if d not in order]
            print(f"\r  reranking {n}/{len(questions)}", end="", flush=True)
        runs[f"bge-m3+rerank"] = reranked
        print(f"\r  reranked {len(questions)} queries in {time.time() - t0:.1f}s"
              f" ({RERANK_DEPTH} calls each)      ")

    print()
    results = {name: metrics.evaluate(rankings, questions) for name, rankings in runs.items()}
    print(metrics.compare(results, questions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
