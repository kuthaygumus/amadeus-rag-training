#!/usr/bin/env python3
"""Module 6 — is the default embedder right for your language?

    python exercises/m6_embedding_bakeoff.py

One command, one table. Scores three embedders over the same 20 questions and the same
structure-aware chunks, and breaks the score down by question type. The row that matters is
`tr_en`: a Turkish question whose answer sits in an English document.

Two of the three are here on purpose as bad examples. `all-MiniLM-L6-v2` is the model ChromaDB
installs and uses if you never choose one, so it is what you get by accident. `nomic-embed-text`
is a reasonable-looking English-first choice. Neither errors on a Turkish query. Neither returns
nothing. They return the wrong document, confidently, and no log line tells you.

Helios Air is a fictional airline; the corpus is synthetic training material.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import chunking as C  # noqa: E402
import metrics  # noqa: E402
import retrieval as R  # noqa: E402

CORPUS = ROOT / "corpus" / "2026-Q3"
GOLD = ROOT / "eval" / "gold_questions.jsonl"
STRATEGY = "structure-aware"
TYPE_ORDER = ["tr_tr", "tr_en", "en_en", "exact_token", "multi_hop"]


def hit_at_1_by_type(rankings: dict[str, list[str]], questions: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for q in questions:
        ranked = rankings[q["id"]]
        hit = 1.0 if ranked and ranked[0] in q["gold_doc_ids"] else 0.0
        buckets.setdefault(q["type"], []).append(hit)
    return {t: sum(v) / len(v) for t, v in buckets.items()}


def score_ollama(model: str, chunk_ids, chunk_texts, questions):
    retriever = R.DenseRetriever(chunk_ids, chunk_texts, model=model)
    return {q["id"]: C.to_documents(retriever.rank(q["query"])) for q in questions}


def score_chroma_default(chunk_ids, chunk_texts, questions):
    """ChromaDB's out-of-the-box embedder: all-MiniLM-L6-v2, English-only, chosen for you."""
    import chromadb
    from chromadb.utils import embedding_functions

    collection = chromadb.EphemeralClient().create_collection(
        "m6_bakeoff", embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    for i in range(0, len(chunk_ids), 200):
        collection.add(ids=chunk_ids[i:i + 200], documents=chunk_texts[i:i + 200])
    out = {}
    for q in questions:
        result = collection.query(query_texts=[q["query"]], n_results=min(20, len(chunk_ids)))
        out[q["id"]] = C.to_documents(result["ids"][0])
    return out


def main() -> int:
    documents = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
    questions = metrics.load_gold(GOLD)
    chunk_ids, chunk_texts, _ = C.chunk_corpus(documents, STRATEGY)
    print(f"{len(documents)} documents, {len(chunk_ids)} {STRATEGY} chunks, {len(questions)} questions\n")

    runs: dict[str, dict[str, list[str]]] = {}

    for label, model in [("all-MiniLM-L6-v2", None), ("nomic-embed-text", "nomic-embed-text"), ("bge-m3", "bge-m3")]:
        started = time.time()
        try:
            ranked = score_chroma_default(chunk_ids, chunk_texts, questions) if model is None \
                else score_ollama(model, chunk_ids, chunk_texts, questions)
        except Exception as exc:  # noqa: BLE001 — the message matters more than the type
            print(f"  {label:<18} SKIPPED — {exc}")
            if model:
                print(f"                     fix: ollama pull {model}\n")
            continue
        runs[label] = ranked
        print(f"  {label:<18} indexed and queried in {time.time() - started:.1f}s")

    if not runs:
        print("\nNothing ran. Is Ollama up? Try: ollama list")
        return 1

    labels = list(runs)
    width = max(len(x) for x in labels) + 2
    print(f"\n{'metric':<20}" + "".join(f"{x:>{width}}" for x in labels))
    print("-" * (20 + width * len(labels)))
    scores = {x: metrics.evaluate(runs[x], questions) for x in labels}
    for metric in ("hit@1", "recall@5", "MRR"):
        print(f"{metric:<20}" + "".join(f"{scores[x][metric]:>{width}.3f}" for x in labels))

    print(f"\n{'hit@1 by type':<20}" + "".join(f"{x:>{width}}" for x in labels))
    print("-" * (20 + width * len(labels)))
    by_type = {x: hit_at_1_by_type(runs[x], questions) for x in labels}
    for question_type in TYPE_ORDER:
        n = sum(1 for q in questions if q["type"] == question_type)
        if not n:
            continue
        row = "".join(f"{by_type[x].get(question_type, 0.0):>{width}.3f}" for x in labels)
        marker = "   <-- Turkish question, English document" if question_type == "tr_en" else ""
        print(f"{question_type + f' (n={n})':<20}{row}{marker}")

    print("\nThe tr_en row is the one to read. Nothing above errored, and nothing warned you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
