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

Kraken Air is a fictional airline; the corpus is synthetic training material.
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
        # Full depth, not a top-k. The two Ollama columns rank every chunk, so truncating this
        # one would score MRR against a shorter list and quietly flatter the other two.
        result = collection.query(query_texts=[q["query"]], n_results=len(chunk_ids))
        out[q["id"]] = C.to_documents(result["ids"][0])
    return out


def main() -> int:
    documents = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
    questions = metrics.load_gold(GOLD)
    chunk_ids, chunk_texts, _ = C.chunk_corpus(documents, STRATEGY)
    print(f"{len(documents)} documents, {len(chunk_ids)} {STRATEGY} chunks, {len(questions)} questions\n")
    metrics.warn_if_gold_drifted(questions, documents)

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

    conclude(labels, scores, by_type, questions)
    return 0


def conclude(labels, scores, by_type, questions) -> None:
    """Read the closing lines off the table this run printed, never off a remembered result.

    Each paragraph is guarded by the condition it states. If an embedder behaves differently on
    your machine the sentence about it disappears rather than contradicting the numbers above it.
    """
    counts = {t: sum(1 for q in questions if q["type"] == t) for t in TYPE_ORDER}

    tr_en = {x: by_type[x]["tr_en"] for x in labels if "tr_en" in by_type[x]}
    if len(tr_en) >= 2:
        n = counts["tr_en"]
        print("\nThe tr_en row is the one to read: "
              + ", ".join(f"{x} {v:.3f}" for x, v in tr_en.items()) + ".")
        best = max(tr_en, key=lambda x: tr_en[x])
        blind = [x for x, v in tr_en.items() if v == 0.0]
        if blind and tr_en[best] > 0:
            print(f"{' and '.join(blind)} answered none of the {n} Turkish questions whose answer "
                  f"is in an English document; {best} answered {round(tr_en[best] * n)} of {n}.")

    en_en = {x: by_type[x]["en_en"] for x in labels if "en_en" in by_type[x]}
    if len(en_en) >= 2 and len({round(v, 3) for v in en_en.values()}) == 1:
        value = next(iter(en_en.values()))
        print(f"\nThe en_en row is the counterweight: all {len(en_en)} score {value:.3f} there. An "
              f"English-only test would have reported these embedders interchangeable.")

    overall = {x: scores[x]["hit@1"] for x in labels}
    if len(overall) >= 2:
        step = 1 / len(questions)
        top, bottom = max(overall, key=lambda x: overall[x]), min(overall, key=lambda x: overall[x])
        gap = overall[top] - overall[bottom]
        print(f"\nOverall hit@1 runs from {overall[bottom]:.3f} ({bottom}) to {overall[top]:.3f} "
              f"({top}) — {round(gap / step)} of {len(questions)} questions apart.")

    print("\nNothing in the table above errored and nothing warned you. A wrong document comes "
          "back with exactly the confidence of a right one; only the score tells you which.")


if __name__ == "__main__":
    raise SystemExit(main())
