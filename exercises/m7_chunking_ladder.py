#!/usr/bin/env python3
"""Module 7 — why did retrieval bring back garbage, and what actually fixes it?

    python exercises/m7_chunking_ladder.py

One command, one table. Scores the same 20 questions and the same embedder against five ways of
cutting the same 28 documents, then strips boilerplate and scores again.

Read the table twice. Once down the hit@1 column, which is the story everyone expects: bigger
number, better retrieval. Then down the `exact_token` column, which is the story nobody tells:
cutting the corpus into fixed 280-character pieces loses one of the four questions that hinge on
a literal flight code or fare basis code, while the average goes up. A recursive or a
structure-aware split gets that question back. The point is not the size of the loss — it is
that the average alone would never have shown you it happened.

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
EMBEDDER = "bge-m3"
TYPE_ORDER = ["tr_tr", "tr_en", "en_en", "exact_token", "multi_hop"]

LADDER = [
    ("whole documents", None),
    ("fixed 280", "fixed-280"),
    ("fixed 280 + overlap 60", "fixed-280+overlap60"),
    ("recursive 600", "recursive-600"),
    ("structure-aware 900", "structure-aware"),
]


def rank_all(documents, strategy, questions):
    if strategy is None:
        ids, texts = list(documents), list(documents.values())
    else:
        ids, texts, _ = C.chunk_corpus(documents, strategy)
    retriever = R.DenseRetriever(ids, texts, model=EMBEDDER)
    ranked = {q["id"]: C.to_documents(retriever.rank(q["query"])) for q in questions}
    return len(ids), ranked


def by_type(ranked, questions):
    buckets: dict[str, list[float]] = {}
    for q in questions:
        r = ranked[q["id"]]
        buckets.setdefault(q["type"], []).append(1.0 if r and r[0] in q["gold_doc_ids"] else 0.0)
    return {t: sum(v) / len(v) for t, v in buckets.items()}


def main() -> int:
    documents = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
    questions = metrics.load_gold(GOLD)
    print(f"{len(documents)} documents, {len(questions)} questions, embedder {EMBEDDER}\n")

    header = f"{'strategy':<24}{'chunks':>8}{'hit@1':>9}{'recall@5':>10}{'MRR':>8}   " + \
             "".join(f"{t.replace('_', '-'):>13}" for t in TYPE_ORDER)
    print(header)
    print("-" * len(header))

    rows = []
    for label, strategy in LADDER:
        started = time.time()
        try:
            n, ranked = rank_all(documents, strategy, questions)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<24} SKIPPED — {exc}")
            continue
        s = metrics.evaluate(ranked, questions)
        t = by_type(ranked, questions)
        rows.append((label, s, t))
        print(f"{label:<24}{n:>8}{s['hit@1']:>9.3f}{s['recall@5']:>10.3f}{s['MRR']:>8.3f}   "
              + "".join(f"{t.get(x, 0.0):>13.3f}" for x in TYPE_ORDER)
              + f"   ({time.time() - started:.0f}s)")

    if not rows:
        print("\nNothing ran. Is Ollama up, and is bge-m3 pulled? Try: ollama list")
        return 1

    # The same best strategy, with the legal boilerplate removed first.
    cleaned = {k: C.strip_boilerplate(v) for k, v in documents.items()}
    raw_chars = sum(len(v) for v in documents.values())
    clean_chars = sum(len(v) for v in cleaned.values())
    n, ranked = rank_all(cleaned, "structure-aware", questions)
    s = metrics.evaluate(ranked, questions)
    t = by_type(ranked, questions)
    print(f"{'  + boilerplate stripped':<24}{n:>8}{s['hit@1']:>9.3f}{s['recall@5']:>10.3f}{s['MRR']:>8.3f}   "
          + "".join(f"{t.get(x, 0.0):>13.3f}" for x in TYPE_ORDER))
    removed = 100 * (raw_chars - clean_chars) / raw_chars
    print(f"\nStripping the legal footer removed {removed:.1f}% of the characters.")

    whole = next((r for r in rows if r[0] == "whole documents"), None)
    fixed = next((r for r in rows if r[0] == "fixed 280"), None)
    best = next((r for r in rows if r[0] == "structure-aware 900"), None)

    if whole and fixed:
        et_before, et_after = whole[2].get("exact_token", 0), fixed[2].get("exact_token", 0)
        h_before, h_after = whole[1]["hit@1"], fixed[1]["hit@1"]
        print(f"\nThe column to argue about: cutting into fixed 280-character pieces moved hit@1 "
              f"{h_before:.3f} -> {h_after:.3f} while exact_token went {et_before:.3f} -> {et_after:.3f}.")
        if et_after < et_before:
            print("The average improved. A category went backwards. Both are in the same row.")

    # The naive strategies tend to land on the same score; say so only if they actually did.
    naive = [r for r in rows if r[0] in ("fixed 280", "fixed 280 + overlap 60", "recursive 600")]
    if len(naive) == 3 and len({round(r[1]["hit@1"], 3) for r in naive}) == 1 and best:
        print(f"\nAll three naive strategies — fixed size, fixed size with overlap, and a recursive "
              f"splitter — stopped at exactly {naive[0][1]['hit@1']:.3f}.")
        print(f"Only splitting on the document's own structure passed it, at {best[1]['hit@1']:.3f}. "
              f"A smarter naive splitter bought nothing.")

    if fixed and best:
        print(f"\nAnd the metrics disagree on purpose: recall@5 is HIGHEST for the worst strategy "
              f"({fixed[1]['recall@5']:.3f} for fixed 280 against {best[1]['recall@5']:.3f} for "
              f"structure-aware). Cutting into more pieces gives the right document more chances to "
              f"appear somewhere in the top five, while ranking it first gets harder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
