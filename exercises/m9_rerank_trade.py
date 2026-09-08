#!/usr/bin/env python3
"""Module 9 — when does a reranker actually pay?

    python exercises/m9_rerank_trade.py            # the full measurement, 640 model calls
    python exercises/m9_rerank_trade.py --quick    # a reduced run, clearly labelled as reduced

The wrong question is "does reranking help?". Run that experiment and you get a number that
looks like snake oil. The right question is "help WHAT?" — so this runs the SAME reranker over
four retrieval setups of deliberately different quality: a weak embedder and a strong one,
crossed with bad chunking and good chunking.

Guess before you look: does it help all four, none, or some?

Every call is local. Measured: 324 s for the full run and 65-80 s for --quick, on an M-series Mac
with the model already resident. A CPU-only laptop is considerably slower, which is why --quick
exists and why the cost is part of the lesson rather than a footnote to it.

Helios Air is a fictional airline; the corpus is synthetic training material.
"""

from __future__ import annotations

import argparse
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

SETUPS = [
    ("weak   : nomic + fixed-280",        "fixed-280",       "nomic-embed-text"),
    ("weak   : nomic + structure-aware",  "structure-aware", "nomic-embed-text"),
    ("strong : bge-m3 + fixed-280",       "fixed-280",       "bge-m3"),
    ("strong : bge-m3 + structure-aware", "structure-aware", "bge-m3"),
]


def rank_of_gold(ranking: list[str], gold: list[str]) -> int | None:
    for position, doc in enumerate(ranking, 1):
        if doc in gold:
            return position
    return None


def run_setup(documents, questions, strategy, model, depth, progress):
    ids, texts, _ = C.chunk_corpus(documents, strategy)
    pieces = dict(zip(ids, texts))
    retriever = R.DenseRetriever(ids, texts, model=model)
    before = {q["id"]: retriever.rank(q["query"]) for q in questions}
    after, calls = {}, 0
    for n, q in enumerate(questions, 1):
        candidates = before[q["id"]][:depth]
        order = R.pointwise_rerank(q["query"], candidates, pieces)
        calls += len(candidates)
        after[q["id"]] = order + [c for c in before[q["id"]] if c not in order]
        progress(n, len(questions))
    to_docs = lambda r: {k: C.to_documents(v) for k, v in r.items()}
    return to_docs(before), to_docs(after), calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="4 candidates instead of 8, strong setups only — a REDUCED run")
    args = parser.parse_args()

    depth = 4 if args.quick else 8
    setups = [s for s in SETUPS if s[0].startswith("strong")] if args.quick else SETUPS

    documents = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
    questions = metrics.load_gold(GOLD)
    total_calls = len(setups) * len(questions) * depth

    if args.quick:
        print("*** REDUCED RUN — 4 candidates per question, strong setups only. ***")
        print("*** The direction holds; the numbers are NOT the ones on the slides. ***\n")
    print(f"{len(setups)} setups x {len(questions)} questions x {depth} candidates "
          f"= {total_calls} model calls\n")

    header = f"{'retrieval setup':<36}{'hit@1':>17}{'MRR':>18}"
    print(header)
    print(f"{'':<36}{'before':>8}{'after':>9}{'before':>9}{'after':>9}   verdict")
    print("-" * 92)

    CLEAR = "\r\033[K" if sys.stdout.isatty() else ""
    results, deltas = {}, {}
    for label, strategy, model in setups:
        started = time.time()

        def progress(n, total, label=label):
            # Only animate on a real terminal; piped or redirected output stays readable.
            if sys.stdout.isatty():
                print(f"\r  {label} … {n}/{total}", end="", flush=True)
            elif n == total:
                print(f"  {label} … {total}/{total}", flush=True)

        try:
            before, after, calls = run_setup(documents, questions, strategy, model, depth, progress)
        except Exception as exc:  # noqa: BLE001
            print(f"{CLEAR}{label:<36} SKIPPED — {exc}")
            if "nomic" in model or "bge" in model:
                print(f"{'':<36} fix: ollama pull {model}")
            continue
        b, a = metrics.evaluate(before, questions), metrics.evaluate(after, questions)
        delta = a["MRR"] - b["MRR"]
        verdict = "HELPED" if delta > 0.02 else ("hurt" if delta < -0.02 else "no change")
        results[label] = (before, after)
        deltas[label] = delta
        print(f"{CLEAR}{label:<36}{b['hit@1']:>8.3f}{a['hit@1']:>9.3f}{b['MRR']:>9.3f}{a['MRR']:>9.3f}"
              f"   {verdict} ({delta:+.3f})  [{calls} calls, {time.time() - started:.0f}s]")

    strong = next((v for k, v in results.items() if k.startswith("strong") and "structure" in k), None)
    if strong:
        before, after = strong
        was_first = [q for q in questions if rank_of_gold(before[q["id"]], q["gold_doc_ids"]) == 1]
        demoted = [q for q in was_first
                   if (rank_of_gold(after[q["id"]], q["gold_doc_ids"]) or 99) > 1]
        was_not = [q for q in questions if q not in was_first]
        promoted = [q for q in was_not
                    if (rank_of_gold(after[q["id"]], q["gold_doc_ids"]) or 99)
                    < (rank_of_gold(before[q["id"]], q["gold_doc_ids"]) or 99)]
        print("\nQuestion by question, on the strongest setup:")
        print(f"  gold already at rank 1: {len(was_first):>2} questions — reranking pushed "
              f"{len(demoted)} of them down ({', '.join(q['id'] for q in demoted) or 'none'})")
        print(f"  gold below rank 1:      {len(was_not):>2} questions — reranking pulled "
              f"{len(promoted)} of them up ({', '.join(q['id'] for q in promoted) or 'none'})")

    # The closing line is read off the verdicts this run produced, not asserted ahead of them.
    if not deltas:
        print("\nNothing ran, so there is nothing to conclude. Is Ollama up? Try: ollama list")
        return 1
    weak_d = {k: v for k, v in deltas.items() if k.startswith("weak")}
    strong_d = {k: v for k, v in deltas.items() if k.startswith("strong")}
    print()
    if (weak_d and strong_d and all(v > 0.02 for v in weak_d.values())
            and all(v < -0.02 for v in strong_d.values())):
        print("That is the trade, and this run showed both halves of it: MRR rose on every weak "
              "setup and fell on every strong one.")
        print("A reranker levels toward its own ceiling. It pays when your retriever is worse "
              "than it, and costs you when your retriever is better.")
    elif strong_d and not weak_d and all(v < -0.02 for v in strong_d.values()):
        print("Both strong setups lost MRR. That is half the trade: a reranker levels toward its "
              "own ceiling, and these retrievers were already above it.")
        print("This reduced run carries no weak setup, so it cannot show the other half. The full "
              "command does.")
    else:
        print("Read the verdict column rather than a rule — this run moved MRR by " + ", ".join(
            f"{k.split(':')[-1].strip()} {v:+.3f}" for k, v in deltas.items()) + ".")
        print("A reranker levels a ranking toward its own ceiling, so which way it moves depends "
              "on which side of that ceiling the retriever was already on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
