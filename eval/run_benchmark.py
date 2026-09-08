#!/usr/bin/env python3
"""Score the retrievers over the gold set and print the tables that go on the projector.

Sections 1, 3, 4 and 5 of eval/RESULTS.md come out of these commands (sections 2 and 6 come
from notebooks/05_chunking_and_noise.py and notebooks/03_stuff_the_prompt.py instead):

    python eval/run_benchmark.py --skip-rerank   # document level: BM25 vs each embedder vs RRF
    python eval/run_benchmark.py                 # the same, plus a rerank pass (slow)
    python eval/run_benchmark.py --chunking      # the chunking ladder            (RESULTS.md 1)
    python eval/run_benchmark.py --chunking --embed-model nomic-embed-text
                                                 # the ladder on the weak embedder (RESULTS.md 3)
    python eval/run_benchmark.py --fusion        # dense vs BM25 vs RRF on chunks (RESULTS.md 4)
    python eval/run_benchmark.py --rerank-sweep  # one reranker, four setups      (RESULTS.md 5)

Without a section flag the run is at document level, which is the shape the gold set is written
against and the number module 5 opens on. `--chunking` is the chunk-level version of the same
measurement, and the two are meant to be read against each other.

Every section prints what it cost — model calls and wall-clock seconds — because on this course
the price of a technique is part of its result. `--rerank-sweep` is off by default for that
reason: it is 8 model calls per question per setup, four setups, and it takes minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunking as C  # noqa: E402
import metrics  # noqa: E402
import retrieval as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "2026-Q3"
GOLD = ROOT / "eval" / "gold_questions.jsonl"

# 8 candidates means 8 model calls per query. The cost quoted on the module 9 and 10 pages is
# this number, so it lives in one place.
RERANK_DEPTH = 8

# (row label, chunking strategy or None for whole documents, strip boilerplate first, short label)
LADDER = [
    ("whole documents", None, False, "whole"),
    ("fixed-280", "fixed-280", False, "fixed"),
    ("fixed-280+overlap60", "fixed-280+overlap60", False, "fixed+ovl"),
    ("recursive-600", "recursive-600", False, "recursive"),
    ("structure-aware", "structure-aware", False, "structure"),
    ("structure-aware + strip", "structure-aware", True, "str+strip"),
]

SWEEP = [
    ("weak   : nomic + fixed-280", "fixed-280", "nomic-embed-text"),
    ("weak   : nomic + structure-aware", "structure-aware", "nomic-embed-text"),
    ("strong : bge-m3 + fixed-280", "fixed-280", "bge-m3"),
    ("strong : bge-m3 + structure-aware", "structure-aware", "bge-m3"),
]


def load_corpus() -> tuple[list[str], list[str], dict[str, str]]:
    paths = sorted(CORPUS.glob("*.md"))
    if not paths:
        sys.exit(f"no documents in {CORPUS}")
    ids = [p.stem for p in paths]
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return ids, texts, dict(zip(ids, texts))


def dense_or_skip(ids: list[str], texts: list[str], model: str) -> R.DenseRetriever | None:
    """Build a dense retriever, or explain precisely why it could not be built.

    A model that is not pulled is a one-line fix and no reason to lose the rest of the table,
    so the run continues without it. A server that is not running stops everything, because
    nothing after this point can work either.
    """
    try:
        return R.DenseRetriever(ids, texts, model=model)
    except R.OllamaError as e:
        if e.kind == "missing_model":
            print(f"  {model}: not installed — skipping it.")
            print(f"      fix it with:  ollama pull {model}")
            return None
        raise


def cost(t0: float, calls0: dict[str, int]) -> str:
    embed = R.CALLS["embed"] - calls0["embed"]
    chat = R.CALLS["chat"] - calls0["chat"]
    return f"{time.time() - t0:.1f}s, {embed} embed calls, {chat} chat calls"


def snapshot() -> dict[str, int]:
    return dict(R.CALLS)


def progress(text: str) -> None:
    """Overwrite one line while a slow loop runs — but only on a terminal, so a redirected
    log does not collect a hundred half-finished progress lines."""
    if sys.stdout.isatty():
        print(f"\r  {text}", end="", flush=True)


def table(header: list[str], rows: list[list[str]], widths: list[int]) -> str:
    """Fixed-width table. A negative width means left-align that column."""
    def cell(text: str, width: int) -> str:
        return text.ljust(-width) if width < 0 else text.rjust(width)

    line = "".join(cell(h, w) for h, w in zip(header, widths))
    return "\n".join([line, "-" * len(line.rstrip())]
                     + ["".join(cell(c, w) for c, w in zip(row, widths)) for row in rows])


def chunk_rankings(documents: dict[str, str], strategy: str | None, strip: bool,
                   questions: list[dict], model: str) -> tuple[dict, int] | None:
    """Index the corpus under one chunking strategy and rank every gold question against it.

    `strategy=None` indexes whole documents, which is the top rung of the ladder and the
    condition the document-level run uses.
    """
    docs = {k: C.strip_boilerplate(v) for k, v in documents.items()} if strip else documents
    if strategy is None:
        ids, texts = list(docs), list(docs.values())
        to_docs = list
    else:
        ids, texts, _ = C.chunk_corpus(docs, strategy)
        to_docs = C.to_documents
    dense = dense_or_skip(ids, texts, model)
    if dense is None:
        return None
    return {q["id"]: to_docs(dense.rank(q["query"])) for q in questions}, len(ids)


# --------------------------------------------------------------------------- sections

def section_documents(ids, texts, documents, questions, args) -> None:
    runs: dict[str, dict[str, list[str]]] = {}

    bm25 = R.BM25(ids, texts)
    runs["BM25"] = {q["id"]: bm25.rank(q["query"]) for q in questions}

    dense_by_model = {}
    for model in [m.strip() for m in args.embed_models.split(",") if m.strip()]:
        t0, c0 = time.time(), snapshot()
        dense = dense_or_skip(ids, texts, model)
        if dense is None:
            continue
        rankings = {q["id"]: dense.rank(q["query"]) for q in questions}
        dense_by_model[model] = rankings
        runs[model] = rankings
        print(f"  {model}: indexed and queried in {cost(t0, c0)}")

    if "bge-m3" in dense_by_model:
        runs["RRF(bge-m3+BM25)"] = {
            q["id"]: R.rrf([dense_by_model["bge-m3"][q["id"]], runs["BM25"][q["id"]]])
            for q in questions
        }

    if not args.skip_rerank and "bge-m3" in dense_by_model:
        t0, c0 = time.time(), snapshot()
        reranked = {}
        for n, q in enumerate(questions, 1):
            top = dense_by_model["bge-m3"][q["id"]][:RERANK_DEPTH]
            order = R.pointwise_rerank(q["query"], top, documents)
            reranked[q["id"]] = order + [d for d in dense_by_model["bge-m3"][q["id"]] if d not in order]
            progress(f"reranking {n}/{len(questions)}")
        runs["bge-m3+rerank"] = reranked
        print(f"\r  rerank at depth {RERANK_DEPTH}: {cost(t0, c0)}                    ")

    print()
    results = {name: metrics.evaluate(rankings, questions) for name, rankings in runs.items()}
    print(metrics.compare(results, questions))


def section_chunking(documents, questions, args) -> None:
    print(f"## chunking ladder ({args.embed_model}, scored at document level)\n")
    rows, by_type_rows, types = [], {}, []
    for label, strategy, strip, _short in LADDER:
        t0, c0 = time.time(), snapshot()
        got = chunk_rankings(documents, strategy, strip, questions, args.embed_model)
        if got is None:
            print(f"  the ladder needs {args.embed_model}; nothing to print.")
            return
        rankings, n_chunks = got
        r = metrics.evaluate(rankings, questions)
        rows.append([label, str(n_chunks), f"{r['hit@1']:.3f}", f"{r['recall@5']:.3f}",
                     f"{r['MRR']:.3f}", f"{time.time() - t0:.0f}s"])
        by_type_rows[label] = r["by_type"]
        types = sorted(set(types) | set(r["by_type"]))
        print(f"  {label}: {n_chunks} chunks, {cost(t0, c0)}")

    print()
    print(table(["strategy", "chunks", "hit@1", "recall@5", "MRR", "time"],
                rows, [-26, 8, 9, 10, 9, 8]))

    print("\nhit@1 by question type — an average that improves can hide a category that "
          "went the other way\n")
    head = ["type"] + [short for _, _, _, short in LADDER]
    type_rows = [[t] + [f"{by_type_rows[label][t]['hit@1']:.3f}" if t in by_type_rows[label] else "-"
                        for label, _, _, _ in LADDER] for t in types]
    print(table(head, type_rows, [-14] + [12] * len(LADDER)))

    raw = sum(len(v) for v in documents.values())
    clean = sum(len(C.strip_boilerplate(v)) for v in documents.values())
    strip_pct = 100 * (raw - clean) / raw
    print(f"\nboilerplate strip removes {raw - clean} of {raw} characters ({strip_pct:.1f}%).")


def section_fusion(documents, questions, args) -> None:
    print(f"## fusion over structure-aware chunks ({args.embed_model})\n")
    ids, texts, _ = C.chunk_corpus(documents, "structure-aware")
    t0, c0 = time.time(), snapshot()
    dense = dense_or_skip(ids, texts, args.embed_model)
    if dense is None:
        print(f"  fusion needs {args.embed_model}; nothing to print.")
        return
    bm25 = R.BM25(ids, texts)
    dense_rank = {q["id"]: dense.rank(q["query"]) for q in questions}
    bm25_rank = {q["id"]: bm25.rank(q["query"]) for q in questions}
    runs = {
        "dense only": {k: C.to_documents(v) for k, v in dense_rank.items()},
        "BM25": {k: C.to_documents(v) for k, v in bm25_rank.items()},
        "RRF of both": {q["id"]: C.to_documents(R.rrf([dense_rank[q["id"]], bm25_rank[q["id"]]]))
                        for q in questions},
    }
    print(f"  {len(ids)} chunks, {cost(t0, c0)}\n")
    print(metrics.compare({n: metrics.evaluate(r, questions) for n, r in runs.items()}, questions))


def section_rerank_sweep(documents, questions, args) -> None:
    print(f"## one reranker ({R.CHAT_MODEL}) over four retrieval setups, depth {RERANK_DEPTH}")
    print(f"   cost: {RERANK_DEPTH} model calls per question per setup — "
          f"{RERANK_DEPTH * len(questions) * len(SWEEP)} calls in total\n")

    def score(ranking):
        return metrics.evaluate({k: C.to_documents(v) for k, v in ranking.items()}, questions)

    rows = []
    for label, strategy, model in SWEEP:
        t0, c0 = time.time(), snapshot()
        ids, texts, _ = C.chunk_corpus(documents, strategy)
        pieces = dict(zip(ids, texts))
        dense = dense_or_skip(ids, texts, model)
        if dense is None:
            continue
        base = {q["id"]: dense.rank(q["query"]) for q in questions}
        after = {}
        for n, q in enumerate(questions, 1):
            order = R.pointwise_rerank(q["query"], base[q["id"]][:RERANK_DEPTH], pieces)
            after[q["id"]] = order + [c for c in base[q["id"]] if c not in order]
            progress(f"{label}: {n}/{len(questions)}")
        before, post = score(base), score(after)
        delta = post["MRR"] - before["MRR"]
        verdict = "HELPED" if delta > 0.02 else ("hurt" if delta < -0.02 else "no change")
        rows.append([label, f"{before['hit@1']:.3f}", f"{post['hit@1']:.3f}",
                     f"{before['MRR']:.3f}", f"{post['MRR']:.3f}",
                     f"   {verdict} ({delta:+.3f})"])
        print(f"\r  {label}: {cost(t0, c0)}                              ")

    print()
    print(table(["retrieval setup", "hit@1 before", "after", "MRR before", "after", "   verdict"],
                rows, [-36, 14, 9, 13, 9, -22]))


# --------------------------------------------------------------------------- entry point

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-rerank", action="store_true",
                        help="document-level section only: skip the rerank pass")
    parser.add_argument("--embed-models", default="bge-m3,nomic-embed-text",
                        help="document-level section: embedders to compare, comma separated. "
                             "A model that is not pulled is reported and skipped.")
    parser.add_argument("--embed-model", default="bge-m3",
                        help="embedder for --chunking and --fusion (default: bge-m3)")
    parser.add_argument("--chunking", action="store_true", help="print the chunking ladder")
    parser.add_argument("--fusion", action="store_true",
                        help="print dense vs BM25 vs RRF over structure-aware chunks")
    parser.add_argument("--rerank-sweep", action="store_true",
                        help="print the reranker over four retrieval setups (slow)")
    parser.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="run on the first N gold questions only, to check a command works. "
                             "Numbers from a smoke run are NOT comparable to eval/RESULTS.md.")
    args = parser.parse_args()

    ids, texts, documents = load_corpus()
    questions = metrics.load_gold(GOLD)
    if args.smoke:
        gold_total = len(questions)
        questions = questions[:args.smoke]
        print(f"SMOKE RUN: {len(questions)} of {gold_total} gold questions. "
              f"These numbers do not match eval/RESULTS.md and must not be quoted.\n")
    kb = sum(len(t) for t in texts) / 1024
    print(f"corpus  {len(ids)} documents, {kb:.0f} KB")
    print(f"gold    {len(questions)} question{'' if len(questions) == 1 else 's'}\n")

    t0 = time.time()
    try:
        sections = [(args.chunking, section_chunking, (documents, questions, args)),
                    (args.fusion, section_fusion, (documents, questions, args)),
                    (args.rerank_sweep, section_rerank_sweep, (documents, questions, args))]
        if not any(flag for flag, _, _ in sections):
            section_documents(ids, texts, documents, questions, args)
        else:
            for n, (flag, fn, fn_args) in enumerate([s for s in sections if s[0]]):
                if n:
                    print("\n")
                fn(*fn_args)
    except R.OllamaError as e:
        if e.kind == "unreachable":
            return f"\n{e}"
        raise

    print(f"\ntotal   {time.time() - t0:.1f}s, {R.CALLS['embed']} embed calls, "
          f"{R.CALLS['chat']} chat calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
