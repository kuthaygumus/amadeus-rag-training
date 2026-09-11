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

Kraken Air is a fictional airline; the corpus is synthetic training material.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

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

STRIPPED = "  + boilerplate stripped"


class Rung(NamedTuple):
    """One row of the printed table, kept so the closing lines can be read off it.

    Nothing below the table is allowed to assert anything this record does not contain. That is
    the whole contract of this script: the sentences are derived from the run, so a corpus
    change can move the numbers without ever making the commentary wrong.
    """

    label: str
    chunks: int
    scores: dict
    by_type: dict

    @property
    def hit(self) -> float:
        return self.scores["hit@1"]

    @property
    def recall(self) -> float:
        return self.scores["recall@5"]

    @property
    def mrr(self) -> float:
        return self.scores["MRR"]


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
    metrics.warn_if_gold_drifted(questions, documents)

    header = f"{'strategy':<24}{'chunks':>8}{'hit@1':>9}{'recall@5':>10}{'MRR':>8}   " + \
             "".join(f"{t.replace('_', '-'):>13}" for t in TYPE_ORDER)
    print(header)
    print("-" * len(header))

    rows: list[Rung] = []
    for label, strategy in LADDER:
        started = time.time()
        try:
            n, ranked = rank_all(documents, strategy, questions)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<24} SKIPPED — {exc}")
            continue
        rung = Rung(label, n, metrics.evaluate(ranked, questions), by_type(ranked, questions))
        rows.append(rung)
        print(f"{label:<24}{n:>8}{rung.hit:>9.3f}{rung.recall:>10.3f}{rung.mrr:>8.3f}   "
              + "".join(f"{rung.by_type.get(x, 0.0):>13.3f}" for x in TYPE_ORDER)
              + f"   ({time.time() - started:.0f}s)")

    if not rows:
        print("\nNothing ran. Is Ollama up, and is bge-m3 pulled? Try: ollama list")
        return 1

    # The same structure-aware strategy, with the legal boilerplate removed first.
    cleaned = {k: C.strip_boilerplate(v) for k, v in documents.items()}
    raw_chars = sum(len(v) for v in documents.values())
    clean_chars = sum(len(v) for v in cleaned.values())
    n, ranked = rank_all(cleaned, "structure-aware", questions)
    stripped = Rung(STRIPPED, n, metrics.evaluate(ranked, questions), by_type(ranked, questions))
    print(f"{stripped.label:<24}{n:>8}{stripped.hit:>9.3f}{stripped.recall:>10.3f}{stripped.mrr:>8.3f}   "
          + "".join(f"{stripped.by_type.get(x, 0.0):>13.3f}" for x in TYPE_ORDER))
    removed = 100 * (raw_chars - clean_chars) / raw_chars
    print(f"\nStripping the legal footer removed {removed:.1f}% of the characters.")

    every = rows + [stripped]
    conclude(every, rows, stripped, len(questions))
    return 0


def conclude(every: list[Rung], rows: list[Rung], stripped: Rung, n_questions: int) -> None:
    """Say what this run showed — and only what it showed.

    Every paragraph below is guarded by the condition it asserts, so a rung that moves silences
    the sentence about it rather than turning it into a lie on the projector. If a paragraph
    goes missing, that is the result: read the table.
    """
    def find(label: str) -> Rung | None:
        return next((r for r in every if r.label == label), None)

    whole, fixed = find("whole documents"), find("fixed 280")
    overlap, structure = find("fixed 280 + overlap 60"), find("structure-aware 900")

    # 1. An average that improves while a category goes backwards.
    if whole and fixed:
        et_before, et_after = whole.by_type.get("exact_token", 0), fixed.by_type.get("exact_token", 0)
        print(f"\nThe column to argue about: cutting into fixed 280-character pieces moved hit@1 "
              f"{whole.hit:.3f} -> {fixed.hit:.3f} while exact_token went "
              f"{et_before:.3f} -> {et_after:.3f}.")
        if et_after < et_before:
            print("The average improved. A category went backwards. Both are in the same row.")

    # 2. What overlap bought, and what it cost.
    if fixed and overlap:
        print(f"\nOverlap is the rung that argues with itself. The same 280-character cut, with 60 "
              f"characters of overlap, went from {fixed.chunks} chunks to {overlap.chunks} and moved "
              f"hit@1 {fixed.hit:.3f} -> {overlap.hit:.3f}, recall@5 "
              f"{fixed.recall:.3f} -> {overlap.recall:.3f}.")
        if overlap.hit < fixed.hit and overlap.recall > fixed.recall:
            print("The two metrics went opposite ways: more overlapping pieces got the right "
                  "document into the top five more often, and made it harder to rank first.")
        elif overlap.hit < fixed.hit:
            print("Both moved the wrong way. Overlap is the one rung here that costs chunks, costs "
                  "time and returns nothing — do not reach for it by reflex.")

    # 3. The best recall@5 and the best hit@1 need not be the same strategy.
    by_recall = max(every, key=lambda r: r.recall)
    by_hit = max(every, key=lambda r: r.hit)
    if by_recall.label != by_hit.label:
        print(f"\nThe two columns do not pick the same winner. Best recall@5 is {by_recall.recall:.3f}, "
              f"from {by_recall.label.strip()}, which ranks only {by_recall.hit:.3f} at hit@1. Best "
              f"hit@1 is {by_hit.hit:.3f}, from {by_hit.label.strip()}, whose recall@5 is "
              f"{by_hit.recall:.3f}.")
        print("recall@5 only asks whether the answer reached the top five; hit@1 asks whether it "
              "reached the top. A strategy can be good at the first and bad at the second.")
        if by_recall.chunks > by_hit.chunks:
            print(f"The recall winner is the one that cut smaller — {by_recall.chunks} pieces against "
                  f"{by_hit.chunks}. More pieces, more chances to appear; harder to be first.")
    else:
        print(f"\nOn this run one strategy took both columns: {by_hit.label.strip()}, at hit@1 "
              f"{by_hit.hit:.3f} and recall@5 {by_hit.recall:.3f}. That is not the usual shape — the "
              f"two metrics normally disagree, so check whether it survives a different embedder.")

    # 4. Structure-aware, and what stripping boilerplate off it adds.
    if structure and structure.label == max(rows, key=lambda r: r.hit).label:
        print(f"\nSplitting on the document's own structure leads the ladder: hit@1 {structure.hit:.3f} "
              f"and MRR {structure.mrr:.3f}, on {structure.chunks} chunks.")
    if structure:
        d_hit, d_mrr = stripped.hit - structure.hit, stripped.mrr - structure.mrr
        if d_hit > 0:
            print(f"Stripping the legal boilerplate off that same split adds {d_hit:.3f} more hit@1 "
                  f"({structure.hit:.3f} -> {stripped.hit:.3f}) and {d_mrr:+.3f} MRR, for one chunk "
                  f"fewer. Deleting text made retrieval better.")
        elif d_hit == 0:
            print(f"Stripping the boilerplate off that same split left hit@1 where it was "
                  f"({stripped.hit:.3f}); the move it made was {d_mrr:+.3f} of MRR.")
        else:
            print(f"Stripping the boilerplate off that same split cost {-d_hit:.3f} of hit@1 on this "
                  f"run ({structure.hit:.3f} -> {stripped.hit:.3f}). Say so, and keep the raw text.")

    # 5. How wide any of these gaps actually is.
    step = 1 / n_questions
    spread = by_hit.hit - min(r.hit for r in every)
    print(f"\nBefore anyone quotes a digit: every hit@1 above is a count out of {n_questions} "
          f"questions, so the smallest gap that can exist is {step:.3f} — one question. The whole "
          f"ladder spans {spread:.3f}, which is {round(spread / step)} questions of {n_questions}.")
    print("Read the direction a move went, not the digits it moved between. One question either "
          "way is inside the noise of a gold set this size.")


if __name__ == "__main__":
    raise SystemExit(main())
