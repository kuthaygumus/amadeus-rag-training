"""Deterministic retrieval metrics.

Three numbers, reported three times during the day — after naive RAG, after chunking is
fixed, and after reranking. No LLM judge anywhere in here: the same corpus and the same
questions always produce the same numbers, so an improvement is a fact rather than an opinion.

    hit@1     did the top-ranked document turn out to be a gold document?
    recall@5  how much of the gold set showed up in the top 5?
    MRR       one over the rank of the first gold document, averaged.

MRR is the one that shows partial credit: moving a gold document from rank 6 to rank 2 does
not change hit@1 at all, but it moves MRR from 0.17 to 0.50.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_gold(path: str | Path) -> list[dict]:
    """Read gold_questions.jsonl. Raises on a malformed line rather than skipping it —
    a silently dropped question would quietly change every metric."""
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n} is not valid JSON: {e}") from e
        for field in ("id", "query", "gold_doc_ids", "type"):
            if field not in q:
                raise ValueError(f"{path}:{n} question is missing '{field}'")
        questions.append(q)
    return questions


def hit_at_k(ranking: list[str], gold: list[str], k: int = 1) -> float:
    return 1.0 if set(ranking[:k]) & set(gold) else 0.0


def recall_at_k(ranking: list[str], gold: list[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    return len(set(ranking[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranking: list[str], gold: list[str]) -> float:
    for i, doc in enumerate(ranking, 1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def evaluate(rankings: dict[str, list[str]], questions: list[dict]) -> dict:
    """Score one retriever over the whole gold set.

    `rankings` maps a question id to that retriever's ranked document ids, best first.
    A question with no ranking scores zero rather than being skipped — otherwise a
    retriever could look better by failing to answer.
    """
    per_question, by_type = [], {}
    for q in questions:
        ranking = rankings.get(q["id"], [])
        gold = q["gold_doc_ids"]
        row = {
            "id": q["id"],
            "type": q["type"],
            "hit@1": hit_at_k(ranking, gold, 1),
            "recall@5": recall_at_k(ranking, gold, 5),
            "rr": reciprocal_rank(ranking, gold),
            "rank_of_first_gold": next((i for i, d in enumerate(ranking, 1) if d in gold), None),
        }
        per_question.append(row)
        by_type.setdefault(q["type"], []).append(row)

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "n": len(per_question),
        "hit@1": mean(per_question, "hit@1"),
        "recall@5": mean(per_question, "recall@5"),
        "MRR": mean(per_question, "rr"),
        "by_type": {
            t: {"n": len(rows), "hit@1": mean(rows, "hit@1"), "MRR": mean(rows, "rr")}
            for t, rows in sorted(by_type.items())
        },
        "per_question": per_question,
    }


def compare(results: dict[str, dict], questions: list[dict] | None = None) -> str:
    """Render several named runs side by side — the table that goes on the projector.

    Keep the run names short; they become column headers.
    """
    names = list(results)
    width = max(12, max((len(n) for n in names), default=12) + 2)
    lines = []

    head = f"{'metric':<22}" + "".join(f"{n:>{width}}" for n in names)
    lines += [head, "-" * len(head)]
    for metric in ("hit@1", "recall@5", "MRR"):
        lines.append(f"{metric:<22}" + "".join(f"{results[n][metric]:>{width}.3f}" for n in names))

    types = sorted({t for r in results.values() for t in r["by_type"]})
    if types:
        lines += ["", f"{'hit@1 by type':<22}" + "".join(f"{n:>{width}}" for n in names)]
        lines.append("-" * len(head))
        for t in types:
            cells = ""
            for n in names:
                bucket = results[n]["by_type"].get(t)
                cells += f"{bucket['hit@1']:>{width}.3f}" if bucket else f"{'-':>{width}}"
            n_q = next((r["by_type"][t]["n"] for r in results.values() if t in r["by_type"]), 0)
            lines.append(f"{t + f' (n={n_q})':<22}" + cells)

    if questions:
        lines += ["", "questions where every run missed at rank 1:"]
        missed = [
            q["id"] for q in questions
            if all(
                next((r for r in results[n]["per_question"] if r["id"] == q["id"]), {"hit@1": 0})["hit@1"] == 0
                for n in names
            )
        ]
        lines.append("  " + (", ".join(missed) if missed else "(none)"))
    return "\n".join(lines)
