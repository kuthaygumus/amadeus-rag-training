# %% [markdown]
# # 06 · Hybrid search and reranking — when do they actually pay?
#
# We are at hit@1 0.700 with structure-aware chunking and a multilingual embedder — that is the
# recorded number, in `eval/RESULTS.md`. Six questions out of twenty still miss.
#
# Read the figure your own run prints rather than that one. `bge-m3`'s output is not bit-identical
# across Ollama builds and machines, so a single question can land the other way and move hit@1 by
# 0.05. That is the same 0.05 this notebook calls noise at the bottom of the page; it is not a
# broken setup.
#
# The standard advice for that gap is: add keyword search alongside the embeddings, fuse the two
# rankings, then rerank the survivors with a second model. Three techniques, every RAG blog post,
# always framed as improvements.
#
# We are going to build all three and measure them. **Commit to a guess now: how many of the
# three will beat plain retrieval on this corpus?**
#
# Fusion is measured here, because it costs no model call. The reranker is built here — one
# query, six candidates, the scores printed — and measured by `exercises/m9_rerank_trade.py`,
# because the measurement that matters is 160 calls in each of four setups, 640 in all, and
# there is nothing to learn from watching a progress counter for five minutes.

# %%
import sys, time
from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True, replayable=True)
import _cached
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")

DEMO_ID = "q05"          # a Turkish question whose answer sits in an English procedure

chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
# BM25 is pure Python. The dense index is one embedding call per chunk, and a replay needs none.
dense = None if _cached.USE_CACHED else R.DenseRetriever(chunk_ids, chunk_texts)
bm25 = R.BM25(chunk_ids, chunk_texts)
print(f"{len(chunk_ids)} chunks indexed both ways")

# %% [markdown]
# ## Fusion first, because it is free
#
# Fusing two rankings costs no model call at all — it is arithmetic over positions. So we can
# score all twenty questions three ways in the time it takes to embed the corpus once.

# %%
def rank_three_ways() -> dict:
    dense_runs = {q["id"]: dense.rank(q["query"]) for q in questions}
    bm25_runs = {q["id"]: bm25.rank(q["query"]) for q in questions}
    # Fuse at chunk level, then collapse to documents — fusing after the collapse would throw
    # away exactly the rank information RRF works on.
    rrf_runs = {q["id"]: R.rrf([dense_runs[q["id"]], bm25_runs[q["id"]]]) for q in questions}
    to_docs = lambda runs: {k: C.to_documents(v) for k, v in runs.items()}
    return {"dense": to_docs(dense_runs), "bm25": to_docs(bm25_runs), "rrf": to_docs(rrf_runs),
            "candidates": dense_runs[DEMO_ID][:6]}

rankings = _cached.run("06-dense-bm25-rrf", rank_three_ways)

results = {
    "dense only":  metrics.evaluate(rankings["dense"], questions),
    "BM25 only":   metrics.evaluate(rankings["bm25"], questions),
    "RRF of both": metrics.evaluate(rankings["rrf"], questions),
}
print(metrics.compare(results, questions))

# %% [markdown]
# ## Fusion made it worse
#
# Dense alone: 0.700 on the recorded run. Fusing in BM25: 0.450. Five questions of hit@1, and 0.210
# of MRR, paid for nothing — this is the one comparison today whose margin is far wider than the
# one-question noise floor, so it is the one you can lean on whatever your own dense column says.
#
# Reciprocal rank fusion has no way to know which of its inputs is trustworthy. It rewards
# documents that both rankings agree on, which is a good rule when both rankings are reasonable.
# BM25 over these chunks is not reasonable — look at its `tr_en` row. It scores 0.000, because a
# Turkish query and an English chunk share no tokens to count, and RRF inherits that 0.000 exactly.
# Blending a systematically wrong ranking into a good one drags the good one down.
#
# Worth noting what changed. Over **whole documents** in notebook 04, BM25 scored 0.750 on the
# exact-token questions — not a win over dense retrieval, which was at 1.000 there, but close
# enough to be useful, and it put the bulletin id `SCB-2026-0914` first where dense buried it at
# rank 6. The chunking we did in notebook 05 is what cost it: over these chunks BM25's exact-token
# row is 0.500 and its overall hit@1 falls from 0.400 to 0.300. Two improvements that each helped
# alone, interfering with each other.

# %% [markdown]
# ## Reranking
#
# Different idea. Take the top few chunks and have a language model judge how well each one
# answers the question, then reorder by that judgement.
#
# There is no library here and no magic. A pointwise reranker is a for-loop that asks the model
# one question per candidate. Here is the whole of it, on one query, with the scores printed.

# %%
question = next(q for q in questions if q["id"] == DEMO_ID)
candidates = rankings["candidates"]
gold = question["gold_doc_ids"][0]

print(question["query"], "\n")
for position, chunk_id in enumerate(candidates, 1):
    print(f"  {position}. {chunk_id}")
print(f"\ngold document is {gold}, currently at rank "
      f"{rankings['dense'][DEMO_ID].index(gold) + 1} of {len(rankings['dense'][DEMO_ID])}")

# %% [markdown]
# This is the prompt the reranker sends, once per candidate. Read it before running the cell —
# there is nothing else to a reranker than this.

# %%
print("system:", R.RERANK_SYSTEM)
print("\nuser  : QUERY: " + question["query"])
print("        PASSAGE:\n" + "\n".join("        " + line
                                        for line in chunks[candidates[0]][:300].splitlines()))
print("        Score:")

# %%
def score_every_candidate() -> dict:
    """One model call per candidate, exactly as `R.pointwise_rerank` does it internally."""
    start, rows = time.time(), []
    for chunk_id in candidates:
        reply = R.generate(
            f"QUERY: {question['query']}\n\nPASSAGE:\n{chunks[chunk_id][:900]}\n\nScore:",
            system=R.RERANK_SYSTEM, max_tokens=6)
        digits = "".join(c for c in reply if c.isdigit())
        rows.append({"chunk": chunk_id, "reply": reply.strip(),
                     "score": min(int(digits), 10) if digits else 0})
    return {"rows": rows, "seconds": round(time.time() - start, 1)}

scored = _cached.run("06-pointwise-scores-one-question", score_every_candidate)

for position, row in enumerate(scored["rows"], 1):
    print(f"  {position}. {row['score']:>2}/10   {row['chunk']}")
print(f"\n  {len(scored['rows'])} candidates, {len(scored['rows'])} model calls, "
      f"{scored['seconds']:.1f}s")

# %% [markdown]
# Two things to notice in that column of scores.
#
# The scores are **coarse**. A 3B model asked for an integer from 0 to 10 gives you a handful of
# distinct values and a lot of ties, and ties keep the retriever's original order. This reranker
# can only move a document when the model actually has an opinion, and it usually does not have
# a strong one.
#
# And it is **not free**: one model call per candidate, per question, every time anyone asks
# anything. At depth 8 that is eight calls on top of a retrieval that took milliseconds.

# %%
reordered = [row["chunk"] for row in sorted(scored["rows"],
                                            key=lambda r: -r["score"])]
print("before:", [c.split("#")[0] for c in candidates])
print("after :", [c.split("#")[0] for c in reordered])

# %% [markdown]
# ## The other way to ask, in one call
#
# Hand the model all six passages at once and ask it to put them in order. It looks like the
# obvious saving: one call instead of six.

# %%
def listwise_ranking() -> str:
    return R.generate(
        "QUERY: " + question["query"] + "\n\nPASSAGES:\n" +
        "\n\n".join(f"[{i+1}] {chunks[c][:600]}" for i, c in enumerate(candidates)) +
        "\n\nRanking:",
        system=("You are a relevance ranker. Output ONLY the passage numbers reordered from most "
                "to least relevant, comma-separated. Nothing else."), max_tokens=40).strip()

listwise = _cached.run("06-listwise-ranking-one-question", listwise_ranking)
print("listwise says:", listwise)
print(f"we gave it {len(candidates)} passages")

# %% [markdown]
# Count the numbers it returned against the number of passages we gave it. On the recorded run it
# came back with four indices for six passages — there is no ranking there to use, and two
# passages have silently vanished. Ask the same model about one passage at a time and it returns
# six integers in six calls, which is what the cell above did. That is why the pointwise loop is
# what the rest of the course uses.
#
# **Be fair to the listwise version before retiring it.** The two paths do not have the same
# robustness, and that is our code, not the model's judgement. `R.pointwise_rerank` extracts digits
# with a fallback to 0 and breaks ties on the retriever's original position, so at worst it
# degrades to doing nothing. The listwise cell above has no parser and no fallback at all, so a
# reply that drops passages is scored as a ranking failure rather than a format failure. What this
# comparison measures is the shape of the question *and* the robustness of the code around it, and
# we have not separated the two.
#
# `UNVERIFIED: the older listwise-vs-pointwise head-to-head (2/5 against 5/5) — it was measured on
# a retired 10-document, 5-question probe corpus and no command in this repository reproduces it.`
# The mechanism above is live and reproducible; those two figures are not, so do not quote them.
#
# The mechanism is now on the table. The question is whether it is worth paying for.

# %% [markdown]
# ## The measurement: does it help?
#
# The wrong version of this experiment is "run the reranker on our best setup and see". We could
# do that here — twenty questions at eight candidates, 160 model calls — and it would tell us the
# reranker made things worse, and we would conclude that reranking is snake oil.
#
# It is not. That experiment asks "does reranking help?" when the question is "help **what**?"
#
# So the exercise runs the same reranker over four retrieval setups of deliberately different
# quality: a weak embedder and a strong one, crossed with bad chunking and good chunking.
# **Commit to a guess first: does it help all four, none, or some?**
#
# ```bash
# python exercises/m9_rerank_trade.py            # 640 model calls, about five minutes
# python exercises/m9_rerank_trade.py --quick    # the strong setups only, clearly labelled
# ```
#
# Twenty questions at eight candidates is 160 calls per setup, and there are four setups. It
# prints each setup's own cost in calls and seconds as it finishes, so the price of the
# technique is on the same line as its benefit: 51, 87, 49 and 66 seconds on the machine this was
# recorded on — 253 s of model time, and considerably longer on a CPU-only laptop. The call count
# is fixed; only the clock moves with the hardware.
#
# It needs `nomic-embed-text` as well as `bge-m3` — that is the weak embedder, and it is on the
# pre-work pull list for this reason.

# %% [markdown]
# ## A reranker is a trade, not an upgrade
#
# The split in that table is clean, and it is not about the chunking. It is the embedder. The
# reranker **helped** both setups built on the weak embedder — hit@1 0.300 to 0.400 and 0.350 to
# 0.400 — and **hurt** both setups built on the good one: 0.650 to 0.450, and 0.700 to 0.550. MRR
# moved the same way in all four, without a single exception.
#
# The mechanism is the one you just watched: a 3B model scoring passages from 0 to 10 produces a
# coarse judgement. Where your retriever was already right, imposing that judgement is noise, and
# noise demotes the right answer. Where your retriever was wrong, the same coarse judgement is
# still better than what you had.
#
# **The reranker levels.** Look at the spread rather than the individual rows. Before reranking
# the four setups ran from 0.300 to 0.700 — a range of 0.400. After, they run from 0.400 to
# 0.550, a range of 0.150. The same model pulled the bottom up and the top down, toward whatever
# its own opinion happens to be worth.
#
# So the rule is not "add a reranker". It is: **a reranker pays when your retriever is worse than
# your reranker.** Fix the embedder first, then measure whether you still need one.
#
# The per-question view the exercise prints at the end is the honest version of the same thing.
# On the strongest setup, fourteen of the twenty questions already had the gold document at rank 1
# and reranking pushed four of them down — q06, q10, q16, q18. Six did not start at rank 1, and it
# pulled two of those up, q14 and q20. That is the whole trade, question by question: you buy a
# couple of repairs by breaking things that were not broken.

# %% [markdown]
# ## Before you take this away as a general result
#
# Twenty questions is enough to choose between two designs for this corpus. It is not enough to
# claim anything about reranking in general, and a difference under about 0.05 here is inside the
# noise of the sample.
#
# What does generalise is the method: pick a fixed question set, score three numbers, and run the
# comparison before you adopt the technique. Every one of these tools has a precondition, and the
# material that sells them to you rarely mentions what it is.

# %% [markdown]
# ## What is still broken
#
# Look at the `multi_hop` row in any of these tables, and then distrust it. Getting one of the
# three documents a question needs to rank 1 scores as a hit, and the answer still needs all
# three. Two questions in this set are shaped that way, and nothing we have tried moves them.
#
# > *"XX 1487 is delayed, my CLASSIC class K passenger misconnects onto YY 88 at CDG — can I
# > rebook, and what do they get?"*
#
# The procedure is in one document, whether the partner segment is protected at all is in a
# second, and the penalty is in a third. A single search never brings back all three — notebook 07
# measures it and finds none of the three for one of these questions and two of three for the
# other. Ranking them better does not help, because the problem is not the order — it is that one
# search cannot express this question.
#
# > **Measurably, none of them beat plain retrieval here. But multi-hop still fails.**
