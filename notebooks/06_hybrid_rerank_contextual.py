# %% [markdown]
# # 06 · Hybrid search and reranking — when do they actually pay?
#
# > **Helios Air is a fictional airline.** Everything here is synthetic training material.
#
# We are at hit@1 0.800 with structure-aware chunking and a multilingual embedder. Four questions
# out of twenty still miss.
#
# The standard advice for that gap is: add keyword search alongside the embeddings, fuse the two
# rankings, then rerank the survivors with a second model. Three techniques, every RAG blog post,
# always framed as improvements.
#
# We are going to run all three and measure them. **Commit to a guess now: how many of the three
# will beat plain retrieval on this corpus?**

# %%
import sys, time
from pathlib import Path
sys.path.insert(0, "../eval")
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")

chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)
bm25 = R.BM25(chunk_ids, chunk_texts)
print(f"{len(chunk_ids)} chunks indexed both ways")

# %%
dense_runs = {q["id"]: dense.rank(q["query"]) for q in questions}
bm25_runs = {q["id"]: bm25.rank(q["query"]) for q in questions}
rrf_runs = {q["id"]: R.rrf([dense_runs[q["id"]], bm25_runs[q["id"]]]) for q in questions}

results = {
    "dense only": metrics.evaluate({k: C.to_documents(v) for k, v in dense_runs.items()}, questions),
    "BM25 only":  metrics.evaluate({k: C.to_documents(v) for k, v in bm25_runs.items()}, questions),
    "RRF of both": metrics.evaluate({k: C.to_documents(v) for k, v in rrf_runs.items()}, questions),
}
print(metrics.compare(results, questions))

# %% [markdown]
# ## Fusion made it worse
#
# Dense alone: 0.800. Fusing in BM25: 0.450.
#
# Reciprocal rank fusion has no way to know which of its inputs is trustworthy. It rewards
# documents that both rankings agree on, which is a good rule when both rankings are reasonable.
# BM25 over these chunks is not reasonable — look at its `tr_en` row. It scores zero, because a
# Turkish query and an English chunk share no tokens to count. Blending a systematically wrong
# ranking into a good one drags the good one down.
#
# Worth noting what changed: in notebook 04, BM25 over **whole documents** beat dense retrieval
# on exact tokens. The chunking we did in notebook 05 is what destroyed it — short chunks give
# BM25 too little text for its length normalisation to mean anything. Two improvements that each
# helped alone, interfering with each other.

# %% [markdown]
# ## Reranking
#
# Different idea. Take the top few chunks and have a language model judge how well each one
# answers the question, then reorder by that judgement.
#
# There are two ways to ask. Hand the model all six and say "put these in order", or ask about
# one chunk at a time and give it a score. Try both on a single question first.

# %%
question = next(q for q in questions if q["id"] == "q05")
candidates = dense_runs[question["id"]][:6]
gold = question["gold_doc_ids"][0]
print(question["query"], "\n")
print(f"before: {[c.split('#')[0] for c in candidates]}")
print(f"gold document is {gold}, currently at rank "
      f"{C.to_documents(dense_runs[question['id']]).index(gold) + 1}")

# %%
listwise = R.generate(
    "QUERY: " + question["query"] + "\n\nPASSAGES:\n" +
    "\n\n".join(f"[{i+1}] {chunks[c][:600]}" for i, c in enumerate(candidates)) +
    "\n\nRanking:",
    system=("You are a relevance ranker. Output ONLY the passage numbers reordered from most to "
            "least relevant, comma-separated. Nothing else."), max_tokens=40)
print("listwise says:", listwise.strip())

reordered = R.pointwise_rerank(question["query"], candidates, chunks)
print("pointwise gives:", [c.split("#")[0] for c in reordered])

# %% [markdown]
# ## Score the whole set — pointwise, on top of our best retrieval

# %%
start = time.time()
reranked = {}
for n, q in enumerate(questions, 1):
    top = dense_runs[q["id"]][:8]
    order = R.pointwise_rerank(q["query"], top, chunks)
    reranked[q["id"]] = order + [c for c in dense_runs[q["id"]] if c not in order]
    print(f"\r  {n}/{len(questions)}", end="", flush=True)
print(f"\r  {len(questions)} questions reranked in {time.time() - start:.0f}s "
      f"— 8 model calls each, {time.time() - start and (time.time() - start)/len(questions):.1f}s per question")

results["+ pointwise rerank"] = metrics.evaluate(
    {k: C.to_documents(v) for k, v in reranked.items()}, questions)
print("\n" + metrics.compare({k: results[k] for k in ["dense only", "RRF of both", "+ pointwise rerank"]},
                             questions))

# %% [markdown]
# Reranking made it worse too.
#
# So on this corpus, none of the three techniques beat what we already had. That is not the
# result the blog posts promise, and it would be easy to conclude that reranking is snake oil.
#
# It is not. We measured the wrong thing.

# %% [markdown]
# ## The right experiment
#
# We tested "does reranking help?" What we should have tested is "help *what*?"
#
# Run the same reranker over four retrieval setups of deliberately different quality: a weak
# embedder and a strong one, crossed with bad chunking and good chunking. **Guess first: does the
# reranker help all four, none, or some?**

# %%
def measure(strategy: str, model: str):
    ids, texts, _ = C.chunk_corpus(docs, strategy)
    pieces = dict(zip(ids, texts))
    retriever = R.DenseRetriever(ids, texts, model=model)
    base = {q["id"]: retriever.rank(q["query"]) for q in questions}
    after = {}
    for q in questions:
        order = R.pointwise_rerank(q["query"], base[q["id"]][:8], pieces)
        after[q["id"]] = order + [c for c in base[q["id"]] if c not in order]
    score = lambda r: metrics.evaluate({k: C.to_documents(v) for k, v in r.items()}, questions)
    return score(base), score(after)

print(f"{'retrieval setup':<38}{'hit@1':>16}{'MRR':>18}")
print(f"{'':<38}{'before':>8}{'after':>8}{'before':>9}{'after':>9}   verdict")
print("-" * 92)
for label, strategy, model in [
    ("weak   : nomic + fixed-280",       "fixed-280",       "nomic-embed-text"),
    ("weak   : nomic + structure-aware", "structure-aware", "nomic-embed-text"),
    ("strong : bge-m3 + fixed-280",      "fixed-280",       "bge-m3"),
    ("strong : bge-m3 + structure-aware","structure-aware", "bge-m3"),
]:
    before, after = measure(strategy, model)
    delta = after["MRR"] - before["MRR"]
    verdict = "HELPED" if delta > 0.02 else ("hurt" if delta < -0.02 else "no change")
    print(f"{label:<38}{before['hit@1']:>8.3f}{after['hit@1']:>8.3f}"
          f"{before['MRR']:>9.3f}{after['MRR']:>9.3f}   {verdict} ({delta:+.3f})")

# %% [markdown]
# ## A reranker is a trade, not an upgrade
#
# The split is clean, and it is not about the chunking. It is the embedder. The reranker helped
# both setups using the weak embedder and hurt both setups using the good one.
#
# The mechanism is worth saying out loud, because it generalises. A 3B model scoring passages
# from 0 to 10 produces a coarse judgement — lots of ties, limited resolution. Where your
# retriever was already right, imposing that judgement is pure noise and it demotes the right
# answer. Where your retriever was wrong, the same coarse judgement is still better than what
# you had.
#
# **The reranker levels, and it levels to its own ceiling.** It pulled both weak setups up to
# roughly 0.45 hit@1 and dragged both strong ones down toward the same place. That number is
# roughly what this model's opinion is worth. If your retrieval is already better than your
# reranker's opinion, adding it can only lose.
#
# So the rule is not "add a reranker". It is: **a reranker pays when your retriever is worse than
# your reranker.** Fix the embedder first, then measure whether you still need one.
#
# And the price is real. Eight model calls and about four seconds per query, on top of retrieval
# that took milliseconds — paid on every question, to make this system worse.

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
# Look at the `multi_hop` row in any of these tables. Two questions, and nothing we have tried
# moves them.
#
# > *"H9 1487 is delayed, my CLASSIC class K passenger misconnects onto AU 88 at CDG — can I
# > rebook, and what do they get?"*
#
# The procedure is in one document, whether the partner segment is protected at all is in a
# second, and the penalty is in a third. Retrieval returns one of the three. Ranking them better
# does not help, because the problem is not the order — it is that one search cannot express this
# question.
#
# > **Measurably, none of them beat plain retrieval here. But multi-hop still fails.**
