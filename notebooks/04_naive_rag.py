# %% [markdown]
# # 04 · From keyword search to naive RAG
#
# We know the model does not have our data, and we know we cannot afford to hand it the entire
# rule book on every question. So we need to find the right piece and hand over only that.
#
# Start with the least clever thing that could possibly work.

# %%
import sys, time
from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True)
import _cached
import retrieval as R, metrics

CORPUS = Path("../corpus/2026-Q3")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
doc_ids, texts = list(docs), list(docs.values())
print(f"{len(docs)} documents, {sum(map(len, texts)) / 1024:.0f} KB")

# %% [markdown]
# ## Part 1 — keyword search
#
# BM25 scores a document by how often the query's words appear in it, discounted by how common
# those words are everywhere else, and adjusted for document length. Three ideas, no training,
# no model. `eval/retrieval.py` writes it out rather than importing a library so you can read it.
#
# It should do well on something no embedding will ever nail: an exact flight code.

# %%
bm25 = R.BM25(doc_ids, texts)
for query in ["XX 1487", "SCB-2026-0914"]:
    print(f"{query!r:>18} -> {bm25.rank(query)[:3]}")

# %% [markdown]
# Exactly right, both times. Now the query an actual call-centre agent would type.
#
# **Guess first:** the agent asks, in Turkish, what a passenger pays if they cancel. The answer
# is in an English document that uses the phrase "cancellation penalty". Does BM25 find it?

# %%
paraphrase = "Musteri bileti iptal ederse ne oder?"
ranking = bm25.rank(paraphrase)
print(f"{paraphrase!r}\n  top 3: {ranking[:3]}")
print(f"  where the right document ended up: rank {ranking.index('fare_classic_shorthaul') + 1}")

# %% [markdown]
# The rank is bad. Look at the score behind it, because the score is the argument — a rank of 14
# could mean "nearly, but thirteen documents were slightly better".

# %%
scores = bm25.scores(paraphrase)
best = sorted(zip(scores, doc_ids), reverse=True)[:3]
for score, name in best:
    print(f"  {score:6.3f}  {name}")
print(f"  {scores[doc_ids.index('fare_classic_shorthaul')]:6.3f}  fare_classic_shorthaul"
      "   <-- the document that answers the question")

# %% [markdown]
# **Exactly zero.** Not "ranked low" — scored nothing at all, and the rank it got is just where
# ties land in the sort.
#
# The reason is not subtle. The query tokenises to *musteri, bileti, iptal, ne, oder*. The document
# says *cancellation*, *penalty*, *EUR*. Three of those query words appear nowhere in the corpus at
# all, so BM25 skips them; the other three exist in the Turkish macros but not in this English fare
# sheet, so their term frequency in it is zero and they contribute zero too. Nothing is left to add
# up. BM25 is not bad at meaning — it has no notion of meaning at all.

# %% [markdown]
# ## Part 2 — embeddings
#
# So match on meaning instead. An embedding model turns a piece of text into a list of numbers
# positioned so that things which mean similar things land near each other. Then "find the
# relevant document" becomes "find the nearest vector", which is just arithmetic.

# %%
start = time.time()
dense = R.DenseRetriever(doc_ids, texts)          # bge-m3, 1024 numbers per document
print(f"embedded {len(doc_ids)} documents in {time.time() - start:.1f}s")

ranking = dense.rank(paraphrase)
print(f"\n{paraphrase!r}\n  top 3: {ranking[:3]}")
print(f"  right document: rank {ranking.index('fare_classic_shorthaul') + 1}")

# %% [markdown]
# ## Part 3 — the whole naive pipeline
#
# Retrieve the top few documents, paste them into the prompt, ask the question. That is RAG.
# There is no more to the basic idea than this.

# %%
def naive_rag(question: str, k: int = 3) -> str:
    hits = dense.rank(question)[:k]
    context = "\n\n".join(f"[SOURCE: {h}.md]\n{docs[h][:1500]}" for h in hits)
    answer = R.generate(
        f"SOURCES:\n{context}\n\nQUESTION: {question}",
        system=("Answer ONLY from the sources. Cite the source filename in square brackets. "
                "If the sources do not contain the answer, say you do not know. Be brief."),
    )
    return f"retrieved: {hits}\n{answer}"

print(naive_rag("CLASSIC K sinifi iptal cezasi ne kadar?"))

# %% [markdown]
# It works. The model that invented four different numbers in notebook 00 now reads a document
# and answers from it, with a filename attached.
#
# So we are done — except we have not measured anything, and "it worked on the question I tried"
# is not a result.

# %% [markdown]
# ## Part 4 — measuring it
#
# `eval/gold_questions.jsonl` holds twenty questions with the documents that should come back for
# each. Three numbers, computed the same way every time, no model judging anything:
#
# - **hit@1** — was the top-ranked document a right one?
# - **recall@5** — how much of the right set showed up in the top five?
# - **MRR** — one over the rank of the first right document, averaged. This is the one that
#   shows partial credit: moving a document from rank 6 to rank 2 leaves hit@1 untouched but
#   moves MRR from 0.17 to 0.50.
#
# We will report these same three numbers twice more today, on this same set.

# %%
questions = metrics.load_gold("../eval/gold_questions.jsonl")

def score_both() -> dict:
    return {
        "BM25":   metrics.evaluate({q["id"]: bm25.rank(q["query"]) for q in questions}, questions),
        "bge-m3": metrics.evaluate({q["id"]: dense.rank(q["query"]) for q in questions}, questions),
    }

results = _cached.run("04-bm25-vs-dense-whole-documents", score_both)
print(metrics.compare(results, questions))

# %% [markdown]
# Read the by-type rows, not just the top line.
#
# BM25 scores **zero** on every Turkish question whose answer is in an English document — there
# is no shared token to count — where dense retrieval gets a third of them.
#
# Be careful with the exact-token row, because the folklore has it backwards. BM25 scores 0.750
# there and dense scores **1.000**. A whole document is long enough that the flight code still
# sits in text the embedding can use, so BM25 does not win this category outright. Where it does
# win is one query at a time, and Part 5 below has the example: the bulletin id `SCB-2026-0914`,
# rank 1 for BM25 and rank 6 for dense.
#
# Neither retriever is better outright; they fail at different things, which is worth remembering
# when someone tells you keyword search is obsolete.
#
# And the headline number is not good. Read it off your own run rather than off this line — twenty
# questions means hit@1 moves in steps of 0.05, and embedding output is not bit-identical across
# Ollama builds. On the recorded run dense retrieval put the right document first for twelve of the
# twenty. Eight did not.

# %% [markdown]
# ## Part 5 — four failures you can see
#
# An average tells you something is wrong. It does not tell you what. Look at the actual output.

# %% [markdown]
# **Failure 1 — the confidently wrong answer.**
#
# Ask for the K class cancellation penalty. The right document comes back first. Watch the answer
# anyway.

# %%
print(naive_rag("CLASSIC K sinifi iptal cezasi kac euro?"))

# %% [markdown]
# The retrieval was right and the answer is still wrong. It talked about the wrong booking
# classes, or the wrong route band, or both.
#
# The document is 3,923 characters and we handed over the first 1,500. Run the numbers on where
# that cut falls: the column heading is at character 1,131 and survives, the O and T rows follow
# it, and the K row starts at character 1,743 — outside the slice. So the model was handed a
# labelled table that does not contain the row it was asked about, and it answered from a row
# that was there. It did not hedge. Nothing in the answer marks it as a guess.
#
# Remember this one. Module 7 is about exactly this failure.

# %% [markdown]
# **Failure 2 — two versions of the truth.**

# %%
hits = dense.rank("misconnect meal voucher amount")[:4]
print("retrieved:", hits, "\n")
for h in hits:
    if "misconnect_v" not in h:
        continue
    lines = docs[h].splitlines()
    version = next(l for l in lines if l.startswith("Version:"))
    i = next(n for n, l in enumerate(lines) if "meal voucher" in l)
    amount = " ".join(" ".join(lines[i:i + 2]).split())
    print(f"  {h}\n    {version}\n    {amount}")

# %% [markdown]
# Both revisions of the same procedure came back, and they disagree about the money. One says
# EUR 10, the other EUR 15. Only one of them is in force.
#
# A human reading the two documents would spot `Superseded` in the header. The retriever has no
# concept of a document being superseded — it ranked both by similarity, and similarity is
# exactly what these two have in common.

# %% [markdown]
# **Failure 3 — identifiers blur together.**
#
# Semantic similarity is the whole point of embeddings. It is also the problem, because two
# bulletin numbers that differ by one digit mean completely different things while looking
# almost identical.

# %%
for query, should_be in [("XX 1487", "bulletin_scb_2026_0914"),
                         ("SCB-2026-0914", "bulletin_scb_2026_0914"),
                         ("YY 88", "interline_xx_yy"),
                         ("booking class K", "fare_classic_shorthaul")]:
    d_rank = dense.rank(query); b_rank = bm25.rank(query)
    print(f"{query:<16} dense: rank {d_rank.index(should_be) + 1:<3} (top: {d_rank[0]})")
    print(f"{'':<16} bm25 : rank {b_rank.index(should_be) + 1:<3} (top: {b_rank[0]})")

# %% [markdown]
# The flight number it handles. The bulletin id it does not — it puts a different bulletin first
# and buries the right one. And `booking class K` returns the corporate travel policy, because
# that document also talks about booking classes at length.
#
# This is where the keyword search we discarded twenty minutes ago earns its place back.

# %% [markdown]
# **Failure 4 — the Turkish question that lands somewhere else entirely.**

# %%
question = next(q for q in questions if q["id"] == "q05")   # one of the six tr_en questions
ranking = dense.rank(question["query"])
print(question["query"], "\n")
print(f"  top 3     : {ranking[:3]}")
print(f"  should be : {question['gold_doc_ids'][0]}"
      f"  (came back at rank {ranking.index(question['gold_doc_ids'][0]) + 1})")

# %% [markdown]
# A Turkish agent asks what a passenger waiting four hours is owed. The answer is in an English
# procedure, and the documents that come back ahead of it are about staff expense claims, or a fare
# sheet, or baggage — all of them plausibly "money after a travel disruption", none of them the
# answer. On the recorded run the right procedure came back at rank 18 of 28.
#
# The Turkish query and the English document share almost no vocabulary, so keyword search is
# hopeless here by construction. But the embedding was supposed to bridge that gap, and it only
# partly does. Six of our twenty questions are shaped like this one, and by the by-type table
# above, two thirds of them fail.
#
# Whether that is the embedder's fault or ours is the question module 6 answers.

# %% [markdown]
# ## What we learned
#
# The pipeline is fine. Retrieval is what is broken, in four distinct ways:
#
# 1. we handed the model a slice of a document that lost the heading it needed
# 2. we returned two contradicting versions with nothing to distinguish them
# 3. exact identifiers get smoothed into their neighbours
# 4. a Turkish question does not reach an English document
#
# None of these is fixed by a bigger model, and only the second is really about the corpus. The
# rest are decisions we made about what to put in the index and how to cut it up — which is
# where the next three modules go.
#
# > **RAG works. Retrieval is bringing back garbage.**
