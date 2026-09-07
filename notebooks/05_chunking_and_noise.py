# %% [markdown]
# # 05 · Chunking, noise, and measuring the difference
#
# > **Helios Air is a fictional airline.** Everything here is synthetic training material.
#
# Notebook 04 ended with a wrong answer that came from a *correct* retrieval. The right document
# was ranked first, and the model still said the wrong number. That is the failure we chase here,
# because it is the one nobody warns you about.

# %%
import sys
from pathlib import Path
sys.path.insert(0, "../eval")
import retrieval as R, chunking as C, metrics

CORPUS = Path("../corpus/2026-Q3")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
sheet = docs["fare_classic_shorthaul"]
print(f"{len(docs)} documents · the fare sheet we care about is {len(sheet)} characters")

# %% [markdown]
# ## Part 1 — find the damage
#
# Cut that fare sheet every 280 characters, the way the simplest splitter would, and go looking
# for the row that answers our question.

# %%
pieces = C.fixed(sheet, size=280, overlap=0)
print(f"{len(pieces)} chunks\n")
for n, piece in enumerate(pieces):
    has_row = "| K |" in piece and "EUR 90" in piece
    has_header = "Booking class" in piece and "Cancellation penalty" in piece
    if has_row or has_header:
        print(f"chunk {n}: {'THE K ROW' if has_row else ''}{'THE COLUMN HEADER' if has_header else ''}")

# %% [markdown]
# They are in different chunks.
#
# **Guess before the next cell:** which one gets retrieved, and what does the model do with it?

# %%
k_chunk = next(p for p in pieces if "| K |" in p and "EUR 90" in p)
print(k_chunk)

# %% [markdown]
# Read that as the model reads it. Rows of letters and euro amounts, four numbers per row, and
# nothing anywhere saying which column is the change penalty and which is the cancellation
# penalty. The row survived perfectly. The thing that gave it meaning did not.

# %%
answer = R.generate(
    f"SOURCE:\n{k_chunk}\n\nQUESTION: CLASSIC K sinifi iptal cezasi kac euro?",
    system="Answer only from the source. Be brief.")
print(answer)
print("\nthe correct answer is EUR 90")

# %% [markdown]
# It does not say "I can't tell which column that is". It picks one and states it.
#
# This is the shape of the most expensive kind of RAG bug: retrieval looked fine, the document
# was right, the chunk contained the answer, and the output is wrong with no signal that it is.

# %% [markdown]
# ## Part 2 — the fix that does not work
#
# Everyone's first instinct is overlap: let each chunk carry the tail of the one before it, so
# nothing falls between the cracks. Try it.

# %%
for size, overlap in [(280, 0), (280, 60), (280, 120), (400, 80), (600, 100)]:
    pieces = C.fixed(sheet, size, overlap)
    k = next((n for n, p in enumerate(pieces) if "| K |" in p and "EUR 90" in p), None)
    h = next((n for n, p in enumerate(pieces) if "Booking class" in p and "Cancellation penalty" in p), None)
    together = k is not None and k == h
    print(f"size {size:>3} overlap {overlap:>3} -> {len(pieces):>2} chunks · "
          f"header in chunk {h}, K row in chunk {k} · {'FIXED' if together else 'still broken'}")

# %% [markdown]
# Overlap does not remove the boundary. It moves it. There is still a cut somewhere in that
# table, and the header is still on the wrong side of it.
#
# The second instinct is a recursive splitter — the default in every RAG tutorial — which splits
# on paragraphs before it splits on characters.

# %%
pieces = C.recursive(sheet, size=600)
k = next((n for n, p in enumerate(pieces) if "| K |" in p and "EUR 90" in p), None)
h = next((n for n, p in enumerate(pieces) if "Booking class" in p and "Cancellation penalty" in p), None)
print(f"recursive-600 -> {len(pieces)} chunks · header in {h}, K row in {k} · "
      f"{'FIXED' if k == h else 'STILL BROKEN'}")

# %% [markdown]
# Still broken. The table is a single paragraph and it is wider than the split size, so the
# recursive splitter falls through to cutting it by character count like everything else.
#
# Reaching for the standard tool did not help, because the problem was never the algorithm. It
# was that none of these splitters know what a table is.

# %% [markdown]
# ## Part 3 — splitting on structure
#
# The document already tells us where its own boundaries are: headings, numbered rules, sections.
# Cut there instead, and prefix every chunk with the document and section it came from.

# %%
pieces = C.structure_aware(sheet, size=900, title="fare classic shorthaul")
k = next((n for n, p in enumerate(pieces) if "| K |" in p and "EUR 90" in p), None)
h = next((n for n, p in enumerate(pieces) if "Booking class" in p and "Cancellation penalty" in p), None)
print(f"structure-aware -> {len(pieces)} chunks · header in {h}, K row in {k} · "
      f"{'FIXED' if k == h else 'still broken'}\n")
print(pieces[k][:600])

# %%
answer = R.generate(
    f"SOURCE:\n{pieces[k]}\n\nQUESTION: CLASSIC K sinifi iptal cezasi kac euro?",
    system="Answer only from the source. Be brief.")
print(answer)

# %% [markdown]
# Same model, same question, same document. The only thing that changed is where we cut.
#
# Notice the `[fare classic shorthaul > ...]` prefix on the chunk. That line costs almost
# nothing and tells the retriever what this grid of numbers is *about* — which is all that
# "contextual retrieval" means when you strip the branding off it.

# %% [markdown]
# ## Part 4 — measure it, on all twenty questions
#
# One repaired answer is an anecdote. Score every strategy over the whole gold set, with the same
# three numbers we used in notebook 04.

# %%
runs = {}
ids, texts = list(docs), list(docs.values())
whole = R.DenseRetriever(ids, texts)
runs["no chunking"] = {q["id"]: whole.rank(q["query"]) for q in questions}

for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, strategy)
    retriever = R.DenseRetriever(chunk_ids, chunk_texts)
    runs[strategy] = {q["id"]: C.to_documents(retriever.rank(q["query"])) for q in questions}
    print(f"  {strategy:<22} {len(chunk_ids):>4} chunks")

results = {name: metrics.evaluate(r, questions) for name, r in runs.items()}
print("\n" + metrics.compare(results, questions))

# %% [markdown]
# ## What the table says, and what it hides
#
# hit@1 goes from 0.550 to 0.800 and MRR from 0.655 to 0.846. Chunking is the largest single
# improvement available in this entire course, and it costs nothing at query time.
#
# Now read the rest of it, because two rows disagree with the headline.
#
# **Exact tokens got worse before they got better.** Whole documents scored 1.000 on the
# exact-token questions. Fixed-size chunking halved it. A bulletin id means something because of
# the document it sits in; cut that document into fragments and the fragment carrying the id is
# no longer obviously about that flight. Structure-aware chunking gets it back.
#
# **recall@5 is highest for the worst strategy.** Fixed-280 produces 288 chunks against
# structure-aware's 152, so the right document has far more chances to appear somewhere in the
# top five — while being ranked first less often. If you had picked recall@5 as your metric you
# would have shipped the worse system and had a number to justify it.
#
# One number is never enough. That is not a caveat about this corpus; it is the reason the gold
# set reports three.

# %% [markdown]
# ## Part 5 — noise
#
# The corpus is deliberately dirty, the way exported documents actually are: the same legal
# footer on every fare sheet, `Page 3 of 7` artefacts, leftover HTML, duplicated paragraphs.

# %%
import re
noise = {
    "repeated legal footer": sum("LEGAL NOTICE" in t for t in docs.values()),
    "page-number artefacts": sum(len(re.findall(r"Page \d+ of \d+", t)) for t in docs.values()),
    "leftover HTML":         sum(len(re.findall(r"<(div|br|/div)[^>]*>|&nbsp;", t)) for t in docs.values()),
}
for label, count in noise.items():
    print(f"  {label:<24} {count}")

# %% [markdown]
# Why it matters for retrieval: every fare sheet ends with the same 400 characters of legal
# boilerplate. Under fixed-size chunking that produces six near-identical chunks whose only
# distinguishing content is the footer — they compete for space in your top-k while carrying no
# answer to anything.
#
# Structure-aware chunking already suppresses most of this, because boilerplate lands in its own
# section rather than being smeared across every chunk. Stripping it explicitly is the belt to
# that braces, and on a corpus dirtier than this one it is worth doing on its own.

# %% [markdown]
# ## Where we are
#
# We went from 0.550 to 0.800 by changing nothing except where we cut the documents. No new
# model, no extra call, no cost at query time.
#
# But four of the twenty questions still do not put the right document first, and the two that
# need three documents at once are not going to be fixed by better chunking at all.
#
# > **hit@1 is 0.800. But fixed-280 had halved exact-token retrieval — what else is the average hiding?**
