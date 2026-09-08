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
sys.path[:0] = [".", "notebooks"]
# Only the chat model: every cell here either cuts strings or asks one question. The twenty-question
# measurement lives in `exercises/m7_chunking_ladder.py`, which is where the embedding calls are.
import _preflight; _preflight.ready(chat=True, replayable=True)
import _cached
import retrieval as R, chunking as C

CORPUS = Path("../corpus/2026-Q3")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
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
ASK = "CLASSIC K sinifi iptal cezasi kac euro?"
answer = _cached.run("05-answer-from-headerless-chunk", lambda: R.generate(
    f"SOURCE:\n{k_chunk}\n\nQUESTION: {ASK}", system="Answer only from the source. Be brief."))
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
# Cut there instead, and prefix every chunk with the section heading it came from.

# %%
pieces = C.structure_aware(sheet, size=900, title="fare classic shorthaul")
k = next((n for n, p in enumerate(pieces) if "| K |" in p and "EUR 90" in p), None)
h = next((n for n, p in enumerate(pieces) if "Booking class" in p and "Cancellation penalty" in p), None)
print(f"structure-aware -> {len(pieces)} chunks · header in {h}, K row in {k} · "
      f"{'FIXED' if k == h else 'still broken'}\n")
print(pieces[k][:600])

# %%
answer = _cached.run("05-answer-from-structure-aware-chunk", lambda: R.generate(
    f"SOURCE:\n{pieces[k]}\n\nQUESTION: {ASK}", system="Answer only from the source. Be brief."))
print(answer)

# %% [markdown]
# Same model, same question, same document. The only thing that changed is where we cut.
#
# Notice the `[RULE 2A. Reading the schedule ...]` prefix on the chunk: the section heading the
# table sits under, carried into the chunk itself. That line costs almost nothing and tells the
# retriever what this grid of numbers is *about* — which is all that "contextual retrieval" means
# when you strip the branding off it.

# %% [markdown]
# ## Part 4 — one repaired answer is an anecdote
#
# We fixed one question by hand. That proves nothing about the other nineteen, and it certainly
# does not prove that structure-aware chunking is the right default.
#
# So measure it — but not in this notebook. Scoring five chunking strategies over the twenty
# gold questions means embedding the corpus five times, a thousand chunks in all, and there is
# nothing to learn from watching that happen. It is one command:
#
# ```bash
# python exercises/m7_chunking_ladder.py
# ```
#
# About a minute on an M-series Mac and longer on a CPU-only laptop, all of it embedding calls.
# One table: five ways of cutting the same 28 documents, scored with the same three numbers we
# used in notebook 04, broken down by question type, plus a sixth row with the boilerplate
# stripped. Run it now and we will read it together.

# %% [markdown]
# ## What that table says, and what it hides
#
# Down the hit@1 column it is the story everyone expects: whole documents 0.550, fixed-280 0.700,
# structure-aware 0.800. MRR moves 0.654 to 0.844. Chunking is the largest single improvement
# available in this entire course, and it costs nothing at query time.
#
# Now read the rest of it, because two columns disagree with the headline.
#
# **Exact tokens got worse before they got better.** Whole documents score 1.000 on the
# exact-token questions. Fixed-280 drops that to 0.750 while the average was going up. A bulletin
# id means something because of the document it sits in; cut that document into fragments and the
# fragment carrying the id is no longer obviously about that flight. Structure-aware chunking gets
# it back to 1.000.
#
# **recall@5 is highest for the worst strategy.** Fixed-280 produces 294 chunks against
# structure-aware's 154, so the right document has far more chances to appear somewhere in the
# top five — 0.950 against 0.833 — while being ranked first less often. If you had picked
# recall@5 as your metric you would have shipped the worse system and had a number to justify it.
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
#
# **Guess first: how much of the corpus is boilerplate?** Then run the cell — it is a string
# operation, no model involved.

# %%
clean = {name: C.strip_boilerplate(text) for name, text in docs.items()}
raw_chars = sum(map(len, docs.values()))
clean_chars = sum(map(len, clean.values()))
print(f"  {raw_chars:,} characters -> {clean_chars:,} characters "
      f"({100 * (raw_chars - clean_chars) / raw_chars:.1f}% removed)")

# %% [markdown]
# Four per cent, on a corpus written to be dirtier than most exports.
#
# The last row of the ladder you already ran is that same corpus, cleaned and cut structure-aware:
# hit@1 0.800 to 0.850, MRR 0.844 to 0.869. One question out of twenty, which is right at the edge
# of what twenty questions can tell apart, and a fifth of what chunking alone was worth.
#
# So the order matters more than the size. Fix the cut first, then clean; cleaning a badly chunked
# index buys you a fraction of what re-cutting it does.

# %% [markdown]
# ## Where we are
#
# The ladder went from 0.550 to 0.800 by changing nothing except where we cut the documents. No
# new model, no extra call, no cost at query time.
#
# But four of the twenty questions still do not put the right document first, and the two that
# need three documents at once are not going to be fixed by better chunking at all.
#
# > **hit@1 is 0.800 — but three different naive splitters all stopped at 0.700, and the metric
# > that liked fixed-280 best was recall@5. What else is the average hiding?**
