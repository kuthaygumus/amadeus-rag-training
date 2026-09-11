# %% [markdown]
# # 05 · Chunking, noise, and measuring the difference
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
# Start with the size of a step. Twenty questions means hit@1 can only move in multiples of 0.05,
# because 0.05 *is* one question. Every gap on this ladder is one or two questions wide. Read the
# direction; do not read the digit.
#
# Down the hit@1 column the direction is the one you would hope for: whole documents 0.600,
# fixed-280 0.650, structure-aware 0.700. MRR moves 0.669 to 0.772. That is two questions of hit@1
# for a change that costs nothing at query time — worth having, and not the landslide the tutorials
# promise.
#
# Now read the rest of it, because three rows disagree with the headline.
#
# **Overlap goes backwards.** Adding 60 characters of overlap to fixed-280 *drops* hit@1 from 0.650
# to 0.500 — three questions — and MRR from 0.765 to 0.672, while *raising* recall@5 from 0.850 to
# 0.867. The move everyone reaches for first is the only rung on the ladder that loses ground.
#
# **Exact tokens got worse before they got better.** Whole documents score 1.000 on the
# exact-token questions. Fixed-280 drops that to 0.750 while the average was going up, and
# fixed-280-with-overlap to 0.500. A bulletin id means something because of the document it sits
# in; cut that document into fragments and the fragment carrying the id is no longer obviously
# about that flight. Recursive-600 and structure-aware get it back to 1.000.
#
# **The best recall@5 belongs to a strategy that ranks worse.** Recursive-600 tops that column at
# 0.883, sitting at hit@1 0.650 — level with plain fixed-280 and below structure-aware's 0.700,
# which has the *lowest* recall@5 of any strategy that cuts at all, 0.817. More, smaller pieces
# give the right document more chances to appear somewhere in the top five while making it harder
# to rank first. Pick recall@5 as your metric and you ship the worse system with a number to
# justify it.
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
# Measure the footer rather than asserting its size: it is the same block on every sheet that
# carries it, and its length is the thing the next paragraph is about.
footer = next(t[t.index("LEGAL NOTICE"):] for t in docs.values() if "LEGAL NOTICE" in t)
print(f"  {'legal footer, characters':<24} {len(footer)}  (identical on each of "
      f"{noise['repeated legal footer']} sheets)")

# %% [markdown]
# Why it matters for retrieval: every fare sheet ends with the same block of legal boilerplate —
# the cell above prints how long it is. Under fixed-size chunking that produces one near-identical
# chunk per sheet whose only distinguishing content is the footer, and they compete for space in
# your top-k while carrying no answer to anything.
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
# hit@1 0.700 to 0.750, MRR 0.772 to 0.796, 154 chunks down to 153. One question out of twenty —
# right at the edge of what twenty questions can tell apart, and half of what changing the cut was
# worth. The whole gain lands in one category: `en_en` goes 0.500 to 0.750 and nothing else moves.
#
# So the order matters more than the size. Fix the cut first, then clean; cleaning a badly chunked
# index buys you a fraction of what re-cutting it does.

# %% [markdown]
# ## Where we are
#
# The ladder went from 0.600 to 0.700 by changing nothing except where we cut the documents, and
# to 0.750 by then deleting the legal footer. No new model, no extra call, no cost at query time.
#
# But six of the twenty questions still do not put the right document first, and the two that
# need three documents at once are not going to be fixed by better chunking at all — `multi_hop`
# reads 0.000 on every rung of that ladder.
#
# > **hit@1 is 0.700, and every gap on the ladder is one or two questions wide. Adding overlap made
# > it worse. The best recall@5 belongs to a strategy that ranks worse. What else is the average
# > hiding?**
