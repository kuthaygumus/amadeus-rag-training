# %% [markdown]
# # 03 · Just put the whole thing in the prompt
#
# > **Helios Air is a fictional airline.** Everything here is synthetic training material.
#
# The fine-tuned model knew last quarter's rules and could not tell us where it got them. Fine.
# Skip the training entirely: paste the rule book into the question.
#
# This is not a straw man. It is the right first answer, it works, and knowing exactly where it
# stops working is the reason the rest of the day exists.

# %%
import sys, time
from pathlib import Path
sys.path.insert(0, "../eval")
import retrieval as R

CORPUS = Path("../corpus/2026-Q3")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
everything = "\n\n".join(f"[SOURCE: {name}.md]\n{text}" for name, text in docs.items())
print(f"{len(docs)} documents · {len(everything):,} characters · {len(everything)/1024:.0f} KB")

# %% [markdown]
# ## Does it even fit?
#
# The usual claim is that you hit a context limit. Check it rather than repeating it.

# %%
info = R.__dict__["_post"]("show", {"model": R.CHAT_MODEL})
context_length = next((v for k, v in info.get("model_info", {}).items() if k.endswith("context_length")), None)
rough_tokens = len(everything) // 3        # ~3 characters per token for mixed EN/TR text
print(f"model context window : {context_length:,} tokens" if context_length else "context window: unknown")
print(f"our corpus           : ~{rough_tokens:,} tokens")
print(f"                       {'FITS' if context_length and rough_tokens < context_length else 'check'}")

# %% [markdown]
# It fits, comfortably. So the honest version of this module is not "you will hit a wall".
#
# **Before running the next cell, guess:** with all 28 documents in front of it, does the model
# get the answer right?

# %%
question = "CLASSIC K sinifi iptal cezasi kac euro?"
start = time.time()
answer = R.generate(
    f"SOURCES:\n{everything}\n\nQUESTION: {question}",
    system="Answer only from the sources. Cite the source filename. Be brief.")
stuffed_seconds = time.time() - start
print(f"{answer}\n\n({stuffed_seconds:.1f} seconds — the correct answer is EUR 90)")

# %% [markdown]
# ## Careful — it depends which language you ask in
#
# Asked in Turkish, the model cited a *different* fare sheet: the corpus holds six of them, three
# fare families each in a short-haul and a long-haul edition, and it quoted the long-haul one.
# Ask the same thing in English and it gets it right.
#
# That is a real difference and it is worth noticing, but it is one question. Before building an
# argument on it, measure.

# %% [markdown]
# ## The measurement that decides this module
#
# Eight questions whose answers are single verifiable values in the corpus, asked at four context
# sizes. **Guess first: does the model answer better with less context or with more?**
#
# Everyone expects less. A model handed twenty thousand tokens of mostly irrelevant text has more
# plausible-looking wrong material to pick from — that is the standard argument for retrieval, and
# it is repeated everywhere.

# %%
CASES = [
    ("CLASSIC short-haul, booking class K: cancellation penalty in EUR?", "90",  ["70", "195", "120", "155"]),
    ("CLASSIC short-haul, booking class K: change penalty in EUR?",       "70",  ["90", "180", "155"]),
    ("CLASSIC short-haul, booking class M: cancellation penalty in EUR?", "120", ["90", "240", "175"]),
    ("CLASSIC short-haul, booking class O: cancellation penalty in EUR?", "60",  ["40", "120"]),
    ("CLASSIC short-haul, booking class K: no-show penalty in EUR?",      "180", ["90", "70"]),
    ("Under the CURRENT misconnect SOP, meal voucher value in EUR?",      "15",  ["10"]),
    ("Under the CURRENT misconnect SOP, hotel is offered after how many hours?", "6", ["8"]),
    ("Flight H9 1487 IST-CDG: what is the NEW departure time?",           "11:20", ["08:35"]),
]

import chunking as C
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)
top = lambda q, k: "\n\n".join(f"[SOURCE: {c.split('#')[0]}.md]\n{chunks[c]}" for c in dense.rank(q)[:k])

conditions = [("top-1 chunk", lambda q: top(q, 1)), ("top-3 chunks", lambda q: top(q, 3)),
              ("top-5 chunks", lambda q: top(q, 5)), ("whole corpus", lambda q: everything)]
tally = {name: 0 for name, _ in conditions}

print(f"{'question':<50}" + "".join(f"{n:>14}" for n, _ in conditions))
print("-" * 106)
for q, right, wrong in CASES:
    row = ""
    for name, build in conditions:
        a = R.generate(f"SOURCES:\n{build(q)}\n\nQUESTION: {q}",
                       system="Answer ONLY from the sources. Give the number. Be brief.", max_tokens=80)
        ok = right in a and not any(w in a for w in wrong)
        tally[name] += ok
        row += f"{'OK' if ok else 'X':>14}"
    print(f"{q[:48]:<50}{row}")
print("-" * 106)
print(f"{'correct':<50}" + "".join(f"{str(tally[n]) + '/8':>14}" for n, _ in conditions))
print(f"{'characters of context':<50}" +
      "".join(f"{len(build(CASES[0][0])):>14,}" for _, build in conditions))

# %% [markdown]
# ## More context answered better, not worse
#
# Monotonic, in the direction opposite to the folklore. On this corpus, with this model, there is
# no distractor penalty to find.
#
# Part of the low end is a recall problem rather than a reading problem — a single 239-character
# chunk often does not contain the answer at all. But the comparison that matters is top-5 against
# everything, and everything still wins.
#
# So be careful with the argument you make here. **Retrieval does not earn its place on this corpus
# by making answers better.** If we told the room it did, someone would run this cell and catch us.

# %% [markdown]
# Look at what it got wrong, too. Asked for the class K *change* penalty it answered `EUR 155` — a
# number that appears in no cell of that table. Given a nine-column filed tariff it does not simply
# read the wrong column; sometimes it produces a value between two of them.
#
# That is a failure of reading, not of retrieval. No amount of better search fixes it, and it is
# worth remembering when the day gets enthusiastic about retrieval metrics: a perfect retrieval
# score still hands the answer to a model that has to read a table.

# %% [markdown]
# ## So why not just stuff the prompt?
#
# Three reasons, and accuracy is not among them.

# %%
one_doc = f"[SOURCE: fare_classic_shorthaul.md]\n{docs['fare_classic_shorthaul']}"
print(f"  whole corpus : {len(everything):>7,} chars  ~{len(everything)//3:>6,} tokens")
print(f"  top-5 chunks : {len(top(CASES[0][0], 5)):>7,} chars  ~{len(top(CASES[0][0], 5))//3:>6,} tokens")
print(f"  ratio        : {len(everything)/len(top(CASES[0][0], 5)):>7.0f}x")

# %% [markdown]
# **One — the tokens.** Roughly thirty times as many per question. On a hosted API that is the
# invoice, directly. Locally it is memory, and it is per concurrent user, which is what a call
# centre is.
#
# **Two — the cold query.** The first stuffed question took **75.8 seconds** here, because every
# token has to be processed before generation starts. Ask a second question against the same corpus
# and it is fast: the server keeps the processed prefix cached. So the latency argument is weaker
# than it looks on one laptop — but the cache turns over on every corpus reissue, every restart,
# and every user with a different prefix.
#
# **Three — it does not scale, and that one is arithmetic.**

# %%
per_query_tokens = rough_tokens
print(f"{'corpus':>16}  {'tokens/query':>14}  fits in {context_length:,}?")
for factor, label in [(1, "ours (28 docs)"), (10, "280 docs"), (100, "2,800 docs"), (1000, "28,000 docs")]:
    n = per_query_tokens * factor
    print(f"{label:>16}  {n:>14,}  {'yes' if context_length and n < context_length else 'NO':>6}")

# %% [markdown]
# One order of magnitude and the window is gone. Ours is 28 documents; a real airline rule book is
# tens of thousands, plus every bulletin ever issued.
#
# The order the failures actually arrive in: **token cost first, the context limit second, and
# answer quality not at all — until the corpus is big enough that the right document cannot be in
# the prompt in the first place.** That last clause is the whole argument, and it is honest.

# %% [markdown]
# ## What we actually want
#
# Not "give the model everything", because at ten times this size we cannot. Not "bake it into the
# weights", because they freeze. Something narrower:
#
# > At the moment a question is asked, find the small part of the corpus that answers it, and send
# > only that.
#
# Which leaves the only hard part, and the one the rest of the day is about.
#
# **How do you find the right piece?**
#
# > **It fit, and it answered. But I paid for the entire book to do it, and at ten times this size it will not fit at all.**
