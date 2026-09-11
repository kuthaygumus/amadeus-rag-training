# %% [markdown]
# # 03 · Just put the whole thing in the prompt
#
# The fine-tuned model knew last quarter's rules and could not tell us where it got them. Fine.
# Skip the training entirely: paste the rule book into the question.
#
# This is not a straw man. It is the right first answer, it works, and knowing exactly where it
# stops working is the reason the rest of the day exists.
#
# **This notebook is deliberately slow, because the slowness is the lesson.** Two cells send the
# whole corpus to the model. The corpus is 78,310 characters; the prompt built from it below is
# 79,309, because joining the documents adds a `[SOURCE: name.md]` header to each one. Every one
# of those characters has to be read before a single token comes back. The elapsed seconds are
# printed next to every result — read them, they are the argument.
#
# End to end against a freshly started Ollama it took just under three minutes on the machine it
# was recorded on, almost all of it in two cold calls. Run it a second time and it is a fraction
# of that, because the server keeps what it has already read — which is itself part of the lesson.
# `QUICK=1` halves the question set if the room's laptops are struggling; `USE_CACHED=1` replays
# the recorded run without calling the model at all.

# %%
import sys, time
from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True, replayable=True)
import _cached
import retrieval as R

CORPUS = Path("../corpus/2026-Q3")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
everything = "\n\n".join(f"[SOURCE: {name}.md]\n{text}" for name, text in docs.items())
print(f"{len(docs)} documents · {len(everything):,} characters · {len(everything)/1024:.0f} KB")

# Load the model into memory before anything is timed. What this module turns on is the cost
# of a long prompt, not the cost of starting a 3B model, and the two are easy to confuse.
if not _cached.USE_CACHED:
    R.generate("ready", max_tokens=1)

# %% [markdown]
# ## Does it even fit?
#
# The usual claim is that you hit a context limit. Check it rather than repeating it.

# %%
def model_window() -> dict:
    info = R.__dict__["_post"]("show", {"model": R.CHAT_MODEL})
    return {"context_length": next((v for k, v in info.get("model_info", {}).items()
                                    if k.endswith("context_length")), None)}

context_length = _cached.run("03-model-context-window", model_window)["context_length"]
rough_tokens = len(everything) // 3        # deliberately generous: the real ratio measured
                                           # ~3.75 chars/token, so this over-counts by ~25%
print(f"model context window : {context_length:,} tokens" if context_length else "context window: unknown")
print(f"our corpus           : ~{rough_tokens:,} tokens")
print(f"                       {'FITS' if context_length and rough_tokens < context_length else 'check'}")

# Advertised is not effective. Ollama serves a call at its own default window unless you ask for
# one, and a default of 4,096 would quietly drop most of this prompt and answer confidently off
# whatever was left. So every whole-corpus call below is *served* at a window we asked for, sized
# from the estimate above rather than at the model's maximum: big enough to hold the prompt, small
# enough not to reserve 32k of KV cache on a laptop that has not got it to spare.
STUFFED_CTX = min(context_length, rough_tokens + 1024) if context_length else None

# %% [markdown]
# It fits, comfortably. So the honest version of this module is not "you will hit a wall".
#
# That estimate is deliberately pessimistic. Asked to read this exact prompt, the server reported
# `prompt_eval_count` 21,170 — the tokeniser is more efficient on this text than one token per
# three characters. Both numbers are far inside the window, which is the only thing the check
# needs to establish.
#
# **Before running the next cell, guess:** with all 28 documents in front of it, does the model
# get the answer right?

# %%
ASKED = [("Turkish", "CLASSIC K sinifi iptal cezasi kac euro?"),
         ("English", "CLASSIC class K: what is the cancellation penalty in EUR?")]

def ask_the_whole_corpus() -> list[dict]:
    out = []
    for language, question in ASKED:
        start = time.time()
        answer = R.generate(
            f"SOURCES:\n{everything}\n\nQUESTION: {question}",
            system="Answer only from the sources. Cite the source filename. Be brief.",
            num_ctx=STUFFED_CTX)
        out.append({"language": language, "question": question, "answer": answer,
                    "seconds": round(time.time() - start, 1)})
    return out

stuffed = _cached.run("03-cold-stuffed-query", ask_the_whole_corpus)
for row in stuffed:
    print(f"[{row['language']}, {row['seconds']:.1f}s] {row['answer']}\n")
print("short-haul CLASSIC K cancels for EUR 90; long-haul CLASSIC K cancels for EUR 195")

# %% [markdown]
# ## Two things happened there, and only one of them is about accuracy
#
# **Read the two timings, and read them again on a second run.** They are unstable on purpose.
# A stuffed question against a freshly loaded model has to process all 79,309 characters before
# it can write a single token; that is the slow one, and on the machine this was recorded on it
# was the slowest call in the course. The next question against the same corpus reuses the prefix
# the server has already processed and comes back in a couple of seconds — and so does the *first*
# question if you run this notebook twice in a row, because the cache is still there.
#
# That instability is not noise to be hidden. It is the honest form of the prompt-caching
# objection, and it is why the measurement cell below sends all of its whole-corpus questions
# consecutively instead of interleaving them with short ones.
#
# **Neither answer is wrong, and neither is complete.** The question never said short-haul or
# long-haul. The corpus holds six fare sheets — three fare families, each in a short-haul and a
# long-haul edition — and the model picked one and quoted it, without saying that it had picked.
# That is the failure mode worth naming: not a wrong number, an unmarked choice between two right
# ones. Handing the model everything did not make it ask.
#
# One question is an anecdote either way. Measure.

# %% [markdown]
# ## The measurement that decides this module
#
# Eight questions whose answers are single verifiable values in the corpus, asked at four context
# sizes. **Guess first: does the model answer better with less context or with more?**
#
# Everyone expects less. A model handed twenty thousand tokens of mostly irrelevant text has more
# plausible-looking wrong material to pick from — that is the standard argument for retrieval, and
# it is repeated everywhere.
#
# **One black box, declared.** To compare "the whole book" against "a few short passages" we need
# something that picks the passages. We have not built one — that is the entire second half of the
# day. So the next cell borrows one, and this module deliberately does not explain it: not how it
# cuts the documents up, not how it decides which pieces look relevant, not why the pieces it picks
# are often the wrong ones. Modules 5, 6 and 7 build that machinery one failure at a time, and each
# of those modules is worth more if you meet it as a question rather than as a recap.
#
# For the next twenty minutes it is a function that takes a question and hands back one, three or
# five short passages. **Do not read the passage columns as a verdict on retrieval.** They are here
# to give the last column something to be compared against, and the last column is what this module
# is arguing about.

# %%
CASES = [
    ("CLASSIC short-haul, booking class K: cancellation penalty in EUR?", "90",  ["70", "195", "120", "155"]),
    ("CLASSIC short-haul, booking class K: change penalty in EUR?",       "70",  ["90", "180", "155"]),
    ("CLASSIC short-haul, booking class M: cancellation penalty in EUR?", "120", ["90", "240", "175"]),
    ("CLASSIC short-haul, booking class O: cancellation penalty in EUR?", "60",  ["40", "120"]),
    ("CLASSIC short-haul, booking class K: no-show penalty in EUR?",      "180", ["90", "70"]),
    ("Under the CURRENT misconnect SOP, meal voucher value in EUR?",      "15",  ["10"]),
    ("Under the CURRENT misconnect SOP, hotel is offered after how many hours?", "6", ["8"]),
    ("Flight XX 1487 IST-CDG: what is the NEW departure time?",           "11:20", ["08:35"]),
]

if _cached.QUICK:                       # QUICK=1 halves the question set; the table says so
    CASES = CASES[:4]

# ---------------------------------------------------------------------------- the black box --
# Everything between these two rules is machinery this module does not explain. It exists only to
# produce a "few short passages" condition for the table below. If you want it opened now rather
# than in modules 5-7, it is `eval/chunking.py` and `eval/retrieval.py`; nothing in it is hidden,
# it is only being kept closed.
import chunking as C
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
# Embedding the passages is the only model call outside the measurement itself, so a replay
# skips it rather than needing Ollama to print numbers it already has.
dense = None if _cached.USE_CACHED else R.DenseRetriever(chunk_ids, chunk_texts)

def passages(question: str, n: int) -> str:
    """`n` short passages that this black box thinks are relevant to `question`."""
    return "\n\n".join(f"[SOURCE: {c.split('#')[0]}.md]\n{chunks[c]}"
                       for c in dense.rank(question)[:n])
# -------------------------------------------------------------------------- end of the box --

# The whole-corpus condition runs first, and all of its questions run back to back. Ollama keeps
# the processed prefix of the previous request, so eight questions against the same 79,309
# characters cost one prefill and seven cache hits. Looping the other way round — question
# outside, condition inside — puts three short prompts between every pair of long ones, throws
# that cache away each time, and turns this cell into eight cold queries instead of one.
#
# This cell still pays one full prefill even though the cell above already sent the same corpus,
# because the cached prefix starts at the system message and this one asks for a bare number
# rather than a citation. Change any byte near the front of a prompt and the cache is gone.
conditions = [("whole corpus", lambda q: everything), ("top-1 chunk", lambda q: passages(q, 1)),
              ("top-3 chunks", lambda q: passages(q, 3)), ("top-5 chunks", lambda q: passages(q, 5))]
COLUMNS = ["top-1 chunk", "top-3 chunks", "top-5 chunks", "whole corpus"]
# The keys above are what `cached_runs.json` was recorded under and must not change. What the room
# reads is this, which names no machinery it has not met yet.
LABEL = {"top-1 chunk": "one passage", "top-3 chunks": "three passages",
         "top-5 chunks": "five passages", "whole corpus": "the whole book"}

def measure_context_sizes() -> dict:
    grid, seconds, answers = {}, {}, {}
    for name, build in conditions:
        start = time.time()
        grid[name], answers[name] = [], []
        for q, right, wrong in CASES:
            a = R.generate(f"SOURCES:\n{build(q)}\n\nQUESTION: {q}",
                           system="Answer ONLY from the sources. Give the number. Be brief.",
                           max_tokens=80,
                           num_ctx=STUFFED_CTX if name == "whole corpus" else None)
            grid[name].append(bool(right in a and not any(w in a for w in wrong)))
            # Keep the text, not only the verdict: a claim about what the model said should be
            # something the room can read off the run rather than take on trust.
            answers[name].append(" ".join(a.split()))
        seconds[name] = time.time() - start
        print(f"  {LABEL[name]:<16} {sum(grid[name])}/{len(CASES)} correct   {seconds[name]:>6.1f}s"
              f"   {seconds[name] / len(CASES):>5.1f}s per question", flush=True)
    return {"grid": grid, "seconds": seconds, "answers": answers,
            "questions": [q for q, _, _ in CASES],
            "chars": {name: len(build(CASES[0][0])) for name, build in conditions}}

measured = _cached.run("03-context-size-vs-accuracy", measure_context_sizes,
                  note=f"{len(CASES)} questions x {len(conditions)} context sizes")

print(f"\n{'question':<50}" + "".join(f"{LABEL[n]:>16}" for n in COLUMNS))
print("-" * 114)
for row, q in enumerate(measured["questions"]):
    print(f"{q[:48]:<50}" + "".join(f"{'OK' if measured['grid'][n][row] else 'X':>16}" for n in COLUMNS))
print("-" * 114)
n_cases = len(measured["questions"])
print(f"{'correct':<50}" + "".join(f"{str(sum(measured['grid'][n])) + '/' + str(n_cases):>16}" for n in COLUMNS))
print(f"{'characters of context':<50}" + "".join(f"{measured['chars'][n]:>16,}" for n in COLUMNS))
print(f"{'seconds for all ' + str(n_cases) + ' questions':<50}" +
      "".join(f"{measured['seconds'][n]:>16.1f}" for n in COLUMNS))
print(f"\n  {sum(measured['seconds'].values()):.0f} seconds of model time for this table, "
      f"{measured['seconds']['whole corpus']:.0f} of it in the whole-corpus column alone.")

# %% [markdown]
# ## More context answered better, not worse
#
# Non-decreasing, in the direction opposite to the folklore. On this corpus, with this model, there
# is no distractor penalty to find. Read the character counts on the same table: the passage columns
# are two to three orders of magnitude smaller, and one of them is a single short passage that often
# does not contain the answer at all, so part of the low end is a *recall* problem rather than a
# reading problem. That is the black box's fault, and the black box is not on trial today.
#
# The comparison that matters is five passages against everything, and everything still wins.
#
# So be careful with the argument you make here. **Retrieval does not earn its place on this corpus
# by making answers better.** If we told the room it did, someone would run this cell and catch us.

# %% [markdown]
# Look at what it actually said, too, not only at the ticks. Take the class K *change* penalty —
# the second row — and read the four answers side by side.

# %%
row = 1 if len(measured["questions"]) > 1 else 0
print(measured["questions"][row], "  (correct: EUR 70)\n")
for name in COLUMNS:
    verdict = "OK " if measured["grid"][name][row] else "X  "
    print(f"  {verdict} {LABEL[name]:<16} {measured['answers'][name][row][:110]}")

# %% [markdown]
# The wrong answers are not noise. Every one of them is a real cell from this corpus. `EUR 155`
# is the class K change penalty on the CLASSIC *long-haul* sheet — grep the corpus for it and it
# is there, `fare_classic_longhaul.md` — and `EUR 90` is the class K *cancellation* penalty on
# the short-haul one. Given a nine-column filed tariff and six near-identical sheets, the model
# does not invent numbers. It reads a real cell from the wrong row, the wrong column, or the
# wrong document, and reports it in the same confident voice as a right one.
#
# That is a **generation** failure whenever the right sheet was in front of it, and it is not one
# a better search fixes. When the right sheet was never handed over at all, the same wrong number
# has a different cause and better search does fix it. Two causes, one symptom — and telling them
# apart needs the machinery we have kept in the box, so park the question here and pick it up in
# module 7, which is built on exactly this failure.
#
# One thing worth carrying out of this cell either way: nothing measured in this course scores
# whether the answer was right. Every metric in the day is a retrieval metric.

# %% [markdown]
# ## So why not just stuff the prompt?
#
# Three reasons, and accuracy is not among them.

# %%
top5_chars = measured["chars"]["top-5 chunks"]
print(f"  the whole book : {len(everything):>7,} chars  ~{len(everything)//3:>6,} tokens")
print(f"  five passages  : {top5_chars:>7,} chars  ~{top5_chars//3:>6,} tokens")
print(f"  ratio          : {len(everything)/top5_chars:>7.0f}x")

# %% [markdown]
# Both character counts are measured on the **first** of the eight questions, not averaged across
# them, so the ratio is one question's ratio rather than a constant of the corpus.
#
# **One — the tokens.** Thirty-seven times as many per question, on the question above. On a hosted
# API that is the invoice, directly. Locally it is memory, and it is per concurrent user, which is
# what a call centre is.
#
# **Two — the latency, and only on a cold cache.** Read the seconds the cells above printed on
# this machine today. On a freshly loaded model the first stuffed question costs a minute or more,
# because every token is processed before generation starts. Every question after it against the
# same corpus costs seconds. That is why the measurement cell runs all its whole-corpus questions
# consecutively instead of interleaving them with short ones — loop the other way round and every
# long prompt pays the cold price again.
#
# So be careful how hard you lean on this one. On a single laptop asking a series of questions
# about the same corpus, prompt caching makes the latency argument almost disappear. It comes back
# whenever the prefix changes: a corpus reissue, a restart, a second user with different documents
# in front of theirs. A call centre is the second case, not the first.
#
# **Three — it does not scale, and that one is arithmetic.**

# %%
per_query_tokens = rough_tokens
header = f"fits in {context_length:,}?" if context_length else "fits?"
print(f"{'corpus':>16}  {'tokens/query':>14}  {header}")
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
