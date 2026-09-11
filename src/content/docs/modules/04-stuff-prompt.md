---
title: "4. Just Stuff the Prompt"
description: "The whole rule book fits in the context window, and it answers every question right. So why is this not the design?"
---

## Gate question

> **If I cannot retrain, why not put everything in the prompt?**

Module 3 left us a model with the rule book in its weights and stale anyway. It was fine-tuned on `corpus/2026-Q2/`, where the CLASSIC short-haul class K row reads **EUR 120** and 18 of the 695 generated training pairs teach that number; the current sheet, `FR-CL-SH-2026Q3-014`, says **EUR 90**. `kraken-q2` has not been built yet, so the stale answer is what the training data teaches rather than one anyone has recorded — but the corpus diff is real, the book reissues quarterly, and nobody wants to own a training run every three months to change one row.

So take the obvious shortcut. Do not touch the weights. Paste the rule book into the prompt.

<div class="presenter-note">
Before running the first cell, make the room commit: "Will 28 documents fit in the context window — yes or no? Hands up for no." Most hands go up. They are wrong, and that is the module. Get the guess on the record before the number lands on screen. This slot is 10:32–10:43, eleven minutes, and it holds three things: the fit line, the eight-question table, and the 37x ratio. Everything under *Going deeper* is reading material, not stage time.
</div>

## It fits, and it answers

The Q3 corpus is **28 documents, 78,310 characters, 76 KB**. The notebook concatenates every file in `corpus/2026-Q3/` under a `[SOURCE: name.md]` header, so the prompt that leaves is **79,309 characters, 77 KB** — the 999-character difference is those headers and the blank lines between documents. `qwen2.5:3b` reports a **32,768-token** window. The notebook's deliberately generous three-characters-per-token estimate puts the prompt at **~26,400**; the notebook records the server's own `prompt_eval_count` for that exact prompt as **21,170**. Both are far inside the window, with room left for the question and the answer.

And it answers. Over eight questions whose answers are single verifiable values in the corpus, the whole corpus in one prompt scored **8/8** — including the two the corpus was built to trap.

Read the first cell's two answers before the room draws the wrong conclusion from them. Neither question says short-haul — the notebook's strings are `CLASSIC K sinifi iptal cezasi kac euro?` and `CLASSIC class K: what is the cancellation penalty in EUR?` — and on the recorded run **both answered EUR 195**, off the long-haul sheet, one citing `FR-CL-LH-2026Q3-021` and the other citing nothing. That is not a wrong number. The corpus holds six fare sheets, three families in a short-haul and a long-haul edition each, and the model picked one and quoted it without saying it had picked. Which sheet it lands on is not stable between runs. Handing it everything did not make it ask.

This course could have told you a comfortable lie here — that the corpus is too big, that you hit a token wall, that RAG is therefore necessary. At this size it is not true. A 76 KB rule book fits in a modern context window with room to spare, stuffing it works, and it works better than anything else on this page. Say so plainly; half the room already suspects it.

The failure is not correctness. It is the bill, the clock, and what happens at ten times this size.

## What one question costs

**~26,400 tokens** by the notebook's estimate, 21,170 by the server's. That is the price of one question — of every question, because the model is stateless and reads the whole prompt again each time. Twenty agents on a shift, ten questions each, is 200 queries: about **4.2 million tokens** of rule book pushed through the model in one shift by the server's count, 5.3 million by the notebook's, to produce at most 16k tokens of answer. That is arithmetic from the corpus size, not a meter reading, but it is the arithmetic anyone does the first time they see the invoice.

The clock says the same thing. With the model already loaded but the corpus never seen, the first stuffed question took **84.2 s**; the second question against the same corpus came back in **1.0 s**, because Ollama had kept the prefix it had already processed.

Someone will say prompt caching, and it is the strongest objection on this page — strong enough that you should not lean on the seconds harder than they hold. On one laptop asking a run of questions about one unchanging corpus, caching nearly erases the latency argument. It fixes less than it looks, in three ways: the corpus is reissued quarterly, so the cache is invalidated on Revenue Management's schedule and not yours; a cache hit still carries the decode-side cost of a long key-value cache; and it is an optimisation of the wrong shape, making it cheaper to pay for all 28 documents rather than stopping you paying for them. The cost returns whenever the prefix changes — a reissue, a restart, a second agent with different documents in front of theirs — and a call centre is that case, not the first. The table below carries it with no cold start in it at all: eight questions cost **75.4 s** against the whole corpus and **10.2 s** against five retrieved chunks.

## Now scale it

This corpus has 28 documents because it has to fit on a laptop and in one day. A real rule book is tens of thousands: every fare family on every route band, every SOP revision, every bulletin, every interline agreement, in every language the centre answers in. The notebook prints that arithmetic — 280 documents is **264,360 tokens**, eight times past the window at one order of magnitude, and 28,000 documents is about **26 million**. No production window takes that, and none will take it cheaply enough to spend on every call in a queue. There the wall is real, and it is not one you engineer over. It is one you route around.

<div class="presenter-note">
The sentence not to garble: "It fits, it answers, and it answers better than retrieval does. The problem is that you pay for the entire book to answer one question — on every question." Say it once, slowly, with the token count still on screen. Slow laptop: `QUICK=1`. Ollama down: `USE_CACHED=1` (every replayed cell prints a `[CACHED]` banner).
</div>

## What the run says about accuracy

This is the easiest claim on the page to get wrong, so it is measured rather than asserted. The same eight questions, asked at four context sizes:

| context given to the model | characters (question 1) | correct | seconds for all 8 |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 6.6 |
| top-3 chunks | 1,256 | 3/8 | 7.7 |
| top-5 chunks | 2,144 | 3/8 | 10.2 |
| **the whole corpus, as one prompt** | **79,309** | **8/8** | **75.4** |

Where do the top-k rows come from? From a retrieval step this course has not built yet. The notebook calls the pieces *chunks* and does not explain the word; cutting documents up is module 7, choosing which pieces sit nearest a question is module 6, and both are set up to fail in front of you first. Treat those three columns as a black box on loan from later in the day — on loan deliberately, because the comparison is only fair to retrieval if retrieval gets its best shot. Read the column, not the machinery.

Non-decreasing, and in the direction opposite to the folklore. **More context answered better here, not worse**, and the stuffed prompt is the extreme case of it. On this corpus, with this model, there is no distractor penalty to find. Note also what top-5 bought over top-3: nothing — the same three questions came back right, for 888 more characters of context.

Part of the low end is a recall problem rather than a reading problem: a single 239-character chunk often does not contain the answer at all. But the comparison that matters is top-5 against everything, and everything wins 8/8 against 3/8.

Look at *how* the retrieved columns lost, because it is not noise. Asked for the class K short-haul **change** penalty, where the answer is `EUR 70`, top-3 and top-5 both answered `EUR 155` — a real cell, the class K row of the CLASSIC **long-haul** sheet. Retrieval handed the model the wrong one of six near-identical sheets and the model read it faithfully; with everything in the prompt the right sheet was in there too, and it answered `EUR 70`. That is a document-selection failure, and an argument for retrieval that retrieves *well* rather than an argument against long contexts.

The trap this corpus was built for did not fire either, and that is worth saying plainly. `sop_misconnect_v3.md` is marked **Superseded** — meal voucher **EUR 10**, hotel after **8 hours**. `sop_misconnect_v4.md` is marked **Current**: **EUR 15** and **6 hours**. Both sit in `corpus/2026-Q3/`. Handed both at once, the stuffed model read the metadata and answered from the current one on both questions; top-5 got the hotel threshold right and the meal voucher wrong. The trap is real — stuffing hands the model a document and its replacement together and relies on it to notice one word — but on this run the model noticed, and one run is not a licence to rely on that.

So state the gate carefully. **Retrieval does not earn its place on this corpus by making answers better; on this run it makes them worse.** It earns it on tokens, on the clock, and on scale. Claim otherwise and someone in the room runs this cell and catches you.

<div class="presenter-note">
Take a vote before the table lands: "Everything in the prompt, or five retrieved chunks — which answers more of the eight right?" Most rooms say the five chunks, because the distractor argument is everywhere. Then show the row: 8/8 against 3/8. The second vote is the SOP one — "the corpus holds a superseded procedure and the current one, both in the prompt; which number comes back?" — and the honest answer is that it came back right, which is not what the trap was written for. Open `sop_misconnect_v3.md` and `sop_misconnect_v4.md` side by side anyway: they are one word of metadata apart, and the room needs to see how thin the thing the model noticed is. What you must not say is that stuffing degrades accuracy. The table is on the same screen and someone will check.
</div>

## What you run

Open `notebooks/03_stuff_the_prompt.py` in VS Code and run the blocks with `Shift+Enter`; the `# %%` markers are the block boundaries.

**Terminal (repo root) — green before you start:**

```bash
python scripts/verify_setup.py
```

**What you should see** in the VS Code Interactive window. The fit line first:

```
model context window : 32,768 tokens
our corpus           : ~26,436 tokens
                       FITS
```

Then the two stuffed answers with their elapsed seconds beside them, the eight-question table ending in `1/8  3/8  3/8  8/8`, and last the cell the module rests on — already in the file, so you are reading it, not typing it:

**VS Code — `notebooks/03_stuff_the_prompt.py`, the last block:**

```python
top5_chars = measured["chars"]["top-5 chunks"]
print(f"  whole corpus : {len(everything):>7,} chars  ~{len(everything)//3:>6,} tokens")
print(f"  top-5 chunks : {top5_chars:>7,} chars  ~{top5_chars//3:>6,} tokens")
print(f"  ratio        : {len(everything)/top5_chars:>7.0f}x")
```

```
  whole corpus :  79,309 chars  ~26,436 tokens
  top-5 chunks :   2,144 chars  ~   714 tokens
  ratio        :      37x
```

The 79,309 is the **prompt**, not the corpus, and the 2,144 is the top-5 context for the *first* of the eight questions — the notebook measures characters on one question, not across the set — so 37x is one question's ratio, not a constant of the corpus.

**Roughly how long.** About three minutes against a freshly started Ollama, almost all of it inside two long calls. **If that is too slow on your machine**, restart the Python terminal, put `import os; os.environ["QUICK"] = "1"` in a block above the first one and run again: the question set halves and the printed table says it was reduced. If Ollama is not answering at all, use `USE_CACHED=1` the same way and the notebook replays end to end.

## What the numbers said

<div class="measured">

| what | measured |
| --- | --- |
| the Q3 corpus | 28 documents, 78,310 characters, 76 KB |
| the prompt those 28 documents build | 79,309 characters, 77 KB — the extra 999 are `[SOURCE: …]` headers and blank lines |
| `qwen2.5:3b` context window | 32,768 tokens, read from `/api/show` |
| that prompt as tokens | ~26,400 by the notebook's 3-characters-per-token estimate; 21,170 by the server's own `prompt_eval_count`, as the notebook records it |
| correct, 8 single-value questions | top-1 1/8 · top-3 3/8 · top-5 3/8 · whole corpus 8/8 |
| seconds for those 8 questions | 6.6 · 7.7 · 10.2 · 75.4 — 100 s of model time for the table, 75 of it in one column |
| first stuffed question, corpus unseen | 84.2 s; the next question against the same corpus 1.0 s |
| the two stuffed answers in the first cell | both EUR 195, off the long-haul sheet, on a question that never said which |
| the superseded-SOP trap | it did not fire: the stuffed prompt answered EUR 15 and 6 hours, from the Current SOP |
| tokens per query at scale | 280 documents 264,360 · 28,000 documents ~26 million, against a 32,768-token window |
| the number at stake in the corpus | Q2 sheet EUR 120, Q3 sheet EUR 90 |

</div>

Measured on one M-series Mac with the models already resident, `qwen2.5:3b` generating; the top-k columns are retrieved by machinery this course does not build until module 7. The run is stored in `notebooks/cached_runs.json` and replays with `USE_CACHED=1`; the full table is `eval/RESULTS.md` section 6.

## Going deeper

On accuracy, the mechanism people reach for is "lost in the middle": material in the middle of a long prompt is used less reliably than material at the start or the end. It is a real effect in the literature, of unclear magnitude on any specific corpus — and on this one we measured for it in a 79,309-character prompt and did not find it at all.

One middle ground is worth naming, because someone will build it: stuff a *scoped* prompt. Not the whole book, but everything for one fare family, resolved by a deterministic filter — the ticket already says the fare basis is `KSHEU26`. That is retrieval done with metadata instead of embeddings, and for structured domains like tariffs it often beats a vector search. At ten million documents the shape is the same, only wider — cheap selection first, then a small window read carefully — and the question stops being "how much fits" and becomes "what is my recall at the selection stage".

## Exit line

> It fit, and it answered — better than retrieval did. But I paid for the entire book to do it, and at ten times this size it will not fit at all. I need to select just the right piece.
