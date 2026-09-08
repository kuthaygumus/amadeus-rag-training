---
title: "4. Just Stuff the Prompt"
description: "The whole rule book fits in the context window and it answers correctly. So why is this not the design?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **If I cannot retrain, why not put everything in the prompt?**

We left the last module with a model that had the rule book in its weights and was wrong anyway. It was fine-tuned on `corpus/2026-Q2/`, so it answers **EUR 120** for a cancelled CLASSIC **K** ticket. The current sheet, `FR-CL-SH-2026Q3-014`, says **EUR 90**. The fix is a retrain, the rule book is reissued quarterly, and nobody wants to own a training run every three months to change one number in one row.

So take the obvious shortcut. Do not touch the weights. Paste the rule book into the prompt.

<div class="presenter-note">
Before running the first cell, make the room commit: "Will 28 documents fit in the context window — yes or no? Hands up for no." Most hands go up. They are wrong, and that is the module. Get the guess on the record before the number lands on screen. This slot is 10:32–10:43, eleven minutes, and it holds three things: the fit line, the eight-question table, and the 37x ratio. Everything under *Going deeper* is reading material, not stage time.
</div>

## It fits. That is not the problem.

The Q3 corpus is **28 documents, 78,310 characters, 76 KB**. Concatenate every file in `corpus/2026-Q3/`, put the question after it, send it to `qwen2.5:3b`. The notebook labels each document as it joins them, so the prompt that leaves is slightly larger than the corpus: **79,309 characters, 77 KB**. The 999-character difference is exactly the `[SOURCE: name.md]` headers and the blank lines between documents. The model reports a context window of **32,768 tokens**; that prompt estimates to about **26,400** at roughly three characters per token. It fits, with room left over for the question and the answer.

And it answers. Asked for the CLASSIC short-haul class K cancellation penalty with all 28 documents in front of it, it returns **EUR 90** — the current sheet, the number the fine-tuned model got wrong.

One thing on screen is worth naming before the room draws the wrong conclusion from it. The notebook's first question is the Turkish one, and it does not say short-haul. The model answers **EUR 195** and cites the long-haul sheet. That is not a wrong number; the corpus holds six fare sheets, three families in a short-haul and a long-haul edition each, and the model picked one and quoted it without saying it had picked. Handing it everything did not make it ask.

This course could have told you a comfortable lie here — that the corpus is too big, that you hit a token wall, that RAG is therefore necessary. At this size it is not true. A 76 KB rule book fits in a modern context window with room to spare, and stuffing it works. Say so plainly; half the room already suspects it.

The failure is not correctness. It is the bill, the clock, and what happens at ten times this size.

## The arithmetic

The notebook estimates the token count from characters — `len(everything) // 3`, about three characters per token for mixed English and Turkish — and prints it beside the model's advertised window. That is an estimate, and the page calls it one. If you want the model's own count instead, Ollama's `/api/chat` response carries `prompt_eval_count`; the notebook does not read it.

About **26,400 tokens**, then. That is the price of one question. Not of loading the corpus once — of every question, because the model is stateless and reads the whole prompt again each time.

Multiply. Twenty agents on a shift, ten questions each, is 200 queries. At ~26,400 tokens of prompt per query that is about **5.3 million tokens** of rule book pushed through the model in one shift, to produce at most 16k tokens of answer — the notebook caps each answer at 80. That is arithmetic from the corpus size, not a meter reading, but it is the arithmetic anyone does the first time they see the invoice.

Someone will say prompt caching, and it is the strongest objection on this page. A static prefix can be cached and a cached prefix is cheap to re-read. It fixes less than it looks. The corpus is reissued quarterly, so the cache is invalidated on Revenue Management's schedule, not yours; a cache hit still carries the decode-side cost of a long key-value cache; and it is an optimisation of the wrong shape, making it cheaper to pay for all 28 documents rather than stopping you paying for them.

## And the clock

The notebook prints elapsed seconds next to every call, because the seconds are half the argument. On the machine the run was recorded on, with the model already loaded but the corpus never seen, the first stuffed question took **72.5 s**. The second question against the same corpus came back in **0.9 s** — Ollama had kept the prefix it had already processed.

Do not lean harder on that than it holds. On one laptop asking a run of questions about one unchanging corpus, prompt caching makes the latency argument nearly disappear. It comes back whenever the prefix changes: a quarterly reissue, a restart, a second agent with different documents in front of theirs. A call centre is the second case, not the first.

The eight-question table below shows the same cost with no cold start in it at all: eight questions cost **74.2 s** against the whole corpus and **8.9 s** against five chunks.

## Now scale it

This corpus has 28 documents because it has to fit on a laptop and in one day. A real rule book is tens of thousands: every fare family on every route band, every SOP revision, every schedule bulletin, every interline agreement, in every language the centre answers in.

Hold the ratio. 28 documents is ~26,400 tokens per query, so 280 documents is about **264,000** — already eight times past the 32,768-token window, at one order of magnitude. 28,000 documents is roughly **76 MB**, on the order of **26 million tokens**. No production context window takes that, and none will take it cheaply enough to spend on every call in a queue. There the wall is real, and it is not one you engineer over. It is one you route around.

<div class="presenter-note">
The sentence not to garble: "It fits. That is not the problem. The problem is that you pay for the entire book to answer one question — on every question." Say it once, slowly, with the token count still on screen. If a laptop is too slow for the measurement cell, `QUICK=1` halves the question set and the printed table says it was reduced. If Ollama is unreachable, `USE_CACHED=1` replays the recorded run out of `notebooks/cached_runs.json` without calling the model at all; every replayed cell prints a `[CACHED]` banner naming the date and the machine, so a replay can never be passed off as a live run.
</div>

## What the run says about accuracy

This is the easiest claim on the page to get wrong, so it is measured rather than asserted. Eight questions whose answers are single verifiable values in the corpus, asked at four context sizes:

| context given to the model | characters | correct | seconds for all 8 |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 4.7 |
| top-3 chunks | 1,256 | 3/8 | 7.1 |
| top-5 chunks | 2,144 | 4/8 | 8.9 |
| **the whole corpus, as one prompt** | **79,309** | **7/8** | **74.2** |

Monotonic, and in the direction opposite to the folklore. **More context answered better here, not worse.** On this corpus, with this model, there is no distractor penalty to find.

Part of the low end is a recall problem rather than a reading problem — a single 239-character chunk often does not contain the answer at all. But the comparison that matters is top-5 against everything, and everything wins 7/8 against 4/8.

Look at *how* top-5 lost, because it is not noise. Asked for the class K change penalty it answered `EUR 155`; asked for the class K cancellation penalty it answered `EUR 195`. Both are real cells — the class K row of the CLASSIC **long-haul** sheet — and the question said short-haul. Retrieval handed the model the wrong one of six near-identical sheets and the model read it faithfully. With everything in the prompt the right sheet was in there too.

So state the gate carefully. **Retrieval does not earn its place on this corpus by making answers better.** It earns it on tokens, on the clock, and on scale. Claim otherwise and someone in the room runs this cell and catches you.

## The one the stuffed model lost

The single question the whole corpus got wrong is the one this corpus was built to trap. `sop_misconnect_v3.md` is marked **Superseded**: meal voucher **EUR 10**, hotel after **8 hours**. `sop_misconnect_v4.md` is marked **Current**: **EUR 15** and **6 hours**. Both sit in `corpus/2026-Q3/`.

Asked "under the CURRENT misconnect SOP, hotel is offered after how many hours?", the stuffed model answered **8** — the superseded number. Top-5 answered **6**, because retrieval had handed it the current chunk and not the old one. Both conditions got the meal voucher right at **EUR 15**.

One question out of eight is a hazard, not a trend, and it does not reverse the table. It is worth naming because it is the narrow thing a long context actually costs you here: when the corpus holds a document and its replacement, stuffing hands the model both and relies on it to notice one word of metadata. Retrieval, when it retrieves well, never shows it the old one.

<div class="presenter-note">
Before you show the two SOP revisions, ask: "The corpus contains a superseded procedure and the current one. If both are in the prompt, which number does the model give you?" Take two guesses, then put the answer row on screen — it answered 8 hours, from the file marked Superseded. Then open `sop_misconnect_v3.md` and `sop_misconnect_v4.md` side by side; they are one word of metadata apart. What you must not say is that stuffing degrades accuracy overall. The table in the section above says the opposite, it is on the same screen, and someone will check.
</div>

## What you run

Open `notebooks/03_stuff_the_prompt.py` in VS Code and run the blocks with `Shift+Enter`. The `# %%` markers are the block boundaries; nothing here needs a notebook server, and the dependencies are `numpy` and `chromadb` and nothing else.

```bash
python scripts/verify_setup.py     # green before you start
```

**What you should see.** The fit line first:

```
model context window : 32,768 tokens
our corpus           : ~26,436 tokens
                       FITS
```

Then the two stuffed answers with their elapsed seconds beside them, then the eight-question table ending in `1/8  3/8  4/8  7/8`, then the cell the whole module rests on:

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

Two labels in that output are looser than they look. The 79,309 is the **prompt**, not the corpus:
the 28 files hold 78,310 characters and the `[SOURCE: …]` headers the join adds account for the
other 999. And the 2,144 is the top-5 context for **these eight questions**, so 37x is this
module's ratio, not a constant of the corpus.

**Roughly how long.** About three minutes end to end against a freshly started Ollama, almost all of it inside two long calls. Run it a second time and it is a fraction of that, because the server keeps the prefix it has already read — which is itself part of the lesson.

## What the numbers said

<div class="measured">

| what | measured |
| --- | --- |
| the Q3 corpus | 28 documents, 78,310 characters, 76 KB |
| the prompt those 28 documents build | 79,309 characters, 77 KB — the extra 999 are `[SOURCE: …]` headers and blank lines |
| `qwen2.5:3b` context window | 32,768 tokens, read from `/api/show` |
| that prompt as tokens | ~26,400, estimated at 3 characters per token — not the model's own count |
| correct, 8 single-value questions | top-1 1/8 · top-3 3/8 · top-5 4/8 · whole corpus 7/8 |
| seconds for those 8 questions | 4.7 · 7.1 · 8.9 · 74.2 |
| first stuffed question, corpus unseen | 72.5 s; the next question against the same corpus 0.9 s |
| the stuffed prompt against top-5 chunks, these 8 questions | 79,309 / 2,144 characters = 37x the tokens per question |
| the one the stuffed model lost | hotel threshold: answered 8 h off the Superseded SOP, the Current one says 6 h |
| the number at stake | Q2 sheet EUR 120, Q3 sheet EUR 90 |

</div>

Recorded 2026-09-08 on an M-series Mac, `qwen2.5:3b` generating and `bge-m3` retrieving over structure-aware chunks. The run is stored in `notebooks/cached_runs.json` and replays with `USE_CACHED=1`.

The 72.5 s and the 0.9 s are the same corpus, one prompt with the prefix already processed and one without. That spread is what stuffing costs per query, and it is also why the cost is so easy to demo away on one laptop.

## Going deeper

A long prompt is expensive twice over because prefill and decode scale differently. Prefill is self-attention over the whole sequence — quadratic in the naive formulation, and even with flash-attention kernels, work that grows with every token you add. Decode is different: each generated token attends over a key-value cache whose size grows linearly with the prompt. A 26k-token prompt does not just cost a slow start. It makes every token after it slower and fatter in memory, for the whole answer.

That is why the caching argument only half works. Prefix caching removes repeated prefill. It does not remove the KV cache the long prefix produced, which stays resident and gets attended over on every decode step. When you batch requests on a GPU, the KV cache is usually what caps concurrency, not the compute. Stuffing 26k tokens per user is a direct cut to how many users fit.

On accuracy, the mechanism people reach for is "lost in the middle": material in the middle of a long prompt is used less reliably than material at the start or the end. It is a real effect in the literature, of unclear magnitude on any specific corpus — and on this one we measured for it and did not find it, in a 79,309-character prompt with a 3B model. The cheap mitigation if you do ship a stuffed system is ordering: put the most likely relevant document last, immediately before the question, instead of trusting the model to scan evenly.

At 10 million documents the answer is not a bigger window. It is a two-stage funnel — cheap wide selection, then a small expensive model reading a small window — and the question stops being "how much fits" and becomes "what is my recall at the selection stage", because anything the funnel drops is unreachable however good the generator is. One middle ground is worth naming, because someone will build it: stuff a *scoped* prompt. Not the whole book, but everything for one fare family, resolved by a deterministic filter — the ticket already says the fare basis is `KSHEU26`. That is retrieval done with metadata instead of embeddings, and for structured domains like tariffs it often beats a vector search.

None of this makes fine-tuning pointless. It taught the model the shape of the job: the vocabulary, the tone, that a fare basis code is a thing. Stuffing gives it facts it does not have. They fail in different directions, and the design that wins puts the shape in the weights and the facts in the context.

## Exit line

> It fit, and it answered. But I paid for the entire book to do it, and at ten times this size it will not fit at all. I need to select just the right piece.
