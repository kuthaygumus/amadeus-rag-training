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
Before running the first cell, make the room commit: "Will 28 documents fit in the context window — yes or no? Hands up for no." Most hands go up. They are wrong, and that is the module. Get the guess on the record before the number lands on screen. Budget 25 minutes for this page.
</div>

## It fits. That is not the problem.

The Q3 corpus is **28 documents, 75 KB**. Concatenate every file in `corpus/2026-Q3/`, put it in front of the question, send it to `qwen2.5:3b`. It answers **EUR 90**. It answers the meal voucher question. It gets the multi-hop ones that need two documents, because both documents are right there.

This course could have told you a comfortable lie here — that the corpus is too big, that you hit a token wall, that RAG is therefore necessary. At this size it is not true. A 75 KB rule book fits in a modern context window with room to spare, and stuffing it works. Say so plainly; half the room already suspects it.

The failure is not correctness. It is the bill.

## The arithmetic

Read the token count off the model, not off a tokenizer. Ollama's `/api/chat` response carries `prompt_eval_count` — the tokens it actually read — and `prompt_eval_duration`, the time it spent reading them. This matters practically: HuggingFace weight downloads are policy-blocked on the corporate network, so you cannot `pip install` a tokenizer to count tokens. The model already counted them.

75 KB of mixed English and Turkish prose lands around **20-25k tokens**. That is the price of one question. Not of loading the corpus once — of every question, because the model is stateless and reads the whole prompt again each time.

Multiply. Twenty agents on a shift, ten questions each, is 200 queries. At roughly 22k tokens of corpus per query that is about 4.4 million tokens of rule book pushed through the model in one shift to produce maybe 20k tokens of answer. That is arithmetic from the corpus size, not a meter reading — but it is the arithmetic anyone does the first time they see the invoice.

Someone will say prompt caching, and it is the strongest objection on this page. A static prefix can be cached and a cached prefix is cheap to re-read. It fixes less than it looks. The corpus is reissued quarterly, so the cache is invalidated on Revenue Management's schedule, not yours; a cache hit still carries the decode-side cost of a long key-value cache; and it is an optimisation of the wrong shape, making it cheaper to pay for all 28 documents rather than stopping you paying for them.

## Now scale it

This corpus has 28 documents because it has to fit on a laptop and in one day. A real rule book is tens of thousands: every fare family on every route band, every SOP revision, every schedule bulletin, every interline agreement, in every language the centre answers in.

Hold the ratio. 28 documents is 75 KB, so 28,000 documents is roughly **75 MB** — on the order of **20-25 million tokens**. No production context window takes that, and none will take it cheaply enough to spend on every call in a queue. There the wall is real, and it is not one you engineer over. It is one you route around.

<div class="presenter-note">
The sentence not to garble: "It fits. That is not the problem. The problem is that you pay for the entire book to answer one question — on every question." Say it once, slowly, with the token count still on screen. If the notebook errors instead of answering, check `num_ctx` before debugging anything else — Ollama's default context window is smaller than this corpus, and an overflowing prompt is silently truncated from the front. The notebook sets it explicitly for that reason. If Ollama misbehaves anyway, the notebook ships with saved output cells: read the token count and the timing off the saved run and keep moving.
</div>

<div class="presenter-note">
Ask before you show the two SOP revisions: "The corpus contains a superseded procedure and the current one. If both are in the prompt, which number does the model give you?" Take two guesses from the room, then open `sop_misconnect_v3.md` and `sop_misconnect_v4.md` side by side on screen — EUR 10 and EUR 15, one word of metadata apart. Do not claim we measured that stuffing degrades accuracy. We did not, and someone will check.
</div>

## The second cost, stated honestly

There is a subtler argument against stuffing, and it is easy to overclaim. A model handed 25k tokens of mostly irrelevant text tends to answer worse than the same model handed the right 500 tokens: more text means more plausible-looking wrong material competing for attention.

This corpus has the trap built in. `sop_misconnect_v3.md` is marked **Superseded** and puts the meal voucher at **EUR 10**. `sop_misconnect_v4.md` is marked **Current** and says **EUR 15**. Both sit in `corpus/2026-Q3/`. Stuff everything and you hand the model both numbers, then rely on it to notice one word of metadata.

One measured hint that a model can misread with the answer in front of it: `llama3.2:3b` answered **EUR 70** instead of **EUR 90** for the CLASSIC K cancellation penalty *with the column header in its context*. It read the change-penalty column. That is why it is banned from this course.

Be straight about the limit of that evidence. **This page has no measurement of the distractor effect on this corpus.** The gold set was never run under full-corpus stuffing against retrieved context, so "long context degrades accuracy here" is borrowed from the literature, not from our numbers. What would settle it: run the same 20 gold questions twice at fixed generation settings, once with the whole corpus in the prompt and once with the top-5 structure-aware chunks, and compare answer accuracy. Until then the cost argument stands alone and the accuracy argument is a hypothesis.

## What you run

Notebook: `03_stuff_the_prompt.ipynb`.

```bash
cd ~/amadeus-rag-training
python scripts/verify_setup.py          # green before you start
jupyter lab notebooks/03_stuff_the_prompt.ipynb
```

Inside the notebook:

```python
from pathlib import Path
from eval.retrieval import generate

corpus = "\n\n".join(
    p.read_text() for p in sorted(Path("corpus/2026-Q3").glob("*.md"))
)
print(len(corpus), "characters across 28 documents")

question = "Helios CLASSIC K short-haul: how much is the cancellation penalty?"
answer = generate(f"{corpus}\n\nQuestion: {question}\nAnswer from the documents above.")
```

Time it, and read `prompt_eval_count` off the raw Ollama response so the token number sits on screen next to the stopwatch.

## What the numbers said

<div class="measured">

| what | measured |
| --- | --- |
| Q3 corpus stuffed into the prompt | 28 documents, 75 KB |
| tokens that represents | roughly 20-25k (arithmetic from corpus size, not a meter reading) |
| `qwen2.5:3b` on a short retrieved context | 3/3 correct, 0.9 s |
| `gemma3:4b`, same questions | 2/3, 1.9 s |
| `qwen3:4b`, same questions | correct, 11.6 s (reasoning tokens) |
| `llama3.2:3b`, header in context | EUR 70 where the answer is EUR 90 — banned from this course |
| the number at stake | Q2 sheet EUR 120, Q3 sheet EUR 90 |

</div>

The 0.9 s is the anchor: that is `qwen2.5:3b` answering from a short context. Whatever the stopwatch reads with 20-25k tokens in front of it, compare it to that. The gap is what stuffing costs per query.

## Going deeper

A long prompt is expensive twice over because prefill and decode scale differently. Prefill is self-attention over the whole sequence — quadratic in the naive formulation, and even with flash-attention kernels, work that grows with every token you add. Decode is different: each generated token attends over a key-value cache whose size grows linearly with the prompt. A 25k-token prompt does not just cost a slow start. It makes every token after it slower and fatter in memory, for the whole answer.

That is why the caching argument only half works. Prefix caching removes repeated prefill. It does not remove the KV cache the long prefix produced, which stays resident and gets attended over on every decode step. When you batch requests on a GPU, the KV cache is usually what caps concurrency, not the compute. Stuffing 25k tokens per user is a direct cut to how many users fit.

On accuracy, the mechanism people reach for is "lost in the middle": material in the middle of a long prompt is used less reliably than material at the start or the end. Treat it as a real effect of unclear magnitude on any specific corpus — including this one, where we did not measure it. The cheap mitigation if you do ship a stuffed system is ordering: put the most likely relevant document last, immediately before the question, instead of trusting the model to scan evenly.

At 10 million documents the answer is not a bigger window. It is a two-stage funnel — cheap wide selection, then a small expensive model reading a small window — and the question stops being "how much fits" and becomes "what is my recall at the selection stage", because anything the funnel drops is unreachable however good the generator is. One middle ground is worth naming, because someone will build it: stuff a *scoped* prompt. Not the whole book, but everything for one fare family, resolved by a deterministic filter — the ticket already says the fare basis is `KSHEU26`. That is retrieval done with metadata instead of embeddings, and for structured domains like tariffs it often beats a vector search.

None of this makes fine-tuning pointless. It taught the model the shape of the job: the vocabulary, the tone, that a fare basis code is a thing. Stuffing gives it facts it does not have. They fail in different directions, and the design that wins puts the shape in the weights and the facts in the context.

## Exit line

> It fit. But I paid for all of it on every query. I need to select just the right piece.
