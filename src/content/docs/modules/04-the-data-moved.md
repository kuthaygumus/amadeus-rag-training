---
title: "4. The Data Moved"
description: "The fine-tuned model is frozen on last quarter's book. Retrain every quarter, or hand the model the whole book on every question — we run the second one and watch what it costs."
---

## Gate question

> **The fine-tuned model is frozen on Q2 and the Q3 rule book just arrived — now what?**

Module 3 ended on a model with the rule book in its weights. `kraken-q2` was fitted to `corpus/2026-Q2/`, where the CLASSIC short-haul class K row reads **EUR 120**. The current sheet, `FR-CL-SH-2026Q3-014`, says **EUR 90**. The model does not know. It cannot know: weights froze the moment training stopped, and nothing since has touched them. (`kraken-q2` is still `UNVERIFIED: not built yet`, so EUR 120 is what its 695 training pairs teach, not an answer anyone has recorded — the corpus diff underneath it is real.)

When this course was first pitched, a Principal Engineer listened to the fine-tuning plan and said one sentence: **"the data set changes every three months."** That sentence is this module. Everything before it was about getting knowledge *into* a model. Everything after it is about the fact that the knowledge does not stay still.

Open `corpus/DELTA.md`. Q2 to Q3 is seven documents that did not exist last quarter, three routine policy reissues, one SOP flipped from Current to Superseded, and one changed table row. Twenty-one documents became twenty-eight. One number moved from 120 to 90, and the whole model is wrong about it.

<div class="presenter-note">
Twelve minutes, of which the live request is two — the rest is the wait, and the wait is the content. Open with the Principal Engineer's sentence on the projector and nothing else; let the room read it before you speak. Then a show of hands on two moves: "retrain every quarter, or put the whole book in every prompt?" Count both camps out loud. Most rooms split; a few say "both are obviously wrong", and those people should be asked to hold their answer until the end of the module, because they are right for a reason they cannot yet name.
</div>

## Move (a): retrain every quarter

Fine — the book is reissued, so reissue the model. Look at what that means, in steps rather than in money.

Regenerate the training pairs from `corpus/2026-Q3/`. Run the Colab session again. Evaluate the result against a gold set nobody has written for Q3 yet. Convert to GGUF, quantise, `ollama create`, and roll the new file out to every laptop and every server that holds the old one. Then do it again in three months — and DELTA.md says "three months" is generous: the Q3 folder alone holds six schedule bulletins issued between 2 and 30 September. The book does not change quarterly. It changes weekly, with a quarterly reissue on top.

And at the end of the retrain you have exactly what module 3 had: a confident answer and a citation reconstructed from weights, naming an edition nobody can open. **Retraining buys you the current number and still no source.** The next time someone asks why the answer is 90, the model cannot show them.

## Move (b): put the whole book in the prompt

So do not touch the weights. Leave the model as it is and hand it the rule book at question time — all of it, every question.

The obvious objection is that it will not fit. It does. The Q3 corpus is 28 markdown files, about 79 000 characters; as one prompt that comes to about **21 000 tokens**, and the api asks Ollama for a 65 536-token window (`numCtx: 65536` in `src/routes/chat.ts`). There is room for the question and the answer with a third of the window to spare.

Run it. The request is already in the collection.

**Bruno — `01-bare-llm` › `stuff-the-whole-corpus`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "stuffCorpus": true,
  "edition": "2026-Q3"
}
```

Now wait. On the 12 Sep run this took **60–120 seconds cold** on an M-series Mac; on your laptop it may take longer. Nothing is broken. Ollama is reading 21 000 tokens of tariff before it writes the first word of the answer, and part of the first wait is probably Ollama reloading `gemma3:4b` with the bigger window (`UNVERIFIED: not timed separately`).

When the response lands, read four fields. `documents` is 28 — the assertion in the request checks that. `promptTokens` is the number this module turns on, about 21 000. `ms` is what you just sat through. And `answer` — on the 12 Sep run it said **EUR 90** and cited `[fare_classic_shorthaul]`; it usually does. Your wording will differ.

This course could have told you a comfortable lie here: the corpus is too big, you hit a token wall, therefore RAG. At 28 documents that is not true. **It fits, and it answers.** Say so plainly; half the room already suspects it. The failure is not accuracy. It is the number in `ms`, the number in `promptTokens`, and what both become when the book is real-sized.

<div class="presenter-note">
Press Send and keep talking. Do not fill the silence with apology and do not switch windows — the room has to sit through the minute, because the minute is the argument. Use it: "every agent on the floor, every question, this wait." When the answer lands, point at `promptTokens` before `answer`. Then hand the laptop to a volunteer and have them press Send again, unchanged. It comes back in about a second. Ask the room why. Someone will say "cache"; that is the next section. Then have the same volunteer change one word of the question and send again — slow. If Ollama is down on the projector laptop, read the three numbers off this page and say so; the argument is arithmetic, not a demo.
</div>

## Run it twice

The second identical request comes back in about **a second**. Ollama kept the processed prefix of the last prompt — the whole rule book, tokenised and attended — and only had to read the part that changed, which was nothing.

Change one word of the question and it is slow again. On the 12 Sep run it was; `UNVERIFIED: why — the question sits at the end of the prompt, after the documents, so in principle the document prefix should survive a changed question. Observed once, not explained. Your laptop may behave differently.`

Someone will now say "prompt caching", and it is the strongest objection on this page. On one laptop, asking a run of questions against one unchanging prompt, caching nearly erases the clock. It fixes less than it looks. The cache lives in one Ollama process; a second agent's laptop has its own cold start. It is invalidated on Revenue Management's schedule and not yours — every bulletin, every reissue. And it is an optimisation of the wrong shape: it makes paying for all 28 documents cheaper. It does not stop you paying for them.

## Now scale it

28 documents exist because the corpus has to fit on a laptop and in one day. A real rule book is every fare family on every route band, every SOP revision, every bulletin, every interline agreement, in every language the centre answers in. Call it **10 000 documents**, and keep the arithmetic in tokens and seconds — no prices, because we have not measured any.

28 documents cost about 21 000 tokens, so **750 tokens per document**. 10 000 documents is **7.5 million tokens per question** — 114 times the 65 536-token window the api asks for, and no production window takes it. The clock scales the same way: this laptop read 21 000 tokens in 60–120 s, so 7.5 million tokens is six to twelve hours — for one answer, before the cache, on every cold start.

Now put it on a floor. Twenty agents, ten questions each, one shift: 200 questions. At 28 documents that is 4.2 million tokens of rule book read to produce perhaps 10 000 tokens of answer. At 10 000 documents it is **1.5 billion tokens a day**, to answer 200 questions whose answers are 200 table cells.

There the wall is real, and it is not one you engineer over. It is one you route around.

<div class="presenter-note">
The sentence not to garble, said once, slowly, with `promptTokens` still on the projector: "It fit, and it answered. The problem is that I paid for the entire book to answer one question — and I will pay for it again on the next question, and at real size it will not fit at all." Then the turn: "So which of the 28 documents did the model actually need?" One. About 750 tokens of the 21 000. That is the whole of module 5 — do not build it here, just leave the number in the air.
</div>

## What you run

Nothing new to install. The `api` and `chroma` containers from [setup](/modules/00-setup/) are already up; this request does not use Chroma at all — it reads the edition straight off disk and builds one prompt.

**Bruno — `01-bare-llm` › `stuff-the-whole-corpus`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "stuffCorpus": true,
  "edition": "2026-Q3"
}
```

Send it three times and read the response each time:

1. **Cold.** `documents` 28 · `promptChars` about 79 000 · `promptTokens` about 21 000 · `ms` in the tens of thousands · `answer` — does it say EUR 90, and does it name `[fare_classic_shorthaul]`?
2. **Identical, again.** Same `promptTokens`; `ms` drops to about a thousand. That is Ollama's prefix cache.
3. **One word changed** in `question`. Watch `ms` come back up.

Then one more, for the room that wants to see the stale book: set `"edition": "2026-Q2"` and send. Expect **EUR 120**, cold again, because the prefix changed. Same model, same question, different book — the answer moved with the data, which is the one thing `kraken-q2` cannot do.

**If Ollama is not answering**, the request fails fast with a connection error from the api. Read the numbers in the next section instead; this module's argument is arithmetic and survives without the demo.

## What the numbers said

<div class="measured">

| what | measured |
| --- | --- |
| the Q3 corpus | 28 documents, about 79 000 characters |
| that corpus as one prompt | about 21 000 prompt tokens, by Ollama's own `prompt_eval_count` |
| window the api asks for | 65 536 tokens (`numCtx: 65536` in `src/routes/chat.ts`) |
| first stuffed question, cold | 60–120 s on an M-series Mac, `gemma3:4b` |
| the identical request, repeated | about 1 s — Ollama's prefix cache |
| one word of the question changed | slow again |
| the answer | usually EUR 90, citing `[fare_classic_shorthaul]` |
| the number at stake | Q2 sheet EUR 120, Q3 sheet EUR 90 (`corpus/DELTA.md`) |
| scaled by arithmetic, not measured | 750 tokens per document → 10 000 documents ≈ 7.5 million tokens per question; 200 questions a day ≈ 1.5 billion |

</div>

Measured 12 Sep 2026, one run each, on the trainer's M-series Mac with the models already pulled. Times drift on other machines; token counts do not.

## Going deeper

The strict prompt from module 5 — "answer ONLY from the context, otherwise say I don't know" — is not used here. With 80 KB of context a 4B model told to abstain unless certain will abstain, on everything. `/chat` stuff mode uses a plain instruction instead (`STUFF_SYSTEM` in `chat.ts`). That is its own small lesson: the more you give a small model, the less you can ask of it.

One middle ground is worth naming, because someone will build it: stuff a *scoped* prompt. Not the whole book, but everything for one fare family, selected by a deterministic filter — the ticket already says the fare basis is `KSHEU26`. That is retrieval done with metadata instead of embeddings, and in structured domains like tariffs it often beats a vector search. At ten million documents the shape is the same, only wider: cheap selection first, then a small window read carefully. The question stops being "how much fits" and becomes "what is my recall at the selection stage" — which is module 5's question, asked early.

Long-context models move the wall; they do not remove it. A million-token window reads 1 300 of our documents, not 10 000, and reads them on every question. The cache argument above applies unchanged.

## Exit line

> I cannot retrain every quarter, and I cannot pay for the whole book on every question. The model needed one document out of 28. Hand it only the right piece — [module 5](/modules/05-simple-rag/).
