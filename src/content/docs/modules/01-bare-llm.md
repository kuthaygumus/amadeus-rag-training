---
title: "1. The Bare LLM Wall"
description: "One request to the model on its own: it answers the Kraken Air penalty question confidently, specifically, without a source — and wrong. Then you change the question and it answers again."
---

## Gate question

> **Does the model know *my* data?**

<div class="presenter-note">
Before anyone clicks, put the question on screen and make the room commit: will a 4B model answer the Kraken Air penalty question, or refuse? Show of hands, count it out loud, write the two numbers on the board. Most rooms split. The request settles it in one click — it answers, with a number — and the refuse camp's surprise is what carries the module. Two minutes for this, no more.
</div>

A model on the laptop. No retrieval, no documents, nothing but the weights. We ask it the one question this whole day is built around — the one a Kraken Air call-centre agent gets ten times a shift:

> Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?

The real figure is in `corpus/2026-Q3/fare_classic_shorthaul.md`, RULE 2A table, row K: **EUR 90**, a flat amount per passenger, not a percentage. Last quarter the same row said EUR 120. Kraken Air is fictional — there is no website, no forum thread, no press release the model could have read. It has never seen this number.

Run the request and read the `answer` field. There is no recorded transcript on this page on purpose: the model is `gemma3:4b`, it runs at `temperature: 0`, and it will still say something different on your machine than it did on the trainer's. **What does not change is the shape of the reply — confident, specific, unsourced, and wrong.** On the 12 Sep run it named an amount of the "€50" kind, stated as fact, in one clean sentence. Your number will differ. That it differs is the point: a model that held the fact would say EUR 90 every time, on every laptop.

Look at what is *not* in the reply. No "I am not certain". No "according to the Kraken Air fare rules". No document name, no date, no quarter. It reads exactly like the reply you would get for a question it does know the answer to.

Now change the question. Anything about the two airlines:

- when flight XX 1487 departs
- the checked-baggage allowance on Wyvern Overseas in FLEX
- what the Kraken Air lounge in Istanbul is called
- the no-show penalty for the same K ticket

It will answer every one of them, in the same register, and none of it exists anywhere except in the reply you are reading. The one that stings: the no-show penalty *is* in the same table, two columns to the right of the cancellation penalty — EUR 180 — and the model has no more access to that cell than to any other.

<div class="presenter-note">
Someone will be typing the same question into a bigger model to show it does better. Let them — a frontier model is more likely to hedge, and it still cannot know that this quarter's number is 90 and not 120, because no model has read a document that did not exist when it was trained. Move the argument there, do not defend the 4B model.
</div>

## Why it cannot do otherwise

A language model predicts the next token from learned weights: billions of parameters compressing the statistical shape of everything it read. There is no lookup table inside, no source document, no timestamp on any fact. When the question arrives, the weights produce the most plausible continuation — and for "how much is the cancellation penalty for booking class K", the most plausible continuation is *an amount*. Text that answers fare questions with a number is attested millions of times in the training data. Text that answers them with "I have not read Kraken Air's fare rules" is not.

Sometimes a model does decline. That is not a knowledge check either. Refusal is a trained behaviour, fitted to the *form* of a question — "I don't know" was rewarded during instruction tuning for questions shaped like a lookup of a private or volatile fact. Reword the same question and the refusal goes away, because nothing inside consulted anything either time. *Going deeper* picks this up.

**The danger is not that it fails. The danger is that it fails in exactly the same voice it uses when it succeeds.** Same fluency, same certainty, same absence of a citation. Nothing in the output tells you which of the two you are holding. An agent who reads the reply cannot tell, a customer cannot tell, and a test that checks for a 200 and a non-empty string cannot tell either — Bruno's assertion on this request goes green on the wrong answer.

That is the wall. Everything after this is an attempt to get over it, and the day tries them in order:

1. **Train the model on our rules** — module 3. It works, until the data moves.
2. **Put the whole rule book in the prompt** — module 4 gives it all the data in one go, and shows what that costs per question.
3. **Hand it the right piece at answer time** — module 5 onward. The only one that survives the corpus being reissued next quarter.

## What you run

One request. Your containers are up from [setup](/modules/00-setup/) and `00-health` was four `"ok"`.

**Bruno — `01-bare-llm` › `ask-about-kraken`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

Press send. The reply takes a few seconds on a warm model, longer on the first call of the day while Ollama loads the weights. Read these fields:

- `mode` — `"bare"`. Nothing was retrieved; the api passed your question to Ollama with a one-line system prompt and nothing else.
- `model` — `"gemma3:4b"`.
- `answer` — the sentence the module is about. Is there an amount? Is it EUR 90? Is there a source?
- `promptTokens` — a small number: the one-line system prompt plus your question. Remember it; module 4 makes it ~21 000.
- `ms` — how long the model took.

The assertions on the *Tests* tab pass. They check the status and the mode, not the answer — there is nothing in the response a test *could* check the answer against.

**Then edit the body and send again.** Change `question` to anything about Kraken Air or Wyvern Overseas: a departure time, a baggage rule, a lounge, the refund deadline for a FLEX fare. Send three or four. Count how many times it declines.

**If Ollama is not answering.** `00-health` names the missing piece — Ollama unreachable, or a model not pulled. Do not debug it now: read this page, watch the trainer's projector, and rejoin at module 2, which needs no model at all. Fix it in the break; the lab from module 5 onward needs it.

<div class="presenter-note">
Twelve minutes, of which the live request is two. When a participant's Ollama is down, the fallback is your projector: run the request on your machine, read the answer aloud, then change the question twice in front of the room. If your own machine is down as well, read the page and say "this ran on my machine this morning; you will reproduce it in module 4 with the same request". Expect reports of different numbers — that is a one-sentence answer: the number varies, the pattern does not. Do not let anyone go hunting for a wording that makes it refuse; that is a prior being tuned, not knowledge being checked, and it eats the clock.
</div>

## Going deeper

Think of the weights as compression, not storage. Training squeezes a corpus into a fixed parameter budget, so the frequent and the generic survive at high fidelity while the specific and the rare get reconstructed rather than recalled. A hallucination is a confident guess drawn from a region the training data never constrained. "Cancellation penalties are an amount or a percentage" is generic and survives. "EUR 90 for booking class K on the short-haul CLASSIC sheet in 2026-Q3" is as specific as a fact gets, and no amount of scale moves it across that line — a bigger model changes the odds on the public part of a question and nothing at all on the private part.

Refusal deserves more attention than it usually gets. Instruction tuning rewards "I don't know" for questions that *look like* a lookup of a private or volatile fact, so the signal tracks the wording, not whether the model holds the fact. You can move the line with a system prompt — "answer only if certain" — but you are tuning a prior, not installing a knowledge check, and the lab shows the cost of that later: a strict abstain instruction makes a small model abstain even when the right document is sitting in its context. The alternative is to ground the answer in a retrieved document and hand back the source with it. That is cheaper than asking the model five times to see whether the number moves, and it is auditable — and auditable is what an airline needs.

One thing the bare request cannot show: *where* in the model the Kraken Air number would have to live for it to answer correctly. That is the next module.

## Exit line

> It does not know — and it does not know that it does not know.

<div class="presenter-note">
This is the one sentence you must not garble. Say it slowly, do not add to it, do not explain it — the next module is the explanation. Then straight into module 2 with: "So where is knowledge in this thing, actually? Where would EUR 90 have to be stored?"
</div>
