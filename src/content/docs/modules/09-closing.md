---
title: "9. Closing: Rewind the Chain"
description: "Eight gates read backwards, three decisions to take home, and what is still running on your laptop on Monday."
---

## Not a gate — a rewind

> **Read the chain backwards, out loud, in one breath.**

Nothing runs in this block. Laptops can close. Every module today opened on a failure and closed on
the one thing that failure earned — the question the next module had to answer. Read from the end,
each step is the reason the one before it was not enough.

## The chain, backwards

**8. Freshness** — the Q2 edition answered **EUR 120** and was not wrong, it was stale. Re-ingest
Q3 and the same question reads **EUR 90**: no training run, no new weights, one `POST /ingest`.
That is the whole case for RAG over fine-tuning, and the previous module is what made it fair.

**7. Chunking and noise** — fixed-280 found the right document and read the wrong column — on the
12 Sep run it said "EUR 13"; your number will differ. Cutting to the document's structure put the K
row back under its header, `[fare classic shorthaul > RULE 2A …]`, and the answer became EUR 90.
**Right chunk, right answer** — which earned the right to prove RAG beats fine-tuning where it
matters: when the data moves.

**6. ChromaDB in a container** — we could finally *see* the chunks, and the K-row chunk had no
header. The store is a container with an endpoint you can peek into; that is what turned "retrieval
brought back garbage" into a fault with a location.

**5. Simple RAG** — ingest, retrieve, query: the answer was grounded, with a source named. But look
at what was retrieved. A grounded answer from the wrong chunk is still the wrong answer, delivered
in the voice the model uses when it is right.

**4. The data moved** — the whole corpus fit in the prompt: ~21 000 tokens, 60–120 s cold per
question on the 12 Sep run. You cannot retrain quarterly and you cannot pay for the whole book per
question — so hand the model the right piece. Retrieval is born here, as a cost-saving.

**3. Fine-tuning** — a model that answers Q2 from its own weights works, until the data moves.
`UNVERIFIED: kraken-q2 has not been built; the stale answer is a prediction, not something this room
watched.` The shape of the failure needs no model: weights cannot know a row changed and cannot cite
the row they learned.

**2. How a neural network learns** — weights are a frozen photograph of the training data. Which
raised the only reasonable question: can we re-take the photo with our data?

**1. The bare LLM** — it did not know our data and did not know that it did not know. Asked for the
CLASSIC K cancellation penalty it invented a confident amount — a "€50"-class figure on the 12 Sep
run. So: where *is* knowledge in a model?

Eight gates and one setup, not one of them opened with a definition. **The whole day is one number —
EUR 90 — quoted from a document, with the document named.**

<div class="presenter-note">
Do the rewind standing, no slides, no laptop on the projector. Eight lines, one breath each; point
at the gate on the board as you pass it. Two things to say honestly rather than smoothly: the
fine-tune line is a prediction — <code>kraken-q2</code> was never built — and the numbers they saw
today are one run on one machine, so they should quote the direction, not the digit. About four
minutes. If the day is running late this is the part you keep; cut "what stays on your laptop" to
one sentence instead.
</div>

## Three decisions you take home

**Chunk to the document's structure, not to a character count.** The failure of the day was never
the document — it was the row that lost its header. Markdown headings, table headers, clause numbers
are free context; a fixed-size cutter throws them away and a model-generated "context line" buys them
back at inference cost. Use what the author already wrote.

**Measure retrieval at document level before you touch the model.** A cheap, deterministic check —
did the right document come back in the top 1, the top 5 — tells you where the fault is before any
prompt engineering. Not one metric today scored the *answer*; build that second eval too, because the
first one will never tell you the second is broken.

<div class="measured">

Trainer-side only — `npm run eval` in the lab repository, 20 gold questions, `bge-m3`, document-level
hit@1 / recall@5 / MRR, one run on 12 Sep 2026 (numbers drift on other machines):

| strategy | hit@1 | recall@5 | MRR |
|---|---|---|---|
| fixed-280 | 0.700 | 0.917 | 0.817 |
| fixed-280 + overlap 60 | 0.550 | 0.883 | 0.717 |
| recursive-600 | 0.700 | 0.900 | 0.806 |
| structure-1500 | 0.700 | 0.833 | 0.781 |
| structure-1500 + strip | 0.650 | 0.850 | 0.766 |

Read it honestly: structure-aware chunking does **not** find the right document more often. It fixes
the **chunk** — the K row together with its header — which is what turns "I don't know" into EUR 90.
Overlap loses on every metric. This is why a document-level number alone is not enough, and why you
still read what was retrieved.

</div>

**Re-ingest is the deployment unit, not retraining.** When Q3 replaces Q2 the pipeline does not
change, the model does not change, the prompt does not change. One ingest of the new edition, one
collection name that says what is in it — `kraken-2026-Q3-structure-1500` — and the stale answer is
gone. That is a deploy you can schedule, roll back and diff. A fine-tune is none of those.

## What stays on your laptop

Everything the day ran is still there, and none of it needs the network:

- the `amadeus-rag-lab` clone — corpus, both editions, `corpus/DELTA.md`, the TypeScript pipeline in
  `src/pipeline.ts`, and the trainer's eval in `eval/`;
- two containers — `chroma` with its named volume (your last collection is still in it) and the
  built `api` image. `podman compose up` without `--build` brings them back as they were;
- the Bruno collection, folders `00-health` to `06-freshness`, every request body as you left it;
- the two Ollama models on the host.

On Monday, with no network: start Ollama, **Terminal (repo root of `amadeus-rag-lab`)** →
`podman compose up`, open Bruno, send **Bruno — `00-health` › `health`**. If it returns, the rest of
the collection runs. Change one document in `corpus/2026-Q3/`, re-ingest, ask again — that is the
whole loop, and now it is yours.

## What the day left out

Deliberately: hybrid retrieval and BM25, reranking, contextual retrieval, agentic multi-hop loops.
Each was measured in an earlier version of this course and each has a precondition the lab's
single-pass retrieval does not meet yet. They are written up, with the old numbers and with what you
would add to `src/pipeline.ts` to try each, in [Going Further](/reference/going-further/). Read it
when the one-document question stops being the question you have.

<div class="presenter-note">
The day closes at 15:00 and this block is the last fifteen minutes of it. End on the EUR 90
sentence, then the exit line, then stop talking — no summary slide after it. Before you stop, one
show of hands: "who will run <code>podman compose up</code> on Monday?" Count it and say the count
back; it is the only retention metric you get. The repo link and the going-further page go on the
screen while they pack up, not before. If Ollama is down at this point it does not matter — nothing
here runs.
</div>

## Exit line

> Every step today was born where the previous one stopped being enough. The last one was a
> re-ingest, not a retrain — and that is the version that still runs on Monday.
