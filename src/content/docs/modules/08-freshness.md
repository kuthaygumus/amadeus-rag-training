---
title: "8. Freshness: Why RAG Exists"
description: "The same question against last quarter's rule book and this quarter's: 120 becomes 90 with a re-ingest, not a retrain. This is the module that closes the loop on module 3."
---

## Gate question

> **Module 3's fine-tuned model is frozen on Q2. Why is everything we built in modules 5–7 better than fine-tuning?**

Be honest about where the day stands. Module 3 showed a model that had the rule book in its weights — `kraken-q2`, a LoRA fine-tune on `corpus/2026-Q2/`, `UNVERIFIED: not built yet, shown as a trainer demo`. Modules 5–7 built something far clumsier: a container, an embedding model, a chunker that had to be fixed twice before it found the K row. A fair sceptic in the room should be asking why the clumsy thing wins.

It wins on one axis, and the axis is time. **A fine-tuned model knows what the documents said on the day you trained it. A RAG pipeline knows what they say now.** This module makes that sentence into four requests and two numbers.

<div class="presenter-note">
Twenty-five minutes, of which the four requests are about four; the comparison table takes the rest. Before request 1, ask for hands: "Who still thinks fine-tuning would have been the simpler design?" — a few will, and they are not wrong about simpler; keep them for the table at the end. Ollama down: read this page, the two numbers are on it, and the DELTA diff is a file you can open on the projector without any model.
</div>

## Two editions, one row

`corpus/2026-Q2/` is last quarter's rule book: 21 documents. `corpus/2026-Q3/` is this quarter's: 28. The Q2 edition is generated from Q3 by `scripts/make_q2.py`, and `corpus/DELTA.md` lists every change it makes — so, for once in this course, you know *exactly* what moved between two editions.

The headline is one table cell in `fare_classic_shorthaul.md`, RULE 2A, row K:

```diff
- (Q2) | K | KSHEU26 | none | none | EUR 70 | EUR 120 | EUR 240 | 2 x 23 kg | Yes |
+ (Q3) | K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |
```

Read the whole row, not just the number the day is about. The cancellation penalty went from EUR 120 to EUR 90, and the no-show penalty next to it went from 240 to 180 — because RULE 4 derives no-show from cancellation. One business decision, two cells. The document id also stepped forward, `FR-CL-SH-2026Q2-013` → `FR-CL-SH-2026Q3-014`, and the new sheet's `Supersedes:` line names the old one.

What else changed, exactly as `DELTA.md` says it:

- Seven documents did not exist in Q2 at all: `sop_misconnect_v4.md` and the six winter-season schedule bulletins `bulletin_scb_2026_09xx.md`.
- `sop_misconnect_v3.md` was **Current** in Q2 and is **Superseded** in Q3 — the same file, one word of metadata apart.
- Three policies had a routine reissue: `policy_corporate_travel.md` 6.1 → 6.2, `policy_travel_approval.md` 4.7 → 4.8, `policy_expense_reimbursement.md` 5.3 → 5.4.
- Every edition stamp and every document id was rewritten back one quarter, including cross-references in other documents. Those are not facts about the airline; they are what makes the Q2 folder a Q2 edition.

Everything else is byte-identical. A quarterly reissue of a real rule book looks like this: a handful of real changes buried in a relabelling of everything.

## What a retrain would cost

Put the module 3 pipeline next to the one you are about to run.

To move `kraken-q2` to Q3 you would regenerate the dataset (`scripts/make_finetune_dataset.py`, 695 question–answer pairs for Q2 — the 18 that teach EUR 120 would need to become 90, and you would need to find them), run the LoRA notebook again on a GPU, convert to GGUF, `ollama create` a `kraken-q3`, evaluate it, and ship it to every laptop. Someone owns that. It has a calendar. And at the end of it the model still cannot tell you *which document* it got 90 from — the number is spread across weights, not stored in a row.

To move the RAG pipeline to Q3 you `POST /ingest` with a different `edition`. The response tells you how long it took, in milliseconds, stage by stage.

<div class="presenter-note">
The sentence not to garble: "The model never changed. The documents did, and the answer followed them." Say it after request 4, with `sources[0].source` on screen. Do not oversell the seconds in `stages.total` — a real corpus is tens of thousands of documents and embedding it is minutes to hours, not seconds. The point is not that it is fast; the point is that it is a batch job with no training run, no GPU, no evaluation cycle and no human in it.
</div>

## What you run

Four requests, in order, all in one Bruno folder. Each `/ingest` writes its own Chroma collection, named from its configuration — `kraken-2026-Q2-structure-1500-strip` and `kraken-2026-Q3-structure-1500-strip` — so ingesting Q3 does not overwrite Q2. The most recent ingest becomes the default target for `/query`.

**Bruno — `06-freshness` › `1-ingest-q2`**

```json
{
  "edition": "2026-Q2",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

Same strategy module 7 ended on, applied to last quarter's book. Read `edition` (`2026-Q2`), `documents` (21), `collection`, and `stages` — `read`, `chunk`, `embed`, `store`, `total`, all in ms.

**Bruno — `06-freshness` › `2-query-q2`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

Read `answer` — **EUR 120**. That was the rule in Q2, and that is what a model fine-tuned on Q2 would say forever. Read `sources[0].source` too: `fare_classic_shorthaul`, the Q2 copy. `abstained` should be `false`.

**Bruno — `06-freshness` › `3-ingest-q3`**

```json
{
  "edition": "2026-Q3",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

This is the "retrain". Read `stages.total`. That number, on your laptop, is the entire cost of moving the system from last quarter's rules to this quarter's. `documents` is 28 now; `collection` ends in `2026-Q3-structure-1500-strip`.

**Bruno — `06-freshness` › `4-query-q3`**

Same body as request 2. Read `answer` — **EUR 90**, with `[fare_classic_shorthaul]` as the source, and `sources[0].excerpt` should contain the RULE 2A heading and the K row together, which is what module 7 was for.

**120 → 90 with a re-ingest, not a retrain. The model never changed; the documents did, and the answer followed them — with a source you can open.**

The two collections are both still in Chroma. To hold the two answers side by side, put the collection in the body:

**Bruno — `06-freshness` › `4-query-q3`, body edited**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "collection": "kraken-2026-Q2-structure-1500-strip"
}
```

Same model, same question, same prompt template, same embedding — 120 again. Switch `collection` back to the Q3 name, 90. Nothing about the intelligence changed between the two; only the shelf it was reading from.

<div class="presenter-note">
If time allows, one more thing on the projector: open `corpus/2026-Q2/fare_classic_shorthaul.md` and `corpus/2026-Q3/fare_classic_shorthaul.md` side by side and show that `sources[0].excerpt` is literally that text. That is what "auditable" means here — a support agent can click through to the row. A fine-tuned model gives you a number and a feeling.
</div>

## Fine-tuning against RAG, fairly

Fine-tuning is not the villain of this course. It is the wrong tool for *this* problem, and the table should say where it is the right one.

| | fine-tuning (`kraken-q2`) | RAG (modules 5–7) |
|---|---|---|
| freshness | frozen at training time; Q2 until someone retrains | as fresh as the last `/ingest`; edition is a request parameter |
| citation / auditability | none — the number lives in weights | `sources[{source, score, excerpt}]` on every answer; the row can be opened |
| cost of an update | regenerate dataset → GPU run → GGUF → `ollama create` → eval → redeploy; days, a person, a calendar | one `POST /ingest`; `stages.total` |
| what it is actually good for | style, format, tone, domain terminology, a task shape the base model does not have — things that do not change quarterly | facts that move: tariffs, SOPs, bulletins, anything with a `Supersedes:` line |
| failure mode | confidently states the old fact, with no signal that it is old | retrieves the wrong chunk (module 7) or abstains; both are visible in `debug` and `sources` |
| hallucination risk | same as the base model, plus stale facts delivered with full confidence | reduced when the chunk is right; the prompt restricts the model to the retrieved context, and below the similarity threshold no model call is made at all (`abstained: true`) |

A team that wants the model to *speak* like Kraken Air's agents — the phrasing, the structure of a refund explanation, the Turkish/English register — fine-tunes for that, once, and keeps it for years. A team that wants the model to *know* what the K-row penalty is this quarter never fine-tunes for that. The mature design does both: a model tuned for the shape of the answer, reading facts from documents that are re-ingested on Revenue Management's schedule.

## What the numbers said

<div class="measured">

| what | measured |
| --- | --- |
| the number at stake | Q2 sheet EUR 120, Q3 sheet EUR 90; no-show 240 → 180 with it |
| Q2 edition, `structure` + `stripBoilerplate` | on the 12 Sep run the answer was EUR 120 — your wording will differ |
| Q3 edition, same configuration | EUR 90 `[fare_classic_shorthaul]` on the same run |
| Q3 chunks, `structure-1500` + strip | 132 (trainer-side count; the Q2 edition has fewer documents and fewer chunks — `UNVERIFIED: Q2 count not recorded`) |
| what `DELTA.md` says moved | 1 fare cell (and its derived no-show), 1 SOP status flip, 3 policy version bumps, 7 new documents, every edition stamp and id |

</div>

Measured 12 Sep 2026 on one M-series Mac, `gemma3:4b` generating, `bge-m3` embedding, one run each. Ingest timings were not recorded for that run; read `stages.total` on your own machine and expect it to differ.

## Going deeper

The edition-per-collection design has a second use the day does not have time for. If the business rule is that a ticket is governed by the fare conditions in force when it was issued, then a passenger who bought in Q2 and cancels in Q3 may owe EUR 120, not 90 — and the question "which edition applies?" is a metadata filter, not a retrieval problem. Keeping `kraken-2026-Q2-…` alive and selecting the collection from the ticket's issue date is exactly the scoped retrieval module 4 gestured at. Deleting last quarter's edition the moment this quarter's arrives is the mistake.

The other thing to say about fine-tuning: the two techniques compose. The industry phrase is "fine-tune for form, retrieve for facts". A small model tuned on the shape of a good fare-rule answer, with the actual cells retrieved from the current edition, is the design most production teams land on — and the [going further](/reference/going-further/) page has the next layer up: hybrid search, reranking, and agentic retrieval that decides for itself which edition to open.

## Exit line

> Fine-tuning photographed the rule book. RAG reads it. RAG exists for the case where the data moves faster than you can train — and this quarter it moved.

Gate out: → [9. Closing: Rewind the Chain](/modules/09-closing/).
