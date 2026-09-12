---
title: "Going Further: Hybrid, Rerank, Agentic"
description: "The four techniques every RAG post hands you next — what they are, what an earlier version of this course measured, and what you would add to the lab to try each."
---

> **Beyond the day — not taught on 7 October, not in the lab.** Nothing on this page runs from the
> Bruno collection. It is reading material for the week after, when the one-document question stops
> being the question you have.

The lab stops at a deliberate point: one question, one document, one pass of retrieval, and the
right chunk found. The first thing a search for "improve RAG" will hand you is four words — hybrid,
rerank, contextual, agentic. An earlier version of this course, built on Python notebooks, ran three
of them against the same twenty gold questions over the same 28 Kraken Air / Wyvern Overseas
documents, and measured. One of the four you already shipped on the day without noticing. Two lost.
The fourth solves a problem the lab never shows you, and costs what it costs.

The old numbers survive here only inside the boxes, for one reason: they name a **precondition** each
technique needs, and that is the part that transfers to your system. The digits do not.

<div class="measured">

**About every number on this page.** They come from the earlier Python version of this course
(September 2026, `eval/RESULTS.md`), now in the repository's git history — commit `07c13ba` of
`amadeus-rag-training`. They were produced with a different chat model, a different chunker
configuration (154 structure-aware chunks, not the lab's 132) and scoring code that no longer exists
in the lab. **They are not reproducible with `amadeus-rag-lab`** and are quoted for their direction,
not their digit. Twenty questions decides between two designs; a difference under about 0.05 is noise.

</div>

## BM25, and why it fails on paraphrase

BM25 is the keyword retriever: a term scores high when it is rare across the corpus, its weight
saturates as it repeats inside one document, and long documents are penalised. It finds an identifier
like `XX 1487` instantly and needs no model at all — no embedding call, no GPU, an inverted index
that updates in microseconds.

It fails in exactly one place, and the corpus puts that place in your face: **the query and the
document must share vocabulary.** "iptal edersem ne öderim" has no path to "Cancellation penalty".
A paraphrase in the same language is the same failure in a milder form — "what do I pay to walk away
from this ticket" shares no content word with the RULE 2A table. An embedder crosses both gaps
because it maps meaning, not tokens; BM25 cannot, and chunking makes it worse, because a six-term
query that used to find all six terms in one document now finds them split across three chunks, none
scoring much.

<div class="measured">

Earlier Python course, Sep 2026, 20 gold questions, 6 of them Turkish query over an English document
(`tr_en`), 4 Turkish over Turkish (`tr_tr`):

| retriever over the same 154 structure-aware chunks | hit@1 | recall@5 | MRR | `tr_en` |
|---|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** | 0.667 |
| BM25 | 0.300 | 0.633 | 0.467 | **0.000** |

Over whole documents, before any chunking, BM25 reached 0.400 hit@1 — and **1.000** on the four
`tr_tr` questions against **0.000** on the six `tr_en`. Same corpus, best and worst, decided by which
language the document happened to be in. Chunking cost it the rest: its exact-identifier score fell
from 0.750 to 0.500.

</div>

**To try it in the lab:** build a BM25 index over the same chunk texts `ingest()` writes to Chroma
and return its ranking next to the dense one from `retrieve()` in `src/pipeline.ts`, so the two can
be compared on the same `hits` shape.

## Hybrid retrieval and reciprocal rank fusion — why fusion diluted

Hybrid means running both retrievers and merging. The merge everyone uses is reciprocal rank fusion:
cosine similarity and a BM25 score share no unit, so you throw the scores away and keep the
positions. Each document earns `1 / (k + rank)` from every ranking it appears in, `k = 60` by
convention, and the sums are sorted. It is genuinely elegant, and it has a precondition nobody
states out loud.

Do the arithmetic before you assume `k = 60` protects you. Rank 1 is worth `1/61`; rank 3 is nearly
the same; rank 154 is still worth **28 % of rank 1**. When both retrievers rank the *same* full set
of chunks — which they do, there is no "only one of them found it" case — fusion is a sum of two
smoothed reciprocal ranks, and the weaker ranking votes on everything, all the way down. **Reciprocal
rank fusion assumes both inputs are independently sound.** A good retriever and one that is
systematically wrong on a third of your corpus do not average to something in between; they inherit
the failure, because an ordering always looks like an opinion.

<div class="measured">

Earlier Python course, Sep 2026:

| retriever over the same 154 chunks | hit@1 | recall@5 | MRR | `tr_en` |
|---|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** | 0.667 |
| BM25 | 0.300 | 0.633 | 0.467 | 0.000 |
| RRF of the two | 0.450 | 0.683 | 0.579 | **0.000** |

Dense alone 0.750; adding the second retriever cost 0.300 — six questions. On `tr_en`, 0.667 and
0.000 fused to 0.000. Fusion won exactly once: at **document** level, where BM25 has whole documents
to match, RRF took the best recall@5 on the page — 0.733 against dense's 0.717 — and paid for that
0.016 with 0.100 of hit@1.

</div>

When hybrid does earn its keep: at millions of documents BM25 returns as a cheap **first stage** that
hands a few hundred candidates to an expensive second stage. There it generates candidates rather
than voting on the answer, and recall@k is its metric. That is not the hybrid measured above.

**To try it in the lab:** fuse the two rankings by position inside `retrieve()` before the threshold
filter, and let how deep the weaker ranking is allowed to vote be the knob you actually sweep.

## Reranking — listwise, pointwise, and when a reranker pays

A reranker takes the top candidates and asks a model to judge them. The cheapest one is a chat model
with a prompt, and there are two shapes. **Listwise**: one call, all candidates, "put these in
order". **Pointwise**: one call per candidate, "score this passage 0–10 for this question", ties
keeping the retriever's order so the reranker only moves a document when it has an opinion.

The earlier course made both calls on the same question over the same six candidates. Pointwise
returned six usable integers in six calls. Listwise returned `1,4,2,5` — four indices for six
passages, nothing to rank with. **A small model answers a narrow question well and a wide one badly.**
Pointwise also degrades to doing nothing when the reply is unparseable; listwise has no such floor.
Pointwise costs six calls where listwise costs one.

Then the number that decides whether to rerank at all. The same pointwise reranker was run over four
retrieval setups of deliberately different quality:

<div class="measured">

Earlier Python course, Sep 2026 — the reranker was a 3B chat model scoring the top 8, 20 questions:

| retrieval setup | hit@1 before | after | MRR before | after | verdict |
|---|---|---|---|---|---|
| weak embedder + fixed-280 | 0.350 | **0.450** | 0.515 | **0.544** | helped |
| weak embedder + structure-aware | 0.350 | **0.400** | 0.490 | **0.540** | helped |
| `bge-m3` + fixed-280 | 0.700 | 0.500 | 0.817 | 0.680 | hurt |
| `bge-m3` + structure-aware | **0.750** | 0.600 | 0.817 | 0.725 | hurt |

Per question, on the strongest setup: 15 of 20 already had the gold document at rank 1 and reranking
pushed **4** of them down; of the 5 that did not, it pulled **1** up. Cost: 8 model calls per
question, 160 per setup, 640 for the table; the four setups took 191 s, 127 s, 51 s and 71 s on one
M-series Mac.

</div>

It helped both weak setups and hurt both strong ones, without exception. What split the table was
not the chunking — both strategies appear on both sides — but the embedder. **A reranker levels, and
it levels toward its own ceiling.** Before, the setups spread from 0.350 to 0.750; after, from 0.400
to 0.600. Handed candidates from a retriever of that quality, that reranker's output landed in that
band whatever it was given. If your retriever is worse than the pairing, imposing the model's opinion
is an upgrade. If it is already better, imposing it can only lose — and the two rows where it
"helped" would still have been the wrong design, because the right move there was to fix the
embedder.

The production reranker is not a chat model but a **cross-encoder** — query and passage read together
in one forward pass, trained on relevance labels. `bge-reranker-v2-m3` is the multilingual one and
pairs with the lab's embedder. The earlier course did not measure it; "it would clear 0.750" is a
hypothesis, and the lesson survives either answer, because a stronger judge still has a ceiling you
locate relative to your retriever.

**To try it in the lab:** between `retrieve()` and the prompt assembly in the `/query` route, add one
`gemma3:4b` call per hit that returns a 0–10 integer, re-sort the hits by it with ties keeping the
retriever's order, and count the calls you just added.

## Contextual retrieval — you already shipped it

Contextual retrieval means prefixing each chunk with enough surrounding context that it stands
alone, usually by having a model write a line about where the chunk sits in the document — at one
generation call per chunk, at ingest time.

The lab's `structure` strategy already prefixes every chunk with its heading path:
`[fare classic shorthaul > RULE 2A …]` — document title plus the heading the chunk sits under. Same
mechanism, taken from the document instead of generated, at zero inference cost. It is why the K row
and its column header survive in the same chunk, and why module 7 turned "EUR 13" into EUR 90 on the 12 Sep run. On
documents that carry headings, check that a generated context line buys you something Markdown did
not give you free; on documents that do not — scanned PDFs, chat logs, flat text — it is the first
thing to try.

**To try it in the lab:** in `ingest()`, before embedding, call the chat model once per chunk for a
one-line "where this sits" prefix and ingest the result as a new strategy next to `structure`, so
`npm run eval` can compare the paid prefix against the free one.

## Agentic RAG — the loop, multi-hop, and the bill

Every lever on the day acts on a single pass: embed the question once, get one neighbourhood back. A
better embedder moves the point; a better chunker sharpens each candidate; a reranker reorders what
came back. None of them adds a second query. Some questions need one.

The earlier course's gold set held two such questions. One of them, as a caller would say it: a
Kraken flight is late, the CLASSIC K passenger misses the Wyvern connection at CDG and waits five
hours — what is owed during the wait, is the Wyvern leg repriced, and what is the change fee if the
passenger would rather voluntarily move to tomorrow? Three questions in one coat. The duty of care is
in `sop_misconnect_v4.md`; whether the Wyvern coupon survives is Clause 4 of `interline_xx_yy.md`;
the change fee is in `fare_classic_shorthaul.md`. No file holds two of the three, on purpose: the SOP
states no monetary value of any kind, and the interline clause does not price anything. The
documents point at each other. Real rule books do.

### Which fare sheet — RULE 7

Two fare sheets could govern one ticket, and the corpus settles it in a rule that appears on both —
`fare_classic_shorthaul` RULE 7 and `fare_classic_longhaul` RULE 7: **the sheet is chosen by the
transaction, not by the document.** A voluntary change to a single coupon is assessed on the sheet
for the band of the Kraken sector held; a cancellation or refund of the whole journey is assessed on
the long-haul sheet. So the misconnect question's change fee is short-haul CLASSIC K: **EUR 70**. A
cancellation of the same journey would be long-haul CLASSIC K: **EUR 195**. Quoting EUR 90 — the
day's number, the short-haul cancellation — would be a right row on the wrong sheet. A single pass
cannot notice a rule buried at the end of both documents; a tool that knows whether it is pricing a
change or a cancellation picks the sheet *before* retrieval.

### The loop

Four steps, no framework. **Decompose** — one model call turns the question into standalone
sub-questions, capped at four. **Retrieve per sub-question** — each gets its own embedding and its
own top-k against the same collection: three points instead of one. **Check sufficiency** — one
model call answers `YES`, or `NO` plus one short query for what is missing; narrow, the same way
pointwise beat listwise. **Answer with citations**, naming the document behind each number.
Retrieval stops being a step before the model and becomes a tool the model calls — it decides when
to stop and what to ask next; your code decides what to do with that.

<div class="measured">

Earlier Python course, recorded run of 9 Sep 2026, one M-series Mac:

| | measured |
|---|---|
| `multi_hop` questions in the gold set | 2 of 20 |
| their hit@1, single pass, every chunking strategy | 0.500 — and 0.000 before chunking |
| misconnect question: gold documents reached, single pass → loop | **0 of 3 → 2 of 3**, 4.9 s |
| whole-journey cancellation question: single pass → loop | 2 of 3 → 2 of 3, 7.9 s |
| model calls per question, loop | 6–10, three or four of them embeddings |

The loop never found `interline_xx_yy`. The answer it wrote anyway got the meal voucher and the
EUR 70 change fee right, silently dropped the hotel threshold and the repricing question, and
volunteered a cancellation penalty nobody asked about. Two questions cannot measure a method; they
show that one exists and that it is not free.

</div>

### The bill

**More calls** — six to ten where simple RAG made one, and a loop multiplies per-call latency by the
round count instead of adding to it. **Non-determinism** — the plan is generated, so two runs can
decompose, retrieve and cite differently; log every sub-question, that list is what you debug.
**More passes, more chances to pull the wrong thing** — the corpus holds `sop_misconnect_v3.md`,
superseded, saying EUR 10 and an 8-hour hotel threshold against v4's EUR 15 and 6 hours; filter on
version before chunks reach the model, as a property of the tool, not a hope about the ranking.
**The loop that never ends** — if the sufficiency check can always say "still missing something", it
will. Bound it four ways: a hard round cap, a cap on sub-questions, a stop when a round brings back
nothing new, and on the last round an answer from what it has that *states* what it could not find.
An honest partial answer beats an infinite loop, and beats the confident invented one module 1
watched the bare model produce.

And the caveat the whole course carries: every metric above scores **retrieval** — whether the right
document came back. None scores whether the sentence built on it was right. The recorded loop run is
the proof that the two can disagree.

**To try it in the lab:** add an `/agent` route that calls `retrieve()` in a loop — decompose prompt,
per-sub-question retrieval, a narrow sufficiency prompt — with a hard round cap and a log of every
generated sub-question in the `debug` field, and put a router in front so single-hop questions never
pay for decomposition.

## Where to read the old numbers

The full tables — the chunking ladder with its per-type breakdown, three embedders, BM25 and fusion
at both granularities, the four-setup rerank sweep — are in `eval/RESULTS.md` at commit `07c13ba` of
the `amadeus-rag-training` repository's git history, alongside the notebooks that produced them.
Read them as the earlier course's measurement of the earlier course's setup. The lab's own numbers
are the trainer-side `npm run eval` in `amadeus-rag-lab/eval/results.md`, and those are the only
ones that describe what is on your laptop.
