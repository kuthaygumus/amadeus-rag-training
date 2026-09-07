---
title: "10. Agentic RAG — The Finale"
description: "What if one shot is not enough?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **What if one shot is not enough?**

One question in the gold set has failed every method we built today. It failed with keyword
search, with dense retrieval, with the structure-aware chunking that took hit@1 to **0.800**,
and after reranking. The two `multi_hop` questions score **0.500** at best under every
single-shot method in this course — and 0.500 flatters them, because document-level scoring
calls it a hit when *one* gold document lands at rank 1. This question needs three.

Here it is, as an agent would type it:

> H9 1487 gecikti, CLASSIC K sınıfındaki yolcum CDG'de AU 88 bağlantısını kaçırıyor ve yeni
> uçuşa kadar 5 saat bekleyecek. Bu beklemede kendisine ne vermem gerekiyor, Aurora bacağı
> yeniden fiyatlandırılır mı, bir de yolcu bugünkü seferi hiç istemeyip kendi isteğiyle yarına
> geçmek isterse ondan ne kadar değişiklik ücreti alırım?

Three questions wearing one coat. The care during the wait is in `sop_misconnect_v4.md`.
Whether the Aurora coupon survives is Clause 4 of `interline_h9_au.md`. The change fee is in
`fare_classic_shorthaul.md`. No file holds two of the three, and the corpus is built that way
on purpose: the SOP states it "establishes no monetary penalty, waiver or refund value of any
kind", and Clause 4.4 states the agreement does not determine the value of any voucher. The
documents point at each other. Real rule books do.

<div class="presenter-note">
Put the Turkish question on screen and read it out loud, slowly, the way a caller says it. Ask
the room: "how many documents does this need?" Take answers — you will get one and two. Only
then reveal three, and show the two sentences where the SOP and the agreement each refuse to
answer the other's part. 4 minutes, laptops closed, nothing runs yet.
</div>

## Why one pass returns one document

Embed that question and you get one vector. One point, but the question has three centres of
mass — duty of care, interline protection, fare penalty. The point lands between them, nearest
whichever the phrasing weighted most, and the top five come back from that one neighbourhood.
One retrieval pass gets you one of the three.

Nothing in the pipeline can fix this, because nothing in the pipeline is broken. A better
embedder moves the point; it does not split it. A better chunker sharpens each candidate; it
does not add a second query. Reranking reorders what was retrieved, and what was retrieved
never held the other two documents. Every lever we pulled today acts on a single pass. The
single pass is the problem.

## The loop

Four steps, no new machinery.

**Decompose.** One model call turns the question into sub-questions, one per line, capped at
four: what care is owed for a five-hour wait; is the AU segment protected and is it repriced;
what is the CLASSIC K short-haul voluntary change penalty.

**Retrieve per sub-question.** Each gets its own embedding and its own top-k against the same
`bge-m3` index over the same structure-aware chunks from module 7. Three points instead of one.

**Check sufficiency.** One model call answers `YES` or names what is missing. A single token,
not prose — the same reason pointwise reranking beat listwise **5/5 against 2/5**: a small
model answers a narrow question well and a wide one badly.

**Answer with citations,** naming the document behind each number.

The loop returns: a **EUR 15** meal voucher, because the wait exceeds three hours
(`sop_misconnect_v4` 3.4), and no hotel, because five hours does not exceed six (3.5); the
Aurora segment is protected and reaccommodated **without repricing** (`interline_h9_au` 4.1,
4.2); and the voluntary move to tomorrow costs **EUR 70** (`fare_classic_shorthaul`, CLASSIC K,
change penalty).

That last number closes the day on itself. **EUR 70 is the wrong answer we have been chasing
since module 7** — the change column mistaken for the cancellation column. Here it is right,
because the passenger really is asking for a voluntary change, and the loop can tell the two
columns apart only because structure-aware chunking kept the header attached to the K row. The
final answer is correct because of a decision made three modules ago about where to cut a
string.

<div class="presenter-note">
Before running the loop, have the room write down the three sub-questions they expect. Then run
the decompose cell alone and compare. It will match nobody exactly, and that is the lesson: the
plan is generated, not written. The sentence not to garble: <strong>retrieval stopped being a
step before the model and became a tool the model calls.</strong> Say it once and let it sit.
</div>

## What actually changed

Until now the shape was fixed — retrieve, then generate. Retrieval happened before the model,
decided by us, exactly once. Now the model issues the queries, reads what came back, and
decides whether to go again. Retrieval became a tool with a caller, and control flow moved out
of our code into the model's output. That is the whole definition of an agent, and no library
was involved: a loop, a stopping condition, and a function the model can ask us to run.

## The bill

**More calls.** One decompose, three retrievals, one sufficiency check, one answer: six model
calls where naive RAG made one. At the measured `qwen2.5:3b` cost of **0.9 s** per generation
that is arithmetic, not a measurement, and it lands in single-digit seconds — the order of the
**~4 s** a rerank pass already costs at 8 calls per query. Swap in `qwen3:4b`, measured at
**11.6 s** because it emits reasoning tokens, and the same loop takes a minute. In a loop,
per-call latency multiplies.

**Non-determinism.** The plan is generated, so two runs can decompose, retrieve and cite
differently. Temperature 0, and log every sub-question — that list is the artefact you debug.
Without it a wrong answer is unattributable.

**More passes, more chances to retrieve the wrong thing.** The corpus holds
`sop_misconnect_v3.md`, superseded, saying **EUR 10** and an **8-hour** hotel threshold against
v4's EUR 15 and 6 hours. Each extra pass is another opportunity to pull it. Filter on the
version line before chunks reach the model.

**The loop that never ends.** If the sufficiency check can always say "still missing
something", it will. Bound it three ways: at most two extra rounds, at most four sub-questions,
and a hard rule that on the last round the model answers with what it has and states what it
could not find. An honest partial answer beats an infinite loop, and beats a confident invented
one — which is what module 1 measured the bare model doing when it invented a "20-30% ceza".

<div class="presenter-note">
Break the loop live: edit the sufficiency prompt so it can never say YES, re-run, watch the
round counter hit the cap and stop. Ten seconds, and it is the part they need in production. If
the laptops are slow, run the loop once on the projector and have them read the logged
sub-questions. 35 minutes: gate 4, one-pass failure 6, loop 12, break it 5, rewind 8.
</div>

## What you run

Notebook: `07_agentic_rag.ipynb`

```bash
ollama serve                                  # already running from module 0
jupyter lab notebooks/07_agentic_rag.ipynb
```

The same functions the benchmark uses. Nothing here is a new dependency:

```python
from retrieval import DenseRetriever, generate
from chunking import chunk_corpus, to_documents
from metrics import load_gold

gold = load_gold()
q19 = next(q for q in gold if q["id"] == "q19")     # the three-document question

ids, texts, parents = chunk_corpus(documents, "structure-aware")
index = DenseRetriever(ids, texts)

subqs = decompose(q19["query"], max_parts=4)        # one generate() call
seen = {}
for _ in range(3):                                  # hard round cap
    for sq in subqs:
        for cid in index.rank(sq)[:3]:
            seen[cid] = texts[ids.index(cid)]       # dedupe by chunk id
    missing = check_sufficient(q19["query"], seen)  # "YES", or what is absent
    if missing == "YES":
        break
    subqs = [missing]
print(answer_with_citations(q19["query"], seen))
```

Run the same question through single-pass retrieval in the cell above, so both outputs sit on
one screen. One document against three.

## What the numbers said

<div class="measured">

| | measured |
|---|---|
| `multi_hop` questions in the gold set | 2 of 20 |
| gold documents required by q19 | 3 (`sop_misconnect_v4`, `interline_h9_au`, `fare_classic_shorthaul`) |
| best `multi_hop` score, any single-shot method | **0.500** |
| best single-shot overall (bge-m3 + structure-aware) | hit@1 **0.800**, MRR **0.846** |
| pointwise vs listwise reranking, probe corpus | 5/5 vs 2/5 (MRR 1.000 vs 0.600) |
| rerank pass cost | 8 model calls, ~4 s per query |
| `qwen2.5:3b` generation | 0.9 s, 3/3 correct |
| `qwen3:4b` generation | 11.6 s (reasoning tokens) |
| superseded `sop_misconnect_v3` | EUR 10 voucher, 8-hour hotel threshold |
| current `sop_misconnect_v4` | EUR 15 voucher, 6-hour hotel threshold |

The loop is **not in the benchmark table**. Two multi-hop questions cannot measure a method;
they can only show that one exists. What would settle it: fifty labelled multi-hop questions,
scored on whether *all* gold documents were retrieved rather than whether one was. Until then
the claim on this page is the narrow one — a single pass returned one of three documents, and
the loop returned three.

</div>

## Going deeper

Decomposition is query rewriting with a budget. Everything measured today about retrieval
quality still applies to each sub-question, now written by the model rather than a human —
better on average, because its phrasing drifts toward the vocabulary of the corpus, and worse
in the tail, because a badly rewritten sub-question fails silently. There is no error signal
for asking the wrong thing well.

The sufficiency check is the interesting part and the weakest part. You are asking a 3B model
to judge the completeness of its own evidence, and module 9 measured what that opinion is
worth: imposed on a ranking already better than itself, **it levels to its own ceiling**, 0.800
down to 0.650. The mitigation is to narrow the judgement — not "is this enough?" but "does the
retrieved text state a euro amount for a voluntary change?" A checklist derived from the
sub-questions converts a fuzzy call into a lookup. Same lesson as pointwise beating listwise,
one level up.

Notice what the loop does not need: no planner framework, no tool-calling API, no agent class.
Six functions from `eval/retrieval.py` and a `for` loop with a break. Most agent frameworks are
this plus retries, tracing and a schema. Adopt one when you need the tracing, not to obtain the
behaviour.

At 10 million documents the loop stops being free. Every extra pass is another full ANN search,
so cap the fan-out and cache at sub-question level — the same three sub-questions recur across
thousands of misconnect cases, and a cache keyed on the normalised sub-question beats any model
swap. Put a router in front, because most questions are single-hop and should never pay for
decomposition. And make the tools explicit and typed — `search_sop`, `search_fare_rules`,
`search_interline`, each with its own filter — so the model picks a corpus instead of hoping
one index covers everything. The version filter that keeps v3 out belongs there, as a property
of the tool, not a hope about the ranking.

The honest limit: no number of passes produces a document nobody wrote. If the interline
partner's SOP conflicts with ours and no paragraph resolves it, the loop's failure mode is to
keep searching for text that does not exist. Bound it, and make it say so.

## Rewinding the day

Read the chain backwards, out loud, in one breath.

The model did not know our data and did not know that it did not know — it invented a "20-30%
ceza". So we looked at what training is, and found weights are a frozen photograph. So we
fine-tuned, and it worked, and then Q2 became Q3: **EUR 120 became EUR 90**, the weights still
said 120, and they could cite nothing. So we put the whole rule book in the prompt, and it fit,
and we paid for all 75 KB on every query. So we selected — and retrieval brought back garbage.
So we changed the embedder, because the default was English-only and nobody warned us. So we
changed the chunker, and found the header, not the row, was the failure. So we put it in
ChromaDB to make it real. So we added hybrid and reranking, and measured that a reranker is a
trade, not an upgrade. And then one question needed three documents at once, and every
single-shot method in the course scored 0.500 on it.

Ten gates, not one of them a definition. Each was a wall the previous module walked into. The
whole day is one number: **EUR 90**, quoted from a document, with the document named.

<div class="presenter-note">
Do the rewind standing, no slides, no laptop. It is the last eight minutes of the day and the
only part anyone repeats to a colleague. End on the EUR 90 sentence, then the exit line, then
stop talking. No summary slide after it.
</div>

## Exit line

> You just watched RAG turn into an agent.
