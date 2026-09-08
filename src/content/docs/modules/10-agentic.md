---
title: "10. Agentic RAG — The Finale"
description: "What if one shot is not enough?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **What if one shot is not enough?**

Two questions in the gold set are typed `multi_hop`, and neither of them moved all day. Their
hit@1 is **0.000** on whole documents and **0.500** on every chunking strategy on the ladder —
fixed size, fixed size with overlap, recursive, structure-aware, all four the same. It is 0.500
on all three embedders module 6 compared, including the one that took everything else to 0.800.
Pointwise reranking did not shift it either: on the strongest setup, q19's first gold document
moved from rank 6 to rank 5, and q20's was already at rank 1.

That 0.500 flatters them, because document-level scoring calls it a hit when *one* gold document
lands at rank 1. Here is what the number is hiding. Put q19 through the best single pass we
have — `bge-m3` over structure-aware chunks — and the top five documents come back as:

```
faq_en_general   macro_tr_rebook   codeshare_h9_au_conditions
sop_denied_boarding   macro_tr_misconnect
```

Of the three documents the question needs, that is **zero of three**. And it does not look like
a failure. `faq_en_general` and `macro_tr_rebook` are both about exactly this situation. They
refuse to carry a figure and send the agent somewhere else instead. A single number at rank 1
would have looked fine on a spot check.

Here is the question, as an agent would type it:

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

### Which fare sheet — RULE 7

The passenger holds one traffic document with a European Helios sector and an intercontinental
Aurora sector on it, so two fare sheets could govern. The corpus settles it in a rule that
appears on both of them, `fare_classic_shorthaul` RULE 7 and `fare_classic_longhaul` RULE 7:

> the sheet is chosen by the transaction and not by the document. A voluntary change to a single
> coupon is assessed on the sheet for the band of the Helios sector held [...] A cancellation or
> refund of the journey as a whole is assessed on the LONG-HAUL CLASSIC sheet.

q19 asks for a voluntary change on the European sector, so the short-haul sheet governs and the
CLASSIC K change penalty is **EUR 70**. q20 — the other multi-hop question, on the Atlantic
band — asks what to collect when the passenger abandons the journey, so the long-haul sheet
governs and the CLASSIC K cancellation penalty is **EUR 195**. Quoting the short-haul
cancellation, EUR 90, would be a right row on the wrong sheet.

<div class="presenter-note">
Put the Turkish question on screen and read it out loud, slowly, the way a caller says it. Ask
the room: "how many documents does this need?" Take answers — you will get one and two. Only
then reveal three, and show the two sentences where the SOP and the agreement each refuse to
answer the other's part. RULE 7 is worth thirty seconds on the projector because two questions
in the gold set turn on it. 4 minutes, laptops closed, nothing runs yet.
</div>

## Why one pass returns one document

Embed that question and you get one vector. One point, but the question has three centres of
mass — duty of care, interline protection, fare penalty. The point lands between them, nearest
whichever the phrasing weighted most, and the top five come back from that one neighbourhood.
One retrieval pass gets you one neighbourhood, and on q19 it was the wrong one.

Nothing in the pipeline can fix this, because nothing in the pipeline is broken. A better
embedder moves the point; it does not split it. A better chunker sharpens each candidate; it
does not add a second query. Reranking reorders what was retrieved, and what was retrieved
never held the other two documents. Every lever we pulled today acts on a single pass. The
single pass is the problem.

## The loop

Four steps, no new machinery.

**Decompose.** One model call turns the question into standalone sub-questions, capped at four:
what care is owed for a five-hour wait; is the AU segment protected and is it repriced; what is
the CLASSIC K short-haul voluntary change penalty.

**Retrieve per sub-question.** Each gets its own embedding and its own top-k against the same
`bge-m3` index over the same structure-aware chunks from module 7. Three points instead of one.

**Check sufficiency.** One model call answers `YES`, or `NO` followed by one short query for
what is missing. A narrow question, not an open one — the same reason pointwise reranking beat
listwise on the retired probe corpus (5/5 against 2/5 across 5 questions and 10 documents; the
direction is real, the magnitude is unproven at that size). A small model answers a narrow
question well and a wide one badly.

**Answer with citations,** naming the document behind each number.

## What the run actually produces

The loop reaches the documents. On q19 it goes from **zero of three** to **three of three**.
Then read the answer it writes, because this is where the finale stops being clean.

In the recorded run the model quoted the six-hour hotel threshold with the fifteen-euro meal
amount attached to it, as if that were what a hotel costs, and then contradicted itself about
the change fee across its last two sentences. Your run will word it differently — the
decomposition is generated, so this is the one part of the day that is not deterministic — but
the shape recurs. Look for the shape, not the wording.

What the documents actually say, each with the paragraph it comes from:

- a **EUR 15** meal voucher, because the wait exceeds three hours — `sop_misconnect_v4` 3.4
- **no hotel**, because five hours does not exceed six — `sop_misconnect_v4` 3.5
- the Aurora segment is protected and reaccommodated **without repricing** — `interline_h9_au`
  4.1 and 4.2
- the voluntary change costs **EUR 70** — `fare_classic_shorthaul`, RULE 7 and the CLASSIC K row

Notice which two the model slid between. **EUR 70** is the change penalty and **EUR 90** is the
cancellation penalty, one column apart on the same K row — the exact confusion module 7 was
about. Structure-aware chunking is why the header is still attached to that row at all, so the
column is there to be read. It was read wrong anyway. Retrieval was fixed. Reading was not.

That is the honest close of the day, and it is worth saying out loud rather than letting the
finale look tidier than it is. Every metric in this course scores **retrieval** — whether the
right document came back. Not one of them scores whether the answer was right. Those are two
systems with two failure modes, and we have been measuring one of them. If you carry one thing
into your own project: build the retrieval eval first, because it is cheap and deterministic,
then build a second one for the answers, because the first will never tell you the second is
broken.

**And the loop does not always win.** On q20 it reached two of the three documents — the same
two a single pass reached. `fare_classic_longhaul`, the sheet RULE 7 selects for a cancellation
of the whole journey, never came back, in either run we tried. Two questions cannot measure a
method. They can show that one exists, and that it is not free.

<div class="presenter-note">
Before running the loop, have the room write down the three sub-questions they expect. Then run
the decompose cell alone and compare. It will match nobody exactly, and that is the lesson: the
plan is generated, not written. Then run the answer cell and read it out with the corpus open —
do not skip past the bad answer to get to the applause. The sentence not to garble:
<strong>retrieval stopped being a step before the model and became a tool the model calls.</strong>
Say it once and let it sit.
</div>

## What actually changed

Until now the shape was fixed — retrieve, then generate. Retrieval happened before the model,
decided by us, exactly once. Now the model issues the queries, reads what came back, and
decides whether to go again. Retrieval became a tool with a caller, and control flow moved out
of our code into the model's output. That is the whole definition of an agent, and no library
was involved: a loop, a stopping condition, and a function the model can ask us to run.

## The bill

**More calls.** One decompose, one retrieval per sub-question, one sufficiency check per round,
one answer: six to ten model calls where naive RAG made one. Three of those passes are
embedding calls rather than generations, so they are the cheap part; the generations are not.
For scale, the reranker we priced in module 9 costs **8 model calls per query** and took
**65.2 seconds across the twenty questions** on the strongest setup — the loop is the same
order, not a new category. What is different is that a loop multiplies per-call latency by the
round count instead of adding to it, so a slower generation model does not cost you a little
more. It costs you the round count more.

**Non-determinism.** The plan is generated, so two runs can decompose, retrieve and cite
differently. Temperature 0, and log every sub-question — that list is the artefact you debug.
Without it a wrong answer is unattributable.

**More passes, more chances to retrieve the wrong thing.** The corpus holds
`sop_misconnect_v3.md`, superseded, saying **EUR 10** and an **8-hour** hotel threshold against
v4's EUR 15 and 6 hours. Each extra pass is another opportunity to pull it. Filter on the
version line before chunks reach the model.

**The loop that never ends.** If the sufficiency check can always say "still missing
something", it will. The notebook bounds it three ways: a hard `MAX_ROUNDS`, a cap of four
sub-questions, and a stop when a round brings back nothing new. Add the fourth bound in
production — on the last round the model answers with what it has and states what it could not
find. An honest partial answer beats an infinite loop, and beats a confident invented one,
which is what module 1 watched the bare model do when it invented a "20-30% ceza".

<div class="presenter-note">
Break the loop live: edit the sufficiency prompt so it can never say YES, re-run, watch the
round counter hit the cap and stop. Ten seconds, and it is the part they need in production.
28 minutes: gate 4, one-pass failure 5, the loop 10, reading the answer 5, breaking it 4. The
rewind below is not in this budget — it belongs to the 15-minute closing block at 14:33.
</div>

## What you run

**You watch this one.** The trainer drives it; the notebook is in the repo and runs on your own
laptop afterwards with the same two dependencies as every other module. If you want to follow
along live, nothing stops you — but the room is better spent watching one screen here.

Open `notebooks/07_agentic_rag.py` in VS Code and run the blocks with `Shift+Enter`. `ollama
serve` has been running since module 0.

- **what you should see** — the single-pass cell prints `of the 3 documents needed, retrieval
  found 0`; after the loop, `final coverage:` lists all three; the comparison table at the end
  prints q19 as `0 → 3` and q20 as `2 → 2`
- **roughly how long** — the loop cells are seconds each; the two-question comparison at the end
  ran in under 15 seconds on an M-series Mac, longer on a CPU-only laptop

The same functions the benchmark uses. Nothing here is a new dependency:

```python
import _preflight; _preflight.ready(chat=True, embed=True)   # cwd -> notebooks/, eval/ on the path
from pathlib import Path
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)

task = [q for q in questions if q["type"] == "multi_hop"][0]      # q19

MAX_ROUNDS = 3
gathered: dict[str, str] = {}
for sub in decompose(task["query"]):                              # one R.generate() call
    for chunk_id in dense.rank(sub)[:3]:
        gathered.setdefault(chunk_id, sub)                        # dedupe by chunk id

for round_number in range(MAX_ROUNDS):                            # hard round cap
    enough, verdict = sufficient(task["query"], list(gathered))   # "YES", or NO + a query
    if enough:
        break
    follow_up = verdict.split("\n")[-1].lstrip("NO").strip(" .:,-") or task["query"]
    before = len(gathered)
    for chunk_id in dense.rank(follow_up)[:3]:
        gathered.setdefault(chunk_id, follow_up)
    if len(gathered) == before:
        break                                                     # nothing new came back
```

`decompose` and `sufficient` are the two prompts written in the notebook, so this block runs after
those two cells and not before them. Everything else — `generate`, `DenseRetriever`,
`chunk_corpus`, `load_gold` — is the code module 5 onwards already ran. The single-pass cell sits
directly above, so both outputs land on one screen: zero documents against three.

## What the numbers said

<div class="measured">

| | measured |
|---|---|
| `multi_hop` questions in the gold set | 2 of 20 |
| gold documents q19 needs | 3 — `sop_misconnect_v4`, `interline_h9_au`, `fare_classic_shorthaul` |
| `multi_hop` hit@1, whole documents | 0.000 |
| `multi_hop` hit@1, all four chunked strategies on the ladder | 0.500 |
| `multi_hop` hit@1, all three embedders on structure-aware chunks | 0.500 |
| q19, rank of first gold document before / after pointwise rerank | 6 / 5 |
| best single-shot overall (bge-m3 + structure-aware) | hit@1 **0.800**, MRR **0.844** |
| q19 gold documents reached — single pass / loop | **0 of 3** / **3 of 3** |
| q20 gold documents reached — single pass / loop | 2 of 3 / 2 of 3 |
| pointwise vs listwise rerank, retired probe corpus of 10 documents and 5 questions | 5/5 vs 2/5 (MRR 1.000 vs 0.600) — direction real, magnitude unproven at n=5 |
| rerank pass cost, strongest setup | 8 model calls per query, 65.2 s across 20 questions |
| superseded `sop_misconnect_v3` | EUR 10 voucher, 8-hour hotel threshold |
| current `sop_misconnect_v4` | EUR 15 voucher, 6-hour hotel threshold |

The loop is **not in the benchmark table**. Two multi-hop questions cannot measure a method;
they can only show that one exists. What would settle it: fifty labelled multi-hop questions,
scored on whether *all* gold documents were retrieved rather than whether one was, and a second
eval that scores the answer rather than the retrieval. Until then the claim on this page is the
narrow one — on q19 a single pass returned none of the three documents and the loop returned all
three, and on q20 the loop returned what the single pass already had.

</div>

## Going deeper

Decomposition is query rewriting with a budget. Everything measured today about retrieval
quality still applies to each sub-question, now written by the model rather than a human —
better on average, because its phrasing drifts toward the vocabulary of the corpus, and worse
in the tail, because a badly rewritten sub-question fails silently. There is no error signal
for asking the wrong thing well.

The sufficiency check is the interesting part and the weakest part. You are asking a 3B model
to judge the completeness of its own evidence, and module 9 measured what that opinion is
worth: imposed on a ranking already better than itself, **it levels to its own ceiling**, hit@1
0.800 down to 0.600 and MRR 0.844 down to 0.717. The mitigation is to narrow the judgement —
not "is this enough?" but "does the retrieved text state a euro amount for a voluntary change?"
A checklist derived from the sub-questions converts a fuzzy call into a lookup. Same lesson as
pointwise beating listwise, one level up.

Notice what the loop does not need: no planner framework, no tool-calling API, no agent class.
Two things out of `eval/retrieval.py` — `generate` and `DenseRetriever` — plus three prompts you
write in the notebook and a `for` loop with a break. Most agent frameworks are this plus
retries, tracing and a schema. Adopt one when you need the tracing, not to obtain the behaviour.

At 10 million documents the loop stops being free. Every extra pass is another full ANN search,
so cap the fan-out and cache at sub-question level — the same three sub-questions recur across
thousands of misconnect cases, and a cache keyed on the normalised sub-question beats any model
swap. Put a router in front, because most questions are single-hop and should never pay for
decomposition. And make the tools explicit and typed — `search_sop`, `search_fare_rules`,
`search_interline`, each with its own filter — so the model picks a corpus instead of hoping
one index covers everything. The version filter that keeps v3 out belongs there, as a property
of the tool, not a hope about the ranking. So does RULE 7's distinction: a tool that knows
whether it is pricing a change or a cancellation can pick the sheet before retrieval, instead of
asking a 3B model to notice a rule buried at the end of both.

The honest limit: no number of passes produces a document nobody wrote. If the interline
partner's SOP conflicts with ours and no paragraph resolves it, the loop's failure mode is to
keep searching for text that does not exist. Bound it, and make it say so.

## Rewinding the day

This is the closing block, not part of module 10's 28 minutes. Read the chain backwards, out
loud, in one breath.

The model did not know our data and did not know that it did not know — it invented a "20-30%
ceza". So we looked at what training is, and found weights are a frozen photograph. So we
fine-tuned, and it worked, and then Q2 became Q3: **EUR 120 became EUR 90**, the weights still
said 120, and they could cite nothing. So we put the whole rule book in the prompt, and it fit,
and we paid for all 78,310 characters of it on every query. So we selected — and retrieval
brought back garbage. So we changed the embedder, because the default was English-only and
nobody warned us. So we changed the chunker, and found the header, not the row, was the
failure. So we put it in ChromaDB to make it real. So we added hybrid and reranking, and
measured that a reranker is a trade, not an upgrade. And then two questions needed three
documents at once, and no single-shot method in the course got them above **0.500** — on q19,
zero of the three documents it needed came back at all.

Ten gates, not one of them a definition. Each was a wall the previous module walked into. The
whole day is one number: **EUR 90**, quoted from a document, with the document named — and one
caveat, which is that we measured whether the document came back, never whether the sentence
built on it was right.

<div class="presenter-note">
Do the rewind standing, no slides, no laptop. It is the last part of the day and the only part
anyone repeats to a colleague. End on the EUR 90 sentence, then the exit line, then stop
talking. No summary slide after it. This is the 15-minute closing block at 14:33, which also
carries the repo link and the handout.
</div>

## Exit line

> Every step today was born where the previous one stopped being enough. The last one turned
> RAG into an agent — and that is exactly where the agent day picks it up.
