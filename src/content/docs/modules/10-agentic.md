---
title: "10. Agentic RAG — The Finale"
description: "What if one shot is not enough?"
---

## Gate question

> **What if one shot is not enough?**

Two questions in the gold set are typed `multi_hop`, and neither of them moved all day. Their
hit@1 is **0.000** on whole documents and **0.500** on every rung of the chunking ladder — fixed
size, fixed size with overlap, recursive, structure-aware, with and without boilerplate stripping
— chunking is the only thing that moves it off zero, and 0.500 is not a win. Fusion undoes even
that: BM25 and RRF fall back to 0.000 on it over chunks, and only dense retrieval holds the 0.500.
Across the three embedders on structure-aware chunks the row reads 0.500, 0.500 and 0.500 —
MiniLM, `nomic-embed-text` and `bge-m3` alike — which `eval/RESULTS.md` says to read as one of the
two questions becoming reachable once documents are chunked, not as anyone's capability.

Reranking is the only other lever pulled today, and on the strongest setup it pulled exactly one
question up from below rank 1 — q14, neither of the `multi_hop` pair. `UNVERIFIED: what reranking does to the multi_hop
row at document level — the default benchmark run that prints it was not part of the re-run, so
the earlier claim that it lifted the row from 0.000 to 0.500 is withdrawn.`

Document-level scoring would flatter these two if they ever scored at all: it calls it a hit when
*one* of three gold documents lands at rank 1. They never got that far. Put q19 through the best
single pass we have — `bge-m3` over structure-aware chunks — and of the three documents it needs,
the top five hold **zero**.

And it does not look like a failure. q19's own `why` field names what comes back instead —
`macro_tr_misconnect`, `macro_tr_rebook`, `faq_en_general` — all about exactly this situation, all
stating the no-repricing rule, none willing to carry a figure. A plausible document at rank 1
passes a spot check. The question scores zero while looking right.

The question itself, from `eval/gold_questions.jsonl`, exactly as stored:

> XX 1487 gecikti, CLASSIC K sınıfındaki yolcum CDG'de YY 88 bağlantısını kaçırıyor ve yeni
> uçuşa kadar 5 saat bekleyecek. Bu beklemede kendisine ne vermem gerekiyor, Wyvern bacağı
> yeniden fiyatlandırılır mı, bir de yolcu bugünkü seferi hiç istemeyip kendi isteğiyle yarına
> geçmek isterse ondan ne kadar değişiklik ücreti alırım?

Three questions wearing one coat. The care during the wait is in `sop_misconnect_v4.md`; whether
the Wyvern coupon survives is Clause 4 of `interline_xx_yy.md`; the change fee is in
`fare_classic_shorthaul.md`. No file holds two of the three, and the corpus is built that way on
purpose: the SOP "establishes no monetary penalty, waiver or refund value of any kind", and
Clause 4.4 does not determine the value of any voucher. The documents point at each other. Real
rule books do.

### Which fare sheet — RULE 7

Two fare sheets could govern one traffic document, and the corpus settles it in a rule that
appears on both — `fare_classic_shorthaul` RULE 7 and `fare_classic_longhaul` RULE 7:

> the sheet is chosen by the transaction and not by the document. A voluntary change to a single
> coupon is assessed on the sheet for the band of the Kraken sector held [...] A cancellation or
> refund of the journey as a whole is assessed on the LONG-HAUL CLASSIC sheet.

So q19, a voluntary change on the European sector, takes the short-haul sheet: CLASSIC K change
penalty **EUR 70**. q20, a cancellation of the whole journey, takes the long-haul sheet: CLASSIC K
cancellation penalty **EUR 195**. Quoting the short-haul cancellation, EUR 90, would be a right row
on the wrong sheet.

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
On q19 it was the wrong one.

Nothing in the pipeline can fix this, because nothing in the pipeline is broken. A better
embedder moves the point; it does not split it. A better chunker sharpens each candidate; it
does not add a second query. Reranking reorders what was retrieved, and what was retrieved never
held the other two documents. Every lever we pulled today acts on a single pass. The single pass
is the problem.

## The loop

Four steps, no new machinery.

**Decompose.** One model call turns the question into standalone sub-questions, capped at four:
what care is owed for a five-hour wait; is the YY segment protected and is it repriced; what is
the CLASSIC K short-haul voluntary change penalty.

**Retrieve per sub-question.** Each gets its own embedding and its own top-k against the same
`bge-m3` index over the same structure-aware chunks from module 7. Three points instead of one.

**Check sufficiency.** One model call answers `YES`, or `NO` followed by one short query for what
is missing. A narrow question, not an open one — the same reason module 9's pointwise reranking
worked and listwise did not: handed six passages and asked to put them in order, `qwen2.5:3b`
returned `1,4,2,5`, four indices for six passages; asked to score them one at a time it returned
six usable integers in six calls and 3.3 s. A small model answers a narrow question well and a
wide one badly.

**Answer with citations,** naming the document behind each number.

Until now the shape was fixed — retrieve, then generate, exactly once, decided by us. Now
retrieval has a caller, and no library was involved: a loop, a stopping condition, and a function
the model can ask us to run. Be precise about how far that goes. The model decides *when to stop*
and *what to ask next*; our code decides what to do with that. Tool *selection* is still ours,
because there is only one tool — and that is the next thing to make explicit.

## What the run actually produces

The loop reaches two of the three documents. On q19 it goes from **0 of 3** to **2 of 3** —
`fare_classic_shorthaul` and `sop_misconnect_v4` come back, two more search rounds for the fare
basis of booking class K bring back nothing new, and `interline_xx_yy` never arrives. Then read
the answer it writes anyway, because this is where the finale stops being clean.

In the recorded run the model gets the meal voucher and the change penalty right, drops the hotel
threshold and the repricing question without saying it could not find them, and adds a
**cancellation** penalty of **EUR 90** for **booking class M** that nobody asked about. Your run
will word it differently — the decomposition is generated, so this is the one part of the day
that is not deterministic — but the shape recurs: a document the loop never found becomes a
silent gap in the answer instead of a stated one. Look for the shape, not the wording.

What the documents actually say, each with the paragraph it comes from:

- a **EUR 15** meal voucher, because the wait exceeds three hours — `sop_misconnect_v4` 3.4
- **no hotel**, because five hours does not exceed six — `sop_misconnect_v4` 3.5
- the Wyvern segment is protected and reaccommodated **without repricing** — `interline_xx_yy`
  4.1 and 4.2
- the voluntary change costs **EUR 70** — `fare_classic_shorthaul`, RULE 7 and the CLASSIC K row

Notice what the model added that nobody asked for. **EUR 70** is the change penalty for class K,
the question actually on the table; **EUR 90** is the short-haul cancellation penalty, one column
over on the same row — module 7's exact confusion, except this time nothing handed it the wrong
cell, it volunteered one. Structure-aware chunking is why the header stayed attached to the row it
did retrieve, so the change penalty came out right. Retrieval still fell one document short.
Reading, on what it had, was still not clean.

That is the honest close of the day, and it is worth saying out loud rather than letting the
finale look tidier than it is. Every metric in this course scores **retrieval** — whether the
right document came back. Not one scores whether the answer was right. Two systems, two failure
modes, and we measured one of them. Build the retrieval eval first, because it is cheap and
deterministic; then build a second one for the answers, because the first will never tell you the
second is broken.

**And the loop does not always win.** On q20 it reached two of the three documents — the same
two a single pass reached. `fare_classic_longhaul`, the sheet RULE 7 selects for a cancellation
of the whole journey, never came back. Two questions cannot measure a method. They can show that
one exists, and that it is not free.

<div class="presenter-note">
Before running the loop, have the room write down the three sub-questions they expect. Then run
the decompose cell alone and compare. It will match nobody exactly, and that is the lesson: the
plan is generated, not written. Then run the answer cell and read it out with the corpus open —
do not skip past the bad answer to get to the applause. The sentence not to garble:
<strong>retrieval stopped being a step before the model and became a tool the model calls.</strong>
Say it once and let it sit.
</div>

## The bill

**More calls.** One decompose, one retrieval per sub-question, one sufficiency check per round,
one answer: six to ten model calls where naive RAG made one — three or four of them cheap
embedding calls, the rest generations. For scale, module 9's reranker costs **8 model calls per
question**, 160 for a 20-question pass, and on the strongest setup that pass took **71 seconds**;
the full four-setup sweep was 640 calls and 440 seconds. The loop is the same order, not a new
category. What differs is that a loop multiplies per-call latency by the round count instead of
adding to it, so a slower generation model does not cost you a little more. It costs you the
round count more.

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
28 minutes: gate 4, one-pass failure 5, the loop 10, reading the answer 5, breaking it 4. This
module is on the never-cut list — the trainer-drives saving is already spent as the default, so
there are no minutes here to recover. The rewind below is not in this budget — it belongs to the
15-minute closing block at 14:33.
</div>

## What you run

**You watch this one.** The trainer drives it; the notebook is in the repo and runs on your own
laptop afterwards on the same two dependencies as every other module. Follow along live if you
want — but the room is better spent on one screen here.

**VS Code, repository root as the open folder — `notebooks/07_agentic_rag.py`:** open it, put the
cursor in the first `# %%` block and press `Shift+Enter` down the file. Ollama has been serving
since module 0 — the app on macOS, your session's background service on Windows; you have not had
to start it by hand.

- **what you should see** — the single-pass cell prints `of the 3 documents needed, retrieval
  found 0`; after the loop, `final coverage:` lists two of the three — `interline_xx_yy` stays
  missing; the comparison table at the end prints q19 as `0 → 2` and q20 as `2 → 2`
- **roughly how long** — the loop cells are seconds each; in the recorded run the two-question
  comparison took 4.9 s for q19 and 7.9 s for q20 on an M-series Mac, longer on a CPU-only laptop

**If a cell will not run.** The closing comparison cell is recorded, under the key
`07-single-shot-vs-agentic` in `notebooks/cached_runs.json` — 2026-09-09, M-series Mac.

**Terminal (repo root):**

```bash
USE_CACHED=1 python notebooks/07_agentic_rag.py
```

`notebooks/_cached.py` then replays that cell instead of calling the model, printing a `[CACHED]`
banner with the date and machine so a replay can never be passed off as a live run. It is the only
recorded cell here: the decompose, sufficiency and answer cells above it still call the model, so
a cached run gives you the outcome — q19 `0 → 2`, q20 `2 → 2` — and this page supplies the
walkthrough.

<div class="presenter-note">
<strong>Fallback, and when to reach for it.</strong> This is the lowest-energy slot of the day,
it is your laptop alone on the projector, and the rewind that follows is the part the room takes
home. The moment you are debugging rather than teaching — Ollama not answering, a model missing,
a cell outliving the room's patience — stop, set <code>USE_CACHED=1</code>, and re-run. The
recorded run puts the closing table on screen with a <code>[CACHED]</code> banner and you narrate
the loop off this page. Say out loud that it is a replay; the banner is on screen anyway. Do not
debug live in front of the room during the finale.
</div>

The loop itself, and nothing else — the corpus, chunking and retriever cells above it are the
same ones module 5 onwards already ran:

**VS Code — `notebooks/07_agentic_rag.py`, the block after the two prompt cells:**

```python
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

`decompose` and `sufficient` are the two prompts written in the notebook, and `dense` and
`questions` come from the indexing cell above, so this block runs after them and not before. One
defect to carry into your own code: `lstrip("NO")` strips a *character set*, not a prefix, so a
follow-up beginning "NOT enough on hotels" arrives as "T enough on hotels". The notebook has it
the same way. Strip the prefix with a regex.

## What the numbers said

<div class="measured">

| | measured |
|---|---|
| `multi_hop` questions in the gold set | 2 of 20 |
| gold documents q19 needs | 3 — `sop_misconnect_v4`, `interline_xx_yy`, `fare_classic_shorthaul` |
| `multi_hop` hit@1, whole documents | 0.000 |
| `multi_hop` hit@1, every rung of the chunking ladder | **0.500** |
| `multi_hop` hit@1, three embedders on structure-aware chunks | **0.500** · **0.500** · **0.500** — MiniLM, `nomic-embed-text`, `bge-m3`, identically |
| `multi_hop` hit@1, dense / BM25 / RRF | dense **0.500** over chunks, 0.000 over whole documents; BM25 and RRF 0.000 at both granularities |
| the setup the loop runs on (`bge-m3` + structure-aware) | hit@1 **0.750**, MRR **0.817** |
| best rung on the ladder (structure-aware + boilerplate stripped) | hit@1 **0.800**, MRR **0.841** |
| q19 gold documents reached — single pass / loop | **0 of 3** / **2 of 3**, 4.9 s |
| q20 gold documents reached — single pass / loop | 2 of 3 / 2 of 3, 7.9 s |
| rerank pass cost, strongest setup | 8 model calls per question, 160 per setup, 71 s across the 20 questions |
| superseded `sop_misconnect_v3` | EUR 10 voucher, 8-hour hotel threshold |
| current `sop_misconnect_v4` | EUR 15 voucher, 6-hour hotel threshold |

The ladder, embedder, fusion and rerank rows are `eval/RESULTS.md`; the q19 and q20 rows are the
recorded run in `notebooks/cached_runs.json`, 2026-09-09.

The loop is **not in the benchmark table**. Two multi-hop questions cannot measure a method;
they can only show that one exists. What would settle it: fifty labelled multi-hop questions,
scored on whether *all* gold documents were retrieved rather than whether one was, and a second
eval that scores the answer rather than the retrieval. Until then the claim on this page is the
narrow one — on q19 a single pass returned none of the three documents and the loop returned two
of the three, and on q20 the loop returned what the single pass already had.

</div>

## Going deeper

Decomposition is query rewriting with a budget, and everything measured today still applies to
each sub-question — now written by the model rather than a human. Better on average, because its
phrasing drifts toward the vocabulary of the corpus; worse in the tail, because there is no error
signal for asking the wrong thing well.

The sufficiency check is the interesting part and the weakest part. You are asking a 3B model to
judge the completeness of its own evidence, and module 9's reranker result applies here unchanged:
a 3B model's opinion levels to its own ceiling. The mitigation is to narrow the judgement — not
"is this enough?" but "does the retrieved text state a euro amount for a voluntary change?" A
checklist derived from the sub-questions converts a fuzzy call into a lookup. Same lesson as
pointwise beating listwise, one level up.

Notice what the loop does not need: no planner framework, no tool-calling API, no agent class.
Two things out of `eval/retrieval.py` — `generate` and `DenseRetriever` — plus three prompts you
write in the notebook and a `for` loop with a break. Most agent frameworks are this plus
retries, tracing and a schema. Adopt one when you need the tracing, not to obtain the behaviour.

At 10 million documents the loop stops being free, because every extra pass is another full ANN
search. Cap the fan-out and cache at sub-question level — the same three sub-questions recur
across thousands of misconnect cases, and a cache keyed on the normalised sub-question beats any
model swap. Put a router in front, because most questions are single-hop and should never pay for
decomposition. And make the tools explicit and typed — `search_sop`, `search_fare_rules`,
`search_interline`, each with its own filter. The version filter that keeps v3 out belongs there,
as a property of the tool rather than a hope about the ranking, and so does RULE 7: a tool that
knows whether it is pricing a change or a cancellation picks the sheet before retrieval, instead
of asking a 3B model to notice a rule buried at the end of both.

The honest limit: no number of passes produces a document nobody wrote. If the interline
partner's SOP conflicts with ours and no paragraph resolves it, the loop's failure mode is to
keep searching for text that does not exist. Bound it, and make it say so.

## Rewinding the day

This is the closing block, not part of module 10's 28 minutes. Read the chain backwards, out
loud, in one breath.

The model did not know our data and did not know that it did not know — asked for the CLASSIC K
cancellation penalty it invented a "20-30% ceza". So we looked at what training is, and found
weights are a frozen photograph. So we built a fine-tune — 695 generated pairs drilling the Q2
answer — and then Q2 became Q3 and that row moved to **EUR 90**. We never got to watch it be
wrong: `UNVERIFIED: the stale fine-tuned answer — kraken-q2 has not been built and no run of it
exists.` What module 3 shows without a model is the shape of the problem: weights have no way to
know a row changed, and no way to cite the row they learned. So we put the whole rule book in the
prompt, and it fit, and we paid for all **79,309** characters of it on every query. So we
selected — and retrieval brought back garbage. So we checked the embedder we had been handed, and
found the one every framework picks by default scores **0.000** on all six Turkish-question,
English-document pairs. So we changed the chunker, and found the header, not the row, was the
failure. So we put it in ChromaDB to make it real — and watched its silent default embedder get
two questions wrong in two lines of code. So we added hybrid and reranking, and measured that a
reranker levels to its own ceiling: a trade, not an upgrade. And then two questions needed three
documents at once, and the `multi_hop` row reads **0.500** on every chunked rung and **0.000**
only before chunking — on q19, not one of the three documents it needs came back at all in a
single pass.

Nine gates and one setup, not one of them a definition. Each was a wall the previous module
walked into. The whole day is one number: **EUR 90**, quoted from a document, with the document
named — and one caveat, which is that we measured whether the document came back, never whether
the sentence built on it was right.

<div class="presenter-note">
Do the rewind standing, no slides, no laptop. It is the last part of the day and the only part
anyone repeats to a colleague. Two things to say honestly rather than smoothly: the fine-tune
line is a prediction, not something this room watched — <code>kraken-q2</code> was never built —
and the gaps between the ladder's rungs are one question wide, so quote the direction and not the
digit. End on the EUR 90 sentence, then the exit line, then stop talking. No summary slide after
it. This is the 15-minute closing block at 14:33, which also carries the repo link and the
handout.
</div>

## Exit line

> Every step today was born where the previous one stopped being enough. The last one turned
> RAG into an agent — and that is exactly where the agent day picks it up.
