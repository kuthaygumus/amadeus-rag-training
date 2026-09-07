---
title: "7. Chunking, Noise and Measurement"
description: "Why did retrieval bring back garbage?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **Why did retrieval bring back garbage?**

Module 5 left us with a pipeline that worked and answers that did not. The retriever pulled
back text that was topically right and useless: a legal footer, half a revision history, a grid
of euro amounts with nothing to say which column was which. Nobody had touched the embedder,
the model or the prompt. The only thing between corpus and retriever was the code nobody argues
about — the splitter.

<div class="presenter-note">
Before running anything, put the K-class question back on screen and ask the room to vote on
what is broken. Expect "the model is too small" and "we need a better embedder". Write both on
the board. Neither is the answer, and the point of the next 40 minutes is that they were
confident. Do not show the ladder yet.
</div>

## The ladder

Same 28 documents, same `bge-m3` embeddings, same 20 questions, scored at document level. The
only variable is how the text was cut. Whole documents: hit@1 **0.550**. Blind 280-character
chunks: 0.650. Add 60 characters of overlap: 0.700. A recursive splitter, the one every tutorial
reaches for: 0.700. Split on the document's own headings: **0.800**, MRR 0.655 → **0.846**.

A 45% relative gain in hit@1, touching no model, costing nothing at query time, in about thirty
lines of Python. Chunking is the biggest lever measured all day, and the one people leave on a
default.

## The failure is the header, not the row

Everyone assumes fixed-size chunking cuts a table row in half. It does not. Here is chunk 6 of
`fare_classic_shorthaul.md` under a blind 280-character split:

```
 | none | none | EUR 65 | EUR 85 | EUR 170 | 1 x 23 kg | Yes |
| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |
| M | MSHEU26 | none | none | EUR 90 | EUR 120 | EUR 240 | 2 x 23 kg | Yes |

Page 3 of 7

RULE 3. Involuntary cases. Where the cancellation
```

The row is intact. Every number the answer needs is there. Missing is the line naming the
columns — `| Booking class | Fare basis | ... | Change penalty | Cancellation penalty | ...` —
which landed in **chunk 4**, two chunks earlier, and is never retrieved.

So the model reads `EUR 70 | EUR 90 | EUR 180` and guesses. It answers **EUR 70**, the change
penalty, where the cancellation penalty is **EUR 90**. It does not hedge. From where it sits
nothing is ambiguous: there is a first number, and first numbers are usually the answer.

<div class="presenter-note">
Ask out loud before revealing the boundary: "the row survived — so why is the answer wrong?"
Give them fifteen seconds. Someone will get it, and it lands ten times harder from a
participant. The sentence not to garble: <strong>the row survived; the header did not travel
with it.</strong>
</div>

**Overlap does not fix it.** Sixty characters of overlap adds 75 chunks (288 → 363) and moves
the boundary: the header now appears in chunks 4 and 5, because the overlap copies it, and the
K row is in chunk 7. Still two apart. Overlap hedges against
cutting a sentence; it does nothing about a reference 900 characters away.

**Recursive splitting does not fix it either.** This is the uncomfortable one: it is the
default in every framework and what most of the room already ships. It splits on paragraphs,
then lines, then sentences, and keeps prose intact — header in chunk 4, K row in chunk 5.
Adjacent, still separate. The reason cannot be tuned away: the table is wider than the split
size. A 600-character window cannot hold a nine-column header plus seven rows, so a cut lands
inside the table whatever the separator logic says.

**Structure-aware splitting fixes it.** Cut at the document's own boundaries — `##` headings,
`RULE n.`, `SECTION n` — and the table stays with the rule that introduces it: header and K row
both in chunk 4. Then prefix each chunk with its document title and section heading, so a bare
grid of numbers arrives labelled `[fare classic shorthaul > ## SECTION 4 — VOLUNTARY CHANGES
AND CANCELLATIONS]`. That prefix is the whole of what is sold as "contextual retrieval": tell
each fragment where it came from. It costs a string concatenation.

## Two reasons never to report one number

**Fixed-280 raises the average and halves a category.** Exact-token queries — flight codes,
bulletin ids — score **1.000** over whole documents and **0.500** under fixed-280, back to
1.000 only under structure-aware. Headline hit@1 rose 0.550 → 0.650 while a whole class of
question got twice as bad underneath it.

**Recall@5 is highest for the worst strategy.** Fixed-280 scores **0.950** against
structure-aware's 0.850. Not a fluke — arithmetic: 288 small pieces give the gold document more
chances to appear in the top five while making it harder to rank first. Pick recall@5 and blind
chunking wins; pick hit@1 and it loses by 0.150. Both are honest. Choosing between them is
where you decide whether you are measuring or advertising.

## Noise is measurable, not cosmetic

The corpus carries the mess a real export carries, on purpose. Of 28 documents, **23** have a
`Page 3 of 7` artefact stranded mid-table, **16** carry leftover HTML (`<br>`, `&nbsp;`,
`<div class="legal">`), and the six fare rule sheets close with the same legal footer word for word. Exactly one
document repeats a paragraph inside itself: `macro_en_refund.md`, where "read the cancellation
column, not the change column" appears twice in a row.

`strip_boilerplate()` in `eval/chunking.py` removes the legal blocks, the page artefacts and the
leftover markup, and the corpus loses **4.2%** of its characters — 77,114 down to 73,841. A small
cut, and it moves the metric on both strategies:

| | chunks | hit@1 | MRR |
|---|---|---|---|
| fixed-280, as-is | 288 | 0.650 | 0.789 |
| fixed-280, cleaned | 275 | **0.700** | **0.802** |
| structure-aware, as-is | 152 | 0.800 | 0.846 |
| structure-aware, cleaned | 151 | **0.850** | **0.871** |

Thirteen chunks disappear from the blind strategy and one from the structure-aware one, because
structure-aware chunking had already isolated the boilerplate in its own sections rather than
smearing it through every chunk. Both gain the same five points of hit@1 — cleaning helps whatever
you do, and 0.850 with MRR 0.871 is the best this corpus reaches anywhere in the course.

We have not measured a retrieval delta from cleaning alone: on 20 questions it would sit inside
the noise floor. What would settle it is the same benchmark on a gold set big enough that 0.05
means something, with cleaning as the only variable.

## What you run

Notebook: `05_chunking_and_noise.ipynb`

```bash
jupyter lab notebooks/05_chunking_and_noise.ipynb
python eval/run_benchmark.py     # the same comparison, from the command line
```

It calls the same functions the benchmark calls, from `eval/chunking.py`:

```python
from chunking import fixed, recursive, structure_aware, chunk_corpus, to_documents
from metrics import load_gold, evaluate, compare

for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = chunk_corpus(documents, strategy)
    print(strategy, len(texts))
```

The cell that matters most needs no model at all: for each strategy, print which chunk holds
the column header and which holds the K row.

<div class="presenter-note">
40 minutes: ladder 8, header demo 12, overlap and recursive 8, metric disagreement 6, noise 6.
If the benchmark is slow on the room's laptops, read the retrieval numbers from cache and run
the chunk-boundary cell live — it is string slicing, it cannot fail. The boundary cell is the
demo; the benchmark is the evidence.
</div>

## What the numbers said

<div class="measured">

| strategy | hit@1 | recall@5 | MRR | chunks |
|---|---|---|---|---|
| whole documents | 0.550 | 0.717 | 0.655 | 28 |
| fixed 280 | 0.650 | **0.950** | 0.789 | 288 |
| fixed 280 + overlap 60 | 0.700 | 0.867 | 0.795 | 363 |
| recursive 600 | 0.700 | 0.900 | 0.816 | 195 |
| **structure-aware 900** | **0.800** | 0.850 | **0.846** | 152 |

| strategy | header stays with the K row? | exact-token hit@1 |
|---|---|---|
| whole documents | n/a — nothing was cut | 1.000 |
| fixed 280 | no (header chunk 4, K row chunk 6) | 0.500 |
| fixed 280 + overlap 60 | no (chunks 4-5 vs chunk 7) — overlap moves the boundary | — |
| recursive 600 | no (chunk 4 vs chunk 5) — the table is wider than the split | — |
| structure-aware 900 | yes (both in chunk 4) | 1.000 |

</div>

## Going deeper

The retriever never sees a document. It sees one vector per chunk — one point for the whole
span. Average a nine-column header, seven fare rows and a page-number artefact into a single
1024-dimensional point and you get something close to everything about short-haul fares and
specific about nothing. Chunk size therefore trades against precision independently of model
quality: a longer chunk carries more context for the generator and a blurrier vector for the
retriever. Structure-aware splitting wins because it makes those two pressures agree — a
section is both the natural unit of meaning and the natural unit of embedding.

The heading prefix does two jobs. The vector moves, because "VOLUNTARY CHANGES AND
CANCELLATIONS" pulls the chunk toward cancellation queries it would otherwise miss. And the
generator's input improves, because the text now states its own provenance. Published
contextual-retrieval work generates that sentence of context with an LLM per chunk; we get most
of the effect from a heading already in the file. Generate it when your documents have no
structure to borrow — scanned PDFs, chat logs, ticket dumps. Borrow it when they do.

At scale you stop chunking tables at all. The durable fix for a filed tariff is a different
representation: parse the table once and emit one row per chunk with the header expanded into
it — `CLASSIC short-haul, booking class K, fare basis KSHEU26, change penalty EUR 70,
cancellation penalty EUR 90`. The ambiguity that produced the wrong answer cannot occur,
because the column name now sits next to the value. That is work per document type, and it is
what separates a demo from a system.

At 10 million documents nothing changes conceptually and everything changes operationally.
Chunking becomes a versioned offline batch job: change the splitter and you re-embed the
corpus, so you need to know which strategy produced any given vector. Keep the parent id on
every chunk and fetch neighbours at query time — retrieve the chunk, serve the section — which
recovers most of what small chunks lose without paying the blur. And stop trusting 20
questions: sample real queries from logs, label a few hundred, report stratified by query type,
because the exact-token collapse we just measured is invisible in any average.

One caution on our own numbers. Twenty questions decide between two designs; they do not
publish. Recursive-600 versus fixed-280-with-overlap is 0.000 in hit@1 and 0.021 in MRR — one
question moving, meaningless. Whole documents versus structure-aware is 0.250, five questions,
real. Read the size of the gap before the ranking.

<div class="presenter-note">
If someone says "we use a RecursiveCharacterTextSplitter and it is fine" — agree, then point at
the recursive-600 row: 0.700, better than blind splitting, 0.100 worse than structure-aware,
and it still separates the header from the K row. A good default, not a solution for tables.
Do not let it become a framework argument; the measurement is the argument.
</div>

## Exit line

> hit@1 is 0.800. But fixed-280 had halved exact-token retrieval — what else is the average hiding?
