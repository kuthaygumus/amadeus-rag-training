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
the board. Neither is the answer, and the point of the next 35 minutes is that they were
confident. Do not show the ladder yet.
</div>

## The ladder

Same 28 documents, same `bge-m3` embeddings, same 20 questions, scored at document level. The
only variable is how the text was cut. Whole documents: hit@1 **0.550**. Blind 280-character
chunks: 0.700. Add 60 characters of overlap: 0.700. A recursive splitter, the one every tutorial
reaches for: 0.700. Split on the document's own headings: **0.800**, MRR 0.654 → **0.844**.

Read the middle of that ladder again. Three different naive strategies — fixed size, fixed size
with overlap, and a recursive splitter — all stop at exactly 0.700. Reaching for a smarter naive
splitter buys nothing here. Only cutting on the document's own structure gets past it.

Bottom of the ladder to the top is a 45% relative gain in hit@1, touching no model, costing
nothing at query time, using the three splitters in `eval/chunking.py`. It is not the largest
lever the day measured — on these same structure-aware chunks, module 6's embedder swap was worth
0.450 (`nomic-embed-text` 0.350 against `bge-m3`'s 0.800) where chunking is worth 0.250. It is the
one people leave on a default.

## The failure is the header, not the row

Everyone assumes fixed-size chunking cuts a table row in half. It does not. Here is chunk 6 of
`fare_classic_shorthaul.md` under a blind 280-character split — chunk numbers on this page are
0-based, the way `enumerate` prints them:

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

So the model reads a grid of euro amounts and guesses. Asked for the CLASSIC K cancellation
penalty, which is **EUR 90**, it answers **EUR 65** — the value the notebook's cached run holds,
and the same on three re-runs. EUR 65 is the change penalty of the row above, and that row is
class W, whose class label and fare basis were cut off at the chunk boundary. Wrong column and
wrong row. It does not hedge, because from where it sits nothing is ambiguous: there is a grid
of numbers and one of them is the answer.

<div class="presenter-note">
Ask out loud before revealing the boundary: "the row survived — so why is the answer wrong?"
Give them fifteen seconds. Someone will get it, and it lands ten times harder from a
participant. The sentence not to garble: <strong>the row survived; the header did not travel
with it.</strong> If the model answers something other than EUR 65, do not fight it — any wrong
number makes the point, and the point is that it never says it cannot tell.
</div>

**Overlap does not fix it.** Sixty characters of overlap adds 74 chunks across the corpus
(294 → 368) and moves the boundary. In the fare sheet the full header line now lands in chunk 5,
with its first two column names copied into the tail of chunk 4, and the K row is in chunk 7.
Still two apart. Overlap hedges against cutting a sentence; it does nothing about a reference
612 characters away.

**Recursive splitting does not fix it either.** This is the uncomfortable one: it is the
default in every framework and what most of the room already ships. It splits on paragraphs,
then lines, then sentences, and keeps prose intact — header in chunk 4, K row in chunk 5.
Adjacent, still separate. The reason cannot be tuned away: the table is wider than the split
size. A 600-character window cannot hold a nine-column header plus seven rows, so a cut lands
inside the table whatever the separator logic says.

**Structure-aware splitting fixes it.** Cut at the document's own boundaries — `##` headings,
`RULE n.`, `SECTION n` — and the table stays with the rule that introduces it: header and K row
both in chunk 4. Each chunk is then prefixed with the heading of the section it came from, so a
bare grid of numbers arrives already labelled:

```
[RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
| Booking class | Fare basis | Advance purchase | ... | Change penalty | Cancellation penalty | ...
```

The prefix is that heading line itself, not a path: `structure_aware()` falls back to the
document title only for the text above the first heading, and does not repeat the title on every
chunk. That prefix is the whole of what is sold as "contextual retrieval": tell each fragment
where it came from. It costs a string concatenation.

## Two reasons never to report one number

**Fixed-280 raises the average and degrades a category.** Exact-token queries — flight codes,
bulletin ids — score **1.000** over whole documents and **0.750** under fixed-280, and stay at
0.750 when you add overlap. Recursive-600 and structure-aware both bring them back to 1.000.
Headline hit@1 rose 0.550 → 0.700 while one of the four exact-token questions went the other
way underneath it. One question is a small thing to lose. The point is that nothing in the
headline number tells you it happened; only the per-type breakdown does.

**Recall@5 is highest for the worst strategy.** Fixed-280 scores **0.950** against
structure-aware's 0.833. Not a fluke — arithmetic: 294 small pieces give the gold document more
chances to appear in the top five while making it harder to rank first. Pick recall@5 and blind
chunking wins; pick hit@1 and it loses by 0.100. Both are honest. Choosing between them is
where you decide whether you are measuring or advertising.

## Noise is measurable, not cosmetic

The corpus carries the mess a real export carries, on purpose. Of 28 documents, **23** carry a
page-number artefact such as `Page 3 of 7` — **18** of them stranded on a line of their own
mid-table, which is the form `strip_boilerplate()` matches — **16** carry leftover HTML (`<br>`,
`&nbsp;`, `<div class="legal">`), and the six fare rule sheets close with the same 431-character
legal footer, byte for byte identical. Exactly one document repeats a body paragraph inside
itself: `macro_en_refund.md`, where the instruction to read "the **cancellation** column, not the
change column" appears twice in a row. Two more documents repeat a page footer, which is already
counted under the artefacts above.

`strip_boilerplate()` in `eval/chunking.py` removes the legal blocks, the page artefacts and the
leftover markup, and the corpus loses **4.2%** of its characters — 78,310 down to 75,037. A small
cut, and it moves the metric:

| | chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| structure-aware, as-is | 154 | 0.800 | 0.833 | 0.844 |
| structure-aware, cleaned | 153 | **0.850** | 0.833 | **0.869** |

One chunk disappears from the structure-aware index. Fourteen disappear from the blind one
(294 → 280 — a string count, no model involved), because structure-aware chunking had already
isolated the boilerplate in its own sections rather than smearing it through every chunk. We
scored the cleaned corpus under structure-aware only, so there is no cleaned fixed-280 row here.

That delta is measured, with cleaning as the only variable: **+0.050** hit@1 — one question of
twenty — **+0.025** MRR, and recall@5 unchanged. Twenty questions cannot separate 0.050 from
noise, so read the direction as plausible and the size as unproven. What would settle it is the
same benchmark on a gold set big enough that 0.05 means something.

## What you run

The mechanism and the measurement are two different artefacts in this module.

**The mechanism.** Open `notebooks/05_chunking_and_noise.py` in VS Code with the Microsoft
Python extension and run the blocks with Shift+Enter. Everything except two model calls is
string slicing, so it cannot stall on a slow laptop. The cell that matters most needs no model
at all: for each strategy, print which chunk holds the column header and which holds the K row.

```python
import chunking as C

pieces = C.fixed(sheet, size=280, overlap=0)
for n, piece in enumerate(pieces):
    if "| K |" in piece and "EUR 90" in piece:
        print(f"chunk {n}: THE K ROW")
    if "Booking class" in piece and "Cancellation penalty" in piece:
        print(f"chunk {n}: THE COLUMN HEADER")
```

*What you should see:* `chunk 4: THE COLUMN HEADER` and `chunk 6: THE K ROW`. Two chunks apart,
and the answer generated from the K-row chunk alone is EUR 65 where the correct answer is
EUR 90. *Roughly:* twelve minutes at the pace the notebook is written for.

**The measurement.** One command, one table:

```bash
python exercises/m7_chunking_ladder.py
```

*What you should see:* five ladder rows plus a sixth for the cleaned corpus, each with chunks,
hit@1, recall@5, MRR and the five question types, then the line
`Stripping the legal footer removed 4.2% of the characters.` The line to stop on is the one
reporting `exact_token went 1.000 -> 0.750 while hit@1 went 0.550 -> 0.700`. *Roughly:* two to
three minutes, nearly all of it embedding calls.

The corpus-wide chunk counts, if you want them in the notebook as well:

```python
for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = C.chunk_corpus(documents, strategy)
    print(strategy, len(texts))       # 294 · 368 · 197 · 154
```

<div class="presenter-note">
35 minutes: ladder 6, header demo 12, overlap and recursive 6, metric disagreement 6, noise 5.
Start the ladder command running before you open the header demo — it takes two to three minutes
and the demo covers it. If it fails on someone's laptop, read the numbers off this page and run
the boundary cell live: it is string slicing, it cannot fail. The boundary cell is the demo; the
ladder is the evidence. This measurement is not one to cut when the day runs late.
</div>

## What the numbers said

<div class="measured">

| strategy | chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| whole documents | 28 | 0.550 | 0.717 | 0.654 |
| fixed 280 | 294 | 0.700 | **0.950** | 0.814 |
| fixed 280 + overlap 60 | 368 | 0.700 | 0.883 | 0.799 |
| recursive 600 | 197 | 0.700 | 0.900 | 0.816 |
| **structure-aware 900** | 154 | **0.800** | 0.833 | **0.844** |

hit@1 by question type, same ladder:

| strategy | tr_tr | tr_en | en_en | exact_token | multi_hop |
|---|---|---|---|---|---|
| whole documents | 1.000 | 0.333 | 0.250 | 1.000 | 0.000 |
| fixed 280 | 0.750 | 0.667 | 0.750 | 0.750 | 0.500 |
| fixed 280 + overlap 60 | 1.000 | 0.667 | 0.500 | 0.750 | 0.500 |
| recursive 600 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 |
| structure-aware 900 | 1.000 | 0.667 | 0.750 | 1.000 | 0.500 |

Where the header lands in `fare_classic_shorthaul.md`, 0-based, no model involved:

| strategy | chunks in that file | header | K row | together? |
|---|---|---|---|---|
| whole documents | 1 | — | — | yes — nothing was cut |
| fixed 280 | 15 | 4 | 6 | no |
| fixed 280 + overlap 60 | 18 | 5 | 7 | no — overlap moves the boundary |
| recursive 600 | 10 | 4 | 5 | no — the table is wider than the split |
| structure-aware 900 | 10 | 4 | 4 | yes |

</div>

Twenty questions is not a benchmark. It is large enough to choose between two designs and far
too small to publish; a difference under about 0.05 is inside this sample's noise.

## Going deeper

The retriever never sees a document. It sees one vector per chunk — one point for the whole
span. Average a nine-column header, seven fare rows and a page-number artefact into a single
1024-dimensional point and you get something close to everything about short-haul fares and
specific about nothing. Chunk size therefore trades against precision independently of model
quality: a longer chunk carries more context for the generator and a blurrier vector for the
retriever. Structure-aware splitting wins because it makes those two pressures agree — a
section is both the natural unit of meaning and the natural unit of embedding.

The heading prefix does two jobs. The vector moves, because the heading's own words — "Reading
the schedule", "fare basis code", "change" — join a grid of numbers that otherwise says nothing
about what it is for. And the generator's input improves, because the text now states its own
provenance. Published contextual-retrieval work generates that sentence of context with an LLM
per chunk; we get most of the effect from a heading already in the file. Generate it when your
documents have no structure to borrow — scanned PDFs, chat logs, ticket dumps. Borrow it when
they do.

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
because the exact-token dip we measured here is invisible in any average.

One caution on our own numbers. Twenty questions decide between two designs; they do not
publish. Recursive-600 against fixed-280-with-overlap is 0.000 in hit@1 and 0.017 in MRR — one
question moving, meaningless. Whole documents against structure-aware is 0.250, five questions,
real. Read the size of the gap before the ranking.

<div class="presenter-note">
If someone says "we use a RecursiveCharacterTextSplitter and it is fine" — agree, then point at
the recursive-600 row: 0.700, better than whole documents, 0.100 worse than structure-aware, and
tied with the two strategies it was supposed to beat. It still separates the header from the K
row. A good default, not a solution for tables. Do not let it become a framework argument; the
measurement is the argument.
</div>

## Exit line

> hit@1 is 0.800, and 0.850 once the boilerplate is stripped — the best this corpus reaches all
> day, and the exact-token questions only held up because we stopped splitting blindly. That
> result is 154 vectors sitting in a Python list inside one notebook process. Where do they live
> when that process is gone?
