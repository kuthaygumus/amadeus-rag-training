---
title: "7. Chunking, Noise and Measurement"
description: "Why did retrieval bring back garbage?"
---

## Gate question

> **Why did retrieval bring back garbage?**

The embedder is settled and the number did not move: `bge-m3` over 28 whole documents is still
hit@1 **0.600**. So it was not the embedder. Module 5 left us with a pipeline that worked and
answers that did not — retrieval put `fare_classic_shorthaul` at rank 1, a hit, and the model
still answered for the wrong booking class, because the prompt carried the first 1,500 characters
of a 3,923-character document and the row holding the answer sits past that cut. We cut blindly,
once, and lost the answer. The only thing between the corpus and the prompt is the code nobody
argues about — the cut.

<div class="presenter-note">
Before running anything, put the K-class question back on screen and ask the room to vote on
what is broken. Expect "the model is too small" and "we need a better embedder". Write both on
the board. Neither is the answer, and the point of the next 35 minutes is that they were
confident. Do not show the ladder yet.
</div>

## The ladder

Same 28 documents, same `bge-m3` embeddings, same 20 questions, scored at document level. The
only variable is how the text was cut. Whole documents: hit@1 **0.600**. Blind 280-character
chunks: 0.700. Add 60 characters of overlap: **0.550**. A recursive splitter, the one every
tutorial reaches for: 0.700. Split on the document's own headings: **0.750**. Strip the
boilerplate first and that same split reaches **0.800**.

Read the third rung again. Overlap is the one move on this ladder that goes backwards, and it
goes backwards on everything: hit@1 falls 0.700 → 0.550, MRR 0.817 → 0.720, and recall@5 falls
too, 0.917 → 0.883. It also costs you 74 more chunks and the time to embed them. Its hit@1 is
below the rung where you did not chunk at all. Overlap is sold as the safe default — the move you
make when you are not sure, because surely a little redundancy cannot hurt. On this corpus it is
the only change that costs on every axis at once.

Before reading any gap here as a result: hit@1 over 20 questions moves in steps of 0.05, because
0.05 *is* one question. Every gap on this ladder is one question wide except the overlap drop,
which is three. So the direction is the lesson and the digit is not — that is not a hedge bolted
onto the result, it is the size of the instrument that produced it.

Bottom to top is three questions of twenty, touching no model, costing nothing at query time,
using the splitters in `eval/chunking.py`. It is not the largest lever the day measured — on
these same structure-aware chunks, module 6's embedder choice was worth 0.400
(`nomic-embed-text` 0.350 against `bge-m3`'s 0.750). It is the one people leave on a default.

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
penalty, which is **EUR 90**, the recorded run answered **EUR 65** — the change penalty of the
row above, and that row is class W, whose class label and fare basis were cut off at the chunk
boundary. Wrong column and wrong row. Generation is not seeded, so which wrong cell comes back
can differ between runs; that it is confidently wrong, and never hedges, does not.

<div class="presenter-note">
Ask out loud before revealing the boundary: "the row survived — so why is the answer wrong?"
Give them fifteen seconds. Someone will get it, and it lands ten times harder from a
participant. The sentence not to garble: <strong>the row survived; the header did not travel
with it.</strong> If the model answers something other than EUR 65, do not fight it — any wrong
number makes the point, and the point is that it never says it cannot tell.
</div>

**Overlap does not fix it, and it charges for the attempt.** Sixty characters of overlap adds 74
chunks across the corpus (294 → 368) and moves the boundary. In the fare sheet the full header
line now lands in chunk 5, with its first two column names copied into the tail of chunk 4, and
the K row is in chunk 7. Still two apart. Overlap hedges against cutting a sentence; it does
nothing about a reference 612 characters away — and those 368 near-duplicate pieces are what
drags hit@1 down to 0.550, the worst ranking on the ladder.

**Recursive splitting does not fix it either.** This is the uncomfortable one: it is the default
in every framework and what most of the room already ships. It splits on paragraphs, then lines,
then sentences, and keeps prose intact — header in chunk 4, K row in chunk 5. Adjacent, still
separate. The reason cannot be tuned away: the table is wider than the split size. A
600-character window cannot hold a nine-column header plus seven rows, so a cut lands inside the
table whatever the separator logic says. Hold on to this rung; its recall@5 is 0.900, second only
to plain fixed-280's 0.917 and well above anything that ranks better than it.

**Structure-aware splitting fixes it.** Cut at the document's own boundaries — `##` headings,
`RULE n.`, `SECTION n` — and the table stays with the rule that introduces it: header and K row
both in chunk 4. Each chunk is then prefixed with the heading of the section it came from, so a
bare grid of numbers arrives already labelled:

```
[RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
| Booking class | Fare basis | Advance purchase | ... | Change penalty | Cancellation penalty | ...
```

The prefix is that heading line itself, not a path: `structure_aware()` falls back to the
document title only above the first heading. That prefix is the whole of what is sold as
"contextual retrieval" — tell each fragment where it came from — and it costs a string
concatenation. It leads both ranking columns, hit@1 0.750 and MRR 0.817, on the lowest recall@5
of any chunked strategy, 0.833.

## Two reasons never to report one number

**An average that improves can hide a category that got worse.** The four `exact_token`
questions — a literal flight number or bulletin id, plus one fare-table lookup — score **1.000**
over whole documents. Fixed-280 drops them to 0.750, and adding overlap drops them to **0.500**,
two of the four gone. Recursive-600 and structure-aware are both back at 1.000. So fixed-280
raised the headline average, 0.600 → 0.700, and paid for it out of exactly the category the room
will ask about first; the overlap rung paid twice, in the average and in the category at once.
Nothing in a headline number tells you either happened. Only the per-type table does.

**The best recall@5 does not belong to the best ranker.** Fixed-280 tops that column at
**0.917** while sitting at hit@1 0.700 — and all three naive cuts beat structure-aware on
recall@5, 0.917, 0.883 and 0.900 against 0.833. Arithmetic, not a fluke: more small pieces give
the gold document more chances to appear in the top five while making it harder to rank first.
Pick recall@5 and blind chunking wins; pick hit@1 and it loses. Both are honest. Choosing
between them is where you decide whether you are measuring or advertising.

## Noise is measurable, not cosmetic

The corpus carries the mess a real export carries, on purpose, and the notebook counts it across
the 28 documents: **6** documents close with the same legal footer, byte for byte identical;
**25** page-number artefacts such as `Page 3 of 7`, some stranded on a line of their own
mid-table, which is the form `strip_boilerplate()` matches; and **42** scraps of leftover HTML
(`<br>`, `&nbsp;`, `<div class="legal">`). One document repeats a body paragraph inside itself —
`macro_en_refund.md`, where the instruction to read "the **cancellation** column, not the change
column" appears twice in a row.

`strip_boilerplate()` in `eval/chunking.py` removes the legal blocks, the page artefacts and the
leftover markup, and the corpus loses **4.2%** of its characters — 78,310 down to 75,037. A small
cut, and it moves the metric:

| | chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| structure-aware, as-is | 154 | 0.750 | 0.833 | 0.817 |
| structure-aware, cleaned | 153 | **0.800** | 0.833 | **0.841** |

Only one chunk disappears from the structure-aware index, because structure-aware chunking had
already isolated the boilerplate in its own sections rather than smearing it through every chunk.
We scored the cleaned corpus under structure-aware only, so there is no cleaned fixed-280 row.
The delta is measured with cleaning as the only variable: **+0.050** hit@1 and **+0.024** MRR,
recall@5 unchanged — one question of twenty, so read the direction, not the size. And the whole
gain lands in one place: `en_en` goes 0.500 → 0.750 and nothing else moves.

## What you run

The mechanism and the measurement are two different artefacts in this module.

**VS Code — `notebooks/05_chunking_and_noise.py`, the `# %%` block after the first markdown
cell:** run the blocks with Shift+Enter. Everything except two model calls is string slicing, so
it cannot stall on a slow laptop. The cell that matters most needs no model at all: for each
strategy, print which chunk holds the column header and which holds the K row. `sheet` and `C`
come from the notebook's first block — this is a cell in the file, not something to type.

```python
pieces = C.fixed(sheet, size=280, overlap=0)
print(f"{len(pieces)} chunks\n")
for n, piece in enumerate(pieces):
    has_row = "| K |" in piece and "EUR 90" in piece
    has_header = "Booking class" in piece and "Cancellation penalty" in piece
    if has_row or has_header:
        print(f"chunk {n}: {'THE K ROW' if has_row else ''}{'THE COLUMN HEADER' if has_header else ''}")
```

**What you should see.** `chunk 4: THE COLUMN HEADER` and `chunk 6: THE K ROW`. Two chunks apart,
and the answer generated from the K-row chunk alone is EUR 65 where the correct answer is EUR 90.
**Roughly how long.** Twelve minutes at the pace the notebook is written for.

**Terminal (repo root):** one command, one table.

```bash
python exercises/m7_chunking_ladder.py   # leave it running while you work through the notebook
```

**What you should see.** Five ladder rows plus a sixth for the cleaned corpus, each with chunks,
hit@1, recall@5, MRR and the five question types, then the line
`Stripping the legal footer removed 4.2% of the characters.` The line to stop on is
`The column to argue about: cutting into fixed 280-character pieces moved hit@1 0.600 -> 0.700 while exact_token went 1.000 -> 0.750.`
**Roughly how long.** Two to three minutes, nearly all of it embedding calls.

**VS Code — a new `# %%` block at the end of `05_chunking_and_noise.py`,** if you want the
corpus-wide chunk counts there as well:

```python
for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = C.chunk_corpus(docs, strategy)
    print(strategy, len(texts))       # 294 · 368 · 197 · 154
```

<div class="presenter-note">
35 minutes: ladder 6, header demo 12, overlap and recursive 6, metric disagreement 6, noise 5.
Start the ladder command in the repo-root terminal before you open the notebook — it takes two to
three minutes and the demo covers it. If it fails on someone's laptop, read the numbers off this
page and run the boundary cell live: it is string slicing, it cannot fail. The script derives its
own closing paragraphs from the run it just did, so read them off the screen rather than
preparing them: it will name the overlap rung as the one that argues with itself, and it will
name fixed 280 as the recall@5 winner that ranks only 0.700. Both are the arguments this module
turns on, and they arrive in the script's words, not yours — which is the point. The boundary
cell is the demo; the ladder is the evidence. This measurement is not one to
cut when the day runs late.
</div>

## What the numbers said

<div class="measured">

| strategy | chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| whole documents | 28 | 0.600 | 0.717 | 0.677 |
| fixed 280 | 294 | 0.700 | **0.917** | 0.817 |
| fixed 280 + overlap 60 | 368 | **0.550** | 0.883 | 0.720 |
| recursive 600 | 197 | 0.700 | 0.900 | 0.806 |
| structure-aware 900 | 154 | 0.750 | 0.833 | 0.817 |
| **structure-aware + boilerplate stripped** | 153 | **0.800** | 0.833 | **0.841** |

hit@1 by question type, same ladder:

| strategy | tr_tr | tr_en | en_en | exact_token | multi_hop |
|---|---|---|---|---|---|
| whole documents | 1.000 | 0.333 | 0.500 | 1.000 | **0.000** |
| fixed 280 | 1.000 | 0.500 | 0.750 | 0.750 | 0.500 |
| fixed 280 + overlap 60 | 0.750 | 0.500 | 0.500 | **0.500** | 0.500 |
| recursive 600 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 |
| structure-aware 900 | 1.000 | **0.667** | 0.500 | 1.000 | 0.500 |
| + boilerplate stripped | 1.000 | **0.667** | **0.750** | 1.000 | 0.500 |

The `multi_hop` column is worth one sentence on its own. Over whole documents it is 0.000 — not
one of the two multi-hop questions puts the right document first. Every chunked strategy reaches
0.500, and none of them gets past it. Chunking buys you the first of the two and no cut buys the
second, which is the wall module 10 is built to climb.

Where the header lands in `fare_classic_shorthaul.md`, 0-based, no model involved:

| strategy | chunks in that file | header | K row | together? |
|---|---|---|---|---|
| whole documents | 1 | — | — | yes — nothing was cut |
| fixed 280 | 15 | 4 | 6 | no |
| fixed 280 + overlap 60 | 18 | 5 | 7 | no — overlap moves the boundary |
| recursive 600 | 10 | 4 | 5 | no — the table is wider than the split |
| structure-aware 900 | 10 | 4 | 4 | yes |

</div>

The `multi_hop` column is 0.000 in every row, before and after every change on this page. No cut
of the text reaches it, which is the failure module 10 is built on.

## Going deeper

The retriever never sees a document. It sees one vector per chunk — one point for the whole
span. Average a nine-column header, seven fare rows and a page-number artefact into a single
1024-dimensional point and you get something close to everything about short-haul fares and
specific about nothing. Chunk size therefore trades against precision: a longer chunk carries
more context for the generator and a blurrier vector for the retriever. Structure-aware
splitting wins because it makes those two pressures agree — a section is both the natural unit
of meaning and the natural unit of embedding.

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
cancellation penalty EUR 90` — and the ambiguity that produced the wrong answer cannot occur.
That is work per document type, and it turns chunking into a versioned offline batch job: change
the splitter and you re-embed the corpus, so you have to know which strategy produced any given
vector.

<div class="presenter-note">
If someone says "we use a RecursiveCharacterTextSplitter and it is fine" — agree, then point at
the recursive-600 row: hit@1 0.700, recall@5 0.900, and it still separates the header from the
K row. A good default, not a solution for tables. The honest answer to "which is
better" is that one question separates it from structure-aware, and the header demo — which needs
no model and no benchmark — is the part that does not depend on twenty questions. Do not let it
become a framework argument; the measurement is the argument.
</div>

## Exit line

> hit@1 is 0.750, and 0.800 once the boilerplate is stripped — one question apart, so 0.750 is
> the number the rest of the day argues with, and it is the best this corpus reaches all day. A
> splitter bought that, not a model. It is also 154 vectors sitting in a Python list inside one
> notebook process. Where do they live when that process is gone?
