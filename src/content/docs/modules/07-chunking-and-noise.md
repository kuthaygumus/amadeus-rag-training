---
title: "7. Chunking and Noise"
description: "The retriever brought the wrong slice. Fix the slicing, not the model."
---

## Gate question

> **The right document came back and the answer was still wrong. What do we change — the model, or the cut?**

[Module 6](/modules/06-chromadb/) ended with the chunks on screen. `fare_classic_shorthaul.md`
was at rank 1 — a hit — and the chunk that held the K row did not hold the line naming the
columns. The model read `EUR 70 | EUR 90 | EUR 180` with no idea which column was which and
answered from the wrong one. On the 12 Sep run the fixed-280 index produced "EUR 13"; your
number will differ, its confidence will not.

Nobody in the room wrote the splitter. It is the one piece of code between the corpus and the
prompt that was left on a default. **The row survived; the header did not travel with it.** That
is a slicing failure, and slicing is string code — the cheapest thing in the pipeline to change.

<div class="presenter-note">
Before touching Bruno, ask for a vote: "what is broken — the model, the embedder, or something
else?" Expect "model too small" and "better embedder". Write both on the board; neither is the
answer, and the point of the next 30 minutes is that they were confident. Do not reveal the
three strategies yet — let step 1's <code>chunks</code> count be the first surprise.
</div>

## Three ways to cut, cheapest first

All three live in `src/chunking.ts` of `amadeus-rag-lab`, and `/ingest` picks one by name.

**`fixed`** — cut every N characters, blind to what is there. Default 280. This is what
[module 5](/modules/05-simple-rag/) ingested with, and it is the villain: a table row is rarely
torn in half, but the header line lands in the previous chunk and is never retrieved with it.

**`recursive`** — cut at the most natural boundary that fits, then fall back to finer ones:
headings, then paragraphs, then lines, then sentences, then spaces. Default 600. Prose survives.
A table longer than the window does not — a nine-column header plus seven rows is wider than
600 characters, so the cut still lands inside the table, whatever the separator logic says. This
is the default in every framework and what most of the room already ships.

**`structure`** — cut at the document's own headings: `#` lines, `RULE n.`, `SECTION n`,
`Step n.`. A rule stays with the table it introduces. Then every chunk is prefixed with where it
came from — `[doc title > heading]`, the title being the file name with underscores as spaces.
A bare grid of numbers arrives already labelled:

```
[fare classic shorthaul > RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
penalty and the cancellation penalty are distinct amounts and must not be substituted for one
another; the no-show penalty is derived from the cancellation penalty under RULE 4 below.

| Booking class | Fare basis | Advance purchase | Minimum stay | Change penalty | Cancellation penalty | ...
```

The prefix is the heading line itself — for `RULE 2A` that is the whole first line of the
paragraph, because the rule's heading and its first sentence share a line in this corpus. It
costs a string concatenation. It is also the whole of what is sold as "contextual retrieval".

**Why 1500 and not 900.** The `RULE 2A` section — prose, header, seven rows, the stray
`Page 3 of 7` — is 1 140 characters. At a structure chunk size of 900 the section is cut in two,
the table leaves its sentence behind, and the table chunk is not in the top-10 for the day's
question. At 1500 the section is one chunk and ranks. The default is 1500 for that one reason;
it was measured, not guessed.

## Overlap: the safe default that loses

Overlap is sold as the move you make when you are not sure — surely a little redundancy cannot
hurt. Sixty characters of overlap on fixed-280 takes the Q3 corpus from 294 chunks to 368, and
all it does to the fare sheet is move the boundary: the header still sits in an earlier chunk
than the K row. Overlap hedges against cutting a sentence. It does nothing about a reference
hundreds of characters away, and the 368 near-duplicate pieces compete with each other for the
top-K. On the trainer-side eval it loses on every metric. Numbers in the box below.

## Noise is not cosmetic

The corpus carries what a real export carries, on purpose. Open `corpus/2026-Q3/fare_classic_shorthaul.md`
and read the tail: a `<div class="legal">` block — the same LEGAL NOTICE closes every fare-rule
sheet — a `&nbsp;` left over from HTML, and a `Page 3 of 7` stranded on its own line between the
table and `RULE 3`. Six of the 28 documents close with that same legal block.

`stripBoilerplate` in `src/chunking.ts` removes exactly that kind of thing, before chunking: the
legal block, `LEGAL NOTICE` and `DISCLAIMER:` paragraphs, `Page n of n` lines, stray `<div>`
`<br>` `<span>` `<p>` tags and `&nbsp;`, and a `Distribution:` footer. Then it collapses the
blank runs it left.

Why before chunking: boilerplate that survives into the index produces chunks whose only content
is shared with every other sheet. Six near-identical legal tails compete for three top-K slots,
and none of them answers anything. On this corpus the structure chunk count does not move
(132 → 132) because the legal block has no heading of its own — it rides at the end of the last
rule's chunk — so stripping shortens chunks rather than removing them. The vectors move anyway:
a `RULE 7` chunk stops being one-third legal notice.

## What you run

Same 28 documents, same `bge-m3`, same `gemma3:4b`, same Chroma. Only the chunks change.

**Bruno — `05-chunking-and-noise` › `1-reingest-structure`**

```json
{
  "edition": "2026-Q3",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

Read `chunks`: **132**, against the 294 that `02-ingest` built. Read `collection`:
`kraken-2026-Q3-structure-1500-strip` — one collection per configuration, so the fixed-280 index
is still there if you want to go back. Read `stages.embed`: this is the only slow request of the
module, and that is why. `sample` shows the first three chunks with their `[title > heading]`
prefix.

**Bruno — `05-chunking-and-noise` › `2-peek-again`**

`GET /chunks?source=fare_classic_shorthaul&limit=20`. Find the `| K |` row in `chunks[].text`.
This time the header row is in the same chunk, and the chunk starts with
`[fare classic shorthaul > RULE 2A …]`. Compare `chars` with what `02-ingest › peek-chunks`
showed: fewer, longer pieces.

**Bruno — `05-chunking-and-noise` › `3-query-again`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

Read `answer`: EUR 90, with `[fare_classic_shorthaul]` cited. Read `sources[]`: the table chunk
is now among them — the `excerpt` starts with the prefix. Read `debug.prompt` if you want to see
that the model was handed the header and the row together for the first time today.

**Nothing about the model got better. A splitter won this, not a model.**

<div class="presenter-note">
Thirty minutes, of which the live requests are five: 1 is the slow one (it embeds 132 chunks —
read <code>stages.embed</code> aloud), 2 and 3 are seconds. Before revealing the <code>| K |</code>
chunk in step 2, ask: "the row survived in module 6 — so why was the answer wrong?" Fifteen
seconds of silence, then someone says "the header". It lands ten times harder from a
participant. If Ollama is down, <code>/ingest</code> cannot embed and the module has no live
leg — read the page, then on your own machine run <code>npm test</code> in the repo root:
<code>test/chunking.test.ts</code> needs no model and asserts exactly this — the fixed test
checks the K-row chunk does <em>not</em> contain "Cancellation penalty", the structure test checks
the <code>[fare classic shorthaul &gt; ## RULE 3 …]</code> prefix on its toy sheet.
</div>

### Try it

**Recursive.** Put `"strategy": "recursive"` in step 1 (default size 600, 197 chunks), then
rerun 2 and 3. Prose chunks look clean. The K row and its header are still in different chunks,
because the table is wider than the window — and no separator list fixes that.

**topK 5.** Add `"topK": 5` to step 3 and read `sources[]`. Does `fare_classic_longhaul` sneak
in? Long-haul CLASSIC K is EUR 195 — a distractor that is the *right* answer to a slightly
different question. This is where the chat model earns its keep: with both tables in context,
`gemma3:4b` picks 90. The model this course used until 12 Sep, `qwen2.5:3b`, picked 195 in every
prompt variant we tried, which is why it is gone. Retrieval decides what the model sees; the
model still has to read it.

## What the numbers said

<div class="measured">

Trainer-side only — `npm run eval` in the `amadeus-rag-lab` repo, not something participants
run. 20 gold questions, edition 2026-Q3, embedder `bge-m3`, scored at **document** level (the
best chunk of a document counts as a hit for that document). One run each on an M-series Mac,
12 Sep 2026; numbers drift on other machines.

| collection | chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| fixed-280 | 294 | 0.700 | **0.917** | **0.817** |
| fixed-280 + overlap 60 | 368 | **0.550** | 0.883 | 0.717 |
| recursive-600 | 197 | 0.700 | 0.900 | 0.806 |
| structure-1500 | 132 | 0.700 | 0.833 | 0.781 |
| structure-1500 + strip | 132 | 0.650 | 0.850 | 0.766 |

Read it honestly. **Structure-aware chunking does not find the right document more often** —
hit@1 is 0.700 on three of the five rows, and the stripped index is one question lower. What it
fixes is the **chunk**: the K row together with its header, which is what turns a confident wrong
amount into EUR 90. A document-level score cannot see that; step 2 can.

Two things the table does say. Overlap loses on every metric at once and costs 74 more chunks to
do it. And the best recall@5 belongs to the most fragmented index: more small pieces give the gold
document more chances to appear in the top five while making it harder to rank first. Pick
recall@5 and blind chunking wins; pick hit@1 and nothing separates them. Both are honest.

A step of 0.05 is one question in twenty. Read the direction, not the digit.

</div>

## Going deeper

The retriever never sees a document. It sees one 1024-dimensional point per chunk. Average a
header, seven rows and a page number into one point and you get something close to everything
about short-haul fares and specific about nothing. A longer chunk is more context for the
generator and a blurrier vector for the retriever; structure-aware splitting wins because a
section is both the natural unit of meaning and the natural unit of embedding.

The heading prefix does two jobs. The vector moves, because "fare classic shorthaul" and
"Reading the schedule" join a grid that otherwise says nothing about what it is for. And the
generator's input improves, because the text states its own provenance. Published
contextual-retrieval work generates that sentence with an LLM per chunk; we get most of the
effect from a heading already in the file. Generate it when your documents have no structure to
borrow — scanned PDFs, chat logs, ticket dumps. Borrow it when they do.

At scale you stop chunking tables at all. The durable fix for a filed tariff is a different
representation: parse the table once and emit one row per chunk with the header expanded into it
— `CLASSIC short-haul, booking class K, fare basis KSHEU26, change penalty EUR 70, cancellation
penalty EUR 90` — and the ambiguity that produced the wrong answer cannot occur. That is work per
document type, and it makes chunking a versioned offline job: change the splitter and you
re-embed the corpus. The lab already encodes this — the collection name
`kraken-2026-Q3-structure-1500-strip` says which splitter produced every vector in it.

<div class="presenter-note">
If someone says "we use a RecursiveCharacterTextSplitter and it is fine" — agree, then point at
the recursive row: hit@1 0.700, a good default, and step 2 under recursive still separates the
header from the K row. A good default for prose, not a solution for tables. Do not let it become
a framework argument: the peek in step 2 needs no benchmark and no model, and that is the part of
this module that does not depend on twenty questions.
</div>

## Exit line

> Right chunk, right answer — EUR 90, and not one weight changed. But the fine-tuned model of
> module 3 also knows a cancellation penalty for class K. Now prove RAG beats it where it
> matters: [the data moves](/modules/08-freshness/).
