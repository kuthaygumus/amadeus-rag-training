# Measured results

Every number here was produced by the scripts in this repository, running entirely on a laptop.
Sections 1 to 7 score the corpus in `corpus/2026-Q3/` — 28 documents, 76 KB — against the 20
questions in `gold_questions.jsonl` through local Ollama; the appendix at the end records the one
measurement on this course that has nothing to do with retrieval. Kraken Air is a fictional
airline and the corpus is synthetic training material written for this course; nothing in it is an
Amadeus document.

Retrieval is scored at document level: a chunk ranking is collapsed to its parent documents,
best chunk first. Generation and reranking are `qwen2.5:3b`, embeddings are `bge-m3` unless
stated otherwise.

> **Twenty questions is not a benchmark.** One question is worth 0.05, and hit@1 can only move in
> steps of 0.05. The two gaps this page's conclusion actually rests on — structure-aware chunking
> over the best naive strategy, and boilerplate stripping over structure-aware — are exactly one
> question each. The overlap row's drop is three, and the whole ladder spans 0.250, which is five
> questions of twenty. Do not read 0.750 against 0.700 as a result; read it as two configurations
> that disagreed about one question. What held up across runs is the *direction*: structure-aware
> chunking and boilerplate stripping on top, fixed-size-plus-overlap worst on hit@1 and worse than
> not chunking at all, `multi_hop` never above 0.500 anywhere on the page. Quote the direction; do
> not quote the gaps. These numbers are large enough to decide between two designs and far too
> small to publish.

## Reproducing each section

Run these from the repository root, with Ollama serving `bge-m3`, `nomic-embed-text` and
`qwen2.5:3b`. `eval/README.md` carries the same list with the call counts, which do not vary.

Timings are from the re-run that produced this page: one pass of each command, back to back on one
M-series Mac with the models already resident. They are an order of magnitude, not a
specification. Where a command prints its own total, that total is the figure quoted; where it
prints only per-rung timings, those are quoted and no total is invented.

| section | command | notes |
|---|---|---|
| 1 — the ladder | `python exercises/m7_chunking_ladder.py` | the five ladder rungs indexed and queried in 8, 11, 14, 16 and 26 s; the script prints no total of its own. `python eval/run_benchmark.py --chunking` prints the same ladder plus the boilerplate-strip row and reports **70.1 s**, 126 embed calls |
| 2 — the column header | `python notebooks/05_chunking_and_noise.py`, or open it in VS Code and run the blocks | prints no timing of its own; the two answers quoted below are its only model calls |
| 3 — the embedders | `python exercises/m6_embedding_bakeoff.py` | 6.1 s, 16.0 s and 18.5 s to index and query the three embedders, no total printed; the only command in the repository that includes ChromaDB's default embedder |
| 4 — fusion | `python eval/run_benchmark.py --fusion` over chunks, `python eval/run_benchmark.py --skip-rerank` over whole documents | **33.4 s** and **31.5 s**; both are embedding-only, 21 and 42 embed calls |
| 5 — the reranker | `python exercises/m9_rerank_trade.py` (`--quick` on a slow laptop) | 640 model calls, 160 per setup; the four setups took 191, 127, 51 and 71 s, and considerably longer on a CPU-only laptop. `python eval/run_benchmark.py --rerank-sweep` prints the same four setups |
| 6 — context size | `python notebooks/03_stuff_the_prompt.py`, or open it in VS Code and run the blocks | 100 s of model time for the four-column table, 75 of it in the whole-corpus column alone; the four columns took 6.6, 7.7, 10.2 and 75.4 s |
| 7 — by question type | `python eval/run_benchmark.py --chunking` and `python exercises/m6_embedding_bakeoff.py` | both print hit@1 by question type under their own table; no extra run |

The notebooks are percent-format `.py` files. They run end to end as a script, or are opened in
VS Code with the Microsoft Python extension and run block by block; the `# %%` markers are the
block boundaries. The only dependencies are `numpy` and `chromadb`.

## 1. Chunking is the single biggest lever

| | hit@1 | recall@5 | MRR | chunks |
|---|---|---|---|---|
| whole documents, no chunking | 0.600 | 0.717 | 0.677 | 28 |
| fixed 280 chars | 0.700 | **0.917** | 0.817 | 294 |
| fixed 280 + 60 overlap | 0.550 | 0.883 | 0.720 | 368 |
| recursive, 600 chars | 0.700 | 0.900 | 0.806 | 197 |
| structure-aware, 900 chars | 0.750 | 0.833 | 0.817 | 154 |
| **structure-aware + boilerplate stripped** | **0.800** | 0.833 | **0.841** | 153 |

Five things worth stopping on, and one warning that governs all of them: every value in the hit@1
column is a count out of 20, so the smallest gap that can exist is 0.05 — one question. Recall@5
moves in smaller steps only because the two `multi_hop` questions have three gold documents each
and can be part-scored.

**Overlap is the one move on the ladder that goes backwards, and it goes backwards on everything.**
Adding 60 characters of overlap to fixed-280 chunking drops hit@1 from 0.700 to 0.550 — three
questions — MRR from 0.817 to 0.720, and recall@5 from 0.917 to 0.883. There is no column in which
it pays. It also produces the most chunks of any rung, 368 against fixed-280's 294, so it is the
most expensive way to index this corpus as well as the worst way to rank it. Worst of all, 0.550 is
*below the 0.600 you get by not chunking at all*: the safe default, applied by reflex, is the only
configuration on this page that would have been better left off. You only find that out by scoring
it.

**The best recall@5 does not belong to the best ranker.** Fixed-280 tops the recall column at
**0.917** while sitting at hit@1 0.700, level with recursive-600 and below structure-aware. All
three naive cuts beat structure-aware on recall@5 — 0.917, 0.883 and 0.900 against 0.833 — and
none of them out-ranks it. The recall winner is simply the one that cut smaller: 294 pieces
against 153. More pieces, more chances to appear somewhere in the top five; harder to be the one at
the top. recall@5 asks whether the answer arrived at all, hit@1 asks whether it arrived first, and
here they name different winners on purpose.

**Structure-aware chunking passes every naive strategy on hit@1 and ties one of them on MRR.**
Fixed-280 and recursive-600 tie each other at 0.700; structure-aware is 0.750, one question above
both. But its MRR, 0.817, is exactly fixed-280's 0.817 — the ranking-quality metric cannot
separate them — and at 0.833 it shares the lowest recall@5 of any chunked strategy with the
stripped version of itself. One question of hit@1 and a tie on MRR is not a landslide; the point is which metrics move and why, and section 2
shows the mechanism behind them.

**Stripping the boilerplate is the cheapest 0.05 on the page.** Removing the repeated legal footer
takes **4.2%** of the corpus away — 3,273 of 78,310 characters, leaving 75,037 — and moves
structure-aware chunking from hit@1 0.750 to **0.800** and MRR from 0.817 to **0.841**, on one
chunk fewer and with recall@5 unchanged at 0.833. That is the outright top of both ranking columns,
bought by deleting text. One question out of twenty, for a text substitution that costs nothing at
query time. The whole gain lands in one place: `en_en` goes 0.500 → 0.750 and no other category
moves.

**An average that improves can still hide a category that got worse.** The four `exact_token`
questions are three that hinge on a literal identifier — a flight number (`XX 1487`, `XX 3310`) or
a bulletin id (`SCB-2026-0902`) — and one fare-table lookup. Whole documents score **1.000** on
them. Fixed-280 drops to 0.750 and fixed-280-with-overlap to **0.500**, losing two of the four,
while both of those rungs report a recall@5 higher than anything structure-aware achieves. Both
recursive-600 and structure-aware are back at 1.000. So fixed-280 raised the headline average —
0.600 → 0.700, two questions — and paid for it out of exactly the category the room will be asked
about first, and the overlap rung paid twice, in the average and in the category at once. The
average alone would not have told you either happened.

Where structure-aware actually earns its lead is the cross-lingual row: `tr_en` reads 0.667 for
structure-aware against 0.500 for every other cut and 0.333 for whole documents. Chunking is also
the only thing anywhere on this page that moves `multi_hop` off zero — 0.000 whole, 0.500 on every
chunked rung. Section 7 has the full breakdown and the reason not to celebrate that second one.

## 2. The column header, not the table row

The CLASSIC short-haul penalty table is a realistic filed-tariff width. The sheet is 3,923
characters and fixed-280 chunking cuts it into 15 pieces. The row
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` survives intact — but
the header naming the columns lands in an earlier chunk, and the chunk itself opens mid-table on
` | none | none | EUR 65 | EUR 85 | EUR 170 | 1 x 23 kg | Yes |`, a row whose booking class was cut
off with the header. A model then reads a block of euro amounts with no idea which column is the
cancellation penalty or which row is class K. On the recorded run it answered **EUR 65** — a real
cell of that table, the *change* penalty of the row above — where the right answer is **EUR 90**.
Handed the
structure-aware chunk instead, which carries the header, it answered **EUR 90**. Generation is not
seeded, so the particular wrong cell can differ between runs; that it is confidently wrong, and
never hedges, does not.

| strategy | header stays with the K row? |
|---|---|
| fixed 280 | no — header in chunk 4, K row in chunk 6 |
| fixed 280 + 60 overlap | no — header in chunk 5, K row in chunk 7; overlap moves the boundary, it does not remove it |
| recursive 600 | no — header in chunk 4, K row in chunk 5; the table is wider than the split size |
| **structure-aware** | **yes — both in chunk 4** |

The notebook sweeps five fixed sizes before giving up on them — 280/0, 280/60, 280/120, 400/80,
600/100 — and the header and the K row land in different chunks in all five. The usual reflex,
reach for a recursive splitter, does not fix it either. Only splitting on the document's own
structure does, and it also prefixes each chunk with its heading, which is what "contextual
retrieval" amounts to.

## 3. Cross-lingual retrieval is where the default embedder dies

The same 20 questions over the same 154 structure-aware chunks, changing only the embedder:

| embedder | hit@1 | recall@5 | MRR | hit@1 on `tr_en` |
|---|---|---|---|---|
| `bge-m3` | **0.750** | **0.833** | **0.817** | **0.667** |
| `nomic-embed-text` | 0.350 | 0.633 | 0.490 | 0.000 |
| `all-MiniLM-L6-v2` (ChromaDB's default) | 0.350 | 0.600 | 0.467 | 0.000 |

This is the one comparison on the page where the margins are not one question wide: 0.750 against
0.350 is **eight** questions of twenty, and the `tr_en` column is four questions against none —
`bge-m3` answers 4 of the 6, the other two answer 0 of 6. Note also that the two weak embedders do
not separate at all on hit@1; they tie at 0.350 and differ only in the ordering metrics, 0.633
against 0.600 on recall@5 and 0.490 against 0.467 on MRR. Neither is meaningfully better than the
other. Both are simply the wrong tool.

`all-MiniLM-L6-v2` is the model ChromaDB installs and uses **by default** if you never choose one.
`notebooks/08_chromadb.py` reads it straight off a stored vector — `DefaultEmbeddingFunction`, 384
dimensions, loaded from `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx` — and then sends both
of its Turkish questions to `macro_tr_noshow`, the wrong document, where re-indexing the same 28
documents with `bge-m3` puts the first of them on `fare_classic_shorthaul`, the right sheet.
Anyone who runs `pip install chromadb` and starts
adding documents is using an English-only embedder. On the six Turkish-query/English-document
questions it scores 0.000, and so does `nomic-embed-text`. Neither errors. Neither warns. They
return a confident ranking of the wrong documents, and the only way to find out is to measure.

The `en_en` row is the counterweight: all three embedders score 0.500 there. On English questions
against English documents the cheap embedder is not visibly worse. The failure is specific, and an
English-only test set would never have surfaced it.

## 4. Fusion did not help here

Over the 154 structure-aware chunks:

| | hit@1 | recall@5 | MRR |
|---|---|---|---|
| dense only (`bge-m3`) | **0.750** | **0.833** | **0.817** |
| BM25 over the same chunks | 0.300 | 0.633 | 0.467 |
| RRF of the two | 0.450 | 0.683 | 0.579 |

Reciprocal rank fusion assumes both inputs are individually sound. BM25 over short chunks is not —
it collapses on Turkish queries against English documents entirely (0.000 on `tr_en`, and RRF
inherits the 0.000) — and fusing it in drags the dense ranking down with it, six questions of
hit@1 (0.750 → 0.450) and 0.238 of MRR (0.817 → 0.579).

At document level, where BM25 has whole documents to match against, the picture is closer but the
verdict does not change. This is what `--skip-rerank` prints:

| whole documents | hit@1 | recall@5 | MRR | `exact_token` |
|---|---|---|---|---|
| `bge-m3` | **0.600** | 0.717 | **0.677** | **1.000** |
| BM25 | 0.400 | 0.583 | 0.515 | 0.750 |
| RRF of the two | 0.500 | **0.733** | 0.611 | **1.000** |

**BM25 does not beat dense retrieval on exact tokens, even over whole documents.** Dense is at the
ceiling there, 1.000 against BM25's 0.750. The lexical intuition — "a flight code is a string, so
string matching should win" — is worth stating out loud in the room precisely because this corpus
disproves it in one command.

What BM25 does hold is the Turkish-query/Turkish-document case: 1.000 over whole documents and
0.750 over structure-aware chunks. Its failure is cross-lingual, not lexical, and there it is the
same 0.000 at both granularities. Chunking costs it on top of that (hit@1 0.400 → 0.300, MRR
0.515 → 0.467).

Fusion's only gain anywhere on this page is document-level recall@5, and on this run it is not
worth having: RRF buys 0.016 of recall@5 (0.717 → 0.733) for 0.100 of hit@1 (0.600 → 0.500). The
gain is a fraction of one question. The cost is two whole ones.

**On this run the commands agreed exactly, and that agreement is worth reading carefully.** Five
commands score `bge-m3` over the same 154 structure-aware chunks — `--chunking`, `--fusion`,
`exercises/m7_chunking_ladder.py`, `exercises/m6_embedding_bakeoff.py` and
`notebooks/06_hybrid_rerank_contextual.py` — and all five printed hit@1 0.750, recall@5 0.833,
MRR 0.817. Three score it over whole documents — the top rung of `--chunking`, `--skip-rerank` and
`notebooks/04_naive_rag.py` — and all three printed 0.600, 0.717, 0.677. Not one decimal moved
between commands on this pass.

That is a property of this pass and not a guarantee, and the difference matters on the day.
**`bge-m3` served through Ollama is not bit-stable across runs**, so a participant re-running these
commands should expect two of them to disagree by one question — `notebooks/06_hybrid_rerank_contextual.py`
printing dense hit@1 0.750 where `--fusion` prints 0.700, or the reverse. That did not happen on
this pass, which is why it is listed under "Not re-measured" rather than quoted as a figure; it is
named here anyway because the room will see it and needs to know what it is. Nothing is broken and
no setup needs fixing: 0.05 is one question out of twenty, which is exactly the noise floor this
page opens by warning about. Argue about a direction that flipped, never about a decimal that
moved.

## 5. A reranker is a trade, not an upgrade

The same reranker — `qwen2.5:3b`, scoring each candidate 0–10 on its own — applied to four
retrieval setups of different quality:

| retrieval setup | hit@1 before | after | MRR before | after | |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.515 | **0.544** | helps |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.490 | **0.540** | helps |
| `bge-m3` + fixed-280 | 0.700 | 0.500 | 0.817 | 0.680 | hurts |
| `bge-m3` + structure-aware | **0.750** | 0.600 | **0.817** | 0.725 | hurts |

The "before" column is the ladder and the bake-off arriving from the other direction: 0.700 and
0.750 for the two `bge-m3` rows are exactly section 1's fixed-280 and structure-aware rungs, and
the `nomic` + structure-aware row's 0.350 and MRR 0.490 are exactly section 3's `nomic`. Three
commands, three separate runs, same numbers.

The split is clean, and it is not about chunking — it is about the embedder. Before reranking the
four setups span 0.350 to 0.750. After it they span 0.400 to 0.600, and MRR rose on both weak
setups and fell on both strong ones without a single exception.

**It levels, and it levels to its own ceiling.** A 3B model has an opinion worth about that much;
imposing it on a ranking that was already better than its opinion can only lose.

The per-question view on the strongest setup says the same thing without averaging, and here the
script prints it rather than leaving it to arithmetic. Fifteen of the twenty questions already had
the gold document at rank 1; reranking pushed **four** of them down — q06, q10, q16, q18. Five
questions did not start at rank 1, and reranking pulled **one** of those up — q14. Fifteen, minus
four, plus one, is twelve of twenty, which is the after-column's 0.600 exactly. Four losses against
one gain, which is the table's −0.150 hit@1 spelled out one question at a time.

So the rule is not "add a reranker". It is: **a reranker pays when your retriever is worse than
your reranker, and costs you when it is better.** Fix the embedder first, then measure whether you
still need one. The price of the negative result is exact in calls and wildly approximate in
seconds: eight model calls per question, 160 for a 20-question pass over one setup, 640 to fill the
table above. On this run the four setups took 191, 127, 51 and 71 seconds on an M-series Mac — the
same 160 calls each time, and the slowest setup took nearly four times the fastest. The call count
does not vary; the clock does, and by more than you would guess.

This section is one run against the corrected gold set, and unlike the earlier corpus it has not
been repeated. Treat a before/after pair of yours that differs from the table by 0.05 as sampling
rather than as a broken setup; what the run establishes is the sign of each verdict, and all four
signs agreed with each other.

### How reranking was asked matters more than whether it was used
Two ways of asking the same model over the same candidates. `notebooks/06_hybrid_rerank_contextual.py`
shows the failure directly rather than as a score: handed **six** passages and asked to put them in
order, `qwen2.5:3b` replied `1,4,2,5` — four indices for six passages. There is no ranking there to
use. Asked instead to score each passage on its own it returned six integers (4, 2, 2, 5, 5, 3) in
six calls and 3.3 s, and those integers reorder the list.

The older head-to-head that first established this ran on the **retired 10-document, 5-question
probe corpus** and has no counterpart in the current runs:

| | hit@1 | MRR |
|---|---|---|
| listwise — "put these 6 in order" | 2/5 | 0.600 |
| pointwise — "score this one passage 0–10" | 5/5 | 1.000 |

`UNVERIFIED: the 2/5 and 5/5 probe figures — the probe corpus predates the Kraken Air rename and
no command in this repository reproduces them.` They are kept because the mechanism above is live
and reproducible; the two rows are not. Do not quote them.

Asking a small model to hold six passages in its head and order them is asking it to do something
it is bad at. Asking it about one passage at a time costs six calls instead of one and is the
version that held up. That is why `pointwise_rerank()` exists in `retrieval.py` and a listwise
version does not.

## 6. Stuffing the whole corpus into the prompt is *more* accurate here, not less

The usual argument for retrieval is that a long context degrades the answer — the model has more
plausible-looking wrong material competing for attention. We measured it on eight questions whose
answers are single verifiable values in the corpus, at four context sizes:

| context given to the model | characters | correct | seconds for all 8 |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 6.6 |
| top-3 chunks | 1,256 | 3/8 | 7.7 |
| top-5 chunks | 2,144 | 3/8 | 10.2 |
| **every document, in one stuffed prompt** | **79,309** | **8/8** | **75.4** |

That last figure is the **prompt**, not the corpus. The 28 documents are 78,310 characters. The
prompt is 79,309 because `notebooks/03_stuff_the_prompt.py` writes a `[SOURCE: name.md]` line above
each document before joining them with a blank line — 999 characters of headers and separators.
Section 1's boilerplate figures count the corpus and say 78,310. Both numbers are right and they
are not interchangeable — a sentence about the corpus takes 78,310, a sentence about what was sent
to the model takes 79,309.

Non-decreasing, in the direction opposite to the folklore, and this run is the extreme case of it:
the stuffed prompt got **every one of the eight right**, and no amount of extra context ever made
anything worse. At this corpus size and this model size there is no distractor penalty to find at
all. Some of the low end is a recall problem rather than a reading problem — a single 239-character
chunk often does not contain the answer — but the comparison that matters, top-5 against
everything, is 3/8 against 8/8. Note also what top-5 bought over top-3: nothing. The same three
questions came back right in both columns, for 888 more characters of context.

Look at *how* top-5 lost, because it is not noise. Asked for the class K short-haul **change**
penalty, where the answer is `EUR 70`, both top-3 and top-5 answered `EUR 155` — a real cell, the
class K row of `fare_classic_longhaul.md`. Retrieval handed the model the wrong one of six
near-identical fare sheets and the model read it faithfully. With everything in the prompt the
right sheet was in there too, and it answered `EUR 70`. That is a document-selection failure, and
it is an argument for retrieval that retrieves *well* — not against long contexts.

The trap this corpus was built for did not fire on this run, and that is worth saying plainly.
`sop_misconnect_v3.md` is marked Superseded (meal voucher **EUR 10**, hotel after **8 hours**) and
`sop_misconnect_v4.md` is marked Current (**EUR 15**, hotel after **6 hours**); both sit in
`corpus/2026-Q3/`. Handed both at once the stuffed model read the metadata and answered from the
current one, on both questions. Top-5 got the hotel threshold right and the meal voucher wrong.
The trap is real — a stuffed prompt hands the model a document and its replacement together and
relies on it to notice one word — but on this run the model noticed. One run is not a licence to
rely on that.

**So the case for retrieval on this corpus is not accuracy.** It is **37 times** fewer characters
per question (79,309 against 2,144), the clock — the stuffed column took 75.4 s for its eight
questions against 10.2 s for top-5, 9.4 s per question against 1.3 s, and the first stuffed
question against an unseen corpus took 84.2 s where the next question against the same corpus came
back in 1.0 s once the server had the prefix cached — and scale: the notebook puts that prompt at
~26,436 tokens against the 32,768-token window `qwen2.5:3b` advertises, and prints the arithmetic
for ten times as many documents (264,360 tokens, `NO`). Those are good reasons. "It answers better"
is not one of them here — on this run it answers *worse* — and claiming otherwise would be the
easiest thing on this page for someone in the room to check and disprove.

## 7. The same numbers, cut by question type

Every average above hides five much smaller averages. The gold set is four `tr_tr` questions, six
`tr_en`, four `en_en`, four `exact_token` and two `multi_hop`, and both commands below print the
breakdown under their own table. With n=2 and n=4 rows, a single question moves a cell by 0.500 or
0.250; these tables are for spotting a direction, never a margin.

hit@1 by type, down the ladder (`bge-m3`) — `python eval/run_benchmark.py --chunking`, which
`python exercises/m7_chunking_ladder.py` also prints:

| type | whole | fixed-280 | +overlap60 | recursive-600 | structure-aware | + strip |
|---|---|---|---|---|---|---|
| `tr_tr` (n=4) | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| `tr_en` (n=6) | 0.333 | 0.500 | 0.500 | 0.500 | **0.667** | **0.667** |
| `en_en` (n=4) | 0.500 | 0.750 | 0.500 | 0.500 | 0.500 | **0.750** |
| `exact_token` (n=4) | 1.000 | 0.750 | 0.500 | 1.000 | 1.000 | 1.000 |
| `multi_hop` (n=2) | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

hit@1 by type, across the embedders on structure-aware chunks —
`python exercises/m6_embedding_bakeoff.py`:

| type | `all-MiniLM-L6-v2` | `nomic-embed-text` | `bge-m3` |
|---|---|---|---|
| `tr_tr` (n=4) | 0.750 | 0.500 | 1.000 |
| `tr_en` (n=6) | **0.000** | **0.000** | **0.667** |
| `en_en` (n=4) | 0.500 | 0.500 | 0.500 |
| `exact_token` (n=4) | 0.250 | 0.500 | 1.000 |
| `multi_hop` (n=2) | 0.500 | 0.500 | 0.500 |

**`multi_hop` never reaches 1.000 anywhere on this page, and its 0.500 is not a success.** At
document level it is 0.000 for every retriever `--skip-rerank` scores — BM25, `bge-m3`,
`nomic-embed-text` and RRF alike — and 0.000 on the whole-documents rung of the ladder. The moment
the corpus is chunked it reads 0.500, and it reads 0.500 *everywhere*: on all five chunked rungs,
with and without stripping, and for all three embedders in the bake-off including the two that
score 0.000 on `tr_en`. A number that is identical for the best and the worst embedder in the
comparison is not measuring the embedder. It is one of the two questions becoming reachable once
documents are cut into chunks, and the other staying unreachable for everything.

Which is which is printed. `--fusion` lists q05, q07, q11, q14 and **q19** as the questions *every*
run missed at rank 1, and dense retrieval scores 0.500 on `multi_hop` over those same chunks — so
the half it gets is q20 and the half it never gets is q19. `--skip-rerank` lists q05, q06, q07,
q08, **q19 and q20**: at document level both halves are missed by everything, which is the 0.000.
Fusion undoes even the 0.500 — BM25 and RRF are both back at 0.000 on `multi_hop` over chunks.

That ceiling is the honest version of the module 10 gate: a question whose answer has to be
assembled from three documents is not fixed by ranking the documents better. q19's zero is a real
failure rather than a labelling artefact — `eval/README.md` explains the convention that decides
it, which is that `gold_doc_ids` names the documents an amount may legitimately be quoted from
rather than every document that mentions the rule.

Per-type MRR is computed by `metrics.evaluate()` under `by_type` but is not printed by any command
here; only hit@1 by type is. The tables above are what the commands actually put on screen.


## Appendix — the module 2 MNIST run

Module 2 is the only part of the day that does not touch the corpus, so neither the airline rename
nor the gold-set correction could move it — but it was re-run anyway rather than carried over.
Reproduce it with
`python notebooks/01_mnist_tiny_net.py`, or by opening the file in VS Code and running the blocks.
It needs `numpy` and the MNIST cache that `scripts/seed_offline_assets.py` puts in
`notebooks/mnist_data/`; it makes no network call and no model call, and `numpy.random.default_rng(0)`
seeds it, so the accuracies below are exact rather than approximate.

784 inputs, one hidden layer of 128 with ReLU, 10 outputs. Learning rate 0.1, batch size 32,
5 epochs over 60,000 images.

| | |
|---|---|
| parameters | 101,770 — `W1` 100,352 + `b1` 128 + `W2` 1,280 + `b2` 10 |
| accuracy before any training | 9.9% — one in ten, exactly what guessing gets you |
| after epoch 1 | 95.35% |
| after epoch 2 | 96.45% |
| after epoch 3 | 97.26% |
| after epoch 4 | **97.62%** |
| after epoch 5 | 97.47% |
| training wall clock | 0.92 s |
| cross-entropy, first batch to last | 2.35 → 0.03 |

Measured on an M-series Mac, CPU only, no GPU and no framework. Two things in that table are worth
saying out loud: nearly all of the learning happens in the first pass, and epoch 4 scores higher
than epoch 5 — the curve stops improving and starts wobbling. The wall clock is the one figure here
that moves with the hardware; everything else is fixed by the seed.

## Not re-measured

Everything above came out of the re-run against the corrected gold set. These did not, and are
listed here rather than left to look like fresh measurements:

- `UNVERIFIED: the listwise-vs-pointwise probe table in section 5 (2/5 and 5/5, MRR 0.600 and
  1.000) — measured on a retired 10-document, 5-question probe corpus that predates the rename;
  no command in this repository reproduces it.` The mechanism it illustrates *is* re-measured, in
  `notebooks/06_hybrid_rerank_contextual.py`.
- `UNVERIFIED: the cross-command bge-m3 disagreement described at the end of section 4 — on this
  run every command agreed to the third decimal, so the specific pairing named there (notebook 06
  printing 0.750 against the benchmark's 0.700) was not observed in it.` It is stated as the shape
  the instability takes when it appears, not as a figure from this pass; the agreement that *was*
  measured is quoted in full above it.
- `UNVERIFIED: which individual exact_token question the fixed-size rungs lose, and to what rank —
  no command prints per-question ranks for the chunking ladder.` Only the category average
  (1.000 → 0.750 → 0.500) is printed, and that is all section 1 now claims.
- `UNVERIFIED: the rank q14 moved to under reranking — m9_rerank_trade.py prints which questions
  moved and in which direction, not their positions.` Section 5's fifteen-minus-four-plus-one is
  arithmetic on printed counts that lands exactly on the printed after-value, not a printed rank.
- `UNVERIFIED: any effect of reranking on multi_hop — the default python eval/run_benchmark.py run
  with no flags was not part of this re-run.` The earlier claim that its rerank pass lifted
  `multi_hop` at document level is withdrawn until that command is run again.
- `UNVERIFIED: total wall clock for exercises/m7_chunking_ladder.py, m6_embedding_bakeoff.py and
  m9_rerank_trade.py — those three print per-rung and per-setup timings and no total, so only the
  per-rung figures are quoted and no sum is presented as a measurement.`
- `UNVERIFIED: the end-to-end wall clock of notebooks/01_mnist_tiny_net.py — the file prints the
  0.92 s training time but no whole-file total.` The previously quoted 1.34 s has been dropped from
  the appendix table rather than carried.
- `UNVERIFIED: the split of section 6's 999-character difference between prompt and corpus into
  per-document [SOURCE: ...] headers and blank-line separators — 79,309 and 78,310 are both
  printed, the breakdown between them is not.`
- `UNVERIFIED: timings for python eval/run_benchmark.py (no flags), --rerank-sweep,
  --chunking --embed-model nomic-embed-text and m9_rerank_trade.py --quick — not part of this
  re-run.` They are marked the same way in `eval/README.md`.
