# Measured results

Every number here was produced by the scripts in this repository, running entirely on a laptop.
Sections 1 to 7 score the corpus in `corpus/2026-Q3/` against the 20 questions in
`gold_questions.jsonl` through local Ollama; the appendix at the end records the one measurement
on this course that has nothing to do with retrieval. Helios Air is a fictional airline and the
corpus is synthetic training material written for this course; nothing in it is an Amadeus
document.

Retrieval is scored at document level: a chunk ranking is collapsed to its parent documents,
best chunk first. Generation and reranking are `qwen2.5:3b`, embeddings are `bge-m3` unless
stated otherwise.

> **Twenty questions is not a benchmark.** These numbers are large enough to decide between
> two designs and far too small to publish. Where a difference is under about 0.05 it is
> inside the noise of this sample, and the text below says so rather than pretending otherwise.

## Reproducing each section

Run these from the repository root, with Ollama serving `bge-m3`, `nomic-embed-text` and
`qwen2.5:3b`. `eval/README.md` carries the same list with the measured cost of each command and
the options they accept.

Timings are from one run of each command, back to back on one M-series Mac with the models
already resident. They are an order of magnitude, not a specification; `eval/README.md` carries
the same numbers with the call counts, which do not vary.

| section | command | notes |
|---|---|---|
| 1 — the ladder | `python exercises/m7_chunking_ladder.py` | 58–98 s over two runs; also prints the boilerplate-strip row. `python eval/run_benchmark.py --chunking` prints the same ladder in 93 s |
| 2 — the column header | open `notebooks/05_chunking_and_noise.py` in VS Code and run the blocks | a few seconds; the model call is one question |
| 3 — the embedders | `python exercises/m6_embedding_bakeoff.py` | 18 s; the only command in the repository that includes ChromaDB's default embedder |
| 4 — fusion | `python eval/run_benchmark.py --fusion` over chunks, `python eval/run_benchmark.py --skip-rerank` over whole documents | 9–10 s and 11 s; both are embedding-only |
| 5 — the reranker | `python exercises/m9_rerank_trade.py` (`--quick` on a slow laptop) | 640 model calls, 324 s here and considerably longer on a CPU-only laptop. `python eval/run_benchmark.py --rerank-sweep` prints the same four setups in 298 s |
| 6 — context size | open `notebooks/03_stuff_the_prompt.py` in VS Code and run the blocks | about three minutes against a freshly started Ollama; the four context sizes themselves took 4.7, 7.1, 8.9 and 74.2 s on the recorded run |
| 7 — by question type | `python eval/run_benchmark.py --chunking` and `python exercises/m6_embedding_bakeoff.py` | both print hit@1 by question type under their own table; no extra run |

The notebooks are percent-format `.py` files. They are opened in VS Code with the Microsoft
Python extension and run block by block; the `# %%` markers are the block boundaries. The only
dependencies are `numpy` and `chromadb`.

## 1. Chunking is the single biggest lever

| | hit@1 | recall@5 | MRR | chunks |
|---|---|---|---|---|
| whole documents, no chunking | 0.550 | 0.717 | 0.654 | 28 |
| fixed 280 chars | 0.700 | **0.950** | 0.814 | 294 |
| fixed 280 + 60 overlap | 0.700 | 0.883 | 0.799 | 368 |
| recursive, 600 chars | 0.700 | 0.900 | 0.816 | 197 |
| **structure-aware, 900 chars** | **0.800** | 0.833 | **0.844** | 154 |

Three things worth stopping on.

**Three different naive strategies land on exactly the same score.** Fixed size, fixed size with
overlap and a recursive splitter all reach hit@1 0.700, and their MRRs — 0.814, 0.799, 0.816 —
are within 0.017 of each other, which on twenty questions is nothing. Reaching for a smarter
naive splitter buys nothing here. Only splitting on the document's own structure gets past it,
to 0.800.

**An average that improves can still hide a category that got worse.** The four `exact_token`
questions are three that hinge on a literal identifier — a flight number (`H9 1487`, `H9 3310`) or
a bulletin id (`SCB-2026-0902`) — and one fare-table lookup. Whole documents score 1.000 on them,
fixed-280 drops to 0.750, fixed-280 with overlap stays at 0.750, and both recursive-600 and
structure-aware are back at 1.000.

The question that moves is q18, `H9 3310`: its gold bulletin sits at rank 1 over whole documents
and at rank 2 under both fixed-size strategies, and returns to rank 1 under recursive and
structure-aware splitting. The other three never leave rank 1 anywhere on the ladder. So cutting
the corpus into small pieces raised the average and cost one exact-token question on the way —
one question out of four, not a collapse, and a recursive splitter recovers it as well as a
structure-aware one does. The average alone would not have told you it happened.

**Recall@5 is highest for the worst strategy.** Cutting into 294 small pieces gives the gold
document more chances to appear somewhere in the top five (0.950) than structure-aware chunking
does (0.833), while ranking it first gets harder. Picking the metric that flatters your change is
easy; these two disagree on purpose.

The ladder script prints one more row. Stripping the repeated legal footer removes **4.2%** of the
corpus (78,310 → 75,037 characters) and moves structure-aware chunking to hit@1 0.850, recall@5
0.833, MRR 0.869 over 153 chunks. That is one question out of twenty — real, cheap, and right at
the edge of what this sample can distinguish.

## 2. The column header, not the table row

The CLASSIC short-haul penalty table is a realistic filed-tariff width. Under fixed-size chunking
the row `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` survives
intact — but the header naming the columns lands in an earlier chunk, and the chunk itself opens
mid-table on ` | none | none | EUR 65 | EUR 85 | EUR 170 | 1 x 23 kg | Yes |`, a row whose booking
class was cut off with the header. A model then reads a block of euro amounts with no idea which
column is the cancellation penalty or which row is class K. On the recorded run it answered
**EUR 65** — a real cell of that table, the *change* penalty of the row above — where the right
answer is **EUR 90**. Handed the structure-aware chunk instead, which carries the header, it
answered **EUR 90**. Generation is not seeded, so the particular wrong cell can differ between
runs; that it is confidently wrong, and never hedges, does not.

| strategy | header stays with the K row? |
|---|---|
| fixed 280 | no — header in chunk 4, K row in chunk 6 |
| fixed 280 + 60 overlap | no — overlap moves the boundary, it does not remove it |
| recursive 600 | no — the table is wider than the split size |
| structure-aware | yes — both in chunk 4 |

The usual reflex — reach for a recursive splitter — does not fix this. Only splitting on the
document's own structure does, and it also prefixes each chunk with its heading, which is what
"contextual retrieval" amounts to.

## 3. Cross-lingual retrieval is where the default embedder dies

The same 20 questions over the same structure-aware chunks, changing only the embedder:

| embedder | hit@1 | recall@5 | MRR | hit@1 on `tr_en` |
|---|---|---|---|---|
| `bge-m3` | **0.800** | 0.833 | **0.844** | **0.667** |
| `nomic-embed-text` | 0.350 | 0.633 | 0.492 | 0.000 |
| `all-MiniLM-L6-v2` (ChromaDB's default) | 0.350 | 0.583 | 0.443 | 0.000 |

`all-MiniLM-L6-v2` is the model ChromaDB installs and uses **by default** if you never choose one.
Anyone who runs `pip install chromadb` and starts adding documents is using an English-only
embedder. On the six Turkish-query/English-document questions it scores 0.000, and so does
`nomic-embed-text`. Neither errors. Neither warns. They return a confident ranking of the wrong
documents, and the only way to find out is to measure.

## 4. Fusion did not help here

Over the structure-aware chunks:

| | hit@1 | MRR |
|---|---|---|
| dense only (`bge-m3`) | **0.800** | **0.844** |
| BM25 over the same chunks | 0.300 | 0.467 |
| RRF of the two | 0.450 | 0.586 |

Reciprocal rank fusion assumes both inputs are individually sound. BM25 over short chunks is not —
it collapses on Turkish queries against English documents entirely (0.000 on `tr_en`) — and fusing
it in drags the dense ranking down with it.

The claim this section used to make about whole documents does not survive measurement, so here is
what `--skip-rerank` actually prints:

| whole documents | hit@1 | recall@5 | MRR | `exact_token` |
|---|---|---|---|---|
| `bge-m3` | 0.550 | 0.717 | 0.654 | **1.000** |
| BM25 | 0.400 | 0.583 | 0.515 | 0.750 |
| RRF of the two | 0.450 | 0.783 | 0.602 | 1.000 |

**BM25 does not beat dense retrieval on exact tokens, even over whole documents.** Dense is at the
ceiling there, 1.000 against BM25's 0.750. The lexical intuition — "a flight code is a string, so
string matching should win" — is worth stating out loud in the room precisely because this corpus
disproves it in one command.

What BM25 does hold is the Turkish-query/Turkish-document case: 1.000 over whole documents and
0.750 over structure-aware chunks. Its failure is cross-lingual, not lexical, and there it is the
same 0.000 at both granularities. Chunking costs it a little on top of that (hit@1 0.400 → 0.300,
MRR 0.515 → 0.467), though the MRR difference is inside this sample's noise.

Fusion's one genuine gain is at document level, and it is a trade rather than a win: RRF lifts
recall@5 from 0.717 to 0.783 while dropping hit@1 from 0.550 to 0.450. It finds the document more
often and ranks it first less often.

## 5. A reranker is a trade, not an upgrade

The same reranker — `qwen2.5:3b`, scoring each candidate 0–10 on its own — applied to four
retrieval setups of different quality:

| retrieval setup | hit@1 before | after | MRR before | after | |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.503 | **0.543** | helps |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.492 | **0.537** | helps |
| `bge-m3` + fixed-280 | 0.700 | 0.550 | 0.814 | 0.712 | hurts |
| `bge-m3` + structure-aware | **0.800** | 0.600 | **0.844** | 0.717 | hurts |

The split is clean, and it is not about chunking — it is about the embedder. Before reranking the
four setups span 0.350 to 0.800. After it they span 0.400 to 0.600.

**It levels, and it levels to its own ceiling.** A 3B model has an opinion worth about that much;
imposing it on a ranking that was already better than its opinion can only lose.

The per-question view on the strongest setup says the same thing without averaging. Sixteen of the
twenty questions already had the gold document at rank 1; reranking pushed **five** of them down —
q06 (1→3), q10 (1→2), q11 (1→4), q16 (1→2), q17 (1→3). Four questions did not start at rank 1 —
q05 (13), q07 (7), q14 (2), q19 (6) — and reranking pulled **two** of those up: q14 to rank 1 and
q19 to rank 5. Five losses against two gains, which is the table's −0.200 hit@1 spelled out one
question at a time.

So the rule is not "add a reranker". It is: **a reranker pays when your retriever is worse than
your reranker, and costs you when it is better.** Fix the embedder first, then measure whether you
still need one. The price of the negative result is exact in calls and approximate in seconds:
eight model calls per question, 160 for a 20-question pass over one setup, 640 to fill the table
above. Across the two recorded runs a setup took between 47 and 113 seconds on an M-series Mac —
the call count does not vary, the clock does.

Every figure in this section reproduced exactly on a second run: same four before/after pairs,
same five demotions, same two promotions. At temperature 0 the reranker is repeatable on this
corpus, so a number here that does not reproduce on your laptop is a difference in setup rather
than sampling noise.

### How reranking was asked matters more than whether it was used
Two ways of asking the same model over the same candidates. This comparison comes from the earlier
probe corpus — **10 documents and 5 questions**, not the 20 above — and it is the only probe figure
left on this page:

| | hit@1 | MRR |
|---|---|---|
| listwise — "put these 6 in order" | 2/5 | 0.600 |
| pointwise — "score this one passage 0–10" | 5/5 | 1.000 |

Asking a small model to hold six passages in its head and order them is asking it to do something
it is bad at. Asking it about one passage at a time costs six calls instead of one and is the
version that held up. That is why `pointwise_rerank()` exists in `retrieval.py` and a listwise
version does not. At `n=5` the direction is worth acting on and the magnitude is unproven; do not
quote these two rows without the denominator. Everything else the probe produced — including its
suggestion that reranking helps in general — is retired, and the n=20 table above is why.

## 6. Stuffing the whole corpus into the prompt is *more* accurate here, not less

The usual argument for retrieval is that a long context degrades the answer — the model has more
plausible-looking wrong material competing for attention. We measured it on eight questions whose
answers are single verifiable values in the corpus, at four context sizes:

| context given to the model | characters | correct | seconds for all 8 |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 4.7 |
| top-3 chunks | 1,256 | 3/8 | 7.1 |
| top-5 chunks | 2,144 | 4/8 | 8.9 |
| **every document, in one stuffed prompt** | **79,309** | **7/8** | **74.2** |

That last figure is the **prompt**, not the corpus. The 28 documents are 78,310 characters. The
prompt is 79,309 because `notebooks/03_stuff_the_prompt.py` writes a `[SOURCE: name.md]` line
above each document before joining them with a blank line: 945 characters of headers plus the 54
that separate the 28 documents, 999 in total. Section 1's boilerplate figures count the corpus and
say 78,310. Both numbers are right and they are not interchangeable — a sentence about the corpus
takes 78,310, a sentence about what was sent to the model takes 79,309.

Monotonic, in the direction opposite to the folklore. At this corpus size and this model size there
is no distractor penalty to find. Some of the low end is a recall problem rather than a reading
problem — a single 239-character chunk often does not contain the answer at all — but the
comparison that matters, top-5 against everything, still favours everything, 7/8 against 4/8.

Look at *how* top-5 lost, because it is not noise. Asked for the class K change penalty it answered
`EUR 155`; asked for the class K cancellation penalty it answered `EUR 195`. Both are real cells:
they are the class K row of `fare_classic_longhaul.md`, and the questions said short-haul, where
the answers are `EUR 70` and `EUR 90`. Retrieval handed the model the wrong one of six
near-identical fare sheets and the model read it faithfully. With everything in the prompt the
right sheet was in there too, and it got both right. That is a document-selection failure, and it
is an argument for retrieval that retrieves *well* — not against long contexts.

The one question the stuffed prompt lost is the one this corpus was built to trap. `sop_misconnect_v3.md`
is marked Superseded (hotel after **8 hours**) and `sop_misconnect_v4.md` is marked Current (**6
hours**); both sit in `corpus/2026-Q3/`. Asked for the current threshold the stuffed model answered
**8**. Top-5 answered **6**, because retrieval had handed it the current chunk and not the old one.
Stuffing hands the model a document and its replacement together and relies on it to notice one
word of metadata.

**So the case for retrieval on this corpus is not accuracy.** It is **37 times** fewer characters
per question (a 79,309-character prompt against 2,144), the clock — the stuffed column took 74.2 s
for its eight questions against 8.9 s for top-5, and the first stuffed question against an unseen
corpus took 72.5 s where the next question against the same corpus took 0.9 s once the server had
the prefix cached — and scale: that prompt is roughly 26,400 tokens at three characters per token,
against the 32,768-token window `qwen2.5:3b` advertises, so one order of magnitude more documents
does not fit at all. Those are good reasons. "It answers better" is not one of them here, and
claiming it would be the easiest thing on this page for someone in the room to check and
disprove.

## 7. The same numbers, cut by question type

Every average above hides five much smaller averages. The gold set is four `tr_tr` questions, six
`tr_en`, four `en_en`, four `exact_token` and two `multi_hop`, and both commands below print the
breakdown under their own table.

hit@1 by type, down the ladder (`bge-m3`) — `python eval/run_benchmark.py --chunking`, which
`python exercises/m7_chunking_ladder.py` also prints:

| type | whole | fixed-280 | +overlap60 | recursive-600 | structure-aware | + strip |
|---|---|---|---|---|---|---|
| `tr_tr` (n=4) | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| `tr_en` (n=6) | 0.333 | 0.667 | 0.667 | 0.500 | 0.667 | 0.667 |
| `en_en` (n=4) | 0.250 | 0.750 | 0.500 | 0.500 | 0.750 | 1.000 |
| `exact_token` (n=4) | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 |
| `multi_hop` (n=2) | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

hit@1 by type, across the embedders on structure-aware chunks —
`python exercises/m6_embedding_bakeoff.py`:

| type | `all-MiniLM-L6-v2` | `nomic-embed-text` | `bge-m3` |
|---|---|---|---|
| `tr_tr` (n=4) | 0.750 | 0.500 | 1.000 |
| `tr_en` (n=6) | **0.000** | **0.000** | **0.667** |
| `en_en` (n=4) | 0.500 | 0.500 | 0.750 |
| `exact_token` (n=4) | 0.250 | 0.500 | 1.000 |
| `multi_hop` (n=2) | 0.500 | 0.500 | 0.500 |

**`multi_hop` never gets above 0.500.** Two questions, and across every measurement on this page
the row reads 0.000 or 0.500 — never 1.000. Chunking moves it once, from 0.000 over whole
documents to 0.500 as soon as the corpus is cut; reranking moves it once, in the default
`python eval/run_benchmark.py` run over whole documents, where the rerank pass lifts it from
0.000 to 0.500. Nothing takes it further.
On the strongest setup the half that scores is q20, at rank 1 both before and after reranking,
while q19 sits at rank 6 before and rank 5 after and never reaches the top.

That ceiling is the honest version of the module 10 gate: a question whose answer has to be
assembled from two documents is not fixed by ranking the documents better. q19's zero is a real
failure rather than a labelling artefact — `eval/README.md` explains the convention that decides
it, which is that `gold_doc_ids` names the documents an amount may legitimately be quoted from
rather than every document that mentions the rule.

Per-type MRR is computed by `metrics.evaluate()` under `by_type` but is not printed by any command
here; only hit@1 by type is. The tables above are what the commands actually put on screen.


## Appendix — the module 2 MNIST run

Module 2 is the only part of the day that does not touch the corpus, and its figures had no record
here. Reproduce them by opening `notebooks/01_mnist_tiny_net.py` in VS Code and running the blocks,
or in one shot with `python notebooks/01_mnist_tiny_net.py`. It needs `numpy` and the MNIST cache
that `scripts/seed_offline_assets.py` puts in `notebooks/mnist_data/`; it makes no network call and
no model call, and `numpy.random.default_rng(0)` seeds it, so the accuracies below are exact rather
than approximate.

784 inputs, one hidden layer of 128 with ReLU, 10 outputs. Learning rate 0.1, batch size 32,
5 epochs over 60,000 images.

| | |
|---|---|
| parameters | 101,770 — `W1` 100,352 + `b1` 128 + `W2` 1,280 + `b2` 10 |
| accuracy before any training | 9.87%, which the notebook prints rounded as 9.9% |
| after epoch 1 | 95.35% |
| after epoch 2 | 96.45% |
| after epoch 3 | 97.26% |
| after epoch 4 | **97.62%** |
| after epoch 5 | 97.47% |
| training wall clock | 0.85 s |
| whole file, end to end | 1.17 s |
| cross-entropy, first batch to last | 2.35 → 0.03 |

Measured on an M-series Mac, CPU only, no GPU and no framework. Two things in that table are worth
saying out loud: nearly all of the learning happens in the first pass, and epoch 4 scores higher
than epoch 5 — the curve stops improving and starts wobbling. The wall clock is the one figure
here that moves with the hardware; everything else is fixed by the seed.
