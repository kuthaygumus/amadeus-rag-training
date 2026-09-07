# Measured results

Every number here was produced by the scripts in this directory, on the corpus in
`corpus/2026-Q3/` and the 20 questions in `gold_questions.jsonl`, running entirely on a
laptop through local Ollama. Reproduce any of it with `python eval/run_benchmark.py`.

Retrieval is scored at document level: a chunk ranking is collapsed to its parent documents,
best chunk first. Generation is `qwen2.5:3b`, embeddings are `bge-m3` unless stated otherwise.

> **Twenty questions is not a benchmark.** These numbers are large enough to decide between
> two designs and far too small to publish. Where a difference is under about 0.05 it is
> inside the noise of this sample, and the text below says so rather than pretending otherwise.

## 1. Chunking is the single biggest lever

| | hit@1 | recall@5 | MRR | chunks |
|---|---|---|---|---|
| whole documents, no chunking | 0.550 | 0.717 | 0.655 | 28 |
| fixed 280 chars | 0.650 | **0.950** | 0.789 | 288 |
| fixed 280 + 60 overlap | 0.700 | 0.867 | 0.795 | 363 |
| recursive, 600 chars | 0.700 | 0.900 | 0.816 | 195 |
| **structure-aware, 900 chars** | **0.800** | 0.850 | **0.846** | 152 |

Two things worth stopping on.

**Fixed-size chunking makes exact-token retrieval worse while making the average better.**
Flight-code queries score 1.000 over whole documents and **0.500** under fixed-280 chunking,
recovering to 1.000 only once splitting follows the document's structure. An average that
improves can hide a category that collapsed.

**Recall@5 is highest for the worst strategy.** Cutting into 288 small pieces gives the gold
document more chances to appear somewhere in the top five, while ranking it first gets harder.
Picking the metric that flatters your change is easy; these two disagree on purpose.

## 2. The column header, not the table row

The CLASSIC short-haul penalty table is a realistic filed-tariff width. Under fixed-size
chunking the row `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`
survives intact — but the header naming the columns lands in an earlier chunk. A model then
reads two euro amounts with no idea which is the cancellation penalty, and answers **EUR 70**
where the right answer is **EUR 90**. It does not hedge. It is confidently wrong.

| strategy | header stays with the K row? |
|---|---|
| fixed 280 | no |
| fixed 280 + 60 overlap | no — overlap moves the boundary, it does not remove it |
| recursive 600 | no — the table is wider than the split size |
| structure-aware | yes |

The usual reflex — reach for a recursive splitter — does not fix this. Only splitting on the
document's own structure does, and it also prefixes each chunk with its heading, which is what
"contextual retrieval" amounts to.

## 3. Cross-lingual retrieval is where the default embedder dies

Turkish query, English document, six questions:

| embedder | hit@1 on `tr_en` |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

`all-MiniLM-L6-v2` — the model ChromaDB installs and uses **by default** if you never choose
one — scored 2/5 against 4/5 for `bge-m3` on the smaller probe corpus. Anyone who runs
`pip install chromadb` and starts adding documents is using an English-only embedder, and
nothing warns them. On a Turkish corpus it does not error. It just returns the wrong document.

## 4. Fusion did not help here

| | hit@1 | MRR |
|---|---|---|
| structure-aware, dense only | **0.800** | **0.846** |
| BM25 over the same chunks | 0.300 | 0.465 |
| RRF of the two | 0.450 | 0.581 |

Reciprocal rank fusion assumes both inputs are individually sound. BM25 over short chunks is
not — it collapses on Turkish queries entirely (0.000 on `tr_en`) — and fusing it in drags the
dense ranking down with it. This held on the earlier 10-document probe corpus too, where
fusion pushed an exact flight-code query from rank 1 to rank 2.

BM25 is still worth its five minutes: over **whole documents** it beats dense retrieval on
exact tokens. It is the chunking that destroys it, which is its own lesson about how these
choices interact.

## 5. A reranker is a trade, not an upgrade

The same reranker — `qwen2.5:3b`, scoring each candidate 0–10 on its own — applied to four
retrieval setups of different quality:

| retrieval setup | hit@1 before | after | MRR before | after | |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.494 | **0.544** | helps |
| `nomic` + structure-aware | 0.350 | **0.450** | 0.500 | **0.562** | helps |
| `bge-m3` + fixed-280 | 0.650 | 0.550 | 0.789 | 0.708 | hurts |
| `bge-m3` + structure-aware | **0.800** | 0.650 | **0.846** | 0.750 | hurts |

The split is clean, and it is not about chunking — it is about the embedder. The reranker
lifts both weak setups to roughly 0.45 and drags both strong ones down to roughly 0.65.

**It levels, and it levels to its own ceiling.** A 3B model has an opinion worth about that
much; imposing it on a ranking that was already better than its opinion can only lose. Looking
at individual questions shows the same thing: where the right document was already at rank 1,
reranking moved it down five times out of five; where it sat at rank 5 or 6, reranking pulled
it up.

So the rule is not "add a reranker". It is: **a reranker pays when your retriever is worse
than your reranker, and costs you when it is better.** Fix the embedder first, then measure
whether you still need one. Each rerank pass here cost 8 model calls and about 4 seconds per
query, which is a real price to pay for a negative result.

### How reranking was asked matters more than whether it was used
Two ways of asking the same model over the same candidates, on the probe corpus:

| | hit@1 | MRR |
|---|---|---|
| no reranking | 4/5 | 0.833 |
| listwise — "put these 6 in order" | 2/5 | 0.600 |
| pointwise — "score this one passage 0–10" | 5/5 | 1.000 |

Asking a small model to hold six passages in its head and order them is asking it to do
something it is bad at. Asking it about one passage at a time costs six calls instead of one
and is the only version that worked. (This one is `n=5`; treat the direction as real and the
magnitude as unproven.)

## 6. Stuffing the whole corpus into the prompt is *more* accurate here, not less

The usual argument for retrieval is that a long context degrades the answer — the model has more
plausible-looking wrong material competing for attention. We measured it on eight questions whose
answers are single verifiable values in the corpus, at four context sizes:

| context given to the model | characters | correct |
|---|---|---|
| top-1 chunk | 239 | 1/8 |
| top-3 chunks | 1,256 | 3/8 |
| top-5 chunks | 2,144 | 4/8 |
| **the whole corpus** | **78,113** | **5/8** |

Monotonic, in the direction opposite to the folklore. At this corpus size and this model size there
is no distractor penalty to find. Some of the low end is a recall problem rather than a reading
problem — a single 239-character chunk often does not contain the answer at all — but the
comparison that matters, top-5 against everything, still favours everything.

**So the case for retrieval on this corpus is not accuracy.** It is roughly eleven times fewer
tokens per question, a cold first query that takes 75.8 s against 2.8 s, and the fact that one
order of magnitude more documents does not fit in the window at all. Those are good reasons. "It
answers better" is not one of them here, and claiming it would be the easiest thing on this page
for someone in the room to check and disprove.

Worth noting separately: on several of these questions **both** conditions were wrong, and wrong in
an unusual way. Asked for the class K change penalty, the model answered `EUR 155` — a number that
appears in no cell of the table. Given a nine-column filed-tariff table it does not simply read the
wrong column; it sometimes produces a value between two of them. That is a reading failure of the
generation step, entirely separate from whether retrieval found the right document.
