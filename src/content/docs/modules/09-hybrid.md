---
title: "9. Hybrid, Rerank and Contextual"
description: "What if semantic search is not enough?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **What if semantic search is not enough?**

You finished module 8 at **hit@1 0.800**: `bge-m3`, 152 structure-aware chunks, served out of
ChromaDB. Four questions in five come back with the right document first. One in five does not, and
when it does not the generator answers confidently from the wrong page.

There is a standard list of things to put in that gap. Every RAG post says the same three words:
hybrid, rerank, contextual. Add BM25 next to the vector search, put a model on top to reorder the
results, prefix each chunk with its context. We run all three against the same twenty questions. One
of them you already shipped in module 7 without noticing. The other two lose.

<div class="presenter-note">
Before opening the notebook, write on the board <strong>hybrid · rerank · contextual</strong> and ask
for a show of hands: "which of these three moves 0.800 upwards?" Almost every hand goes up for
rerank, most for hybrid. Write the counts next to the words and leave them there — you will point at
them twice. 3 minutes, laptops closed.
</div>

## Bringing BM25 back

BM25 is the keyword retriever from module 5: rare terms score high, term frequency saturates, long
documents get penalised. It found `H9 1487` instantly and scored a flat zero on "iptal edersem ne
öderim". Run it over exactly the chunks the dense retriever is using — the same 152, same index:

```python
bm25 = BM25(chunk_ids, chunk_texts)
```

**hit@1 0.300, MRR 0.465.** On the six Turkish-query questions, **0.000**. Not degraded. Zero. BM25
has no path from `iptal` to `Cancellation penalty`, and chunking gave it none.

## Reciprocal rank fusion, and why it dilutes

Fusion is the fix everyone reaches for, and it is genuinely elegant. The two retrievers do not need
comparable scores — cosine similarity and a BM25 score share no unit — so you throw the scores away
and keep only the positions:

```python
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
```

Each retriever gives every document `1 / (k + rank)` points. `k = 60` flattens the top of the curve,
so rank 1 and rank 3 are worth nearly the same and the deep tail is worth almost nothing. A document
both retrievers like beats a document only one of them likes.

Fuse dense and BM25 over the same chunks: **hit@1 0.450, MRR 0.581.** Dense alone was 0.800. The
second retriever cost us 0.350.

The reason is in the formula. RRF has no notion of how good a retriever is, or how confident it is on
this query. BM25's rank 1 contributes the same `1/61` as the dense retriever's rank 1. On the six
Turkish questions BM25's ranking is noise, and that noise votes at full weight. It cannot vote less
on the queries it does not understand, because the only thing it hands over is an ordering, and an
ordering always looks like an opinion.

That is the precondition nobody states out loud: **reciprocal rank fusion assumes both inputs are
independently sound.** Two decent retrievers failing on different questions cover each other. A good
retriever and one that is systematically wrong on a third of your corpus just average out, and
averaging is not a repair.

<div class="presenter-note">
Get a commitment before the fusion cell runs. "Dense is 0.800, BM25 is 0.300. Fused — over 0.800,
between them, or under 0.300?" Take a rough vote. Most of the room says over 0.800, because fusion
sounds like addition. Then show <strong>0.450</strong> and say nothing for five seconds. If Ollama is
down, every number here is in <code>eval/RESULTS.md</code>; read it off and keep moving. 5 minutes.
</div>

## The twist that keeps BM25 in the toolbox

BM25 is not a bad retriever. Over **whole documents** it beats dense retrieval on exact-token
queries — flight codes, bulletin ids, fare basis codes like `KSHEU26`. Our dense numbers show the
same shape from the other side: exact-token questions score **1.000** over whole documents and fall
to **0.500** under fixed 280-character chunking.

Chunking destroyed it, in two ways. The length normalisation stops working: BM25 divides by document
length over the average, and once every chunk is the same size that term is a constant, so a
discriminator you relied on quietly disappears. And the evidence gets split — a six-term query used
to find all six terms accumulating in one document; after chunking they sit in three chunks, none of
which scores much.

So the honest statement is not "hybrid search is overrated". It is: **hybrid search over chunks, on
a cross-lingual corpus, measured at 0.450 against 0.800 for dense alone.** Change the index unit and
the answer can flip — one retrieval decision deciding another, for the second time today.

## The reranker, and the most surprising number of the day

A reranker takes the top candidates and asks a language model to judge them. Ours is `qwen2.5:3b`,
scoring each candidate independently from 0 to 10 — one call per candidate, ties keeping the
retriever's original order, so it moves a document only when the model has an opinion. Same
reranker, four retrieval setups of different quality.

| retrieval setup | hit@1 before | after |
|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** |
| `nomic` + structure-aware | 0.350 | **0.450** |
| `bge-m3` + fixed-280 | 0.650 | 0.550 |
| `bge-m3` + structure-aware | **0.800** | 0.650 |

It helped both weak setups and hurt both strong ones. Look at what splits the table: not the
chunking, since both strategies appear on both sides, but the **embedder**. It lifts both `nomic`
setups to about 0.45 and drags both `bge-m3` setups down to about 0.65.

**A reranker levels, and it levels to its own ceiling.** A 3B model's opinion about relevance is
worth roughly 0.45 to 0.65 on this corpus. If your retriever is worse than that, imposing the
model's opinion is an upgrade. If your retriever is already better, imposing it can only lose. There
is no third outcome.

The per-question view says the same thing without averaging. Where the gold document was already at
rank 1, reranking moved it **down five times out of five**. Where it sat at rank 5 or 6, reranking
pulled it up. The reranker is not making rankings better or worse; it drags every ranking toward its
own accuracy. The price for that negative result: **8 model calls and about 4 seconds per query.**

<div class="presenter-note">
The sentence not to garble: <strong>a reranker pays when your retriever is worse than your reranker,
and costs you when it is better.</strong> Say it once, slowly, then point back at the show of hands
on the board. Somebody will object that a real reranker is a cross-encoder, not a chat model. They
are right — answer honestly that we did not measure one and it is the first thing to try. Do not
invent a number for it. 3 minutes.
</div>

## How you ask beats whether you ask

The obvious way to rerank is one call: paste six passages in, ask the model to order them. Cheaper,
and it reads better in code. On the ten-document probe corpus, five questions: no reranking **4/5**,
MRR 0.833. Listwise, "put these six in order", **2/5**, MRR 0.600 — worse than doing nothing.
Pointwise, "score this one passage 0 to 10", six calls instead of one: **5/5**, MRR 1.000. This is
`n = 5`; treat the direction as real and the magnitude as unproven.

Ordering six passages means holding six comparisons in working memory and emitting a permutation. A
3B model is bad at that and good at "how relevant is this one passage" — a single judgement against a
fixed scale. Same model, same passages, same information; the difference between helping and hurting
is the shape of the question.

## Contextual retrieval — you already shipped it

Contextual retrieval means prefixing each chunk with enough surrounding context that it stands alone,
usually by having a model write a line about where the chunk sits. Structure-aware chunking already
prefixes every chunk with its own heading path: same mechanism, taken from the document instead of
generated, at zero inference cost. It is why the K row and its column header survive together and the
model reads **EUR 90** instead of **EUR 70**. The 0.800 in module 7 is contextual retrieval's win,
already banked. Generating the context with a model on top is worth testing — but on documents that
carry headings, check you are buying something Markdown did not give you free.

## What you run

Notebook: `06_hybrid_rerank_contextual.ipynb`.

```bash
ollama serve                       # in a second terminal if it is not running
python scripts/verify_setup.py     # must print green before you continue
jupyter lab notebooks/06_hybrid_rerank_contextual.ipynb
```

Everything comes from the shared code:

```python
from eval.retrieval import BM25, DenseRetriever, rrf, pointwise_rerank
from eval.chunking import structure_aware, chunk_corpus
from eval.metrics import load_gold, evaluate

chunks = chunk_corpus(corpus, structure_aware)          # 152 chunks
dense  = DenseRetriever(chunk_ids, chunk_texts)         # bge-m3
bm25   = BM25(chunk_ids, chunk_texts)

gold = load_gold("eval/gold_questions.jsonl")
for name, rank in (("dense", dense.rank), ("bm25", bm25.rank),
                   ("rrf", lambda q: rrf([dense.rank(q), bm25.rank(q)]))):
    print(name, evaluate({g["id"]: rank(g["query"]) for g in gold}, gold))

reranked = {g["id"]: pointwise_rerank(g["query"], dense.rank(g["query"])[:8], texts)
            for g in gold}                              # 8 calls per query, watch the clock
print("reranked", evaluate(reranked, gold))
```

Short of time, run the reranked cell on the `tr_en` subset first; that is where the levelling shows
most clearly.

## What the numbers said

<div class="measured">

| retriever over the same 152 structure-aware chunks | hit@1 | MRR |
|---|---|---|
| dense, `bge-m3` | **0.800** | **0.846** |
| BM25 | 0.300 | 0.465 |
| RRF of the two | 0.450 | 0.581 |
| BM25 on the 6 `tr_en` questions | 0.000 | — |

| same reranker, four retrieval setups | hit@1 before | after | MRR before | after |
|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.494 | **0.544** |
| `nomic` + structure-aware | 0.350 | **0.450** | 0.500 | **0.562** |
| `bge-m3` + fixed-280 | 0.650 | 0.550 | 0.789 | 0.708 |
| `bge-m3` + structure-aware | **0.800** | 0.650 | **0.846** | 0.750 |

| probe corpus, 10 documents, 5 questions | hit@1 | MRR |
|---|---|---|
| no reranking | 4/5 | 0.833 |
| listwise — "put these 6 in order" | 2/5 | 0.600 |
| pointwise — "score this passage 0–10" | 5/5 | 1.000 |

| cost of one rerank pass | |
|---|---|
| model calls per query | 8 |
| added latency per query | ~4 s |

</div>

Corpus: 28 documents, 75 KB. Gold set: 20 questions, 6 of them Turkish-query / English-document.
Generation and reranking `qwen2.5:3b`, embeddings `bge-m3` unless stated, all local through Ollama.
**Twenty questions decides between two designs and does not support a general claim.** Nothing here
says these techniques are bad. It says that on this corpus, with this embedder and this reranker,
they lost, and it names the precondition each one needed.

## Going deeper

The `k = 60` in RRF is a smoothing constant. Small `k` and rank 1 dominates, so fusion behaves like
"trust whichever retriever is most confident"; large `k` flattens it into a popularity vote across
the whole list. 60 comes from the original TREC work and is nearly always left alone. Tuning it would
not have saved us: no value of `k` lets a ranking that is 0.000 on a third of the questions
contribute usefully, because the problem is not the weighting curve but that the input carries no
signal on those queries. Fusion with an uneven pair needs per-query weighting — deciding before you
fuse whether this is a query BM25 can answer — and that router is another model to build and measure.

Our reranker is a chat model scoring passages, the cheapest thing to wire up and the weakest thing
you can call a reranker. The production answer is a cross-encoder: a model that reads query and
passage **together** in one forward pass and outputs a relevance score, trained on relevance labels
rather than prompted into the job. `bge-reranker-v2-m3` is the multilingual one and pairs with the
embedder we already use. It is a stronger judge and would plausibly sit above 0.800 — but we did not
measure it, so treat that as a hypothesis with an experiment attached, not a result. The lesson
transfers either way: it still has a ceiling, and you still have to find out where that ceiling sits
relative to your retriever.

The levelling result generalises past reranking. Any stage that overwrites an upstream ranking
imposes its own accuracy on the output, whatever the input was. That is why "just add a reranker" is
bad advice in the same way "just add a cache" is: the precondition is a fact about your system you
can only learn by measuring. Order matters too — module 6 fixed the embedder, module 7 fixed the
chunking, and together they took hit@1 from 0.550 to 0.800. Everything in this module landed after
the ceiling was already high, which is exactly when these techniques cost. Two rows of the four-setup
table would have shipped a reranker on a genuine improvement, and the design would still be wrong,
because the right move was to fix the embedder instead.

At ten million documents the shape changes and BM25 returns for a different reason. You cannot embed
and score every chunk per query, so a cheap first stage returns a few hundred candidates and an
expensive second stage ranks them. BM25 over an inverted index is a superb first stage —
sub-millisecond, exact on identifiers, trivially updatable — with a cross-encoder over its top 200
behind it. That is not the hybrid we measured: BM25 generates candidates rather than voting on the
answer, and recall@k is its metric rather than hit@1. Same component, different job, different
metric. At 28 documents that architecture has nothing to do and costs four seconds a query.

<div class="presenter-note">
Running late: cut the listwise-versus-pointwise section to one sentence and keep the four-setup
table — the levelling result is what people repeat afterwards. Before you close, point at the hands
still written on the board and read the three verdicts: hybrid lost, reranking levelled, contextual
already shipped in module 7. Then name what still fails — the multi-hop questions in the gold set.
That is module 10, and it opens on a failure you can show in one cell. 2 minutes.
</div>

## Exit line

> Measurably, none of them beat plain retrieval here. But multi-hop still fails.
