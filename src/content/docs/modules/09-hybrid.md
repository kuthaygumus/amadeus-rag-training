---
title: "9. Hybrid, Rerank and Contextual"
description: "What if semantic search is not enough?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **What if semantic search is not enough?**

The last measured retrieval number is **hit@1 0.800**: `bge-m3` over 154 structure-aware chunks,
scored with the in-memory retriever at the end of module 7. Module 8 moved those same vectors into
ChromaDB and deliberately did not re-run the twenty-question benchmark — over 154 vectors the search
is exhaustive, so the ranking should be identical, but module 8 says plainly that "should be" is not
a measurement. So 0.800 is still the last measured number, and it is the one this module argues
about.

0.800 is sixteen questions out of twenty. Here are the four that miss, with the rank the gold
document actually sat at:

| question | type | rank of the gold document |
|---|---|---|
| q05 | `tr_en` | 13 |
| q07 | `tr_en` | 7 |
| q19 | `multi_hop` | 6 |
| q14 | `en_en` | 2 |

Three of the four are not near-misses. And when the top document is wrong the generator does not
hesitate — it answers from the wrong page in the same voice it uses when it is right.

There is a standard list of things to put in that gap. Every RAG post says the same three words:
hybrid, rerank, contextual. Add BM25 next to the vector search, put a model on top to reorder the
results, prefix each chunk with its context. We run all three against the same twenty questions. One
of them you already shipped in module 7 without noticing. The other two lose.

<div class="presenter-note">
This module has <strong>38 minutes</strong> in the agenda and its three measurements — BM25 and
fusion, the four-setup rerank sweep, and the per-question view — are the ones that must not be cut.
Open by putting the four missing questions on the screen: the gate here is a ceiling, not a crash.
Then write on the board <strong>hybrid · rerank · contextual</strong> and ask for a show of hands:
"which of these three moves 0.800 upwards?" Almost every hand goes up for rerank, most for hybrid.
Write the counts next to the words and leave them there — you will point at them twice.
4 minutes, laptops closed.
</div>

## Bringing BM25 back

BM25 is the keyword retriever from module 5: rare terms score high, term frequency saturates, long
documents get penalised. It found `H9 1487` instantly and scored a flat zero on "iptal edersem ne
öderim". Run it over exactly the chunks the dense retriever is using — the same 154, same index:

```python
bm25 = R.BM25(chunk_ids, chunk_texts)
```

**hit@1 0.300, MRR 0.467.** On the six Turkish-query / English-document (`tr_en`) questions,
**0.000**. Not degraded. Zero. BM25 has no path from `iptal` to `Cancellation penalty`, and chunking
gave it none.

It is not Turkish that defeats it. On the four `tr_tr` questions — Turkish query, Turkish document,
shared vocabulary — BM25 scores **0.750**. What it cannot cross is the language boundary between the
query and the document.

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

Fuse dense and BM25 over the same chunks: **hit@1 0.450, MRR 0.586.** Dense alone was 0.800. The
second retriever cost us 0.350.

Look at where it went. On the six `tr_en` questions dense alone scores 0.667 and BM25 scores 0.000 —
and the fused ranking scores **0.000**. Fusion did not average those two. It inherited the failure.

The reason is in the formula. RRF has no notion of how good a retriever is, or how confident it is on
this query. BM25's rank 1 contributes the same `1/61` as the dense retriever's rank 1. On those six
questions BM25's ranking is noise, and that noise votes at full weight. It cannot vote less on the
queries it does not understand, because the only thing it hands over is an ordering, and an ordering
always looks like an opinion.

That is the precondition nobody states out loud: **reciprocal rank fusion assumes both inputs are
independently sound.** Two decent retrievers failing on different questions cover each other. A good
retriever and one that is systematically wrong on a third of your corpus just average out, and
averaging is not a repair.

<div class="presenter-note">
Get a commitment before the fusion cell runs. "Dense is 0.800, BM25 is 0.300. Fused — over 0.800,
between them, or under 0.300?" Take a rough vote. Most of the room says over 0.800, because fusion
sounds like addition. Then show <strong>0.450</strong> and say nothing for five seconds. The `tr_en`
row is the follow-up: 0.667 and 0.000 fused to 0.000. If Ollama is down, every number in this section
is in <code>eval/RESULTS.md</code>; read it off and keep moving — BM25 itself needs no model at all,
so that half of the cell always runs. 8 minutes, including the BM25 section above.
</div>

## What chunking did to BM25

BM25 is not a bad retriever, and this is not the condition it is good in. Over **whole documents** —
the module 5 index, before any chunking — BM25 scored hit@1 **0.400**, MRR **0.515**. On the four
exact-token questions there, the flight codes and bulletin ids and fare basis codes like `KSHEU26`,
it reached **0.750**.

That is the number usually quoted as "BM25 beats embeddings on identifiers". On this corpus it does
not: `bge-m3` over the same whole documents scores **0.550** overall and **1.000** on those same four
exact-token questions. BM25 was close enough to be worth having, and behind, in the one condition it
is supposed to win.

The asymmetry BM25 does have over whole documents is a language one, and it is sharp. On the four
`tr_tr` questions — Turkish query, Turkish document — it scores **1.000**, level with `bge-m3` on
the same four. On the six `tr_en` questions it scores **0.000** against `bge-m3`'s 0.333. Vocabulary
overlap is the whole of what it can use, so it is at its best and at its worst on the same corpus,
decided by which language the document happens to be in.

Chunking removed even that. Over the 154 chunks BM25's exact-token score falls from 0.750 to
**0.500** and its overall MRR from 0.515 to 0.467, in two ways. The length normalisation stops
working: BM25 divides by document length over the average, and once every chunk is the same size that
term is a constant, so a discriminator you relied on quietly disappears. And the evidence gets split —
a six-term query used to find all six terms accumulating in one document; after chunking they sit in
three chunks, none of which scores much.

So the honest statement is not "hybrid search is overrated". It is: **hybrid search over chunks, on a
cross-lingual corpus, measured 0.450 against 0.800 for dense alone.** Change the index unit and every
number in this section moves — one retrieval decision deciding another, for the second time today.
What did not move on this corpus is which of the two retrievers is ahead.

## The reranker, and the number that decides the module

A reranker takes the top candidates and asks a language model to judge them. Ours is `qwen2.5:3b`,
scoring each candidate independently from 0 to 10 — one call per candidate, ties keeping the
retriever's original order, so it moves a document only when the model has an opinion. Same reranker,
four retrieval setups of deliberately different quality.

| retrieval setup | hit@1 before | after | MRR before | after | verdict |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.503 | 0.543 | helps |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.492 | 0.537 | helps |
| `bge-m3` + fixed-280 | 0.700 | 0.550 | 0.814 | 0.712 | hurts |
| `bge-m3` + structure-aware | **0.800** | 0.600 | 0.844 | 0.717 | hurts |

It helped both weak setups and hurt both strong ones. Look at what splits the table: not the
chunking, since both strategies appear on both sides, but the **embedder**.

**A reranker levels, and it levels toward its own ceiling.** Before reranking the four setups ran
from 0.350 to 0.800 — a spread of 0.450. After, they run from 0.400 to 0.600, a spread of 0.200. The
same model pulled the bottom up and the top down. A 3B model's opinion about relevance is worth
roughly 0.40 to 0.60 on this corpus. If your retriever is worse than that, imposing the model's
opinion is an upgrade. If your retriever is already better, imposing it can only lose. There is no
third outcome.

The per-question view says the same thing without averaging. On the strongest setup, **sixteen** of
the twenty questions already had the gold document at rank 1, and reranking pushed **five** of them
down: q06 to rank 3, q10 to rank 2, q11 to rank 4, q16 to rank 2, q17 to rank 3. Of the **four** that
did not have it first, it pulled **two** up: q14 from rank 2 to rank 1, and q19 from rank 6 to rank
5. Only the q14 move changes hit@1, which is how 0.800 becomes 0.600. You bought two repairs by
breaking five things that were not broken.

The price for that negative result: **8 model calls per question, 160 per setup**, and between
**49.3 and 87.6 seconds** per setup on an M-series Mac.

<div class="presenter-note">
Build the mechanism in the notebook first — one query, six candidates, the scores printed — and let
the room see that a reranker is a for-loop with a prompt in it, not a library. Point at the ties in
the score column: a 3B model asked for an integer gives you a handful of distinct values, and ties
keep the retriever's order. Then start <code>exercises/m9_rerank_trade.py</code> and take the guess
while it runs: does it help all four setups, none, or some? The sentence not to garble when it
finishes: <strong>a reranker pays when your retriever is worse than your reranker, and costs you when
it is better.</strong> Say it once, slowly, then point back at the show of hands on the board.
Somebody will object that a real reranker is a cross-encoder, not a chat model. They are right —
answer honestly that we did not measure one and it is the first thing to try. Do not invent a number
for it. On a slow laptop, <code>--quick</code> finishes in a quarter of the calls and says REDUCED at
the top; the two strong rows are the ones the argument needs. 20 minutes, the run included.
</div>

## How you ask, measured at n = 5

There is a cheaper shape for a reranker: one call, all six passages, ask the model to put them in
order. The notebook makes that call on one question and prints what comes back. On our run the model
answered `1,4,2,5` — four numbers for six passages.

Ordering six passages means holding six comparisons in working memory and emitting a permutation. A
3B model is bad at that and good at "how relevant is this one passage" — a single judgement against a
fixed scale. Same model, same passages, same information; the difference is the shape of the
question.

The only quantitative version of that comparison we have is from a retired probe corpus of ten
documents and five questions: listwise, "put these six in order", **2/5** with MRR 0.600; pointwise,
"score this one passage 0 to 10", **5/5** with MRR 1.000. That is `n = 5`. Treat the direction as
real and the magnitude as unproven — and do not read it as evidence that pointwise reranking helps.
The same pointwise reranker measured on these twenty questions took the strong setup from 0.800 down
to 0.600. It is evidence about the shape of the prompt, and nothing else.

## Contextual retrieval — you already shipped it

Contextual retrieval means prefixing each chunk with enough surrounding context that it stands alone,
usually by having a model write a line about where the chunk sits. Structure-aware chunking already
prefixes every chunk with its own heading path: same mechanism, taken from the document instead of
generated, at zero inference cost. It is why the K row and its column header survive in the same
chunk, and the model reads the cancellation penalty out of the right column instead of the one next
to it. The 0.800 in module 7 is contextual retrieval's win, already banked. Generating the context
with a model on top is worth testing — but on documents that carry headings, check you are buying
something Markdown did not give you free.

## What you run

**The mechanism — in the notebook.** Open `notebooks/06_hybrid_rerank_contextual.py` in VS Code and
run the blocks with Shift+Enter (the Microsoft Python extension; there is no Jupyter in this course).

```bash
ollama serve                       # in a second terminal if it is not running
python scripts/verify_setup.py     # must print green before you continue
```

- **what you should see** — `154 chunks indexed both ways`, then the three-way comparison: dense
  0.800, BM25 0.300, RRF 0.450. Then one query's six candidates scored 0 to 10, printed with the
  seconds they took.
- **roughly how long** — one embedding pass over 154 chunks plus six model calls. A few minutes,
  most of it the embedding. BM25 and the fusion cost no model call at all.

**The measurement — one command.**

```bash
python exercises/m9_rerank_trade.py            # the full sweep
python exercises/m9_rerank_trade.py --quick    # slow laptop: strong setups only, depth 4
```

- **what you should see** — four rows, one per retrieval setup, each carrying its own call count and
  seconds: HELPED on the two `nomic` rows, hurt on the two `bge-m3` rows. Then the per-question
  summary: 16 questions had the gold document at rank 1 and reranking pushed 5 of them down.
- **the cost, which is part of the lesson** — 4 setups × 20 questions × 8 candidates = **640 model
  calls**, 160 per setup. Measured per setup: 49.3 s, 52.4 s, 65.2 s and 87.6 s on an M-series Mac,
  a little over four minutes of model time in total, and considerably longer on a CPU-only laptop.
  `--quick` runs the two strong setups at depth 4 — 160 calls — and prints REDUCED at the top so its
  numbers are never mistaken for the ones on this page.

Everything comes from the shared code. Nothing here is a new dependency:

```python
import sys; sys.path.insert(0, "eval")
from pathlib import Path
import chunking as C, metrics, retrieval as R

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("eval/gold_questions.jsonl")

chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")   # 154 chunks
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)      # bge-m3, one embed call per chunk
bm25  = R.BM25(chunk_ids, chunk_texts)

runs = {"dense": {}, "bm25": {}, "rrf": {}}
for q in questions:
    d, b = dense.rank(q["query"]), bm25.rank(q["query"])
    runs["dense"][q["id"]], runs["bm25"][q["id"]] = d, b
    runs["rrf"][q["id"]] = R.rrf([d, b])

for name, run in runs.items():
    # score documents, not chunks: collapse each chunk ranking to its parent documents first
    scored = metrics.evaluate({k: C.to_documents(v) for k, v in run.items()}, questions)
    print(f"{name:<6} hit@1 {scored['hit@1']:.3f}  MRR {scored['MRR']:.3f}")

# the reranker itself: one model call per candidate, on one question
q = next(x for x in questions if x["id"] == "q05")
candidates = runs["dense"][q["id"]][:6]               # 6 candidates -> 6 model calls
print(R.pointwise_rerank(q["query"], candidates, chunks))
```

## What the numbers said

<div class="measured">

| retriever over the same 154 structure-aware chunks | hit@1 | MRR |
|---|---|---|
| dense, `bge-m3` | **0.800** | **0.844** |
| BM25 | 0.300 | 0.467 |
| RRF of the two | 0.450 | 0.586 |
| BM25 on the 6 `tr_en` questions | 0.000 | — |
| RRF on the 6 `tr_en` questions | 0.000 | — |

| whole documents, before any chunking | hit@1 | MRR | exact_token | tr_tr | tr_en |
|---|---|---|---|---|---|
| dense, `bge-m3` | 0.550 | 0.654 | **1.000** | 1.000 | 0.333 |
| BM25 | 0.400 | 0.515 | 0.750 | 1.000 | **0.000** |

| same reranker, four retrieval setups | hit@1 before | after | MRR before | after |
|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.503 | **0.543** |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.492 | **0.537** |
| `bge-m3` + fixed-280 | 0.700 | 0.550 | 0.814 | 0.712 |
| `bge-m3` + structure-aware | **0.800** | 0.600 | 0.844 | 0.717 |

| the strongest setup, question by question | count |
|---|---|
| gold document already at rank 1 | 16 |
| of those, demoted by reranking | 5 — q06, q10, q11, q16, q17 |
| gold document below rank 1 | 4 |
| of those, promoted by reranking | 2 — q14, q19 |

| what one rerank pass costs, at depth 8 | |
|---|---|
| model calls per question | 8 |
| model calls per setup, 20 questions | 160 |
| the full four-setup sweep | 640 |
| measured wall clock per setup | 49.3 s – 87.6 s |

</div>

Corpus: 28 documents, 78,310 characters. Gold set: 20 questions, 6 of them Turkish-query /
English-document. Generation and reranking `qwen2.5:3b`, embeddings `bge-m3` unless stated, all local
through Ollama. **Twenty questions decides between two designs and does not support a general
claim**; a difference under about 0.05 is inside this sample's noise. Nothing here says these
techniques are bad. It says that on this corpus, with this embedder and this reranker, they lost, and
it names the precondition each one needed.

## Going deeper

The `k = 60` in RRF is a smoothing constant. Small `k` and rank 1 dominates, so fusion behaves like
"trust whichever retriever is most confident"; large `k` flattens it into a popularity vote across
the whole list. 60 comes from the original TREC work and is nearly always left alone. Tuning it would
not have saved us: no value of `k` lets a ranking that is 0.000 on six of the twenty questions
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
imposes its own accuracy on the output, whatever the input was. That is why "add a reranker" is bad
advice in the same way "add a cache" is: the precondition is a fact about your system you can only
learn by measuring. Order matters too. Module 7 moved hit@1 from 0.550 to 0.800 by changing nothing
but the chunking — both of those numbers are `bge-m3`. Module 6's lever was the other one: on the
same structure-aware chunks, `nomic-embed-text` scores 0.350 against `bge-m3`'s 0.800. Everything in
this module landed after both levers had already been pulled, which is exactly when these techniques
cost. Two rows of the four-setup table would have shipped a reranker on a genuine improvement —
`nomic` really does go from 0.350 to 0.450 and 0.400 — and the design would still be wrong, because
the right move was to fix the embedder instead.

At ten million documents the shape changes and BM25 returns for a different reason. You cannot embed
and score every chunk per query, so a cheap first stage returns a few hundred candidates and an
expensive second stage ranks them. BM25 over an inverted index is a strong first stage —
sub-millisecond, exact on identifiers, trivially updatable — with a cross-encoder over its top 200
behind it. That is not the hybrid we measured: BM25 generates candidates rather than voting on the
answer, and recall@k is its metric rather than hit@1. Same component, different job, different
metric. At 28 documents that architecture has nothing to do and costs seconds per query.

<div class="presenter-note">
Running late: keep all three measurements and cut the prose instead — the n=5 listwise aside goes to
one sentence, and "Going deeper" is reading material. Before you close, point at the hands still
written on the board and read the three verdicts: hybrid lost, reranking levelled, contextual was
already shipped in module 7. Then name what still fails. Look back at the four questions on the
opening slide: q19 is a multi-hop question and it is still there after everything in this module.
That is module 10, and it opens on a failure you can show in one cell. 6 minutes.
</div>

## Exit line

> Measurably, none of them beat plain retrieval here. But multi-hop still fails.
