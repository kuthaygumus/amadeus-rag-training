---
title: "9. Hybrid, Rerank and Contextual"
description: "What if semantic search is not enough?"
---

## Gate question

> **What if semantic search is not enough?**

The last measured retrieval number is **hit@1 0.750**: `bge-m3` over 154 structure-aware chunks.
Module 7's ladder went one rung further — stripping the legal footer reached 0.800 over 153 chunks —
but everything in this module runs on the unstripped 154, so 0.750 is the number that compares.
Module 8 moved those vectors into ChromaDB and deliberately did not re-measure: over 154 vectors the
search is exhaustive, and "should be identical" is not a measurement.

0.750 is fifteen questions out of twenty. Five miss, not the clean two-of-three-types it used to be:

| question | type |
|---|---|
| q05 | `tr_en` |
| q07 | `tr_en` |
| q11 | `en_en` |
| q14 | `en_en` |
| q19 | `multi_hop` |

Two `tr_en`, two `en_en`, one `multi_hop` — its twin, q20, now lands at rank 1. Every run in this
module still returns these same five — `--fusion` prints them under its table. Only one of their
ranks is printed: q05's `sop_misconnect_v4` sits at **rank 12 of 28**, which is not a near miss.
`UNVERIFIED: the ranks of the other four — no command in this repository prints per-question ranks
for this run.` When the top document is wrong the generator does not hesitate; it answers from the
wrong page in the voice it uses when it is right.

Every RAG post offers the same three words for that gap: hybrid, rerank, contextual — BM25 next to
the vector search, a model on top to reorder the results, a context line in front of each chunk. We
run all three against the same twenty questions. One of them you already shipped in module 7 without
noticing. The other two lose.

<div class="presenter-note">
This module has <strong>38 minutes</strong> in the agenda and its three measurements — BM25 and
fusion, the four-setup rerank sweep, and the per-question view — are the ones that must not be cut.
Open by putting the five missing questions on the screen: the gate here is a ceiling, not a crash.
Then write on the board <strong>hybrid · rerank · contextual</strong> and ask for a show of hands:
"which of these three moves 0.750 upwards?" Almost every hand goes up for rerank, most for hybrid.
Write the counts next to the words and leave them there — you will point at them twice.
4 minutes, laptops closed.
</div>

## Bringing BM25 back

BM25 is the keyword retriever from module 5: rare terms score high, term frequency saturates, long
documents get penalised. It found `XX 1487` instantly and scored a flat zero on "iptal edersem ne
öderim". Run it over exactly the chunks the dense retriever is using — the same 154, same index.

**VS Code — `notebooks/06_hybrid_rerank_contextual.py`, in the indexing block that has just
produced `chunk_ids` and `chunk_texts`:**

```python
bm25 = R.BM25(chunk_ids, chunk_texts)
```

**hit@1 0.300, MRR 0.467.** On the six Turkish-query / English-document (`tr_en`) questions,
**0.000**. Not degraded. Zero. BM25 has no path from `iptal` to `Cancellation penalty`, and chunking
gave it none. It is not Turkish that defeats it: on the four `tr_tr` questions — Turkish query,
Turkish document, shared vocabulary — it scores **0.750**. What it cannot cross is the language
boundary between the query and the document.

## Reciprocal rank fusion, and why it dilutes

Fusion is the fix everyone reaches for, and it is genuinely elegant. The two retrievers do not need
comparable scores — cosine similarity and a BM25 score share no unit — so you throw the scores away
and keep only the positions.

**`eval/retrieval.py` — the whole of the fusion step:**

```python
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
```

Do the arithmetic before you assume `k = 60` protects you: rank 1 is worth `1/61`, rank 3 nearly the
same, and rank 154 is still worth **28% of rank 1**. And both inputs here are complete permutations
of the same 154 chunks, so every chunk sits in both rankings — there is no "only one retriever found
it" case at all. Fusion becomes a sum of two smoothed reciprocal ranks, and the weaker ranking votes
on everything, all the way down.

Fuse dense and BM25 over the same chunks: **hit@1 0.450, MRR 0.579.** Dense alone was 0.750; the
second retriever cost 0.300, six questions. On the six `tr_en` questions dense scores 0.667, BM25
scores 0.000, and the fused ranking scores **0.000**. Fusion did not average those two; it inherited
the failure, and it cannot do otherwise, because BM25 hands over an ordering and an ordering always
looks like an opinion. That is the precondition nobody states out loud — **reciprocal rank fusion
assumes both inputs are independently sound.** A good retriever and one that is systematically wrong
on a third of your corpus just average out, and averaging is not a repair.

Fusion does win something once, and it is worth pricing rather than celebrating. At **document
level**, where BM25 has whole documents to match against, RRF takes the best recall@5 on the page:
**0.733** against dense's 0.717. It buys that **0.016 of recall@5 with 0.100 of hit@1**, 0.600 down
to 0.500. The gain is a fraction of one question; the cost is two whole ones. Even there its `tr_en`
row is 0.000 against dense's 0.333.

<div class="presenter-note">
Get a commitment before the fusion cell runs. "Dense is 0.750, BM25 is 0.300. Fused — over 0.750,
between them, or under 0.300?" Take a rough vote. Most of the room says over 0.750, because fusion
sounds like addition. Then show <strong>0.450</strong> and say nothing for five seconds. The `tr_en`
row is the follow-up: 0.667 and 0.000 fused to 0.000. If Ollama is down, every number in this section
is in <code>eval/RESULTS.md</code>, and <code>USE_CACHED=1</code> replays the whole notebook from
<code>cached_runs.json</code>; read it off and keep moving — BM25 itself needs no model at all, so
that half of the cell always runs. 8 minutes, including the BM25 section above.
</div>

## What chunking did to BM25

BM25 is not a bad retriever, and this is not the condition it is good in. Over **whole documents** —
the module 5 index, before any chunking — it scored hit@1 **0.400**, MRR **0.515** and **0.750** on
the four exact-token questions. That is the number usually quoted as "BM25 beats embeddings on
identifiers". Here it does not: `bge-m3` over the same whole documents scores **0.600** overall and
**1.000** on those same four. The asymmetry BM25 does have is a language one — **1.000** on the four
`tr_tr` questions and **0.000** on the six `tr_en`, at its best and at its worst on one corpus,
decided by which language the document happens to be in.

Chunking cost it the rest: over the 154 structure-aware chunks its exact-token score falls from 0.750
to **0.500** and its MRR from 0.515 to **0.467**. Be precise about why, because module 5 told you a
stronger version. There, over 294 fixed-280 chunks, every chunk really is the average length and
BM25's length term becomes a constant; structure-aware chunks still vary in length, so here the
signal is compressed rather than erased. The other reason holds in both: a six-term query that used
to find all six terms in one document now finds them split across three chunks, none scoring much.

So the honest statement is not "hybrid search is overrated". It is: **hybrid search over chunks, on a
cross-lingual corpus, measured 0.450 against 0.750 for dense alone.** Change the index unit and every
number here moves — one retrieval decision deciding another, for the second time today.

## The reranker, and the number that decides the module

A reranker takes the top candidates and asks a language model to judge them. Ours is `qwen2.5:3b`,
scoring each candidate independently from 0 to 10 — one call per candidate, ties keeping the
retriever's original order, so it moves a document only when the model has an opinion. Same reranker,
four retrieval setups of deliberately different quality.

<div class="measured">

| retrieval setup | hit@1 before | after | MRR before | after | verdict |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.515 | **0.544** | HELPED, +0.029 MRR |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.490 | **0.540** | HELPED, +0.050 |
| `bge-m3` + fixed-280 | 0.700 | 0.500 | 0.817 | 0.680 | hurt, −0.137 |
| `bge-m3` + structure-aware | **0.750** | 0.600 | 0.817 | 0.725 | hurt, −0.092 |

</div>

It helped both weak setups and hurt both strong ones, without exception in either direction. What
splits the table is not the chunking — both strategies appear on both sides — but the **embedder**.

**A reranker levels, and it levels toward its own ceiling.** Before, the four setups ran from 0.350
to 0.750, a spread of 0.400; after, from 0.400 to 0.600, a spread of 0.200. Say the condition with
the band, though: handed candidates from a retriever of this quality, this reranker's output lands
between 0.400 and 0.600 whatever it was given. That band belongs to the **pair**, not to the model —
it only permutes the retriever's own top 8, so a worse candidate list would drag it down too. If your
retriever is worse than that pairing, imposing the model's opinion is an upgrade. If it is already
better, imposing it can only lose.

The per-question view says the same without averaging. On the strongest setup, **fifteen** of the
twenty questions already had the gold document at rank 1, and reranking pushed **four** down: q06,
q10, q16, q18. Of the **five** that did not have it first, it pulled **one** up, q14 — and the
after column's 0.600 is twelve questions at rank 1: fifteen minus four plus one. Four
breakages against one real gain: the table's −0.150 hit@1, one question at a time. `UNVERIFIED: which
rank q14 moved to — the exercise prints direction and count, not position.`

The price: **8 model calls per question, 160 per setup, 640 for the table.** On this run the four
setups took **191 s, 127 s, 51 s and 71 s** in the order they print. `UNVERIFIED: the total wall
clock for the sweep and the timing for --quick — the script prints only these four per-setup times,
and this run recorded neither.` One run on one M-series Mac with the models already resident: an
order of magnitude, not a specification. The call count is fixed, the clock is not, and a CPU-only
laptop is much slower.

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

### How you ask, in one call or six

There is a cheaper shape for a reranker: one call, all six passages, ask the model to put them in
order. The notebook makes both calls on the same question, over the same six candidates, with the
same model. Pointwise returned six integers — 4, 2, 2, 5, 5, 3 — in six calls and 3.3 s. Listwise
returned `1,4,2,5`: four indices for six passages, with no ranking in it to use.

Be fair about what that is not. Our pointwise path parses — an unreadable reply becomes 0 and ties
keep the retriever's order, so it degrades to doing nothing — while the listwise cell has no parser
and no fallback, and it shows the model 600 characters of each passage where pointwise shows 900.
What failed on screen is format compliance, not judgement, and nothing scored the listwise output:
this shows the cheap shape producing something unusable, not ranking worse. The older head-to-head
that first pointed this way ran on a retired ten-document probe corpus and is marked `UNVERIFIED` in
`eval/RESULTS.md`; its numbers are not quoted here.

And none of it argues for reranking at all. The same pointwise reranker, over these twenty questions,
took the strongest setup from 0.750 down to 0.600.

## Contextual retrieval — you already shipped it

Contextual retrieval means prefixing each chunk with enough surrounding context that it stands alone,
usually by having a model write a line about where the chunk sits. Structure-aware chunking already
prefixes every chunk with its own heading path: same mechanism, taken from the document instead of
generated, at zero inference cost. It is why the K row and its column header survive in the same
chunk. The 0.750 in module 7 is contextual retrieval's win, already banked. Generating the context
with a model on top is worth testing — but on documents that carry headings, check you are buying
something Markdown did not give you free.

## What you run

**The mechanism — in the notebook.** Open `notebooks/06_hybrid_rerank_contextual.py` in VS Code with
the repository root as the open folder, put the cursor in a `# %%` block and press `Shift+Enter`; the
output appears in the Interactive window. There is no Jupyter server in this course.

**Terminal (repo root):**

```bash
ollama serve                       # only if it is not already running
python scripts/verify_setup.py     # checks all three models — this sweep needs nomic-embed-text too
```

- **what you should see** — `154 chunks indexed both ways`, then the three-way comparison: dense
  0.750, BM25 0.300, RRF 0.450. Then one query's six candidates scored 0 to 10, with the seconds.
- **roughly how long** — one embedding pass over 154 chunks plus six model calls; a few minutes, most
  of it the embedding. BM25 and the fusion cost no model call at all.

**The measurement — one command. Terminal (repo root):**

```bash
python exercises/m9_rerank_trade.py            # the full sweep
python exercises/m9_rerank_trade.py --quick    # slow laptop: strong setups only, depth 4
```

- **what you should see** — four rows, one per retrieval setup, each carrying its own call count and
  seconds: HELPED on the two `nomic` rows, hurt on the two `bge-m3` rows. Then the per-question
  summary: 15 questions had the gold document at rank 1 and reranking pushed 4 of them down.
- **the cost, which is part of the lesson** — 4 setups × 20 questions × 8 candidates = **640 model
  calls**, 160 per setup — 191 s, 127 s, 51 s and 71 s on the run this page quotes, in that order; no
  total is printed. `--quick` is 2 setups × 20 questions × 4 candidates = 160 calls, and it prints
  REDUCED at the top so its numbers are never mistaken for the ones on this page.

Everything comes from the shared code. The notebook's preflight moves the working directory to
`notebooks/`, which is why its paths climb one level.

**VS Code — the notebook's own cells, condensed:**

```python
import sys; from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True, replayable=True)
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")   # 154 chunks
chunks = dict(zip(chunk_ids, chunk_texts))
dense, bm25 = R.DenseRetriever(chunk_ids, chunk_texts), R.BM25(chunk_ids, chunk_texts)

q = next(x for x in questions if x["id"] == "q05")
d, b = dense.rank(q["query"]), bm25.rank(q["query"])
fused = R.rrf([d, b])                                 # fuse at chunk level, collapse to documents after
print(R.pointwise_rerank(q["query"], d[:6], chunks))  # 6 candidates -> 6 model calls
```

## What the numbers said

<div class="measured">

| retriever over the same 154 structure-aware chunks | hit@1 | recall@5 | MRR |
|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** |
| BM25 | 0.300 | 0.633 | 0.467 |
| RRF of the two | 0.450 | 0.683 | 0.579 |
| BM25 on the 6 `tr_en` questions | 0.000 | — | — |
| RRF on the 6 `tr_en` questions | 0.000 | — | — |

| whole documents, before any chunking | hit@1 | recall@5 | MRR | `exact_token` | `tr_tr` | `tr_en` |
|---|---|---|---|---|---|---|
| dense, `bge-m3` | **0.600** | 0.717 | **0.677** | **1.000** | 1.000 | **0.333** |
| BM25 | 0.400 | 0.583 | 0.515 | 0.750 | 1.000 | 0.000 |
| RRF of the two | 0.500 | **0.733** | 0.611 | **1.000** | 1.000 | 0.000 |

| the strongest setup, question by question | count |
|---|---|
| gold document already at rank 1 | 15 |
| of those, demoted by reranking | 4 — q06, q10, q16, q18 |
| gold document below rank 1 | 5 |
| of those, promoted by reranking | 1 — q14 |

| what one rerank pass costs, at depth 8 | |
|---|---|
| model calls per question | 8 |
| model calls per setup, 20 questions | 160 |
| the full four-setup sweep | 640 |
| wall clock per setup, this run | 191 s, 127 s, 51 s, 71 s |
| the full run, and `--quick` | not printed — only the four per-setup times above were measured |

</div>

Corpus: 28 documents, 78,310 characters. Gold set: 20 questions, 6 of them Turkish-query /
English-document. Generation and reranking `qwen2.5:3b`, embeddings `bge-m3` unless stated, all local
through Ollama. **Twenty questions decides between two designs and does not support a general
claim**; a difference under about 0.05 is inside the noise of a set this size. Nothing here says
these techniques are bad. It says that on this corpus, with this embedder and this reranker, they
lost, and it names the precondition each one needed.

## Going deeper

One condition we never varied: we fused the complete 154-item rankings every time. How deep you let a
weak ranking vote is a knob, and we left it at the bottom of the list.

Our reranker is a chat model scoring passages, the weakest thing you can call a reranker. The
production answer is a cross-encoder: query and passage read **together** in one forward pass,
trained on relevance labels rather than prompted into the job. `bge-reranker-v2-m3` is the
multilingual one and pairs with our embedder. We did not measure it, so "it would clear 0.750" is a
hypothesis with an experiment attached — and the lesson survives either answer, because a stronger
judge still has a ceiling you have to locate relative to your retriever.

The levelling result generalises past reranking: any stage that overwrites an upstream ranking
imposes its own accuracy on the output. That is why "add a reranker" is bad advice in the same way
"add a cache" is — the precondition is a fact about your system you can only learn by measuring.
Order matters too. Everything here landed after module 7's chunking lever (0.600 to 0.750) and module
6's embedder lever (`nomic` at 0.350 against `bge-m3`'s 0.750) had already been pulled, which is
exactly when these techniques cost. Two rows of the table would have shipped a reranker on a genuine
improvement — `nomic` really does go 0.350 to 0.450 and 0.350 to 0.400 — and the design would still
be wrong, because the right move was to fix the embedder.

At ten million documents BM25 returns for a different reason: a cheap first stage returns a few
hundred candidates and an expensive second stage ranks them, and BM25 over an inverted index is a
strong first stage — sub-millisecond, exact on identifiers, trivially updatable. That is not the
hybrid we measured: there it generates candidates rather than voting on the answer, and recall@k is
its metric rather than hit@1.

<div class="presenter-note">
Running late: keep all three measurements and cut the prose instead — the listwise aside goes to
one sentence, and "Going deeper" is reading material. Before you close, point at the hands still
written on the board and read the three verdicts: hybrid lost, reranking levelled, contextual was
already shipped in module 7. Then name what still fails. Look back at the five questions on the
opening slide: q19 is the multi-hop miss and it is still there after everything in this module — its
twin, q20, was already resolved by chunking. That is module 10, and it opens on a failure you can
show in one cell. 6 minutes.
</div>

## Exit line

> Measurably, none of them beat plain retrieval here. But multi-hop still fails.
