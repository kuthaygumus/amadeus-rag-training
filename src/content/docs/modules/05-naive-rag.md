---
title: "5. Keyword Search to Naive RAG"
description: "How do I find the right piece?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **How do I find the right piece?**

The last module ended with a working answer and a bill. All 75 KB fits in the context window,
the model quotes EUR 90 correctly, and you pay for the whole rule book on every question —
including the ones that need one paragraph out of twenty-eight documents. The rule book is
reissued quarterly. At four quarters and six stations the prompt stops fitting long before the
cost stops hurting.

So select. Send three documents instead of twenty-eight. The rest of the day is about that one
word — select — and how many ways it goes wrong.

<div class="presenter-note">
Before opening the notebook, ask the room: "You have 28 documents and a question. Write down,
in one line, how you would pick the three to send." Take two answers out loud. Somebody will say
"search for the keywords". That person has just written the first half of this module, and you
say so. 3 minutes, no laptops open yet.
</div>

## The retriever everyone reaches for first

Keyword search. Count how often the query's words appear in each document, rank by the count.
It is a reasonable first try and it is what sits behind most internal search boxes you have
ever used.

BM25 is that idea with three corrections, and `eval/retrieval.py` writes it out longhand rather
than importing a library, because watching the score come apart is the point:

```python
idf  = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
f    = self.term_freq[i].get(term, 0)
norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_length or 1)
score += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
```

Three corrections, in the order they appear. **Term frequency saturates:** a document saying
"penalty" nine times is not nine times more relevant, so `f` sits inside a fraction that flattens
out, and `k1 = 1.5` sets how fast. **Rare words count for more:** `idf` grows as `df` — the number
of documents containing the term — shrinks, so `penalty` earns almost nothing across the fare
sheets while `KSHEU26`, in one document only, earns nearly everything. **Long documents are
penalised:** `norm` divides by document length against the corpus average, so the 9 KB interline
agreement does not win every query by sheer word count; `b = 0.75` sets how hard that bites.

Run it on `"H9 1487 IST-CDG retiming"` and BM25 puts `bulletin_scb_2026_0914` first, scoring
9.481 against 5.691 for the next document. `1487` sits in four of the twenty-eight documents, but
only one of them is a one-page bulletin, so the length normalisation finishes what the idf
started. The tokenizer is crude on purpose — it splits on anything non-alphanumeric — so `h9` and
`1487` survive as their own tokens, and that crudeness is exactly why it wins.

Now type the question a Turkish agent actually types:

> iptal edersem ne öderim

Zero. Not a bad ranking — a score of exactly 0.000 against `fare_classic_shorthaul`, because
neither `iptal` nor `öderim` is a token in that document. It says `Cancellation penalty`. What
comes back first instead is `macro_tr_noshow` at 7.638 — a Turkish document answering a different
question, top-ranked because it happens to contain the word `iptal`. BM25 does not degrade
gracefully into a near miss: the term is absent, `df` is zero, the loop `continue`s, the document
scores nothing, and something confidently irrelevant takes its place.

Not only a Turkish problem. Ask in English — "what do I pay to give up the ticket" — and the fare
sheet lands at rank 6. `faq_en_general` wins with 14.736 against its 3.699, because `pay` occurs
in exactly one document in this corpus and idf rewards that hard, while `cancellation` — the word
that would have found the answer — sits in fourteen documents and is not in the query at all.
Keyword search matches strings. Your users ask questions.

<div class="presenter-note">
This is the sentence not to garble: <strong>BM25 does not know that "iptal" and "cancellation"
are the same word — it only knows that they are different strings.</strong> Say it once, slowly,
and let the zero sit on screen. Do not soften it into "it works less well". It is zero.
</div>

## Meaning instead of strings

Module 2 built a network that turned 784 pixels into a 128-number vector. An embedding model
does the same to text: `bge-m3` turns any string, Turkish or English, into a fixed-length vector
positioned so that things meaning the same land near each other. Embed the 28 documents once,
embed the question, rank by cosine similarity. That is `DenseRetriever`, fourteen lines.

Ask "iptal edersem ne öderim" now and the short-haul CLASSIC sheet comes back first. No shared
token, no dictionary, no translation step. Why `bge-m3` and not whatever your framework installs
by default is the whole of module 6; take it on trust for the next twenty minutes.

Bolt the generator on and you have the pipeline: embed the query, take the top 3, paste them
into the prompt under a "use only this context" instruction, generate. Roughly thirty lines end
to end. It works, and the score is the point.

<div class="presenter-note">
Before you run the benchmark cell, get a commitment: "hit@1 means the very first document we
retrieve is the right one. Out of 20 questions, what fraction do we get? Shout a number."
People say 0.9. Then show 0.550. That gap is the module. If Ollama has fallen over, the same
table is in <code>eval/RESULTS.md</code> — read it off there and keep moving, do not debug live.
</div>

## What it gets wrong, in front of you

**0.550.** Nearly half the time the first document handed to the model is the wrong one, and the
model then answers from a source that cannot answer the question without knowing it. Three
failures the notebook shows explicitly.

**The confidently wrong number.** Switch the index from whole documents to fixed 280-character
chunks — which is what every tutorial does, and which raises the average score to 0.650 — then
ask for the CLASSIC K cancellation penalty. The retrieved chunk contains the row
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` intact. The header
that names the columns is in the previous chunk. The model sees two euro amounts, no labels, and
answers **EUR 70**. The answer is **EUR 90**. It does not hedge, it does not ask, and nothing in
the output distinguishes this from the times it was right.

**Two versions of the truth.** Ask what meal voucher a five-hour misconnect earns.
`sop_misconnect_v4` says EUR 15 and a hotel after 6 hours. `sop_misconnect_v3` says EUR 10 and a
hotel after 8 hours. Both are in the corpus, both are about misconnects, both look identical to a
similarity score, and they frequently arrive in the same top-3. The only thing separating them is
the line `Version: 3 | Superseded`, which the retriever does not read and the model has no reason
to weigh. Retrieval has no concept of "current".

**The exact code the vectors blur.** Over 28 whole documents the exact-token questions score
**1.000** — flight codes are fine at this scale. Cut the same corpus into 288 chunks and that
category falls to **0.500**. Six schedule bulletins share the same boilerplate and differ by a
handful of tokens; sliced, the slices are near-identical prose and the one carrying `1487` stops
standing out. The queries BM25 was best at are the ones dense search is worst at, and chunking
makes it worse still.

## What you run

Notebook: `04_naive_rag.ipynb`.

```bash
ollama serve                       # in a second terminal, if not already running
ollama pull bge-m3
ollama pull qwen2.5:3b
python scripts/verify_setup.py     # must print green before you continue
jupyter lab notebooks/04_naive_rag.ipynb
```

Inside the notebook you use the shared code rather than rewriting it:

```python
from eval.retrieval import BM25, DenseRetriever, generate
from eval.metrics import load_gold, evaluate

bm25  = BM25(doc_ids, documents)
dense = DenseRetriever(doc_ids, documents)          # bge-m3 through Ollama

bm25.rank("H9 1487 IST-CDG retiming")[:3]
bm25.rank("iptal edersem ne öderim")[:3]            # look at the scores, not the order
dense.rank("iptal edersem ne öderim")[:3]

questions = load_gold("eval/gold_questions.jsonl")
evaluate({q["id"]: dense.rank(q["query"]) for q in questions}, questions)
```

The full comparison across every strategy is `python eval/run_benchmark.py`, which is what
produced every number on this page.

<div class="presenter-note">
Cell timing: the BM25 cells return instantly, embedding the 28 documents takes a few seconds
once, and the benchmark over 20 questions is the only cell worth talking over — start it, then
ask the room for their hit@1 guess while it runs. If you are behind, cut the English "give up the
ticket" probe and go straight from the Turkish zero to the dense retriever; keep all three
failure demos, they are what module 6 and module 7 open on.
</div>

## What the numbers said

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index size |
|---|---|---|---|---|
| dense, whole documents | 0.550 | 0.717 | 0.655 | 28 documents |
| dense, fixed 280-char chunks | 0.650 | 0.950 | 0.789 | 288 chunks |
| BM25 over those chunks | 0.300 | — | 0.465 | 288 chunks |
| dense, structure-aware chunks (module 7) | 0.800 | 0.850 | 0.846 | 152 chunks |

| category | whole documents | fixed 280 |
|---|---|---|
| exact-token questions (flight codes, bulletin ids) | 1.000 | 0.500 |
| BM25 on Turkish-query / English-document questions | — | 0.000 |

</div>

Corpus: 28 documents, 75 KB. Gold set: 20 questions. Embeddings `bge-m3`, generation
`qwen2.5:3b`, everything local through Ollama. Twenty questions decide between two designs and
are far too few to publish; a difference under about 0.05 here is noise.

<div class="presenter-note">
Do not fix anything in this module. Somebody will call out "just add overlap" or "use a proper
splitter" the moment the EUR 70 appears — write both on the whiteboard, say they are measured in
module 7, and that one of them is wrong. Coming back to that whiteboard is how module 7 opens.
</div>

## Going deeper

BM25's `idf` is the part worth understanding properly, because it is why keyword search has not
died. It is an estimate of surprise: `KSHEU26` in a document is strong evidence about that
document, `the` is none. A dense embedding has no equivalent knob. It compresses a passage into
one vector of a few hundred numbers, and a rare identifier is a tiny fraction of that passage's
meaning, so it gets averaged away. That is the mechanism behind the 1.000 to 0.500 collapse —
structural, not a bug in `bge-m3`.

The `b = 0.75` normalisation is the other half of why chunking wrecks BM25. Cut everything to 280
characters and every document is the average length, `norm` goes to 1 everywhere, and the term
that separated a short bulletin from a long agreement stops doing any work. BM25 scored 0.300 over
chunks against 0.550 for dense over whole documents — yet over whole documents BM25 beats dense on
exact tokens. Same algorithm, opposite verdict, and the only thing that changed was what you
indexed. Retrieval choices are not independent, which is why the day measures combinations.

Top-k is a decision you make once and forget you made. Every metric here is k-dependent: recall@5
under fixed-280 chunking is 0.950 while hit@1 is 0.650, so the right document is nearly always in
the top five and often not first. If the generator reliably ignored the four wrong ones you would
ship k=5 and stop. It does not — a wrong document in context is a lie waiting to be quoted, and
the v3/v4 failure is exactly that.

At ten million documents the brute-force loop in `DenseRetriever` dies. You build an approximate
index — HNSW is the usual answer — and trade a recall loss you choose with a parameter for
logarithmic instead of linear search. You pre-filter on metadata before computing any similarity,
which is what actually solves the v3/v4 problem: `status = current` is a one-line filter that no
amount of better embedding achieves. The quarter, the route band and the document version are
structured fields you already have and are throwing away by treating the corpus as flat text.

The naive pipeline is not a strawman. Thirty lines, no framework, no vector database, and it
answers more than half the gold set on a laptop with the network closed. Everything after this is
measured against 0.550, and two of the popular improvements will fail to beat it.

## Exit line

> RAG works. Retrieval is bringing back garbage.
