---
title: "5. Keyword Search to Naive RAG"
description: "How do I find the right piece?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **How do I find the right piece?**

The last module ended with a working answer and a bill. All twenty-eight documents fit in the
context window, the model quotes EUR 90 correctly, and you pay for the whole rule book on every
question — including the ones that need one paragraph out of 78,310 characters. The rule book is
reissued quarterly. At four quarters and six stations the prompt stops fitting long before the
cost stops hurting.

So select. Send three documents instead of twenty-eight. The rest of the day is about that one
word — select — and how many ways it goes wrong.

<div class="presenter-note">
This module runs 10:43–11:26, 43 minutes, and it is on the never-cut list. Rough budget:
3 min opening question · 7 min BM25 mechanism · 6 min the Turkish zero and the English rank 6 ·
5 min the dense retriever · 6 min the pipeline and the benchmark table · 12 min the four failures ·
4 min the by-type reading and the exit.
<br/><br/>
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
9.500 against 5.705 for the next document. `1487` sits in four of the twenty-eight documents, but
only one of them is a one-page bulletin, so the length normalisation finishes what the idf
started. The tokenizer is crude on purpose — it splits on anything non-alphanumeric — so `h9` and
`1487` survive as their own tokens, and that crudeness is exactly why it wins.

Now type the question a Turkish agent actually types:

> iptal edersem ne öderim

Zero. Not a bad ranking — a score of exactly 0.000 against `fare_classic_shorthaul`, because
neither `iptal` nor `öderim` is a token in that document. It says `Cancellation penalty`. What
comes back first instead is `macro_tr_noshow` at 7.662 — a Turkish document answering a different
question, top-ranked because it happens to contain the word `iptal`. BM25 does not degrade
gracefully into a near miss: the term is absent, `df` is zero, the loop `continue`s, the document
scores nothing, and something confidently irrelevant takes its place.

Not only a Turkish problem. Ask in English — "what do I pay to give up the ticket" — and the fare
sheet lands at rank 6. `faq_en_general` wins with 14.811 against its 3.768, because `pay` occurs
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
embed the question, rank by cosine similarity. That is `DenseRetriever`, twelve lines.

Ask "iptal edersem ne öderim" now and the short-haul CLASSIC sheet comes back first. No shared
token, no dictionary, no translation step. Why `bge-m3` and not whatever your framework installs
by default is the whole of module 6; take it on trust for the rest of this module.

Bolt the generator on and you have the pipeline: embed the query, take the top 3, paste them
into the prompt under a "use only this context" instruction, generate. Twelve lines of retriever
and ten of pipeline. It works, and the score is the point.

<div class="presenter-note">
Before you run the benchmark block, get a commitment: "hit@1 means the very first document we
retrieve is the right one. Out of 20 questions, what fraction do we get? Shout a number."
People say 0.9. Then show 0.550. That gap is the module. If Ollama has fallen over, the same
table is in <code>eval/RESULTS.md</code> — read it off there and keep moving, do not debug live.
</div>

## What it gets wrong, in front of you

**0.550.** Nearly half the time the first document handed to the model is the wrong one, and the
model then answers from a source that cannot answer the question without knowing it. Four
failures the notebook shows explicitly.

**Failure 1 — the right document, the wrong answer.** Ask for the CLASSIC K cancellation penalty.
Dense retrieval puts `fare_classic_shorthaul` first, which is correct. The answer is still wrong.
The pipeline pastes `docs[h][:1500]` into the prompt: the first 1,500 characters of a
3,923-character document. The column header sits at character 1,131 and is inside the cut; the row
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` sits at character
1,743 and is not. The model gets a header, the O and T rows, half of Q, and answers from those.
Retrieval scored a hit. The room got a number for a booking class nobody asked about, with no
hedge, no question and nothing in the output marking it as a guess.

**Failure 2 — two versions of the truth.** Ask what meal voucher a five-hour misconnect earns.
`sop_misconnect_v4` says EUR 15 and a hotel after 6 hours. `sop_misconnect_v3` says EUR 10 and a
hotel after 8 hours. Both are in the corpus, both are about misconnects, both look identical to a
similarity score, and they frequently arrive in the same top-3. The only thing separating them is
the line `Version: 3 | Superseded`, which the retriever does not read and the model has no reason
to weigh. Retrieval has no concept of "current".

**Failure 3 — identifiers blur together.** Four short queries, dense against BM25, over the same
28 whole documents. `H9 1487` is rank 1 for both. `SCB-2026-0914` is rank 1 for BM25 and rank 6
for dense, which puts a different bulletin — `bulletin_scb_2026_0921` — first. `booking class K`
is rank 1 for BM25 and rank 5 for dense, which returns the staff travel policy, because that
document also talks about booking classes at length. One vector of a few hundred numbers has to
carry a whole passage's meaning, and a bulletin id is a tiny fraction of that meaning. This is
where the keyword search we discarded twenty minutes ago earns its place back.

**Failure 4 — the Turkish question that lands somewhere else entirely.** Take `q05`: a Turkish
agent asks what a passenger waiting four hours is owed. The answer is in `sop_misconnect_v4`, an
English procedure, and it comes back at **rank 18 of 28**. What arrives instead is the staff
expense policy, the short-haul fare sheet and the Turkish baggage macro — all of them plausibly
"money after a travel disruption", none of them the answer. Keyword search is hopeless here by
construction, but the embedding was supposed to bridge the gap and only partly does: six of the
twenty questions are shaped like this one and dense retrieval scores **0.333** on them, so four
of the six fail. Whether that is the embedder's fault or ours is module 6.

Then read the by-type rows rather than the average. Over 28 whole documents the exact-token
questions score **1.000** — flight codes, bulletin ids and the class-K fare lookup are all found
at rank 1 at this scale. Cut the same corpus into 294 fixed-280 chunks and the average rises to
0.700 while that category drops to **0.750**: one of the four falls out of first place. Six
schedule bulletins share the same boilerplate and differ by a handful of tokens; sliced, the
slices are near-identical prose and the one carrying `1487` stops standing out. BM25 is no refuge
— the retriever you would reach for when the query is an identifier scores 0.750 on that category
over whole documents and **0.250** once you chunk. The average went up and a category went down.
Module 7 is built on exactly that.

## What you run

```bash
ollama list                        # bge-m3 and qwen2.5:3b must both be listed already
python scripts/verify_setup.py     # must print READY before you continue
```

Nothing is downloaded during the day. If a model is missing from `ollama list`, say so now rather
than starting a pull.

Then open `notebooks/04_naive_rag.py` in VS Code and run the blocks with Shift+Enter.

**What you should see.** BM25 scoring exactly `0.000` on `iptal edersem ne öderim`; the same
question answered from `fare_classic_shorthaul` by the dense retriever; then the comparison table,
BM25 at hit@1 **0.400** against bge-m3's **0.550** over the 28 whole documents.

**Roughly how long.** The BM25 blocks return instantly, embedding the 28 documents takes a few
seconds once, and the two generation blocks are the only ones worth talking over. The same
measurement as a single command — `python eval/run_benchmark.py --skip-rerank` — is about 28 s
and 42 embed calls on a warm machine.

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

`python eval/run_benchmark.py --skip-rerank` prints the whole-document rows on this page and does
no chunking at all. The chunk rows come from `python eval/run_benchmark.py --chunking`, which
walks the whole ladder and is module 7's measurement rather than this one. `eval/README.md` lists
every flag and what each one reproduces.

<div class="presenter-note">
Block timing: the BM25 blocks return instantly, embedding the 28 documents takes a few seconds
once, and the benchmark over 20 questions is the only block worth talking over — start it, then
ask the room for their hit@1 guess while it runs. If you are behind, cut the English "give up the
ticket" probe and go straight from the Turkish zero to the dense retriever; keep all four
failure demos, they are what modules 6 and 7 open on.
</div>

## What the numbers said

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index size |
|---|---|---|---|---|
| BM25, whole documents | 0.400 | 0.583 | 0.515 | 28 documents |
| dense, whole documents | 0.550 | 0.717 | 0.654 | 28 documents |
| dense, fixed 280-char chunks | 0.700 | 0.950 | 0.814 | 294 chunks |
| BM25 over those same chunks | 0.300 | 0.617 | 0.483 | 294 chunks |
| dense, structure-aware chunks (module 7) | 0.800 | 0.833 | 0.844 | 154 chunks |

hit@1 by question type:

| question type | BM25, whole docs | dense, whole docs | dense, fixed 280 |
|---|---|---|---|
| `tr_tr` (4) | 1.000 | 1.000 | 0.750 |
| `tr_en` (6) | 0.000 | 0.333 | 0.667 |
| `en_en` (4) | 0.250 | 0.250 | 0.750 |
| `exact_token` (4) | 0.750 | 1.000 | 0.750 |
| `multi_hop` (2) | 0.000 | 0.000 | 0.500 |

</div>

The four `exact_token` questions are not all flight codes. Three are schedule lookups by flight
number or bulletin id; the fourth, `q15`, is a fare-table lookup by booking class — the CLASSIC K
cancellation penalty from failure 1. Read the label as "questions whose answer is one exact string
in one table", not "flight codes".

Corpus: 28 documents, 78,310 characters. Gold set: 20 questions. Embeddings `bge-m3`, generation
`qwen2.5:3b`, everything local through Ollama. Twenty questions decide between two designs and
are far too few to publish; a difference under about 0.05 here is noise.

<div class="presenter-note">
Do not fix anything in this module. Somebody will call out "just add overlap" or "use a proper
splitter" the moment failure 1 appears — write both on the whiteboard, say they are measured in
module 7, and that one of them is wrong. Coming back to that whiteboard is how module 7 opens.
</div>

## Going deeper

BM25's `idf` is the part worth understanding properly, because it is why keyword search has not
died. It is an estimate of surprise: `KSHEU26` in a document is strong evidence about that
document, `the` is none. A dense embedding has no equivalent knob. It compresses a passage into
one vector of a few hundred numbers, and a rare identifier is a tiny fraction of that passage's
meaning, so it gets averaged away. That is the mechanism behind failure 3, and behind the 1.000 to
0.750 slip on exact tokens once you chunk — structural, not a bug in `bge-m3`.

The `b = 0.75` normalisation is the other half of why chunking wrecks BM25. Cut everything to 280
characters and every document is the average length, `norm` goes to 1 everywhere, and the term
that separated a short bulletin from a long agreement stops doing any work. BM25 over the 28 whole
documents scores 0.400; over the 294 chunks cut from those same documents, 0.300 — and on the
exact-token questions, the ones it should own, 0.750 falls to 0.250. What broke it was not the
retriever. It was the decision about what to index.

Read BM25's own profile by category and the asymmetry is sharper than any average. On the four
Turkish-query/Turkish-document questions it scores **1.000** — every one at rank 1, no embedding
model involved, no GPU, no network. On the six Turkish-query/English-document questions it scores
**0.000**, and not narrowly: the right document comes back at ranks 8 to 21 out of 28, so it is
not in the top 5 even once. That is the `iptal` / `cancellation` gap you watched on one query,
measured across six. What BM25 does not do here is beat dense retrieval in any column: over whole
documents dense is at **1.000** on exact tokens against BM25's 0.750, and ties it at 1.000 on
`tr_tr`. Keyword search earns its five minutes on cost, on single-identifier queries like
`SCB-2026-0914`, and on the fact that it never needs a model — not by winning the table.

Top-k is a decision you make once and forget you made. Every metric here is k-dependent: recall@5
under fixed-280 chunking is 0.950 while hit@1 is 0.700, so the right document is nearly always in
the top five and often not first. If the generator reliably ignored the four wrong ones you would
ship k=5 and stop. It does not — a wrong document in context is a lie waiting to be quoted, and
the v3/v4 failure is exactly that.

At ten million documents the brute-force loop in `DenseRetriever` dies. You build an approximate
index — HNSW is the usual answer — and trade a recall loss you choose with a parameter for
logarithmic instead of linear search. You pre-filter on metadata before computing any similarity,
which is what actually solves the v3/v4 problem: `status = current` is a one-line filter that no
amount of better embedding achieves. The quarter, the route band and the document version are
structured fields you already have and are throwing away by treating the corpus as flat text.

The naive pipeline is not a strawman. Twenty-two lines, no framework, no vector database, and it
puts the right document first for eleven of the twenty gold questions on a laptop with the network
closed. Everything after this is measured against 0.550, and two of the popular improvements will
fail to beat it.

## Exit line

> RAG works, and retrieval is bringing back garbage. Before we fix that — I chose this
> embedding model for you, and I never said why.
