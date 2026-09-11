---
title: "5. Keyword Search to Naive RAG"
description: "How do I find the right piece? Keyword search, embeddings, and the four ways retrieval brings back the wrong document."
---

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
3 min opening question · 7 min BM25 mechanism · 6 min the Turkish rank 14 and the English column ·
5 min the dense retriever's partial rescue · 6 min the pipeline and the benchmark table ·
12 min the four failures · 4 min the by-type reading and the exit.
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
of documents containing the term — shrinks, so `penalty` earns almost nothing across fifteen of
the twenty-eight documents while `KSHEU26`, in one document only, earns nearly everything.
**Long documents are penalised:** `norm` divides by document length against the corpus average, so
the 9 KB interline agreement does not win every query by sheer word count; `b = 0.75` sets how
hard that bites.

Before any of that runs, `tokenize()` decides what a word even is: it replaces every
non-alphanumeric character with a space and lowercases the rest. So `XX 1487` becomes two
separate tokens, `xx` and `1487`, and `SCB-2026-0914` becomes four. That is deliberately crude —
a flight code matches on the literal string and on nothing else — and the two halves do very
different amounts of work. `xx` is Kraken Air's own code and appears in 19 of the 28 documents, so
its `idf` is near the floor; `1487` is in 4, so it is the token actually carrying the query.

**VS Code — `notebooks/04_naive_rag.py`, the first block of Part 1:**

```python
bm25 = R.BM25(doc_ids, texts)
for query in ["XX 1487", "SCB-2026-0914"]:
    print(f"{query!r:>18} -> {bm25.rank(query)[:3]}")
```

Both come back right. `'XX 1487'` returns `bulletin_scb_2026_0914` first, and so does
`'SCB-2026-0914'`. Exact identifiers are what keyword search is for.

Now the query an actual call-centre agent types. This is the notebook's own string, at
`04_naive_rag.py:45`, ASCII rather than proper Turkish because that is what people type into a
work terminal:

> Musteri bileti iptal ederse ne oder?

The answer is in `fare_classic_shorthaul`, an English fare sheet. BM25 returns it at **rank 14 of
28**. What comes back first instead is `macro_tr_noshow`, then `macro_tr_rebook`, then
`macro_tr_baggage` — three Turkish documents answering three different questions, top-ranked
because they contain the word `iptal`.

The reason is not subtle and it is not a near miss. The query's terms are either absent from the
whole corpus, so the loop skips them, or absent from *this* document, so `f` is 0 and the term
contributes nothing. Either way there is no partial credit for meaning it. The fare sheet says
`Cancellation penalty`; the query says `iptal`. BM25 does not know those are the same word — it
only knows they are different strings.

Not only a Turkish problem. On the four English-question/English-document questions in the gold
set, BM25 scores hit@1 **0.250** over these same whole documents: three of the four put something
other than the answer first. Keyword search matches strings. Your users ask questions.

<div class="presenter-note">
This is the sentence not to garble: <strong>BM25 does not know that "iptal" and "cancellation"
are the same word — it only knows that they are different strings.</strong> Say it once, slowly,
and leave rank 14 on screen. Do not soften it into "it works less well".
<br/><br/>
If somebody asks to see the actual score, <code>BM25.rank()</code> returns document ids only —
the notebook never prints a number here. <code>bm25.scores(paraphrase)</code> is the call that
would, and it is not in the file; offer it as homework rather than typing it live.
</div>

## Meaning instead of strings

Module 2 built a network that turned 784 pixels into a 128-number vector. An embedding model
does the same to text: `bge-m3` turns any string, Turkish or English, into a vector of 1,024
numbers positioned so that things meaning the same land near each other. Embed the 28 documents
once, embed the question, rank by cosine similarity. That is `DenseRetriever`, twelve lines.

Ask the same Turkish question again and the fare sheet moves from rank 14 to **rank 5**. That is
the honest result, and it is not the clean rescue the story wants. The top three are
`macro_tr_noshow`, `macro_en_refund`, `macro_tr_rebook` — the same wrong Turkish macro BM25 put
first is still first. What changed is that the right document is now inside the top five instead
of buried in the second half.

Sit with the size of that win, because the pipeline you are about to build sends **k=3**. Rank 5
is inside the window `recall@5` scores and outside the window the model actually reads. The
embedding closed most of the gap and did not close it. This is the same `tr_en` weakness the
by-type table reports as 0.333, and what moves it is not a better question — it is module 6's
choice of embedder and module 7's decision about what to index.

Bolt the generator on anyway and you have the pipeline: embed the query, take the top 3, paste
them into the prompt under a "use only this context" instruction, generate. Twelve lines of
retriever and ten of pipeline. It works, and the score is the point.

<div class="presenter-note">
Before you run the benchmark block, get a commitment: "hit@1 means the very first document we
retrieve is the right one. Out of 20 questions, what fraction do we get? Shout a number."
People say 0.9. Then show the number <em>your</em> run printed — the recording is 0.600, and
embedding output is not bit-stable across Ollama builds, so a 0.05 swing here is the same thing
this page already calls noise. That gap is the module. If Ollama has fallen over, the same table
is in <code>eval/RESULTS.md</code> — read it off there and keep moving, do not debug live.
</div>

## What it gets wrong, in front of you

**0.600.** Two questions in five, the first document handed to the model is the wrong one, and the
model then answers from a source that cannot answer the question without knowing it. Four failures
the notebook shows explicitly, in the order its cells print them — you open it in *What you run*,
below.

**Failure 1 — the right document, the wrong answer.** Ask for the CLASSIC K cancellation penalty.
Dense retrieval puts `fare_classic_shorthaul` first, which is correct. The answer on the recorded
run is `For CLASSIC fare family, the cancellation penalty for booking class Q is EUR 180.` — a
real cell of a real table, for a booking class nobody asked about. The pipeline pastes
`docs[h][:1500]` into the prompt: the first 1,500 characters of a 3,923-character document. The
column header sits at character 1,131 and is inside the cut; the row
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` starts at character
1,743 and is not. The model gets the header, the O and T rows, and the first three characters of
the Q row — the Q row starts at 1,497 and the cut lands at 1,500 — and answers from what was
there. Retrieval scored a hit. No hedge, no question, nothing in the output marking it as a guess.

**Failure 2 — two versions of the truth.** Ask what meal voucher a misconnect earns.
`sop_misconnect_v4` says EUR 15, `sop_misconnect_v3` says EUR 10, and both come back together.
The only thing separating them is a `Version: 3 | Superseded` line the retriever does not read.
Retrieval has no concept of "current".

**Failure 3 — identifiers blur together.** Four short queries, dense against BM25, over the same
28 whole documents. `XX 1487` is rank 1 for both. `SCB-2026-0914` is rank 1 for BM25 and rank 6
for dense, which puts a different bulletin — `bulletin_scb_2026_0921` — first. `YY 88` is rank 5
for BM25 and rank 13 for dense. `booking class K` is rank 1 for BM25 and rank 3 for dense, which
returns `policy_corporate_travel` first, because that document also talks about booking classes
at length. One vector of 1,024 numbers has to carry a whole passage's meaning, and a bulletin id
is a tiny fraction of that meaning. This is where the keyword search we discarded twenty minutes
ago earns its place back.

**Failure 4 — the Turkish question that lands somewhere else entirely.** Take `q05`: a
Turkish-speaking agent asks what a passenger waiting four hours is owed. The answer is in
`sop_misconnect_v4`, an English procedure, and it comes back at **rank 18 of 28**. What arrives
instead is `policy_expense_reimbursement`, the short-haul fare sheet and the Turkish baggage
macro — all of them plausibly "money after a travel disruption", none of them the answer. Keyword
search is hopeless here by construction, but the embedding was supposed to bridge the gap and
only partly does: six of the twenty questions are shaped like this one and dense retrieval scores
**0.333** on them, so four of the six fail. Whether that is the embedder's fault or ours is
module 6.

Then read the by-type rows rather than the average. Over 28 whole documents the `exact_token`
questions score **1.000** — flight codes, bulletin ids and the class-K fare lookup are all found
at rank 1 at this scale. Cut the same corpus into 294 fixed-280 chunks and the average rises from
0.600 to 0.700 while that category drops to **0.750**: one of the four falls out of first place.
Six schedule bulletins share the same boilerplate and differ by a handful of tokens; sliced, the
slices are near-identical prose and the one carrying `1487` stops standing out. BM25 is no refuge
— over the 154 structure-aware chunks module 7 ends on, the retriever you would reach for when
the query is an identifier scores **0.500** on that category against 0.750 over whole documents.
The average went up and a category went down. Both gaps are one question wide; the direction is
the lesson, not the digit. Module 7 is built on exactly that.

## What you run

**Terminal (repo root)** — the folder that contains `corpus/`, `notebooks/`, `eval/` and
`exercises/`:

```bash
ollama list                        # bge-m3 and qwen2.5:3b must both be listed already
python scripts/verify_setup.py     # must print READY before you continue
```

Nothing is downloaded during the day. If a model is missing from `ollama list`, say so now rather
than starting a pull.

**VS Code, repository root as the open folder:** open `notebooks/04_naive_rag.py`, put the cursor
in a `# %%` block and press Shift+Enter. The output appears in the Interactive window. There is no
Jupyter server and no browser notebook anywhere in this course.

**What you should see.** `Musteri bileti iptal ederse ne oder?` putting the fare sheet at rank 14
under BM25 and rank 5 under the dense retriever; then the comparison table, BM25 at hit@1
**0.400** against bge-m3's **0.600** over the 28 whole documents.

**Roughly how long.** The BM25 blocks return instantly, embedding the 28 documents takes about
7 s once, and the two generation blocks are the only ones worth talking over. The same
measurement as a single command — `python eval/run_benchmark.py --skip-rerank`, from the terminal
at the repo root — is 31.5 s and 42 embed calls on a warm machine.

The notebook uses the shared code rather than rewriting it. It does not import `eval` as a
package; `_preflight.ready()` moves the working directory to `notebooks/` and puts `eval/` on
`sys.path`, so the imports are flat and every relative path is written from `notebooks/`:

**VS Code — `notebooks/04_naive_rag.py`, the imports and the blocks that use them:**

```python
import _preflight; _preflight.ready(chat=True, embed=True)
import retrieval as R, metrics

bm25  = R.BM25(doc_ids, texts)
dense = R.DenseRetriever(doc_ids, texts)          # bge-m3, 1024 numbers per document

paraphrase = "Musteri bileti iptal ederse ne oder?"
bm25.rank(paraphrase)[:3]
dense.rank(paraphrase)[:3]

questions = metrics.load_gold("../eval/gold_questions.jsonl")
metrics.evaluate({q["id"]: dense.rank(q["query"]) for q in questions}, questions)
```

Every whole-document number on this page reproduces with
`python eval/run_benchmark.py --skip-rerank`; `eval/README.md` lists the other flags and what
each one prints.

<div class="presenter-note">
Block timing: the BM25 blocks return instantly, embedding the 28 documents takes about 7 s once,
and the benchmark over 20 questions is the only block worth talking over — start it, then ask the
room for their hit@1 guess while it runs. If you are behind, go straight from BM25's rank 14 to
the dense retriever's rank 5 and skip the `en_en` aside; keep all four failure demos, they are
what modules 6 and 7 open on.
</div>

## What the numbers said

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index size |
|---|---|---|---|---|
| BM25, whole documents | 0.400 | 0.583 | 0.515 | 28 documents |
| dense (`bge-m3`), whole documents | 0.600 | 0.717 | 0.677 | 28 documents |
| dense, fixed 280-char chunks (module 7) | 0.700 | 0.917 | 0.817 | 294 chunks |
| dense, structure-aware chunks (module 7) | 0.750 | 0.833 | 0.817 | 154 chunks |
| BM25 over those structure-aware chunks | 0.300 | 0.633 | 0.467 | 154 chunks |

hit@1 by question type:

| question type | BM25, whole docs | dense, whole docs | dense, fixed 280 |
|---|---|---|---|
| `tr_tr` (4) | 1.000 | 1.000 | 1.000 |
| `tr_en` (6) | 0.000 | 0.333 | 0.500 |
| `en_en` (4) | 0.250 | 0.500 | 0.750 |
| `exact_token` (4) | 0.750 | 1.000 | 0.750 |
| `multi_hop` (2) | 0.000 | 0.000 | 0.500 |

</div>

The two whole-document rows are what `notebooks/04_naive_rag.py` prints; nothing on this page
chunks anything, so the chunked rows are module 7's and module 9's runs. There is no measured
BM25 run over the 294 fixed-280 chunks — the only BM25-over-chunks measurement in the repository
is `run_benchmark.py --fusion`, over the 154 structure-aware chunks, so that is the row above.

`run_benchmark.py --skip-rerank` scores the same two whole-document retrievers a second time and
prints recall@5 0.583 / MRR 0.515 for BM25 and 0.717 / 0.677 for `bge-m3` — identical to the
notebook, to three decimals, on this pass. `bge-m3` through Ollama is not bit-stable, so do not
expect that exact agreement on every run; a 0.05 swing either way is the same noise this page
already discounts.

The four `exact_token` questions are not all flight codes. Three are schedule lookups by flight
number or bulletin id; the fourth, `q15`, is a fare-table lookup by booking class — the CLASSIC K
cancellation penalty from failure 1. Read the label as "questions whose answer is one exact string
in one table", not "flight codes".

Corpus: 28 documents, 78,310 characters. Gold set: 20 questions. Embeddings `bge-m3`, generation
`qwen2.5:3b`, everything local through Ollama. Twenty questions decide between two designs and
are far too few to publish; one question is worth 0.05, so a difference under about 0.05 here is
noise.

<div class="presenter-note">
Do not fix anything in this module. Somebody will call out "just add overlap" or "use a proper
splitter" the moment failure 1 appears — write both on the whiteboard, say they are measured in
module 7, and that one of them is wrong. Coming back to that whiteboard is how module 7 opens.
</div>

## Going deeper

BM25's `idf` is the part worth understanding properly, because it is why keyword search has not
died. It is an estimate of surprise: `KSHEU26` in a document is strong evidence about that
document, `the` is none. A dense embedding has no equivalent knob. It compresses a passage into
one vector of 1,024 numbers, and a rare identifier is a tiny fraction of that passage's meaning,
so it gets averaged away. That is the mechanism behind failure 3, and behind the 1.000 to 0.750
slip on exact tokens once you chunk — structural, not a bug in `bge-m3`.

The `b = 0.75` normalisation is the other half of why chunking wrecks BM25. Cut everything down to
chunk size and every document is the average length, `norm` goes to 1 everywhere, and the term
that separated a short bulletin from a long agreement stops doing any work. BM25 over the 28 whole
documents scores hit@1 0.400; over the 154 structure-aware chunks, 0.300 — and on the exact-token
questions, the ones it should own, 0.750 falls to 0.500. What broke it was not the retriever. It
was the decision about what to index.

Read BM25's own profile by category and the asymmetry is sharper than any average: 1.000 on
`tr_tr`, 0.000 on `tr_en`, with no embedding model, no GPU and no network anywhere in it. What it
does not do is beat dense retrieval in any column of the table above — not even on exact tokens,
where dense is at the ceiling. Keyword search earns its five minutes on cost, on single-identifier
queries like `SCB-2026-0914`, and on never needing a model. Not by winning.

Top-k is a decision you make once and forget you made. Every metric here is k-dependent: over
whole documents recall@5 is 0.717 while hit@1 is 0.600, so the right document is often in the top
five and not first — the Turkish question at rank 5 is exactly that case. If the generator
reliably ignored the four wrong ones you would ship k=5 and stop. It does not — a wrong document
in context is a lie waiting to be quoted, and the v3/v4 failure is exactly that.

At ten million documents the brute-force loop in `DenseRetriever` dies. You build an approximate
index — HNSW is the usual answer — and trade a recall loss you choose with a parameter for
logarithmic instead of linear search. You pre-filter on metadata before computing any similarity,
which is what actually solves the v3/v4 problem: `status = current` is a one-line filter that no
amount of better embedding achieves. The quarter, the route band and the document version are
structured fields you already have and are throwing away by treating the corpus as flat text.

The naive pipeline is not a strawman. Twenty-two lines, no framework, no vector database, and it
puts the right document first for twelve of the twenty gold questions on a laptop with the network
closed. Everything after this is measured against 0.600, and two of the popular improvements will
fail to beat it.

## Exit line

> RAG works, and retrieval is bringing back garbage — four of the six Turkish-question /
> English-document questions still come back wrong, and the one we watched only climbed to rank 5.
> Before we fix that: I chose this embedding model for you, and I never said why.
