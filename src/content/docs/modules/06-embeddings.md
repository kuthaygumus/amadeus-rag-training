---
title: "6. The Embedding Bake-Off"
description: "Is the default embedder right for *your* language?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **Is the default embedder right for *your* language?**

In module 5 dense retrieval beat keyword search on the Turkish question. I told you the model was
`bge-m3` and told you to take it on trust. Nobody in this room chose it. In your own project
nobody would choose it either — you would `pip install chromadb`, call `add()`, and a model would
be picked for you.

So change that one line and run the same questions again.

```python
dense = DenseRetriever(doc_ids, documents, model="nomic-embed-text")
```

On the six questions where a Turkish agent asks about an English document, hit@1 goes from
**0.667 to 0.000**. Not one of the six. No exception, no warning, no log line, no score that looks
suspicious. The pipeline returns three documents with confident-looking similarity numbers and the
right one is not among them. The generator then answers from what it was given, because that is
its job.

<div class="presenter-note">
Before running the swapped cell, get a commitment: "we change the embedding model and nothing
else. Out of six Turkish questions, how many still land? Hands up for four or more." Most hands go
up. Show 0.000 and say nothing for three seconds. That silence does more than the next paragraph.
4 minutes.
</div>

## What an embedding actually is

An embedding model takes a string and returns a fixed-length list of floating point numbers. That
is the entire contract. `bge-m3` returns **1024** numbers for any input, one word or one page.
`nomic-embed-text` returns **768**. `all-MiniLM-L6-v2` returns **384**. The length never varies
with the input: the model compresses whatever you gave it into that many numbers and drops
everything it does not treat as meaning.

Each number is a coordinate, so 1024 numbers is a point in 1024-dimensional space. You cannot
picture it and you do not have to — the only thing you ever do with those points is ask how close
two of them are. Closeness is cosine similarity, four lines in `eval/retrieval.py`:

```python
def cosine(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / (norm + 1e-12)
```

Multiply position by position, add it up, divide by both lengths. The division is the important
half: it throws away magnitude and keeps only direction. A long document and a one-line query
cannot be compared by distance — the long one is simply further from the origin — but they can be
compared by which way they point. The result runs from -1 to 1, on real text roughly 0.3 to 0.9,
and only the ordering ever matters. 0.656 is not "66% relevant". It is only "higher than 0.466".

The whole day rests on the claim that direction in that space corresponds to meaning. It does,
approximately, inside the model's training distribution. Outside it the numbers keep coming and
quietly stop meaning what you assumed.

## The trap, live

Same query both times, the one a Helios agent would type:

> CLASSIC K sınıfı iptal cezası ne kadar?

Three candidate documents, one embedder at a time. `nomic-embed-text` scores the Turkish no-show
macro **0.690** and puts it first. The correct English fare sheet — the one holding the
`| K | KSHEU26 | ... | EUR 90 | ... |` row — scores **0.466** and lands third. Last of three.
`bge-m3` scores the correct English document **0.656** and puts it first.

Look at what nomic chose. `macro_tr_noshow` is not a random document. It is Turkish, it is about
cancellation penalties, it contains "iptal cezası iki katına çıkar". A human would call it a
plausible neighbour. But it explicitly refuses to state an amount — "tutarlar bu makroda
tekrarlanmaz", use the English fare rule — so the one number the agent needs is not in it. The
retriever picked the right topic in the right language and discarded the only document with the
answer.

## Be precise about what failed

The easy summary is "MiniLM does not know Turkish". That summary is wrong, and it will get you
beaten in the Q&A.

These models handle Turkish-to-Turkish perfectly well. Ask a Turkish question against the Turkish
macros and nomic ranks sensibly — that is exactly how it landed on `macro_tr_noshow` at 0.690.
What it cannot do is match **across** languages. In a monolingual model's space the language of
the text is itself a strong direction: two Turkish sentences about unrelated topics can sit closer
together than a Turkish sentence and its own English translation, because "being Turkish"
outweighs "being about cancellation penalties". Language becomes a bigger axis than meaning.

It is invisible in every metric you would naturally check, because TR→TR still works and EN→EN
still works. Half the corpus, half the questions, silently broken.

ChromaDB's default embedding function is `all-MiniLM-L6-v2`, downloaded on your first `add()` call
without being asked for, and English-only. On the ten-document probe corpus it scored **2/5**
against **4/5** for `bge-m3`. None of that reaches the log.

<div class="presenter-note">
This is the sentence not to garble: <strong>these models are fine at Turkish-to-Turkish — what
they cannot do is match across languages, and your corpus is mixed.</strong> Say it once, slowly,
and write "TR → TR: ok / TR → EN: 0.000" on the whiteboard. If Ollama drops out here, the three
similarity numbers are on this page and in <code>eval/RESULTS.md</code>; read them off and do not
debug in front of the room.
</div>

## What you run

Notebook: `04b_embeddings_bakeoff.ipynb`.

```bash
ollama pull bge-m3
ollama pull nomic-embed-text
python scripts/verify_setup.py     # must print green before you continue
jupyter lab notebooks/04b_embeddings_bakeoff.ipynb
```

The notebook uses the shared code rather than rewriting it:

```python
from eval.retrieval import embed, cosine, DenseRetriever
from eval.metrics import load_gold, evaluate

q = "CLASSIC K sınıfı iptal cezası ne kadar?"
for model in ("nomic-embed-text", "bge-m3"):
    qv, dv = embed([q], model=model)[0], embed(candidates, model=model)
    print(model, [round(cosine(qv, v), 3) for v in dv])   # look at the ORDER

gold = load_gold("eval/gold_questions.jsonl")
tr_en = [g for g in gold if g["type"] == "tr_en"]
for model in ("nomic-embed-text", "bge-m3"):
    r = DenseRetriever(doc_ids, documents, model=model)
    print(model, evaluate({g["id"]: r.rank(g["query"]) for g in tr_en}, tr_en))
```

Print the vector length once — `len(embed(["test"])[0])` — so 1024 stops being a slide and becomes
something you watched arrive over HTTP.

## What the numbers said

<div class="measured">

| embedder | vector length | hit@1 on the 6 `tr_en` questions |
|---|---|---|
| `nomic-embed-text` | 768 | 0.000 |
| `bge-m3` | 1024 | 0.667 |

| query: "CLASSIC K sınıfı iptal cezası ne kadar?" | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| correct English CLASSIC short-haul fare sheet | 0.466 — rank 3 | **0.656 — rank 1** |
| Turkish no-show macro | **0.690 — rank 1** | ranked below |

| whole gold set, 20 questions | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| fixed 280-char chunks, hit@1 | 0.350 | 0.650 |
| structure-aware chunks, hit@1 | 0.350 | 0.800 |

| probe corpus, 10 documents, 5 questions | hit@1 |
|---|---|
| `all-MiniLM-L6-v2` — ChromaDB's silent default | 2/5 |
| BM25 | 3/5 |
| `bge-m3` | 4/5 |

</div>

Corpus: 28 documents, 75 KB. Gold set: 20 questions, 6 of them Turkish-query / English-document,
everything local through Ollama. Six questions are enough to decide a model choice and far too few
to publish: the direction is real, the exact figure is noisy.

Read the third table again. With nomic, better chunking buys **nothing** — 0.350 either way. The
chunking work in module 7 only pays once the embedder can see across languages at all. Fix the
embedder first, then chunk.

## Going deeper

What multilingual training changes is the objective, not the architecture. A monolingual model is
trained to pull together pairs of texts that mean the same thing in one language. `bge-m3` is
trained on parallel and mined pairs across roughly a hundred languages, where the positive pair is
a sentence and its translation. The gradient's whole job is to land the Turkish sentence and the
English sentence on the same point, so language stops being a usable direction — the loss punishes
it. The fix is not more data or a better prompt. It was baked in months before you pulled the
weights.

Dimensionality is not quality. 1024 beats 768 here, which flatters the intuition, but
384-dimensional MiniLM is a strong English retriever that beats far larger models on English
benchmarks; it fails on our corpus for a reason unrelated to width. Width buys capacity to
represent distinctions. It does not decide which distinctions the model was taught to care about.
Treat the dimension as a cost — 1024 float32 is 4 KB per chunk, so our 152 chunks are 600 KB and
ten million would be 40 GB before index overhead — and treat the training data as the quality
signal.

The cost is real, which is why this is a trade and not a free win. `bge-m3` is an
XLM-RoBERTa-large body, several times MiniLM's parameters, and it embeds correspondingly slower.
At 28 documents that is invisible. At ten million it is a re-indexing budget you plan for, and the
point where Matryoshka embeddings — trained so the first 256 numbers of the vector work on their
own — let you shortlist cheaply and rescore the top few hundred at full width.

Evaluating an embedder for your own corpus takes an afternoon and it is the highest-value
afternoon in this subject. Write twenty questions in the language your users actually type, each
labelled with the document that answers it. Twenty is enough — this page decided a model on six.
Group them by the failure you suspect: same-language, cross-language, exact identifiers,
paraphrase. Then loop three or four models through `DenseRetriever` and print hit@1 **per group**,
not the average. The grouping does the work: nomic's overall number is bad but not alarming, and
only the `tr_en` row shows a clean zero.

And 0.667 is not a victory. Two of six are still wrong. The winner of a bake-off is the least bad
option you measured, not a solved problem, and what is still broken at 0.667 is module 7.

<div class="presenter-note">
Somebody will ask "why not translate the query to English first?" Take it seriously — that is what
people did before multilingual embedders got good. Answer with the cost: an extra model call per
query, a new failure mode when the translation drops a term like CLASSIC K, and you still have to
pick an embedder afterwards. Say it is measurable and that we did not measure it. Do not bluff a
number. 2 minutes, then move.
</div>

## Exit line

> The default was wrong for your language, and nobody told you.
