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

So stop trusting it. One command scores three embedders over the same twenty questions and the
same chunks, and breaks the score down by question type:

```text
hit@1 by type         all-MiniLM-L6-v2  nomic-embed-text            bge-m3
--------------------------------------------------------------------------
tr_tr (n=4)                      0.750             0.500             1.000
tr_en (n=6)                      0.000             0.000             0.667   <-- Turkish question, English document
en_en (n=4)                      0.500             0.500             0.750
exact_token (n=4)                0.250             0.500             1.000
multi_hop (n=2)                  0.500             0.500             0.500
```

Read the `tr_en` row. Those are the six questions where a Turkish agent asks about an English
document — the shape of half our corpus. ChromaDB's own default gets none of the six.
`nomic-embed-text`, a reasonable-looking English-first choice, gets none of the six either. No
exception, no warning, no log line, no score that looks suspicious. Both pipelines return
documents with confident-looking similarity numbers and the right one is not among them. The
generator then answers from what it was given, because that is its job.

**Every retrieval number needs its index condition.** The table above is measured over
structure-aware chunks — the strategy module 7 arrives at — held fixed so that the embedder is the
only thing that changes between columns. Change the index and the same model moves: over whole
documents, which is the pipeline you actually have right now, `bge-m3`'s `tr_en` score is
**0.333**, not 0.667. A retrieval figure quoted without the index it was measured on is not a
figure.

<div class="presenter-note">
Slot: 16 minutes at 11:26. This module is first on the cut list, at 8 minutes — the cut plan is in
the last note on this page.<br />
Before running the command, get a commitment: "we change the embedding model and nothing else. Out
of six Turkish questions, how many still land? Hands up for four or more." Most hands go up. Show
the <code>tr_en</code> row and say nothing for three seconds. That silence does more than the next
paragraph. 4 minutes.
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
cannot be compared by distance — the long one is further from the origin whatever it says — but
they can be compared by which way they point. The result runs from -1 to 1, and only the ordering
ever matters. A similarity score means nothing on its own. It means something only beside the
score the other candidate got.

The whole day rests on the claim that direction in that space corresponds to meaning. It does,
approximately, inside the model's training distribution. Outside it the numbers keep coming and
quietly stop meaning what you assumed.

## The trap, live

A score of 0.000 tells you the right document was not first. It does not tell you what was, and
that is the more useful half. Ask the retriever what it returned instead on those six questions,
and `nomic-embed-text` answers the same way almost every time:

```text
q05 ['sop_misconnect_v4']              -> macro_tr_noshow
q06 ['sop_denied_boarding']            -> macro_tr_noshow
q07 ['interline_h9_au']                -> macro_tr_noshow
q08 ['codeshare_h9_au_conditions']     -> macro_tr_noshow
q09 ['macro_en_refund']                -> macro_tr_rebook
q10 ['macro_en_special_assistance']    -> macro_tr_noshow
```

Six questions, six Turkish documents, and five of them the same Turkish document. The topics are
not even close: q05 asks how large a meal voucher a misconnected passenger is owed, and the answer
comes back from the no-show macro.

Look at what it chose. `macro_tr_noshow` is not a random document. It is Turkish, it is about
penalties, it contains "iptal cezası iki katına çıkar". A human would call it a plausible
neighbour for one of these six. But it explicitly refuses to state an amount — *"tutarlar bu
makroda tekrarlanmaz"*, use the English fare rule — so the one number the agent needs is never in
it. The retriever picked the right language and discarded every document with the answer.

## Be precise about what failed

The easy summary is "these models do not know Turkish". That summary is wrong, and it will get you
beaten in the Q&A.

Go back to the `tr_tr` row: Turkish questions against Turkish documents. `all-MiniLM-L6-v2` scores
**0.750** there and `nomic-embed-text` **0.500** — weak, but nowhere near zero. On `tr_en` both
score **0.000**. Six out of six. The failure is not the Turkish. It is **crossing** between
languages.

In a monolingual model's space the language of the text is itself a strong direction: two Turkish
sentences about unrelated topics can sit closer together than a Turkish sentence and its own
English translation, because "being Turkish" outweighs "being about cancellation penalties".
Language becomes a bigger axis than meaning. That is the whole of the failure, and it is exactly
what the `tr_en` list above looks like from the inside.

It is invisible in every metric you would naturally check, because TR→TR still works after a
fashion and EN→EN still works. Half the corpus, half the questions, silently broken.

ChromaDB's default embedding function is `all-MiniLM-L6-v2`, downloaded on your first `add()` call
without being asked for, and English-only. Over the same twenty questions it lands at hit@1
**0.350** and MRR **0.443**, against `bge-m3`'s **0.800** and **0.844**. None of that reaches the
log.

<div class="presenter-note">
This is the sentence not to garble: <strong>these models are not helpless in Turkish — what they
cannot do is match across languages, and your corpus is mixed.</strong> Say it once, slowly, and
write "TR → TR: 0.750 / TR → EN: 0.000" on the whiteboard. If Ollama drops out here, every figure in the
tables on this page is in <code>MEASURED.md</code>; read them off and do not debug in front of the room.
3 minutes.
</div>

## What you run

```bash
python exercises/m6_embedding_bakeoff.py
```

- **what you should see** — a header line reading `28 documents, 154 structure-aware chunks, 20 questions`, then each of the three embedders reporting that it indexed and queried, then two tables. The line that matters is `tr_en (n=6)`, printing `0.000  0.000  0.667` with an arrow pointing at it. Nothing errors and nothing warns
- **roughly how long** — about 20 seconds, most of it embedding 154 chunks three times

Nothing downloads here. All three models are covered by the pre-work: `bge-m3` and
`nomic-embed-text` were pulled at home, and `all-MiniLM-L6-v2` was cached by
`scripts/seed_offline_assets.py`, which `scripts/verify_setup.py` checks. If the MiniLM column
says `SKIPPED`, run the seed script and re-run; do not pull anything in the room.

If the room asks what the wrong pipeline returned instead, this prints the list in *The trap,
live* from the repository root:

```python
import sys; sys.path.insert(0, "eval")
from pathlib import Path
import chunking as C, metrics, retrieval as R

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("corpus/2026-Q3").glob("*.md"))}
ids, texts, _ = C.chunk_corpus(docs, "structure-aware")
tr_en = [q for q in metrics.load_gold("eval/gold_questions.jsonl") if q["type"] == "tr_en"]

r = R.DenseRetriever(ids, texts, model="nomic-embed-text")
for q in tr_en:
    print(q["id"], q["gold_doc_ids"], "->", C.to_documents(r.rank(q["query"]))[0])
```

<div class="presenter-note">
Run it on the projector and let the room run it at the same time; at about 20 seconds nobody is
left waiting. Two things to watch. A laptop whose <code>all-MiniLM-L6-v2</code> column prints
<code>SKIPPED</code> never ran the seed script — take the two-embedder result and move on, the
<code>tr_en</code> row still reads 0.000 against 0.667. And somebody will notice that the default
ties <code>bge-m3</code> on exactly one row, <code>multi_hop</code>, where both score 0.500 on two
questions. Two questions is not a result; say so before it becomes a defence of the default.
5 minutes.
</div>

## What the numbers said

<div class="measured">

Three embedders, 20 questions, structure-aware chunks:

| embedder | vector length | hit@1 | recall@5 | MRR | `tr_en` (n=6) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` — ChromaDB's silent default | 384 | 0.350 | 0.583 | 0.443 | 0.000 |
| `nomic-embed-text` | 768 | 0.350 | 0.633 | 0.492 | 0.000 |
| `bge-m3` | 1024 | **0.800** | **0.833** | **0.844** | **0.667** |

The same model, two indexes — quote the condition or quote nothing:

| `bge-m3` over | hit@1 | `tr_en` (n=6) |
|---|---|---|
| whole documents — the pipeline you have right now | 0.550 | 0.333 |
| structure-aware chunks — where module 7 ends | 0.800 | 0.667 |

Chunking does not rescue the wrong embedder:

| hit@1, 20 questions | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| fixed 280-character chunks | 0.350 | 0.700 |
| structure-aware chunks | 0.350 | 0.800 |

</div>

Corpus: 28 documents, 154 structure-aware chunks. Gold set: 20 questions, 6 of them
Turkish-query / English-document, everything local through Ollama. Twenty questions are enough to
choose between two designs and far too few to publish — a difference under about 0.05 is inside
this sample's noise. The `tr_en` gap is not.

Read the third table again. With `nomic-embed-text`, better chunking buys **nothing** — 0.350
either way. The chunking work in module 7 only pays once the embedder can see across languages at
all. Fix the embedder first, then chunk.

## Going deeper

What multilingual training changes is the objective, not the architecture. A monolingual model is
trained to pull together pairs of texts that mean the same thing in one language. `bge-m3` is
trained on parallel and mined pairs across roughly a hundred languages, where the positive pair is
a sentence and its translation. The gradient's whole job is to land the Turkish sentence and the
English sentence on the same point, so language stops being a usable direction — the loss punishes
it. The fix is not more data or a better prompt. It was baked in months before you pulled the
weights.

Dimensionality is not quality. 1024 beats 768 and 384 here, which flatters the intuition, but
width is not what broke MiniLM: on the four English-question / English-document questions it still
scores 0.500, and it is exactly 0.000 only where a language boundary is crossed. Width buys
capacity to represent distinctions. It does not decide which distinctions the model was taught to
care about. Treat the dimension as a cost — 1024 float32 is 4 KB per chunk, so our 154 chunks are
well under a megabyte and ten million would be about 40 GB before index overhead — and treat the
training data as the quality signal.

The cost is real, which is why this is a trade and not a free win. `bge-m3` is an
XLM-RoBERTa-large body, several times MiniLM's parameters, and it embeds correspondingly slower —
visible even in this run, where it takes about twice as long to index the same 154 chunks. At 28
documents that is nothing. At ten million it is a re-indexing budget you plan for, and the point
where Matryoshka embeddings — trained so the first 256 numbers of the vector work on their own —
let you shortlist cheaply and rescore the top few hundred at full width.

Evaluating an embedder for your own corpus takes an afternoon and it is the highest-value
afternoon in this subject. Write twenty questions in the language your users actually type, each
labelled with the document that answers it. Twenty is enough — this page chose a model on twenty,
and the row that decided it held six. Group them by the failure you suspect: same-language,
cross-language, exact identifiers, paraphrase. Then loop three or four models through
`DenseRetriever` and print hit@1 **per group**, not the average. The grouping does the work:
MiniLM's overall 0.350 is bad but not alarming, and only the `tr_en` row shows a clean zero.

And 0.667 is not a victory. Two of six are still wrong. The winner of a bake-off is the least bad
option you measured, not a solved problem, and what is still broken at 0.667 is module 7.

<div class="presenter-note">
Somebody will ask "why not translate the query to English first?" Take it seriously — that is what
people did before multilingual embedders got good. Answer with the cost: an extra model call per
query, a new failure mode when the translation drops a term like CLASSIC K, and you still have to
pick an embedder afterwards. Say it is measurable and that we did not measure it. Do not bluff a
number. 4 minutes with the closing arithmetic, then move.<br />
<strong>Cutting 16 to 8.</strong> Keep the commitment and the <code>tr_en</code> row (4 min), the
one sentence about crossing languages (1 min), the run and the first table (2 min), the exit line
(1 min). Drop <em>The trap, live</em>, <em>Going deeper</em> and the translation question. Do not
drop the index-condition sentence: module 7 opens on 0.550 and the room has to know why that is
not 0.800.
</div>

## Exit line

> The default was wrong for your language, and nobody told you. We have the right embedder now
> — and over whole documents, which is what we are still running, retrieval is only 0.550.
