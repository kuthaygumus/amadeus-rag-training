---
title: "6. The Embedding Bake-Off"
description: "Is the default embedder right for *your* language?"
---

## Gate question

> **Is the default embedder right for *your* language?**

Module 5 left a wall on the board and did not explain it. Over 28 whole documents `bge-m3` scored
hit@1 **0.600**, and on the six Turkish-question / English-document items it scored **0.333** —
four of the six came back wrong. Two suspects: the embedder nobody in this room chose, or the way
we are cutting the documents. This module eliminates one of them.

Take the one you never chose. I told you the model was `bge-m3` and told you to take it on trust.
In your own project you would `pip install chromadb`, call `add()`, and a model would be picked for
you. So stop trusting it. One command scores three embedders over the same twenty questions and the
same chunks, and breaks the score down by question type:

```text
hit@1 by type         all-MiniLM-L6-v2  nomic-embed-text            bge-m3
--------------------------------------------------------------------------
tr_tr (n=4)                      0.750             0.500             1.000
tr_en (n=6)                      0.000             0.000             0.667   <-- Turkish question, English document
en_en (n=4)                      0.500             0.500             0.500
exact_token (n=4)                0.250             0.500             1.000
multi_hop (n=2)                  0.500             0.500             0.500
```

Read the `tr_en` row. Those are the six questions where a Turkish-speaking agent asks about an
English document — the shape of half our corpus. ChromaDB's own default gets none of the six.
`nomic-embed-text`, a reasonable-looking English-first choice, gets none of the six either. No
exception, no warning, no log line, no score that looks suspicious. Both pipelines return documents
with confident-looking similarity numbers and the right one is not among them. The generator then
answers from what it was given, because that is its job.

**Read the index condition off the banner before you read the table.** That run is over 154
structure-aware chunks — module 7's splitter, an index you do not have yet. That is deliberate: the
embedder is only the single variable if the index is nailed down, so the bake-off nails it to the
best index this course ever builds. It cuts both ways, and both ways are worth having. The two
English-first models were handed the best index in the course and still scored **0.000** across a
language boundary, so no amount of chunking was ever going to rescue them and the embedder question
is settled here rather than in module 7. And the 0.750 is not your number. Yours is still 0.600
over whole documents, and your `tr_en` is still 0.333.

<div class="presenter-note">
Slot: 16 minutes at 11:26. This module is first on the cut list, at 8 minutes — the cut plan is in
the last note on this page.<br />
Put module 5's 0.333 back on the whiteboard before you run anything; the room has to be holding a
number for this module to move one. Then get a commitment: "we change the embedding model and
nothing else. Out of six Turkish questions, how many does it still bring back? Hands up for four or
more." Most hands go up. Show the <code>tr_en</code> row and say nothing for three seconds. That
silence does more than the next paragraph. 4 minutes.
</div>

## What an embedding actually is

An embedding model takes a string and returns a fixed-length list of floating point numbers. That
is the entire contract. `bge-m3` returns **1024** numbers for any input, one word or one page.
`nomic-embed-text` returns **768**. `all-MiniLM-L6-v2` returns **384**. The *input* is not
unlimited, though: MiniLM stops reading at 256 tokens, so part of every 900-character chunk never
reaches it. One more thing the default does not tell you.

Each number is a coordinate, and the only thing you ever do with two of those points is ask how
close they are. Closeness is cosine similarity, four lines in `eval/retrieval.py`:

```python
def cosine(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / (norm + 1e-12)
```

The whole day rests on the claim that direction in that space corresponds to meaning. It does,
approximately, inside the model's training distribution. Outside it the numbers keep coming and
quietly stop meaning what you assumed.

## The trap, live

A score of 0.000 tells you the right document was not first. What came back instead is the more
useful half, and `nomic-embed-text` answers almost the same way every time:

```text
q05 ['sop_misconnect_v4']              -> macro_tr_noshow
q06 ['sop_denied_boarding']            -> macro_tr_noshow
q07 ['interline_xx_yy']                -> macro_tr_noshow
q08 ['codeshare_xx_yy_conditions']     -> macro_tr_noshow
q09 ['macro_en_refund']                -> macro_tr_rebook
q10 ['macro_en_special_assistance']    -> macro_tr_noshow
```

Five of the six land on the same Turkish document. `macro_tr_noshow` is a plausible neighbour —
Turkish, about penalties, containing "iptal cezası iki katına çıkar" — but it explicitly refuses to
state an amount: *"tutarlar bu makroda tekrarlanmaz"*, use the English fare rule. The retriever
picked the right language and discarded every document that had the answer.

## Be precise about what failed

The easy summary is "these models do not know Turkish". It is wrong, and it will get you beaten in
the Q&A. Go back to the `tr_tr` row: on Turkish questions against Turkish documents
`all-MiniLM-L6-v2` scores **0.750** and `nomic-embed-text` **0.500** — weak, but nowhere near zero.
On `tr_en` both score **0.000**. Six out of six. The failure is not the Turkish. It is **crossing**
between languages.

In a monolingual model's space the language of the text is itself a strong direction: two Turkish
sentences about unrelated topics can sit closer together than a Turkish sentence and its own English
translation, because "being Turkish" outweighs "being about cancellation penalties". Language
becomes a bigger axis than meaning, and the `tr_en` list above is what that looks like from the
inside.

It is invisible in every metric you would naturally check — TR→TR still works after a fashion, EN→EN
still works, and all three models tie at 0.500 on `en_en`. Half the corpus, half the questions,
silently broken, and an English-only test set would never have found it. The model doing the
breaking is ChromaDB's default embedding function, `all-MiniLM-L6-v2`, downloaded on your first
`add()` call without being asked for; module 8 shows it choosing for you and shows the bill.

<div class="presenter-note">
This is the sentence not to garble: <strong>these models are not helpless in Turkish — what they
cannot do is match across languages, and your corpus is mixed.</strong> Say it once, slowly, and
write "TR → TR: 0.750 / TR → EN: 0.000" on the whiteboard. If Ollama drops out here, do not debug in
front of the room: <code>eval/RESULTS.md</code> section 3 is this page's first table, and section 7
is the by-type breakdown you were about to run. Read the <code>tr_en</code> line out of section 3 —
0.667 for <code>bge-m3</code>, 0.000 for both others — and say the command produced it on a laptop
last week. 3 minutes.
</div>

## What you run

**No notebook this time.** This module is one command.

**Terminal (repo root):**

```bash
python exercises/m6_embedding_bakeoff.py
```

- **what you should see** — a header line reading `28 documents, 154 structure-aware chunks, 20 questions`, then each of the three embedders reporting that it indexed and queried, then two tables. The line that matters is `tr_en (n=6)`, printing `0.000  0.000  0.667` with an arrow pointing at it. Nothing errors and nothing warns
- **roughly how long** — about 18 seconds, most of it embedding 154 chunks three times

Nothing downloads here: `bge-m3` and `nomic-embed-text` were pulled at home, and
`all-MiniLM-L6-v2` was cached by `scripts/seed_offline_assets.py`, which `scripts/verify_setup.py`
checks. If the MiniLM column says `SKIPPED`, run `python scripts/seed_offline_assets.py` from the
same terminal and run the bake-off again — and if that will not finish in the room, keep going,
because the `tr_en` row still reads 0.000 against 0.667 with two embedders.

<div class="presenter-note">
Run it on the projector and let the room run it at the same time; at about 18 seconds nobody is left
waiting. Two things to watch. A laptop whose <code>all-MiniLM-L6-v2</code> column prints
<code>SKIPPED</code> never ran the seed script — take the two-embedder result and move on. And
somebody will read across the rows for a defence of the cheap models. There are two, and both are
answerable. All three tie at 0.500 on <code>en_en</code>: true, and it is the point — on English
questions the cheap embedder is not visibly worse, which is why nobody catches this. And all three
columns tie at 0.500 on <code>multi_hop</code> too, <code>bge-m3</code> included — that is one
question out of two, identical for the best and the worst model in the comparison, which means
chunking made that question reachable, not the embedder. Say so before it becomes an argument.
5 minutes.
</div>

## What the numbers said

<div class="measured">

Three embedders, 20 questions, structure-aware chunks:

| embedder | vector length | hit@1 | recall@5 | MRR | `tr_en` (n=6) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` — ChromaDB's silent default | 384 | 0.350 | 0.600 | 0.467 | 0.000 |
| `nomic-embed-text` | 768 | 0.350 | 0.633 | 0.490 | 0.000 |
| `bge-m3` | 1024 | **0.750** | **0.833** | **0.817** | **0.667** |

The same model, two indexes — quote the condition or quote nothing:

| `bge-m3` over | hit@1 | `tr_en` (n=6) |
|---|---|---|
| whole documents — the pipeline you have right now | 0.600 | 0.333 |
| structure-aware chunks — where module 7 ends | 0.750 | 0.667 |

</div>

One asterisk on the MiniLM MRR: the two Ollama models are ranked over all 154 chunks, but Chroma is
asked for its top 20, so a gold chunk outside that window scores 0 instead of its true reciprocal
rank. hit@1, recall@5 and the whole `tr_en` column are unaffected; only that MRR reads harsher than
the model deserves.

Corpus: 28 documents, 154 structure-aware chunks; gold set 20 questions, six of them
Turkish-query / English-document, everything local through Ollama. Twenty questions choose between
two designs and are far too few to publish, so a gap under about 0.05 is noise. These two are not:
0.350 against 0.750 is eight questions, and the `tr_en` column is four against none.

So the embedder gets settled before the chunking does, and the second table is why. A better index
was worth 0.333 → 0.667 to the model that can cross languages. It was worth nothing to the other
two, because they were already standing on module 7's index when they scored 0.000.

## Going deeper

What multilingual training changes is the objective, not the architecture. A monolingual model is
trained to pull together texts that mean the same thing in one language. `bge-m3` is trained on
parallel and mined pairs across roughly a hundred languages, where the positive pair is a sentence
and its translation — so the gradient's whole job is to land the Turkish sentence and the English
sentence on the same point, and language stops being a usable direction because the loss punishes
it. Not more data, not a better prompt: it was baked in months before you pulled the weights.

Dimensionality is not quality. 1024 beats 768 and 384 here, which flatters the intuition, but width
is not what broke MiniLM — it still scores 0.500 on English question against English document, and
it is exactly 0.000 only where a language boundary is crossed. Width is a cost, and the cost is
real: like for like through Ollama, `bge-m3` took **18.5 s** to index and query the same 154 chunks
against `nomic-embed-text`'s **16.0 s**.

Evaluating an embedder for your own corpus takes an afternoon and it is the highest-value afternoon
in this subject. Write twenty questions in the language your users actually type, each labelled with
the document that answers it. Group them by the failure you suspect: same-language, cross-language,
exact identifiers, paraphrase. Then loop three or four models through `DenseRetriever` and print
hit@1 **per group**, not the average. The grouping does the work: MiniLM's overall 0.350 is bad but
not alarming, and only the `tr_en` row shows a clean zero.

And 0.667 is not a victory. Two of six are still wrong. The winner of a bake-off is the least bad
option you measured, not a solved problem, and what is still broken at 0.667 is module 7.

<div class="presenter-note">
Somebody will ask "why not translate the query to English first?" Take it seriously — that is what
people did before multilingual embedders got good. Answer with the cost: an extra model call per
query, a new failure mode when the translation drops a term like CLASSIC K, and you still have to
pick an embedder afterwards. Say it is measurable and that we did not measure it. Do not bluff a
number. 4 minutes with the closing arithmetic, then move.<br />
<strong>Cutting 16 to 8.</strong> Keep module 5's 0.333 on the board and the <code>tr_en</code> row
(4 min), the one sentence about crossing languages (1 min), the run and the first table (2 min), the
exit line (1 min). Drop <em>The trap, live</em>, <em>Going deeper</em> and the translation question.
Do not drop the index-condition paragraph: module 7 opens on 0.600 and the room has to know why that
is not 0.750.
</div>

## Exit line

> The default was wrong for your language and nobody told you — and the embedder you already had is
> not what broke module 5. Over whole documents your `tr_en` is still 0.333, and the only suspect
> left is the cut.
