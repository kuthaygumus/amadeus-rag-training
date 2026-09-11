# Exercises — one command, one table

Three measurements in the day are things you *run*, not things you *build*. They live here
rather than in a notebook, because when twenty people need to compare numbers, everyone has to
get the same output and nobody can be halfway down a cell list.

The rule for the day:

> **If it is a measurement, you run one command and we compare numbers.
> If it is a mechanism, we build it together in cells.**

Everything under `notebooks/` is the second kind — open the `.py` in VS Code and run it cell by
cell. Everything here is the first kind.

| Module | Command | What it shows | Measured |
|---|---|---|---|
| **M6** — is the default embedder right for your language? | `python exercises/m6_embedding_bakeoff.py` | Three embedders over the same chunks. The `tr_en` row is the point. | 18–23 s |
| **M7** — why did retrieval bring back garbage? | `python exercises/m7_chunking_ladder.py` | Five ways to cut the same documents, plus boilerplate stripping. Read the `exact_token` column, not only `hit@1`. | ~60 s |
| **M9** — when does a reranker actually pay? | `python exercises/m9_rerank_trade.py` | The same reranker over four retrieval setups of different quality. | 253 s |

Run all three from the repository root — the folder that holds `corpus/`, `eval/`, `exercises/`
and `notebooks/`. Each script finds the corpus and the gold set from its own location rather than
from your working directory, so the command above is the whole contract: stand at the root and
type it exactly as written.

Those times are from runs on one M-series Mac with the models already in memory, and they move —
the same chunking ladder has been timed at 59 s and at 65 s for identical work. A CPU-only laptop is
several times slower and the first call after a pull is slower again, so treat them as an order of
magnitude. What does not vary is the number of model calls: M6 and M7 are embedding only, M9 makes
640 chat calls.

## Before you run anything

These need local Ollama with `bge-m3`, `nomic-embed-text` and `qwen2.5:3b` pulled, plus the two
Python packages in `requirements.txt`. `python scripts/verify_setup.py` tells you if you are
ready. Nothing here reaches the internet: no API key, no HuggingFace, no sign-in.

## If your laptop is slow

`m9_rerank_trade.py` is the expensive one — 640 model calls in its full form, measured at 253 s on
an Apple-silicon Mac and considerably longer on a CPU-only machine. Use:

```
python exercises/m9_rerank_trade.py --quick
```

It scores 4 candidates per question over the two strong setups instead of 8 over all four, and it
prints a banner saying the run is reduced; measured at 82 s. The direction of the result holds —
both strong setups still lose — but the numbers are not the ones on the slides, because a shallower
rerank moves fewer candidates. The script says so in its banner rather than letting you quote them
by mistake, and its closing lines are written from the verdicts that run actually produced, so a
reduced run cannot claim the half of the trade it did not measure.

## Why the numbers on your screen should match the slides

They come from the same code, the same 28 documents in `corpus/2026-Q3` and the same 20 questions
in `eval/gold_questions.jsonl`. The Ollama retrievers are deterministic — exact cosine over every
chunk, no approximate index — and the reranker runs at temperature 0, so the same setup on the same
corpus gives the same table. The one column that goes through ChromaDB's own index, M6's
`all-MiniLM-L6-v2`, is the exception: that is an approximate nearest-neighbour search, so treat a
small move in that column as the search, not as a result.

If a number on your screen disagrees with a number in `eval/RESULTS.md`, that is worth stopping the
room for. Either the slide is stale, or the corpus and the gold set have drifted apart — a gold
`gold_doc_ids` entry naming a document the corpus no longer contains scores zero however good the
retriever is — or your setup differs. All three are worth knowing. Wall-clock seconds are the
exception; those move with the machine.

The scripts themselves will not disagree with their own tables. Every closing sentence the three
print is derived from the run that just happened and guarded by the condition it asserts, so when
a result moves the commentary moves with it — or goes silent — instead of repeating a claim the
table above it no longer supports.

Kraken Air is a fictional airline. The corpus is synthetic training material.
