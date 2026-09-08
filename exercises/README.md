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
| **M6** — is the default embedder right for your language? | `python exercises/m6_embedding_bakeoff.py` | Three embedders over the same chunks. The `tr_en` row is the point. | 18 s |
| **M7** — why did retrieval bring back garbage? | `python exercises/m7_chunking_ladder.py` | Five ways to cut the same documents, plus boilerplate stripping. Read the `exact_token` column, not only `hit@1`. | 58–98 s |
| **M9** — when does a reranker actually pay? | `python exercises/m9_rerank_trade.py` | The same reranker over four retrieval setups of different quality. | 324 s |

Those times are from runs on one M-series Mac with the models already in memory. They move: M7 was
timed at 98 s and then at 58 s for identical work. A CPU-only laptop is several times slower and
the first call after a pull is slower again, so treat them as an order of magnitude. What does not
vary is the number of model calls — M6 and M7 are embedding only, M9 makes 640 chat calls — nor do
the scores.

## Before you run anything

These need local Ollama with `bge-m3`, `nomic-embed-text` and `qwen2.5:3b` pulled, plus the two
Python packages in `requirements.txt`. `python scripts/verify_setup.py` tells you if you are
ready. Nothing here reaches the internet: no API key, no HuggingFace, no sign-in.

## If your laptop is slow

`m9_rerank_trade.py` is the expensive one — 640 model calls in its full form, measured at 324 s on
an Apple-silicon Mac and considerably longer on a CPU-only machine. Use:

```
python exercises/m9_rerank_trade.py --quick
```

It scores 4 candidates per question over the two strong setups instead of 8 over all four, and it
prints a banner saying the run is reduced. Measured at 65 s and 80 s on two runs. The direction of the result holds —
both strong setups still lose — but the numbers are not the ones on the slides: at depth 4 the
strongest setup ends at hit@1 0.650 rather than 0.600. The script says so rather than letting you
quote them by mistake, and its closing lines are written from the verdicts that run produced, so a
reduced run does not claim the half of the trade it did not measure.

## Why the numbers on your screen should match the slides

They come from the same code, the same 28 documents in `corpus/2026-Q3` and the same 20 questions
in `eval/gold_questions.jsonl`. Retrieval is deterministic, and so is the reranker at temperature
0: the full M9 run reproduced its four before/after pairs and its five demotions exactly on a
second pass. So if a number on your screen disagrees with a number in `eval/RESULTS.md`, that is
worth stopping the room for — either the slide is stale or your setup differs, and both are worth
knowing. Wall-clock seconds are the exception; those move with the machine.

Helios Air is a fictional airline. The corpus is synthetic training material.
