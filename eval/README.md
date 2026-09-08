# Evaluation

| file | what it is |
|---|---|
| `gold_questions.jsonl` | the 20 fixed questions that score the day, one JSON object per line |
| `metrics.py` | `hit@1`, `recall@5`, `MRR`, and the breakdown by question type |
| `chunking.py` | the four splitting strategies, and `strip_boilerplate()` |
| `retrieval.py` | the Ollama transport, `DenseRetriever`, `BM25`, `rrf()`, `pointwise_rerank()` |
| `run_benchmark.py` | the runner that prints the tables |
| `RESULTS.md` | the tables, as printed by those commands |

`metrics.py` computes everything deterministically — no LLM judge, so the numbers are
reproducible and do not drift between runs.

The rule: this set does not change between modules. The same 20 questions are scored again at
every point in the day where a design decision changes, so the improvement is measured rather
than asserted.

## How the gold labels are chosen

`gold_doc_ids` names **the documents an amount may legitimately be quoted from**, not every
document that mentions the fact. Several files restate a rule without carrying a figure — the
Turkish rebooking macro states the no-repricing rule, the English FAQ states it too — and they
are deliberately not gold, because an agent who stopped there would have nothing to quote.

This is what makes q19's zero score a real failure rather than a labelling artefact: both
retrievers return a document that *talks about* the rule and neither returns one that *states
the number*. q07 and q09 are decided the same way. Each question's `why` field records the
reasoning for that question, and it is the first thing to read before disputing a score.

## Reproducing the numbers

Sections 1, 3, 4, 5 and 7 of `RESULTS.md` are printed by the commands below. Run them from the
repository root. Ollama has to be running with `bge-m3`, `nomic-embed-text` and `qwen2.5:3b`
pulled; the four Python files here import nothing outside the standard library, so they still
run on a laptop where you cannot install packages.

The other two sections come from the notebooks rather than from here: the column-header check
in section 2 is `notebooks/05_chunking_and_noise.py`, and the context-size table in section 6
is `notebooks/03_stuff_the_prompt.py`.

Three of these measurements belong to `exercises/` rather than to this directory. They are the
ones the room runs as a single command and compares numbers on, and two of them print figures
no command here can produce:

| command | prints | reproduces |
|---|---|---|
| `python exercises/m6_embedding_bakeoff.py` | three embedders over the same structure-aware chunks | RESULTS.md section 3 — the only command in the repository that includes ChromaDB's default `all-MiniLM-L6-v2`, so it is the only one that reproduces that row. Needs `chromadb` installed; the other commands do not |
| `python exercises/m7_chunking_ladder.py` | the five-rung ladder with hit@1 by question type, then the same corpus with its legal footer stripped | RESULTS.md section 1, including the boilerplate-strip row, which `--chunking` prints as a sixth ladder rung |
| `python exercises/m9_rerank_trade.py` | one reranker over four retrieval setups, before and after, then the per-question view on the strongest | RESULTS.md section 5. `--quick` runs a reduced version and says so |

And the runner in this directory:

| command | prints | reproduces |
|---|---|---|
| `python eval/run_benchmark.py --skip-rerank` | BM25 vs each embedder vs RRF, at document level | the whole-document table in RESULTS.md section 4 |
| `python eval/run_benchmark.py` | the same, plus a rerank pass over the top 8 documents | as above, with the cost of reranking |
| `python eval/run_benchmark.py --chunking` | the chunking ladder, with chunk counts and hit@1 by question type | RESULTS.md sections 1 and 7 |
| `python eval/run_benchmark.py --chunking --embed-model nomic-embed-text` | the same ladder on the weak embedder — `tr_en` is 0.000 in every column | RESULTS.md section 3, and the "before" column of section 5 |
| `python eval/run_benchmark.py --fusion` | dense vs BM25 vs RRF over structure-aware chunks | the chunk-level table in RESULTS.md section 4 |
| `python eval/run_benchmark.py --rerank-sweep` | one reranker over four retrieval setups, before and after | RESULTS.md section 5, the same four setups as the M9 exercise |

Section flags combine, and any of them replaces the default document-level run:
`python eval/run_benchmark.py --chunking --fusion`.

Nothing here has a cached fallback: these commands call Ollama, and without it they stop with
the model name and the `ollama pull` line that fixes it. The recorded-run fallback for a laptop
that cannot run a live model is in the notebooks — `USE_CACHED=1`, described in
`notebooks/README.md` — and it replays notebook cells, not these commands.

If a number printed by a fresh run disagrees with `RESULTS.md`, the run is right. The corpus is
versioned and edited: a correction to `corpus/2026-Q3` moves the chunk counts, and the chunk
counts move every row of the ladder. Re-run before quoting.

Every run ends with what it cost — embed calls, chat calls and wall-clock seconds — because on
this course the price of a technique is part of its result.

Every command below was timed by running it, back to back with the others, on one M-series Mac
with the models already resident in memory and other work running on the same machine. Four of
them were run twice and carry both figures. The exercises are in the same table because they are
what the room actually runs:

| command | wall clock | model calls |
|---|---|---|
| `eval/run_benchmark.py --fusion` | 9–10 s | 21 embed |
| `eval/run_benchmark.py --skip-rerank` | 11 s | 42 embed |
| `exercises/m6_embedding_bakeoff.py` | 18 s | 42 embed, plus one local ONNX index for `all-MiniLM-L6-v2` |
| `eval/run_benchmark.py --chunking --embed-model nomic-embed-text` | 26 s | 126 embed |
| `eval/run_benchmark.py --chunking` | 93 s | 126 embed |
| `exercises/m7_chunking_ladder.py` | 58–98 s | 126 embed |
| `exercises/m9_rerank_trade.py --quick` | 65–80 s | 42 embed, 160 chat |
| `eval/run_benchmark.py` (no flags) | 136 s | 42 embed, 160 chat |
| `eval/run_benchmark.py --rerank-sweep` | 298 s | 84 embed, 640 chat |
| `exercises/m9_rerank_trade.py` | 324 s | 84 embed, 640 chat |

**The call counts are exact and do not vary. The seconds are one machine's on one afternoon and
should be read as an order of magnitude.** Where a range is given, the command was run twice and
both figures are real: the chunking ladder took 98 s and then 58 s for identical work. The two
640-call rows are the same measurement through two entry points and differ by 26 seconds; inside
them the four setups took between 47 and 113 seconds each across the two runs, in no consistent
order. The first call after a pull loads the
model into RAM, a CPU-only laptop is several times slower throughout, and anything else running
on the machine shows up here. Time your own before you promise the room a number.

## Options

- `--embed-models bge-m3,nomic-embed-text` — which embedders the document-level run compares.
  A model that is not pulled is named, with the `ollama pull` line that fixes it, and the run
  continues with the models that are present rather than dropping the whole table.
- `--embed-model bge-m3` — the embedder used by `--chunking` and `--fusion`.
- `--smoke N` — run on the first N questions to check that a command works before committing
  minutes to it. A smoke run prints a warning, and its numbers are not comparable to
  `RESULTS.md`; do not quote them.

Section flags do not need a running model to be discovered: `python eval/run_benchmark.py --help`
lists all of them.

## When something fails

`retrieval.py` distinguishes three failures that look identical from the outside:

- **nothing listening on `localhost:11434`** — start the server with `ollama serve`.
- **the server answered, but the model tag is not pulled** — the message names the model and
  the `ollama pull` command. This is the common one, and it used to be reported as if Ollama
  were down, which sent people to restart a server that was already running.
- **the request failed or timed out** — the message quotes what Ollama said.

Helios Air is a fictional airline. The corpus and the gold questions are synthetic training
material; no Amadeus system, customer or production data appears in them.
