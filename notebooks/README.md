# Notebooks

The notebooks participants run, in module order. Every notebook runs **with no network access**:
generation and embeddings go to a local Ollama instance on `localhost:11434`, and the two assets
that do come from the internet are downloaded at home by `scripts/seed_offline_assets.py`.

## How to run one

There are two surfaces in this course, and the **repository root** — the folder holding `corpus/`,
`notebooks/`, `eval/` and `exercises/` — is the anchor of both.

**VS Code, with the repository root as the open folder.** Open the `.py` file and run it cell by
cell with the Python extension. A line reading `# %%` starts a cell; VS Code shows a **Run Cell**
link above each one. There is no Jupyter to install, which is the point — nothing beyond
`requirements.txt` has to go on the machine.

**A terminal, at the repository root.** Every `ollama …` command and everything written as
`python scripts/…`, `python exercises/…` or `python eval/…` runs there, and so does a notebook if
you want it end to end:

    python notebooks/05_chunking_and_noise.py

Every notebook reads `../corpus` and `../eval`, and its first cell moves the working directory to
`notebooks/` whichever of the two you started from, so both forms work.

The `.ipynb` files next to them are generated from the `.py` by `python scripts/build_notebooks.py`
and are there for anyone who does have a notebook environment. **Edit the `.py`, never the
`.ipynb`.** The `.ipynb` carry no saved output; the recorded run lives in `cached_runs.json`. If
VS Code offers to install Jupyter when you open one, say no, close it, and open the `.py` of the
same name.

## Which notebook belongs to which module

The numbering stops matching the module order at module 8, so this table is the map rather than
the filenames.

| module | what you open | surface |
|---|---|---|
| 1 — the bare LLM | `00_bare_llm_fails.py` | VS Code |
| 2 — what a weight is | `01_mnist_tiny_net.py` | VS Code — no model, no network |
| 3 — fine-tuning | `02_finetune_qwen_lora.py` | VS Code |
| 4 — stuff the prompt | `03_stuff_the_prompt.py` | VS Code |
| 5 — naive RAG | `04_naive_rag.py` | VS Code |
| 6 — the embedder | *no notebook* — `python exercises/m6_embedding_bakeoff.py` | terminal, repo root |
| 7 — chunking | `05_chunking_and_noise.py`, then `python exercises/m7_chunking_ladder.py` | both |
| 8 — ChromaDB | `08_chromadb.py` | VS Code |
| 9 — hybrid and rerank | `06_hybrid_rerank_contextual.py`, then `python exercises/m9_rerank_trade.py` | both |
| 10 — agentic RAG | `07_agentic_rag.py` | VS Code |

## Before the day

From the repository root:

    python -m pip install -r requirements.txt
    python scripts/seed_offline_assets.py     # MNIST, 11.6 MB · Chroma's default embedder, 83 MB
    python scripts/verify_setup.py            # Ollama, the models, one embed and one chat call

The models have to be pulled at home: `qwen2.5:3b`, `bge-m3`, `nomic-embed-text`, and
`qwen2.5:1.5b` if the machine has 8 GB of RAM. That is 3.36 GB, and twenty laptops pulling it
at 09:10 on the day costs the morning. The first cell of every notebook checks for the ones it
needs and stops with the exact `ollama pull` line rather than failing forty cells later.

## Notebooks build mechanisms; the exercises print the tables

Three of the day's measurements do not live in a notebook, on purpose. Scoring five chunking
strategies or reranking four retrieval setups means hundreds of model calls behind a progress
counter, and there is nothing to learn from watching that happen cell by cell.

| what | where | cost |
|---|---|---|
| the chunking ladder, plus the boilerplate strip | `exercises/m7_chunking_ladder.py` | 2–3 min, embeddings only |
| the four-setup rerank trade | `exercises/m9_rerank_trade.py` (`--quick` for a labelled reduced run) | 640 model calls, about five minutes |
| the embedder bake-off | `exercises/m6_embedding_bakeoff.py` | ~20 s |

Notebook 05 builds the splitting mechanism and the column-header failure and then sends you to
`m7`. Notebook 06 builds the reranker — one query, six candidates, the scores printed — and then
sends you to `m9`. Neither notebook contains a cell that runs for minutes without output.

## Three environment variables

| | what it does |
|---|---|
| `USE_CACHED=1` | do not call the model; read the measurement cells out of `cached_runs.json` |
| `QUICK=1` | run a deliberately smaller version of the long cells |
| `RECORD_CACHED=1` | run for real and write every measured result into `cached_runs.json` |

**`USE_CACHED=1`** is the fallback for a laptop that is too slow or an Ollama that will not start.
Every replayed cell prints a `[CACHED]` banner with the date the run was recorded, the machine,
and whether it was a full or reduced run, so nobody can mistake a recording for a live result.

All three are read once, when the notebook first imports its helpers, so they have to be set before
the first block runs. In a repo-root terminal that is `USE_CACHED=1 python notebooks/03_….py`. In
VS Code, put this in a new block **above** the notebook's first block and run it before anything
else — and restart the Python terminal first if a block has already executed:

    import os; os.environ["USE_CACHED"] = "1"        # or "QUICK"

How far it gets you depends on the notebook, and the first line the notebook prints says which
case it is. **03, 05 and 06 replay end to end with Ollama switched off** — every model call in
them goes through the cache. In 04, 07 and 08 the measurements replay but the short narrative
cells still call the model, and the preflight line says so rather than letting it be a surprise.
Notebook 00 records nothing and replays nothing: every cell in it is a live call, which is why
the table below says so.

**`QUICK=1`** cuts work honestly and says so. Notebook 03 uses it to halve the eight-question
context table. The exercises have their own `--quick`, which labels its output `REDUCED`; a
reduced run is not comparable with `eval/RESULTS.md`.

**`RECORD_CACHED=1`** is for the trainer, at home, before the day. From the repository root:

    for n in 03 04 05 06 07 08; do RECORD_CACHED=1 python notebooks/${n}_*.py; done

Record without `QUICK=1`. Restart Ollama first if you want notebook 03's recording to show what a
cold first query really costs — with the corpus already in the server's prefix cache it records
one second instead of seventy.

## What each notebook needs

| notebook | models | other | replays offline |
|---|---|---|---|
| `00_bare_llm_fails` | `qwen2.5:3b` | | no — it is all live calls |
| `01_mnist_tiny_net` | none | seeded `mnist_data/` | needs no model at all |
| `02_finetune_qwen_lora` | `qwen2.5:3b` | | probe only |
| `03_stuff_the_prompt` | `qwen2.5:3b`, `bge-m3` | | yes, end to end |
| `04_naive_rag` | `qwen2.5:3b`, `bge-m3` | | measurements only |
| `05_chunking_and_noise` | `qwen2.5:3b` | | yes, end to end |
| `06_hybrid_rerank_contextual` | `qwen2.5:3b`, `bge-m3` | | yes, end to end |
| `07_agentic_rag` | `qwen2.5:3b`, `bge-m3` | | measurements only |
| `08_chromadb` | `bge-m3` | Chroma's `all-MiniLM-L6-v2` cache | the Chroma half |

`nomic-embed-text` is no longer loaded by any notebook — the module 6 and module 9 exercises are
what need it, which is why it stays on the pre-work pull list.

Notebook 08 still runs without the Chroma cache: the two cells that reproduce Chroma's silent
default replay the recorded run and say why, instead of pulling 83 MB mid-module.

## The two helper modules

`_preflight.py` and `_cached.py` are not notebooks. `build_notebooks.py` skips anything whose
name starts with an underscore. Their docstrings are the reference for what they do.

## The two files that are not notebooks either

`kraken_qa_q2.jsonl` and `kraken-q2.Modelfile` are the build inputs for module 3's fine-tune.
The dataset is written by `python scripts/make_finetune_dataset.py`, which walks it out of
`corpus/2026-Q2` so it can never drift from the documents it claims to teach. The Modelfile is
the last step of the chain — LoRA-train, merge, convert to GGUF, quantise, then
`ollama create kraken-q2 -f kraken-q2.Modelfile` — and it pins greedy decoding, because the
module rests on the answer being identical on every run. The whole chain is summarised in the
repository README under *Building `kraken-q2`*.

Kraken Air is a fictional airline. The corpus is synthetic training material.
