# RAG Training Day

A one-day, hands-on RAG course for working developers — built to run **entirely offline** on a
laptop with no administrator rights. No API key, no cloud account, no sign-in.

## Why this exists

Most RAG material teaches the happy path: chunk, embed, retrieve, done. That version collapses
the first time it meets a real corpus. This course is organised around a single rule:

> **No concept is introduced before you have seen the failure that requires it.**

Every module opens where the previous one hit a wall. The numbers below the `#` column are the
module numbers used by the site sidebar and by the handout, so all three agree.

| # | What you have | What breaks on screen | What you now need |
|---|---|---|---|
| 0 | A laptop | *(setup, not a gate)* | Ollama, three models, two packages, two seeded files |
| 1 | A bare LLM | Invents a fare rule, confidently, with no source | Get my own data in |
| 2 | A trained network | Weights fit the data — a weight is a frozen photograph | Put *my* data in the weights |
| 3 | A fine-tuned model | Right for last quarter, wrong for this one, cites nothing | Fresh knowledge without retraining |
| 4 | The whole corpus in the prompt | It fits, and it answers *better* — you pay for all of it on every query | Select only the right piece |
| 5 | Keyword search (BM25), then naive RAG | Misses the paraphrase; then retrieval returns garbage | Understand *why* retrieval failed |
| 6 | The default embedder | English-only, and nothing warns you | Measure the embedder |
| 7 | Naive chunking | Fixed-size raises the average while `exact_token` slips from 1.000 to 0.750; adding overlap loses on both metrics — hit@1 falls three questions (0.700 to 0.550) and recall@5 falls with it (0.917 to 0.883) | Ask what the average hides |
| 8 | Vectors in a Python list | Restart the process and the index is gone | Persist it, then put it behind a port |
| 9 | Hybrid + rerank + contextual | Neither wins here; the reranker levels | Measure before you add |
| 10 | Agentic RAG | — | Bridge to the agent day |

## Everything is measured

The day is not a sequence of demos. A fixed set of 20 gold questions is scored with the same
three metrics — `hit@1`, `recall@5`, `MRR` — every time a design decision changes. Participants
watch the numbers move and say *how much* better, not "better".

Twenty questions is not a benchmark: enough to choose between two designs, far too few to
publish. One question is worth 0.05, so hit@1 can only move in steps of 0.05, and most of the
gaps on the results page are exactly one question wide. Quote the direction, not the digit.
`eval/RESULTS.md` is the single source of truth for every figure the site and the handout print.

Results that contradict the usual advice are kept in, not smoothed over. Overlap, sold everywhere
as the safe default, is the one rung of the chunking ladder that goes backwards on ranking.
Combining a dense retriever with BM25 does not always beat either alone; this course shows the run
where fusion made things worse, and explains the condition under which it helps. A reranker is
shown as a trade rather than an upgrade: it lifts the weak retrieval setups and drags the strong
ones down.

## The two surfaces

There is no Jupyter server, no browser notebook and no cloud console anywhere in this course.
Everything happens on exactly two surfaces, and **the repository root is the anchor of both**:

1. **Terminal, at the repository root** — one terminal, `cd`'d into the folder that contains
   `corpus/`, `notebooks/`, `eval/` and `exercises/`. Every `ollama ...` command and everything
   written as `python scripts/...`, `python exercises/...` or `python eval/...` runs here. Every
   command printed in this file, on the site and in the handout assumes that working directory.
2. **VS Code, with the repository root as the open folder** — `File → Open Folder` on the
   repository root itself, *not* on `notebooks/`, or the `python scripts/...` and
   `python eval/...` commands on every later page break. The notebooks are percent-format `.py`
   files under `notebooks/`: open one, put the cursor inside a `# %%` block, press `Shift+Enter`,
   and the output appears in the Interactive window.

The only Python dependencies are `numpy` and `chromadb`. Everything runs against
`localhost:11434`, with no network access.

## Running it

Requires [Ollama](https://ollama.com) and Python 3.10+. Clone first, then stay at the root:

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
```

Three models, 3.4 GB, pulled once:

```bash
ollama pull qwen2.5:3b        # 1.9 GB — generation and reranking
ollama pull bge-m3            # 1.2 GB — multilingual embeddings
ollama pull nomic-embed-text  # 274 MB — the weaker embedder modules 6 and 9 measure against
ollama pull qwen2.5:1.5b      # 986 MB — optional fallback for a slow machine
```

Then the two packages, and the two files that would otherwise download in the middle of a module:

```bash
python -m pip install -r requirements.txt   # numpy + chromadb, and nothing else
python scripts/seed_offline_assets.py       # MNIST, 11.6 MB · Chroma's ONNX embedder, 83 MB
python scripts/verify_setup.py              # one line: READY or NOT READY
```

`python -m pip`, not bare `pip`: on Windows the two can resolve to different interpreters.
Windows without Python at all has `scripts/verify_setup.ps1`, which reports the same checks.

The whole pre-work puts **about 3.9 GB** on disk: 1.9 + 1.2 + 0.274 GB of models, roughly 400 MB
of installed packages, 83 MB of ONNX and 11.6 MB of MNIST. Add 986 MB if `verify_setup.py` calls
your machine slow and you pull `qwen2.5:1.5b` as well.

Then work through `notebooks/` in module order on the second surface above. Three of the day's
measurements are commands rather than notebooks — see `exercises/README.md`.

## The corpus

The corpus is a synthetic airline rule book: **Kraken Air (XX)**, with **Wyvern Overseas Airways
(YY)** as the partner carrier in the interline and codeshare documents
(`corpus/*/interline_xx_yy.md`, `corpus/*/codeshare_xx_yy_conditions.md`). It ships in two
editions — `corpus/2026-Q2` and `corpus/2026-Q3` — that differ by a named delta recorded in
`corpus/DELTA.md`, which is what makes module 3's stale-fine-tune demonstration possible.

## Building `kraken-q2`

Module 3 needs a model that learned the **2026-Q2** rule book and nothing about Q3. It is on no
registry; it is built once, elsewhere, on a GPU, and handed out on a USB stick:

```bash
python scripts/make_q2.py                 # derive corpus/2026-Q2 from 2026-Q3 by a named delta
python scripts/make_finetune_dataset.py   # walk Q2 into notebooks/kraken_qa_q2.jsonl
# LoRA-train Qwen2.5-1.5B-Instruct on that file, merge, convert to GGUF, quantise
ollama create kraken-q2 -f notebooks/kraken-q2.Modelfile
```

`notebooks/kraken-q2.Modelfile` pins greedy decoding, because the module rests on the answer being
the same every run. Without the model, module 3 still runs: the two probe cells print
`(skipped — kraken-q2 not installed)` — nothing is replayed in their place, notebook 02 has no
recorded run — and the corpus diff that the module is actually about needs no model at all.

## This site

The course site is built with [Astro Starlight](https://starlight.astro.build) and published
on Vercel. Content is maintained in **English and Turkish** with full parity.

```bash
npm install
npm run dev      # local preview
npm run build    # static build into dist/
```

### Presenter mode

Every module page carries presenter notes — what to say, what to watch for, what to do when the
demo misbehaves, and the timing for the slot. They are hidden by default and excluded from the
site search index, so the room never sees them.

| | |
|---|---|
| turn on | append `?presenter=1` to any page URL, or press `Alt+Shift+P` |
| turn off | `?presenter=0`, or `Alt+Shift+P` again |
| while on | a `PRESENTER MODE · Alt+Shift+P` badge sits in the corner |
| remembered | in `localStorage` under `rag-training-presenter`, for the rest of the day |

Turn it on once on the trainer's laptop before the room fills. There is no visible control that
would invite a participant to find it, and the state is set on `<html>` before the body renders,
so the notes never flash on a normal page load. The implementation is the presenter-mode script
in `astro.config.mjs`, with the styling in `src/styles/custom.css`.

Because this README is not open on the training-room laptop at 09:00, the same two lines are
repeated at the top of `handout/handout-tr.md`, in a trainer-only block that is cut before the
participant copies are printed. Keep the two in step.

## Layout

| Path | What is in it |
|---|---|
| `src/content/docs/` | The course site — English at the root, Turkish under `tr/` |
| `notebooks/` | The nine notebooks participants run, as percent-format `.py` plus generated `.ipynb`, the recorded runs in `cached_runs.json`, and the `kraken-q2` build inputs (`kraken_qa_q2.jsonl`, `kraken-q2.Modelfile`) |
| `exercises/` | The three measurements that are one command rather than a notebook: modules 6, 7 and 9 |
| `corpus/` | The synthetic Kraken Air corpus, in a `2026-Q2` and a `2026-Q3` edition, with the delta between them |
| `eval/` | The 20 gold questions, the deterministic retrieval metrics, the benchmark runner and `RESULTS.md` |
| `scripts/` | Corpus generation (`make_q2.py`, `make_finetune_dataset.py`), the offline asset seeder, the notebook builder, and `verify_setup.py` / `verify_setup.ps1` |
| `handout/` | The printed Turkish handout and the pre-work email |
| `docker-compose.yml` | The ChromaDB server module 8 puts behind a port |

## License

MIT — see [LICENSE](./LICENSE).

---

**Kraken Air (XX) and Wyvern Overseas Airways (YY) are fictional airlines.** Every document, fare
rule, flight number and procedure in this repository is synthetic and written for teaching.
**No Amadeus system, customer or production data is used anywhere in this repository.**
