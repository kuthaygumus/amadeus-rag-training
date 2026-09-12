# RAG Training Day

The course site for a one-day RAG training for working developers, built to run **entirely local**
on a laptop with no administrator rights. No API key, no cloud account, no sign-in, no Python on
the participant's machine.

This repository is the **site** — theory, narrative, presenter notes, in English and Turkish. The
**practical part lives in a second repository,
[`amadeus-rag-lab`](https://github.com/kuthaygumus/amadeus-rag-lab)**: a small TypeScript/Express
service plus ChromaDB in two containers, and a Bruno collection that is the day's handout.
Participants install Ollama, Podman Desktop and Bruno, clone the lab, `podman compose up`, and
click through six folders — one gate each.

## Why this exists

Most RAG material teaches the happy path: chunk, embed, retrieve, done. That version collapses
the first time it meets a real corpus. This course is organised around a single rule:

> **No concept is introduced before you have seen the failure that requires it.**

| # | What you have | What breaks on screen | What you now need |
|---|---|---|---|
| 0 | A laptop | *(setup, not a gate)* | Ollama, Podman Desktop, Bruno, two models, one `git clone` |
| 1 | A bare LLM | Answers about our airline, confidently, with no source — wrong | Where *is* knowledge in a model? |
| 2 | A trained network | A weight is a frozen photograph of the training data | Re-take the photograph with *our* data |
| 3 | A fine-tuned model | Right for last quarter, wrong for this one, cites nothing | Fresh knowledge without retraining |
| 4 | The whole corpus in the prompt | It fits — and every question costs the whole rule book, in tokens and seconds | Select only the right piece |
| 5 | Simple RAG | The answer is grounded; the retrieved slice is the wrong one | See what the store actually holds |
| 6 | ChromaDB in a container | The chunk with the number has lost its table header | Cut the documents differently |
| 7 | Structure-aware chunking, noise stripped | Same model, same store, right answer — a splitter won | Prove it beats fine-tuning where it matters |
| 8 | Two editions of the corpus | Q2 → 120, Q3 → 90, with a re-ingest and no retraining | — |
| 9 | Closing | Rewind the chain | What the day left out: `reference/going-further` |

Modules 2 and 3 are trainer-driven demos (MNIST and a LoRA fine-tune, run in Colab before the day).
Everything else the room runs itself, in Bruno.

## The corpus

A synthetic airline rule book: **Kraken Air (XX)**, with **Wyvern Overseas Airways (YY)** as the
partner carrier in the interline and codeshare documents. It ships in two editions —
`corpus/2026-Q2` and `corpus/2026-Q3` — that differ by a named delta recorded in
`corpus/DELTA.md`. `corpus/` here is the source of truth; the lab repository carries a verbatim
copy. The question the whole day turns on is the CLASSIC short-haul booking-class-K cancellation
penalty: **EUR 120 in Q2, EUR 90 in Q3**.

## Building `kraken-q2`

Module 3 needs a model that learned the **2026-Q2** rule book and nothing about Q3. It is on no
registry; it is built once, on a GPU in Colab, before the day:

```bash
python scripts/make_q2.py                 # derive corpus/2026-Q2 from 2026-Q3 by a named delta
python scripts/make_finetune_dataset.py   # walk Q2 into notebooks/kraken_qa_q2.jsonl
# notebooks/02_finetune_qwen_lora.ipynb in Colab: LoRA on Qwen2.5-1.5B-Instruct, merge, GGUF, quantise
ollama create kraken-q2 -f notebooks/kraken-q2.Modelfile
```

`notebooks/kraken-q2.Modelfile` pins greedy decoding, because the module rests on the answer being
the same every run. `notebooks/01_mnist_tiny_net.ipynb` is module 2's demo.

## This site

Built with [Astro Starlight](https://starlight.astro.build) and published on Vercel at
[amadeus-rag-training.vercel.app](https://amadeus-rag-training.vercel.app). Content is maintained in
**English and Turkish** with full parity.

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

Turn it on once on the trainer's laptop before the room fills. The state is set on `<html>` before
the body renders, so the notes never flash on a normal page load. The implementation is the
presenter-mode script in `astro.config.mjs`, with the styling in `src/styles/custom.css`.

## Layout

| Path | What is in it |
|---|---|
| `src/content/docs/` | The course site — English at the root, Turkish under `tr/` |
| `corpus/` | The synthetic Kraken Air corpus, in a `2026-Q2` and a `2026-Q3` edition, with the delta between them |
| `notebooks/` | The two trainer demos (`01_mnist_tiny_net`, `02_finetune_qwen_lora`) and the `kraken-q2` build inputs (`kraken_qa_q2.jsonl`, `kraken-q2.Modelfile`) |
| `scripts/` | Corpus generation: `make_q2.py`, `make_finetune_dataset.py` |
| `handout/` | The pre-work email and the printed Turkish handout |

## License

MIT — see [LICENSE](./LICENSE).

---

**Kraken Air (XX) and Wyvern Overseas Airways (YY) are fictional airlines.** Every document, fare
rule, flight number and procedure in this repository is synthetic and written for teaching.
**No Amadeus system, customer or production data is used anywhere in this repository.**
