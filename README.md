# RAG Training Day

A one-day, hands-on RAG course for working developers — built to run **entirely offline** on a
locked-down corporate laptop. No API key, no cloud account, no sign-in.

> **Helios Air is a fictional airline.** Every document, fare rule, flight number and procedure
> in this repository is synthetic and written for teaching. **No Amadeus system, customer or
> production data is used anywhere in this repository.**

## Why this exists

Most RAG material teaches the happy path: chunk, embed, retrieve, done. That version collapses
the first time it meets a real corpus. This course is organised around a single rule:

> **No concept is introduced before you have seen the failure that requires it.**

Every module opens where the previous one hit a wall.

| # | What you have | What breaks on screen | What you now need |
|---|---|---|---|
| 1 | A bare LLM | Invents a fare rule, confidently, with no source | Get my own data in |
| 2 | A trained network | Weights fit the data — a weight is a frozen photograph | Put *my* data in the weights |
| 3 | A fine-tuned model | Right for last quarter, wrong for this one, cites nothing | Fresh knowledge without retraining |
| 4 | The whole corpus in the prompt | It fits — and you pay for all of it on every query | Select only the right piece |
| 5 | Keyword search (BM25) | Misses the paraphrase | Match meaning, not words |
| 6 | Naive RAG | Retrieval returns garbage, the answer is confidently wrong | Understand *why* |
| 7 | Better chunking + the right embedder | The exact term still slips through | Lexical + ranking |
| 8 | Hybrid + rerank + contextual | Multi-hop still fails in one shot | Retrieval inside a loop |
| 9 | Agentic RAG | — | — |

## Everything is measured

The day is not a sequence of demos. A fixed set of 20 gold questions is scored with the same
three metrics — `hit@1`, `recall@5`, `MRR` — at three separate points in the day. Participants
watch the numbers move and say *how much* better, not "better".

Results that contradict the usual advice are kept in, not smoothed over. Combining a dense
retriever with BM25 does not always beat either alone; this course shows the run where fusion
made things worse, and explains the condition under which it helps.

## Running it

Requires [Ollama](https://ollama.com) and Python. Two models, ~3.1 GB total, pulled once:

```bash
ollama pull qwen2.5:3b     # generation
ollama pull bge-m3         # multilingual embeddings
```

Then work through the notebooks in `notebooks/` in order. They run with no network access.

## This site

The course site is built with [Astro Starlight](https://starlight.astro.build) and published
on Vercel. Content is maintained in **English and Turkish** with full parity.

```bash
npm install
npm run dev      # local preview
npm run build    # static build into dist/
```

## Layout

| Path | What is in it |
|---|---|
| `src/content/docs/` | The course site — English at the root, Turkish under `tr/` |
| `notebooks/` | The notebooks participants run, in module order |
| `corpus/` | The synthetic Helios Air corpus, in a `2026-Q2` and a `2026-Q3` edition |
| `eval/` | The 20 gold questions and the deterministic retrieval metrics |
| `scripts/` | Corpus generation, environment preflight, setup verification |
| `handout/` | Source for the printed Turkish handout |

## License

MIT — see [LICENSE](./LICENSE).
