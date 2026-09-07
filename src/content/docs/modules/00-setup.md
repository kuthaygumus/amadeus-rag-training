---
title: "0. Setup — Before You Arrive"
description: "Install Ollama, pull two models, run one script. Ten minutes, the evening before — and not on the office network in the morning."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

There is no gate question for this module. This is the pre-work, done the evening before, on a network you are not sharing with twenty other people.

## The thing that already failed

The normal way to build a course like this is `pip install transformers`, pull a model from HuggingFace, and go. That was tried on the corporate network here. It does not work. Model weight downloads from HuggingFace are policy-blocked — not slow, not throttled, refused.

The block is pattern-based, not size-based. A 17 KB weight file is refused exactly like a 4 GB one. So the obvious workaround — "use a tiny model instead" — is not a workaround.

`registry.ollama.ai` is not blocked. Measured on the same laptop, on the same network, on the same afternoon: a 1.16 GB model blob came down at 2.36 MB/s and completed.

Those two facts decided the shape of the whole day. Every model in this course is pulled from Ollama and runs on your own machine. No API key, no cloud account, no sign-in, nothing to expense — and the numbers you produce in the exercises are the numbers on the results page, because nobody silently upgraded a hosted model underneath you between now and October.

## What Ollama actually is

It is a local model server, not a framework. You give it a model name, it downloads a quantized copy of the weights, and it listens on `http://localhost:11434` with a small HTTP API: `/api/tags` lists what you have, `/api/embed` turns text into vectors, `/api/chat` answers. The notebooks call it with `urllib` from the standard library. There is no SDK to install and nothing to import that could go looking for the internet at the wrong moment.

That is the whole reason it is here: the one piece of infrastructure that survives the network and keeps the day reproducible.

<div class="presenter-note">
Say this once and do not garble it: <strong>HuggingFace weights are blocked here, Ollama's registry is not, and that is the only reason we are using Ollama.</strong> Someone in the room will ask "why not just use an API?" in the first ten minutes. The answer is not "local is better" — it is "the network already decided this, and I measured it."
</div>

## Install Ollama

**macOS.** Download from [ollama.com](https://ollama.com), drag the app to Applications, launch it once. You get a menu-bar icon and a running server. Confirm in a terminal:

```bash
ollama --version
```

**Windows, without administrator rights.** `OllamaSetup.exe` is a per-user installer. It puts the binaries under `%LOCALAPPDATA%\Programs\Ollama` and does not need elevation. The server starts with your session.

`UNVERIFIED: this has not yet been tested on an Amadeus Windows image. The per-user install path is how the installer is built, not something measured here.` If the installer asks you for administrator credentials, stop. Do not fight it, do not try to talk IT into it at 09:00 on the day — send a message the evening before and you will be paired with someone whose machine is green.

## Pull two models

```bash
ollama pull qwen2.5:3b     # generation and reranking
ollama pull bge-m3         # embeddings, and it handles Turkish
```

About 3.1 GB together. Do this at home, or at least not on the office wifi on the morning of the course. At the 2.36 MB/s that was measured on this network, 3.1 GB is roughly twenty-two minutes for one laptop — and if fifteen people start pulling at 09:00 you are not sharing 2.36 MB/s, you are dividing it.

Both choices were measured, not picked from a blog post. `qwen2.5:3b` answered 3 of 3 test questions correctly at 0.9 s each. `llama3.2:3b`, with the table header **in** its context, still answered EUR 70 where the answer is EUR 90 — it is banned from this course, so do not substitute it. `bge-m3` is here because on the six Turkish-query/English-document questions it scores 0.667 where `nomic-embed-text` scores 0.000.

## Python and the repo

Python 3.10 or newer — the check script uses `dict | None` type syntax, so an older Python fails on line one rather than halfway through the day.

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
pip install -r requirements.txt
python scripts/verify_setup.py
```

Two packages. `numpy`, which module 2 trains a network with, and `chromadb`, the vector database modules 5 and 8 use. Chromadb pulls about 400 MB of wheels, most of it onnxruntime — do it at home, not on the office network alongside nineteen other people doing the same thing.

It has to be the full `chromadb` and not `chromadb-client`, and the reason is the whole of module 6. The thin client cannot embed text locally, so it raises an exception instead of quietly reaching for the wrong model. A loud error would be a kindness. What actually happens with the full package is the thing worth seeing.

## Do you need Podman?

Not to run the notebooks. Notebooks 00 through 07 use Ollama, Python and nothing else, and they will run on a machine with no container runtime installed at all.

Module 8 is the exception, and it is worth doing yourself if you can. It runs ChromaDB as a service rather than as a library:

```bash
podman compose up -d          # or: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

Podman is not native on macOS or Windows — it runs a small Linux VM underneath, which on Windows means WSL2. That needs administrator rights and a reboot. If your machine will not have it by the day, nothing else breaks: watch that module instead and run it later. Everything measured in this course was measured without it.

The compose file lives in the repository root and the container image is about 650 MB. Same advice: pull it at home.

## What you run

No notebook tonight. One script, `scripts/verify_setup.py`. It talks only to `localhost`, so it behaves the same at the office, at home, or on a plane.

```bash
python scripts/verify_setup.py
```

It checks five things and prints one word.

**READY** means: Python is 3.10+, Ollama answered on port 11434, both model tags are present, an embedding call came back with real vectors, and a generation call read the right column of a table. Close the terminal and forget about it.

**NOT READY** means it tells you exactly which line failed and the command that fixes it. Read the arrow, run the command, run the script again.

Two details in that script are deliberate. It matches models on the **full tag**, so holding `qwen2.5:1.5b` does not satisfy the requirement for `qwen2.5:3b`; telling someone they are ready when they have the weaker model is the worst outcome available. And the last check is a correctness test, not a smoke test: the model is shown a two-penalty table and asked for the cancellation penalty on class K. It has to answer 90, not 70. A model that answers 70 makes the chunking exercise in Module 7 look broken when it is working perfectly.

The script also times your machine and says so: under 3 seconds per answer is comfortable, 3 to 8 seconds is usable but exercises will feel slower than the demo, over 8 seconds is slow. If you get "slow", pull the fallback as well:

```bash
ollama pull qwen2.5:1.5b
```

Be honest with yourself about what that costs. The 1.5B model scored 2 of 3 on the same test questions and picked the wrong table row on the multi-hop one. If you run the day on it you will hit one wrong answer that is the model's fault and not the pipeline's. Know which one it is instead of debugging it.

## Where the files actually live

The weights are ordinary files on disk, and they are portable.

| OS | Path |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

Inside are `blobs/` and `manifests/`. Copy the whole `models` directory onto a USB stick from a machine that has the models, drop it in the same place on a machine that does not, restart Ollama, and `ollama list` shows them. That is the offline fallback for anyone whose home connection gave up at 3 GB.

<div class="presenter-note">
<strong>09:10, ten minutes.</strong> Everybody runs <code>verify_setup.py</code> at once, and you walk the room reading screens — not asking, reading. Before they hit enter, ask the room to guess: "how many of us are green?" Get a number out loud. Two things happen: people commit, and the ones who never ran it the night before out themselves.
<br /><br />
<strong>Red laptop triage, in this order.</strong> Ollama not running → open the app. Model missing but the machine is fast → start the pull now, it will finish during Module 1. Model missing and the network is crawling → USB stick, you carry two. Install blocked by admin rights → stop, pair them immediately, do not spend the room's morning on it.
<br /><br />
<strong>Pairing is the fallback, and it is a fine one.</strong> Say it out loud so nobody feels punished: "one laptop between two people is the normal way to do this — one drives, one reads the output and argues." Do not let a red laptop become a person sitting quietly for six hours.
</div>

## What the numbers said

<div class="measured">

| Measured on the corporate network | Result |
|---|---|
| HuggingFace model weights | policy-blocked — including a 17 KB weight file |
| `registry.ollama.ai`, 1.16 GB blob | downloaded at 2.36 MB/s |

| Generation model | Correct | Time per answer |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `gemma3:4b` | 2/3 | 1.9 s |
| `qwen2.5:1.5b` | 2/3 — wrong row on multi-hop | — |
| `qwen3:4b` | correct, but 11.6 s (reasoning tokens) | 11.6 s |
| `llama3.2:3b` | answered EUR 70 with the header in context | banned |

| Embedder | hit@1 on Turkish query / English document (6 questions) |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

</div>

## Going deeper

Ollama serves quantized GGUF weights, which is why a 3-billion-parameter model is a ~2 GB download and runs on a laptop with no GPU. Quantization stores each weight in roughly 4 bits instead of 16, at a small loss of quality. That loss stays invisible on the tasks in this course; it would not stay invisible on a long chain of reasoning, which is one reason the day never asks a 3B model to do anything clever in a single call.

The embedder choice is the one that will bite you outside this room, because it fails silently. `nomic-embed-text` and `all-MiniLM-L6-v2` are English-first models. Given a Turkish query and the English document that answers it, they do not error and they do not return nothing — they return a confidently wrong document. `bge-m3` is trained multilingually, so a Turkish sentence and its English equivalent land near each other in the vector space. The gap is 0.000 against 0.667 on the same six questions, and you cannot see it by reading code. Only by scoring it.

Running everything locally is not ideology, it is measurement. With a hosted endpoint, the number you got in September and the number you get in October are not comparable, and you never learn which change caused what. A pinned local model turns the day into a controlled experiment, which is worth more here than the extra quality a frontier model would bring.

At ten million documents most of this setup does not survive, and it is worth knowing which part breaks first. Generation stays roughly as it is; you still answer one query at a time. Embedding does not — ten million chunks on a laptop is a multi-week job, so it moves to batched GPU inference or a hosted embedding endpoint, and the vectors move into a real vector database instead of an in-process one. The part that hurts is that the embedder becomes a migration cost: changing it means re-embedding the entire corpus. The choice you make casually on day one is the one that is most expensive to reverse, which is a good argument for measuring it on twenty questions now rather than on ten million documents later.

## Exit line

> Everything is installed and nothing is connected to anything yet. Tomorrow starts with the model completely on its own, and one question about a Helios Air fare rule it has never seen.
