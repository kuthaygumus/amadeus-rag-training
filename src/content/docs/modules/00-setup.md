---
title: "0. Setup — Before You Arrive"
description: "Install Ollama, pull three models, seed two offline files, run one script. Half an hour, the evening before — and not on the office network in the morning."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

There is no gate question for this module. This is the pre-work, done the evening before, on a network you are not sharing with twenty other people.

## The thing that already failed

The normal way to build a course like this is `pip install transformers`, pull weights from a model hub, and go. That route was tried while this course was being built, and abandoned. It costs a multi-gigabyte Python stack on every laptop before a single question is answered, and it leaves the day's numbers depending on which version of which library each machine happened to resolve. Ollama costs one download and one process.

That decided the shape of the whole day. Every model in this course is pulled through Ollama and runs on your own machine. No API key, no cloud account, no sign-in, nothing to expense — and the numbers you produce in the exercises are the numbers on the results page, because nobody silently upgraded a hosted model underneath you between now and October.

## What Ollama actually is

It is a local model server, not a framework. You give it a model name, it downloads a quantized copy of the weights, and it listens on `http://localhost:11434` with a small HTTP API: `/api/tags` lists what you have, `/api/embed` turns text into vectors, `/api/chat` answers. The notebooks call it with `urllib` from the standard library. There is no SDK to install and nothing to import that could go looking for the internet at the wrong moment.

That is the whole reason it is here: one download, one process, and a model that answers the same on your laptop in October as it did on the machine that produced the numbers on this site.

<div class="presenter-note">
Say this once and do not garble it: <strong>everything runs on the laptop in front of you, and that is a measurement decision, not an ideology.</strong> Someone in the room will ask "why not use an API?" in the first ten minutes. The answer is not "local is better" — it is "a pinned local model gives us the same number in September and in October, so when a number moves we know what moved it. And nobody needs an account, a key or an approval to sit down."
</div>

## Install Ollama

**macOS.** Download from [ollama.com](https://ollama.com), drag the app to Applications, launch it once. You get a menu-bar icon and a running server. Confirm in a terminal:

```bash
ollama --version
```

**Windows, without administrator rights.** `OllamaSetup.exe` is a per-user installer. It puts the binaries under `%LOCALAPPDATA%\Programs\Ollama` and does not need elevation. The server starts with your session.

`UNVERIFIED: the per-user install path is how the installer is built, not something confirmed on a managed Windows laptop.` If the installer asks you for administrator credentials, stop. Do not fight it, do not try to talk IT into it at 09:00 on the day — send a message the evening before and you will be paired with someone whose machine is green.

The installer adds Ollama to your user PATH, and a terminal window that was already open does not pick that up. If `ollama --version` says the command is not recognised, close that window and open a new one before concluding anything.

## Pull three models

```bash
ollama pull qwen2.5:3b          # generation and reranking
ollama pull bge-m3              # embeddings, and it handles Turkish
ollama pull nomic-embed-text    # the weaker embedder modules 6 and 9 measure against
```

Sizes as `ollama list` reports them: 1.9 GB, 1.2 GB and 274 MB — about 3.4 GB together.

Pull all three. `nomic-embed-text` is not an alternative to `bge-m3`, it is the control: module 6 puts the two side by side, and module 9's notebook loads both. Pulling a model in the room on the day is the single thing this page exists to prevent.

Do this at home, or at least not on the office wifi on the morning of the course. 3.4 GB is a quiet evening for one laptop. Fifteen laptops starting the same pull at 09:00 are not each getting the link — they are dividing it.

The choices were measured, not picked from a blog post. On three grounded questions put to the same fare table, `qwen2.5:3b` answered 3 of 3 at 0.9 s each and nothing else tried matched it. `llama3.2` answered EUR 70 to the Turkish question where the table's K row says EUR 90, with that table in its context — it is out of this course, so do not substitute it. `bge-m3` is here because on structure-aware chunks it scores hit@1 0.667 on the six Turkish-query/English-document questions, where `nomic-embed-text` scores 0.000.

### The fine-tuned model

Module 3 uses `helios-q2`, a fine-tune trained on the Q2 edition of the Helios corpus. It is not on any registry, so `ollama pull helios-q2` will not find it — it is built once on a GPU elsewhere and handed out on a USB stick before the day. Ask for it if you want module 3 to run live on your own machine; installing it is the same copy-and-merge described in [Where the files actually live](#where-the-files-actually-live).

If you do not have it, nothing else in the day changes, and it is worth knowing exactly what you lose. The two probe cells that ask the fine-tune a question print `(skipped — helios-q2 not installed)` and move on. Everything around them still runs: the cells that read the Q2 and Q3 fare sheets off disk and print the two `| K |` rows side by side need no model at all, and that diff is what module 3 is actually about. `verify_setup.py` reports the missing model as a warning, never a failure.

## Python and the repo

Python 3.10 or newer — the check script uses `dict | None` type syntax, so an older Python fails on line one rather than halfway through the day.

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt
```

**No git?** You do not need it. Download [the ZIP](https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip), extract it, and `cd` into the extracted folder. Everything after that is identical.

**`python -m pip`, not bare `pip`.** On Windows the two can resolve to different interpreters, and when they do the packages land where the notebooks cannot see them — a `ModuleNotFoundError` on the day, on a laptop that showed READY the night before. If your shell only knows the `py` launcher, use `py -m pip install -r requirements.txt` and `py scripts/verify_setup.py` throughout.

Two packages. `numpy`, which module 2 trains a network with, and `chromadb`, the vector database module 8 uses. With their dependencies they measure roughly 400 MB once installed. Do it at home, not on the office network alongside nineteen other people doing the same thing.

It has to be the full `chromadb` and not `chromadb-client`, and the reason is the whole of module 6. The thin client cannot embed text locally, so it raises an exception instead of quietly reaching for the wrong model. A loud error would be a kindness. What actually happens with the full package is the thing worth seeing.

## Seed the two files that are neither models nor packages

```bash
python scripts/seed_offline_assets.py
```

Two things the notebooks read are downloads that nothing else in the setup fetches, and both would otherwise land in the middle of a module:

- **MNIST**, measured at 11.6 MB across four archives, into `notebooks/mnist_data/`. Module 2 trains its network on it. The folder is gitignored, so a fresh clone does not have it.
- **Chroma's default embedder.** Module 8 opens a collection without ever saying how to turn text into vectors, and Chroma picks one for you. The first call that needs it pulls an `all-MiniLM-L6-v2` ONNX archive — measured at 83 MB — into `~/.cache/chroma/onnx_models/`.

Run it after `pip install`, because the second half needs `chromadb` importable. It is safe to run twice: anything already on disk is reported and skipped. `verify_setup.py` checks both and reports NOT READY if either is missing, because a missing file here is a live download in the room.

## How you open a notebook

There is no notebook server in this course and nothing to install beyond an editor. Every notebook exists twice: as `notebooks/NN_name.ipynb`, and as `notebooks/NN_name.py` in percent format — the same cells, with `# %%` markers instead of JSON. The `.py` file is the one you run.

**Install VS Code and the Python extension.** [code.visualstudio.com](https://code.visualstudio.com). On Windows take the **User Installer**: it installs into your own profile and asks for no administrator rights. Then open the Extensions view (`Ctrl+Shift+X`, or `Cmd+Shift+X` on macOS), search for **Python**, and install the one published by Microsoft.

**Open the `notebooks` folder, not the repository root.** `File → Open Folder →` `amadeus-rag-training/notebooks`. The notebooks read `../corpus` and `../eval`, so they have to run with `notebooks` as the working directory.

**Pick the interpreter.** `Ctrl+Shift+P` / `Cmd+Shift+P` → `Python: Select Interpreter` → choose the same Python 3.10+ you installed the packages into. It then shows in the status bar at the bottom right.

**Run a block.** Put the cursor inside a block and press `Shift+Enter`. The Python extension sends that block to a Python terminal at the bottom of the window and moves the cursor to the next one. State carries between presses — variables from one block are still there in the next — which is what makes this cell-by-cell rather than a script run.

**What "it worked" looks like.** Open `notebooks/00_bare_llm_fails.py`, put the cursor on a line like this and press `Shift+Enter`:

```python
import os, sys; print(sys.version); print(os.getcwd())
```

A terminal opens at the bottom and prints a version of 3.10 or higher and a path ending in `notebooks`. If the path does not end in `notebooks`, the relative paths in every notebook will miss; fix it once in that terminal with `import os; os.chdir(r"<full path to the notebooks folder>")`.

### When a cell will not run on your machine

Ollama stopped, a model you never pulled, a laptop too slow to sit through a long cell — the measurement cells can be replayed instead of recomputed. Set `USE_CACHED=1` in the environment before the notebook's first block:

```bash
USE_CACHED=1 python 05_chunking_and_noise.py
```

`notebooks/_cached.py` then reads that cell's result out of `notebooks/cached_runs.json` — an ordinary JSON file you can open in an editor — instead of calling the model. It is never silent: every replayed cell prints a `[CACHED]` banner naming the date it was recorded, the machine, and whether that recording was a full or a reduced run. Notebooks 03 through 08 have recorded cells. Notebook 02 does not, which is why a missing `helios-q2` skips its probes rather than replaying them. This keeps a session moving; it is not a way to run the whole day offline.

## Do you need Podman?

Not to run the notebooks. Notebooks 00 through 07 use Ollama, Python and nothing else, and they will run on a machine with no container runtime installed at all.

Module 8 is the exception, and it is worth doing yourself if you can. It runs ChromaDB as a service rather than as a library:

```bash
podman compose up -d          # or: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

Podman is not native on macOS or Windows — it runs a small Linux VM underneath, which on Windows means WSL2. That needs administrator rights and a reboot. If your machine will not have it by the day, nothing else breaks: watch that module instead and run it later. Everything measured in this course was measured without it.

The compose file lives in the repository root. Same advice as the models: pull the image at home.

## What you run

No notebook tonight. One script, `scripts/verify_setup.py`. It talks only to `localhost`, and it asks `urllib` not to route those calls through a system proxy, so it behaves the same at the office, at home, or on a plane.

```bash
python scripts/verify_setup.py
```

**What you should see.** A column of `ok` lines, then the word **READY** on a line of its own.

**How long.** A second or two once Ollama has the model in memory. The first run after a reboot takes longer while it loads.

**Windows, if Python is not installed at all.** The real check cannot start without it, so run the PowerShell bootstrap first. It installs nothing and tells you what is missing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

If that answers `running scripts is disabled on this system`, the execution policy on that machine is set by Group Policy and the `-ExecutionPolicy` switch cannot override it. You do not need administrator rights to get past it — pipe the file in instead, because text arriving on the pipeline is not a script file:

```powershell
Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
```

`UNVERIFIED: the PowerShell script has not yet been run on a managed Windows laptop. Its header comment lists what to test first.`

The Python script checks the whole chain and prints one word.

**READY** means: Python is 3.10+, `numpy` and `chromadb` import, MNIST and Chroma's default embedder are already on disk, Ollama answered on port 11434, all three model tags are present, an embedding call came back with real vectors, and a generation call read the right column of a table. Close the terminal and forget about it.

**NOT READY** means it tells you exactly which line failed and the command that fixes it. Read the arrow, run the command, run the script again.

Some lines print `warn` rather than `ok` or `FAIL` — VS Code not found in the usual places, `helios-q2` not installed. A warning never turns READY into NOT READY. It is something worth knowing, not something that stops the day.

Two details in that script are deliberate. It matches models on the **full tag**, so holding `qwen2.5:1.5b` does not satisfy the requirement for `qwen2.5:3b`; telling someone they are ready when they have the weaker model is the worst outcome available. And the last check is a correctness test, not a smoke test: the model is shown a two-penalty table and asked for the cancellation penalty on class K. It has to answer 90, not 70. The check reads the first number in the reply rather than scanning the whole of it, because 70 is also in the table and "EUR 90 (the change penalty is EUR 70)" is a correct answer. A model that answers 70 makes the chunking exercise in Module 7 look broken when it is working perfectly.

The script also times your machine and says so: under 3 seconds per answer is comfortable, 3 to 8 seconds is usable but exercises will feel slower than the demo, over 8 seconds is slow. If you get "slow", pull the fallback as well:

```bash
ollama pull qwen2.5:1.5b
```

That is another 986 MB. Be honest with yourself about what it costs: the 1.5B model scored 2 of 3 on the same test questions and picked the wrong table row on the multi-hop one. If you run the day on it you will hit one wrong answer that is the model's fault and not the pipeline's. Know which one it is instead of debugging it.

## Where the files actually live

The weights are ordinary files on disk, and they are portable.

| OS | Path |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

Inside are `blobs/` and `manifests/`. To move models between machines — the offline fallback for anyone whose home download gave up, and the only way to get `helios-q2`:

1. **Quit Ollama completely on both machines first.** macOS: the menu-bar icon → Quit. Windows: the system-tray icon → Quit Ollama. On Windows the server starts with your session and holds those files open.
2. **Merge, do not replace.** Copy the *contents* of `blobs/` and `manifests/` into the folders of the same name on the target machine. Copying the whole `models` directory over the top removes the `manifests/` entries of any model that machine already had, which silently unregisters it.
3. Start Ollama again and run `ollama list`. Everything that came across shows up.

<div class="presenter-note">
<strong>09:10, ten minutes.</strong> Everybody runs <code>verify_setup.py</code> at once, and you walk the room reading screens — not asking, reading. Before they hit enter, ask the room to guess: "how many of us are green?" Get a number out loud. Two things happen: people commit, and the ones who never ran it the night before out themselves.
<br /><br />
<strong>Red laptop triage, in this order.</strong> Ollama not running → open the app. <code>ollama</code> not recognised but the script says the server answers → they have a stale terminal window, tell them to open a new one. Model missing but the machine is fast → start the pull now, it will finish during Module 1. Model missing and the network is crawling → USB stick, you carry two. Install blocked by admin rights → stop, pair them immediately, do not spend the room's morning on it.
<br /><br />
<strong>Pairing is the fallback, and it is a fine one.</strong> Say it out loud so nobody feels punished: "one laptop between two people is the normal way to do this — one drives, one reads the output and argues." Do not let a red laptop become a person sitting quietly for six hours.
<br /><br />
<strong>Ten minutes means ten minutes.</strong> Anything not green by 09:20 is a pairing, not a repair. You get the time back nowhere else in the day.
</div>

## What the numbers said

<div class="measured">

| What the pre-work puts on disk | Size |
|---|---|
| `qwen2.5:3b` | 1.9 GB |
| `bge-m3` | 1.2 GB |
| `nomic-embed-text` | 274 MB |
| `numpy` + `chromadb` and their dependencies | about 400 MB |
| Chroma's `all-MiniLM-L6-v2` ONNX archive | 83 MB |
| MNIST, four archives | 11.6 MB |
| **Total** | **about 3.9 GB** |
| `qwen2.5:1.5b`, only if the script says your machine is slow | +986 MB |

Model sizes are what `ollama list` reports for the installed tags. The Python figure is what the packages occupy once installed, so the bytes crossing the network are fewer than that row suggests.

| Generation model | Correct | Average per answer |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `qwen2.5:1.5b` | 2/3 — answered EUR 90 where the M row says EUR 120 | 0.6 s |
| `gemma3:4b` | 2/3 | 2.1 s |
| `qwen3:4b` | 2/3 — empty answer after its reasoning tokens | 10.0 s |
| `llama3.2` | 1/3 — answered EUR 70 where the K row says EUR 90 | 1.0 s |

Three grounded questions against the same fare table, temperature 0, one M-series Mac. Three questions is a smoke test, not a benchmark: enough to reject a model, not enough to rank the ones that survive.

| Embedder | hit@1 on the six Turkish-query / English-document questions |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

Both embedders scored over the same structure-aware chunks, which is the condition module 6 runs. The gap is a property of that measurement, not a fixed property of either model.

</div>

## Going deeper

Ollama serves quantized GGUF weights, which is why a 3-billion-parameter model is a ~2 GB download and runs on a laptop with no GPU. Quantization stores each weight in roughly 4 bits instead of 16, at a small loss of quality. That loss stays invisible on the tasks in this course; it would not stay invisible on a long chain of reasoning, which is one reason the day never asks a 3B model to do anything clever in a single call.

The embedder choice is the one that will bite you outside this room, because it fails silently. `nomic-embed-text` and `all-MiniLM-L6-v2` are English-first models. Given a Turkish query and the English document that answers it, they do not error and they do not return nothing — they return a confidently wrong document. `bge-m3` is trained multilingually, so a Turkish sentence and its English equivalent land near each other in the vector space. On structure-aware chunks the gap is 0.000 against 0.667 on the same six questions, and you cannot see it by reading code. Only by scoring it.

Running everything locally is not ideology, it is measurement. With a hosted endpoint, the number you got in September and the number you get in October are not comparable, and you never learn which change caused what. A pinned local model turns the day into a controlled experiment, which is worth more here than the extra quality a frontier model would bring.

At ten million documents most of this setup does not survive, and it is worth knowing which part breaks first. Generation stays roughly as it is; you still answer one query at a time. Embedding does not — ten million chunks on a laptop is a multi-week job, so it moves to batched GPU inference or a hosted embedding endpoint, and the vectors move into a real vector database instead of an in-process one. The part that hurts is that the embedder becomes a migration cost: changing it means re-embedding the entire corpus. The choice you make casually on day one is the one that is most expensive to reverse, which is a good argument for measuring it on twenty questions now rather than on ten million documents later.

## Exit line

> Everything is installed and nothing is connected to anything yet. Tomorrow starts with the model completely on its own — `notebooks/00_bare_llm_fails.py`, and one question about a Helios Air fare rule it has never seen.
