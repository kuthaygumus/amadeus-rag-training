---
title: "0. Setup — Before You Arrive"
description: "Install Ollama, pull three models, seed two offline files, run one script — half an hour the evening before. And the rule that makes every command on this site work: the repository root is where you stand."
---

There is no gate question for this module. This is the pre-work, done the evening before, on a network you are not sharing with twenty other people. About half an hour, nearly all of it downloading.

## Where every command on this site goes

Read this section once and nothing later in the day is a guess. There are exactly **two surfaces**, and the folder you are about to clone — the one that contains `corpus/`, `notebooks/`, `eval/` and `exercises/` — is the anchor of both.

1. **Terminal, at the repository root.** One terminal window, `cd`'d into that folder, left open for the day. Every `ollama …` command, and everything written as `python scripts/…`, `python exercises/…` or `python eval/…`, runs here.
2. **VS Code, with the repository root as the open folder.** The notebooks are percent-format `.py` files under `notebooks/`. Open one, put the cursor inside a `# %%` block, press `Shift+Enter`; the output appears in the Interactive window.

There is no Jupyter server, no browser notebook and no cloud console anywhere in this course. Every fenced command on every page of this site says in bold which surface it belongs to, immediately above the fence, so you never have to work it out.

**You are in the right terminal** when `ls` — `dir` on Windows — lists `corpus`, `notebooks`, `eval` and `exercises`. If a command answers `No such file or directory`, check that before anything else: it is the wrong window far more often than it is a broken install.

<div class="presenter-note">
Put this on the board at 09:10 and leave it there: <strong>two surfaces, one anchor — the repository root.</strong> The most common way a room loses ten minutes is one person running <code>python scripts/…</code> from inside <code>notebooks/</code> while everyone helps them debug an install that is fine. When a laptop says <code>No such file or directory</code>, ask "which folder is that terminal in?" before you ask anything else.
</div>

## Install Ollama

**macOS.** Download from [ollama.com](https://ollama.com), drag the app to Applications, launch it once. You get a menu-bar icon and a running server.

**Terminal (anywhere — the repository does not exist yet):**

```bash
ollama --version
```

**Windows, without administrator rights.** `OllamaSetup.exe` is a per-user installer: binaries under `%LOCALAPPDATA%\Programs\Ollama`, no elevation, server starts with your session.

`UNVERIFIED: the per-user install path is how the installer is built, not something confirmed on a managed Windows laptop.` If it asks for administrator credentials, stop — send a message the evening before and you will be paired with someone whose machine is green, rather than arguing with IT at 09:00 on the day.

It adds Ollama to your user PATH, and a terminal that was already open does not pick that up. If `ollama --version` says the command is not recognised, close that window and open a new one before concluding anything.

## Pull three models

**Terminal (anywhere):**

```bash
ollama pull qwen2.5:3b          # generation and reranking
ollama pull bge-m3              # embeddings, and it handles Turkish
ollama pull nomic-embed-text    # the weaker embedder modules 6 and 9 measure against
```

Sizes as `ollama list` reports them: 1.9 GB, 1.2 GB and 274 MB — about 3.4 GB together.

Pull all three. `nomic-embed-text` is not an alternative to `bge-m3`, it is the control: module 6 puts the two side by side, and module 9's notebook loads both. Pulling a model in the room on the day is the single thing this page exists to prevent. Fifteen laptops starting the same 3.4 GB pull at 09:00 are not each getting the link, they are dividing it. If your connection gives up, weights are ordinary files and can be copied from a machine that has them — see [Moving models between machines](#moving-models-between-machines).

**Do not substitute a generation model.** Every recorded number on this site was produced with `qwen2.5:3b`. Four others were tried on three grounded questions against the same fare table:

| Generation model | Correct | Average per answer |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `qwen2.5:1.5b` | 2/3 — answered EUR 90 where the M row says EUR 120 | 0.6 s |
| `llama3.2` | 1/3 — answered EUR 70 where the K row says EUR 90 | 1.0 s |

`UNVERIFIED: this three-question comparison has no reproducing script or recorded output in the repository. Treat it as the reason a model was chosen, not as a measurement. Everything inside "What the numbers said" below does have a source.`

The embedder choice is measured, and it is in the box below.

### The fine-tuned model

Module 3 uses `kraken-q2`, a fine-tune trained on the Q2 edition of the Kraken Air corpus. It is on no registry, so `ollama pull kraken-q2` will not find it: it is built once on a GPU elsewhere and handed out on a USB stick before the day.

`UNVERIFIED: kraken-q2 has not been built yet. If it does not exist by 7 October, module 3 runs on its corpus cells, which need no model at all.`

If you are handed the stick, it holds a `.gguf` and a Modelfile. Copy both into `notebooks/` and run the Modelfile's own build line. This is the one command on this site that is **not** run from the repository root, because `FROM ./kraken-q2.gguf` inside the Modelfile is a relative path:

**Terminal (in `notebooks/` — the single exception):**

```bash
ollama create kraken-q2 -f kraken-q2.Modelfile
```

Without it nothing else changes. The two probe cells print `(skipped — kraken-q2 not installed)` and move on; the cells that read the Q2 and Q3 fare sheets off disk and print the two `| K |` rows side by side need no model, and that diff is what module 3 is actually about. `verify_setup.py` reports the missing model as a warning, never a failure.

## Python and the repo

Python 3.10 or newer. The check script's first check is the Python version itself, so an older interpreter gets one clear NOT READY line and the command that fixes it, rather than a failure halfway through the day.

**Terminal (anywhere — this is where the repository root comes from):**

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt
```

That `cd` is the repository root. Everywhere this site says *repo root*, it means this folder.

**No git?** Download [the ZIP](https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip), extract it, and `cd` into the extracted folder. Everything after that is identical.

**`python -m pip`, not bare `pip`.** On Windows the two can resolve to different interpreters, and the packages then land where the notebooks cannot see them — a `ModuleNotFoundError` on the day, on a laptop that showed READY the night before. If your shell only knows the `py` launcher, use `py -m pip` and `py scripts/verify_setup.py` throughout.

Two packages: `numpy`, which module 2 trains a network with, and `chromadb`, the vector database module 8 uses. Measured in `requirements.txt`: 86 MB of wheels across 79 packages, unpacking to roughly 400 MB on disk — again, at home. It has to be the full `chromadb` and not `chromadb-client`: the thin client cannot embed text locally, so it raises instead of quietly reaching for the wrong model, and what the full package does instead is the whole of module 6.

## Seed the two files that are neither models nor packages

**Terminal (repo root):**

```bash
python scripts/seed_offline_assets.py
```

Two things the notebooks read are downloads nothing else in the setup fetches, and both would otherwise land in the middle of a module:

- **MNIST**, measured at 11.6 MB across four archives, into `notebooks/mnist_data/`. Module 2 trains its network on it. The folder is gitignored, so a fresh clone does not have it.
- **Chroma's default embedder.** Module 8 opens a collection without ever saying how to turn text into vectors, and Chroma picks one for you. The first call that needs it pulls an `all-MiniLM-L6-v2` ONNX archive — measured at 83 MB — into `~/.cache/chroma/onnx_models/`.

Run it after `pip install`, because the second half needs `chromadb` importable, and run it twice if you like: anything already on disk is reported and skipped. `verify_setup.py` reports NOT READY if either is missing, because a missing file here is a live download in the room.

## Open the notebooks in VS Code

Every notebook exists twice: as `notebooks/NN_name.ipynb`, and as `notebooks/NN_name.py` in percent format — the same cells, with `# %%` markers instead of JSON. **The `.py` file is the one you run.** The `.ipynb` twins carry no saved output; if you open one, VS Code offers to install Jupyter — say no, close it, and open the `.py` of the same name.

**Install VS Code and the Python extension.** [code.visualstudio.com](https://code.visualstudio.com). On Windows take the **User Installer**: it installs into your own profile and needs no administrator rights. Then open the Extensions view (`Ctrl+Shift+X`, or `Cmd+Shift+X` on macOS), search for **Python**, and install the one published by Microsoft.

**Open the repository root as the folder.** `File → Open Folder →` `amadeus-rag-training` — the folder holding `corpus/`, `notebooks/`, `eval/` and `exercises/`. Not `notebooks/`. VS Code's integrated terminal starts at the folder you opened, so opening the root with `` Ctrl+` `` gives you both surfaces in one window. The notebooks look after their own working directory: the first block of each calls `_preflight`, which moves to `notebooks/` so `../corpus` and `../eval` resolve, and prints the path when it does.

**Pick the interpreter.** `Ctrl+Shift+P` / `Cmd+Shift+P` → `Python: Select Interpreter` → the same Python 3.10+ you installed the packages into. It then shows in the status bar, bottom right.

**Run a block.** Cursor inside a block, `Shift+Enter`. State carries between presses, which is what makes this cell-by-cell rather than a script run.

**What "it worked" looks like.**

**VS Code — `notebooks/00_bare_llm_fails.py`, the first `# %%` block:**

```text
working directory set to .../amadeus-rag-training/notebooks
ready: models qwen2.5:3b
model: qwen2.5:3b
ready
```

The first line appears only if you were not already in `notebooks/`; its absence is fine. The last two are Ollama answering. If you get a block headed `NOT READY` instead, read it: `_preflight` names the missing thing and the command that fixes it.

### When a cell will not run on your machine

Ollama stopped, a model you never pulled, a laptop too slow to sit through a long cell — the measurement cells can be replayed instead of recomputed. Set `USE_CACHED=1` before the notebook's first block.

**Terminal (repo root):**

```bash
USE_CACHED=1 python notebooks/05_chunking_and_noise.py
```

On PowerShell, `$env:USE_CACHED=1` on its own line first, then the `python` line.

`notebooks/_cached.py` then reads that cell's result out of `notebooks/cached_runs.json` — an ordinary JSON file you can open in an editor — instead of calling the model. It is never silent: every replayed cell prints a `[CACHED]` banner naming the date it was recorded, the machine, and whether the recording was a full or a reduced run. Notebooks 03 through 08 have recorded cells; notebook 02 does not, which is why a missing `kraken-q2` skips its probes rather than replaying them. This keeps a session moving; it is not a way to run the whole day offline.

## What you run

No notebook tonight. One script. It talks only to `localhost`, and asks `urllib` not to route those calls through a system proxy, so it behaves the same at the office, at home, or on a plane.

**Terminal (repo root):**

```bash
python scripts/verify_setup.py
```

**What you should see.** A column of `ok` lines, then the word **READY** on a line of its own.

**How long.** A second or two once Ollama has the model in memory; longer on the first run after a reboot.

**READY** means: Python is 3.10+, `numpy` and `chromadb` import, MNIST and Chroma's default embedder are on disk, Ollama answered on port 11434, all three model tags are present, an embedding call came back with real vectors, and a generation call read the right column of a table. Close the terminal and forget about it.

**NOT READY** names the line that failed and the command that fixes it. Read the arrow, run the command, run the script again.

Some lines print `warn` rather than `ok` or `FAIL` — VS Code not found in the usual places, `kraken-q2` not installed. A warning never turns READY into NOT READY.

The script also times your machine: under 3 seconds per answer is comfortable, 3 to 8 usable but slower than the demo, over 8 slow. If you get "slow", pull the fallback as well:

**Terminal (repo root):**

```bash
ollama pull qwen2.5:1.5b
```

Another 986 MB, and it costs you an answer: on the three test questions above, the 1.5B model read the wrong row of the fare table. Know which wrong answer is the model's rather than debugging it as the pipeline's.

**Windows, if Python is not installed at all.** The real check cannot start without it, so run the bootstrap first. It installs nothing and names what is missing.

**PowerShell (repo root):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

If that answers `running scripts is disabled on this system`, Group Policy set the execution policy and `-ExecutionPolicy` cannot override it. No administrator rights are needed — pipe the file in, because text on the pipeline is not a script file:

**PowerShell (repo root):**

```powershell
Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
```

`UNVERIFIED: the PowerShell script has not yet been run on a managed Windows laptop. Its header comment lists what to test first.`

## Do you need Podman?

Not to run the notebooks: 00 through 07 use Ollama, Python and nothing else. Module 8 is the exception — it runs ChromaDB as a service rather than as a library — and if your machine will not have a container runtime by the day, nothing breaks: watch that module and run it later. Everything measured in this course was measured without it. If you do want it, pull the image at home.

**Terminal (repo root — the compose file is there):**

```bash
podman compose up -d          # or: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

## Moving models between machines

Weights are ordinary files on disk, and they are portable — the fallback for anyone whose home download gave up.

| OS | Path |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

1. **Quit Ollama completely on both machines first.** macOS: menu-bar icon → Quit. Windows: system-tray icon → Quit Ollama. On Windows the server starts with your session and holds those files open.
2. **Merge, do not replace.** Copy the *contents* of `blobs/` and `manifests/` into the folders of the same name on the target machine. Copying the whole `models` directory over the top removes the `manifests/` entries of any model that machine already had, which silently unregisters it.
3. Start Ollama again and run `ollama list`. Everything that came across shows up.

<div class="presenter-note">
<strong>09:10, ten minutes.</strong> Everybody runs <code>verify_setup.py</code> at once, and you walk the room reading screens — not asking, reading. Before they hit enter, ask the room to guess: "how many of us are green?" Get a number out loud. Two things happen: people commit, and the ones who never ran it the night before out themselves.
<br /><br />
<strong>Red laptop triage, in this order.</strong> Terminal in the wrong folder → <code>cd</code> to the repository root, and say the two-surface rule to the whole room while you do it. Ollama not running → open the app. <code>ollama</code> not recognised but the script says the server answers → stale terminal window, tell them to open a new one. Model missing but the machine is fast → start the pull now, it will finish during Module 1. Model missing and the network is crawling → USB stick, you carry two. Install blocked by admin rights → stop, pair them immediately, do not spend the room's morning on it.
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

Model sizes are what `ollama list` reports for the installed tags; the Python figure is what the packages occupy once installed, so fewer bytes cross the network than that row suggests. The same sum is computed in `scripts/verify_setup.py`, so the script and this page cannot drift apart.

| Embedder | hit@1 on the six Turkish-query / English-document questions |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

Both embedders scored over the same structure-aware chunks, which is the condition module 6 runs. Six questions is a direction, not a digit — but 0.000 against 0.667 is not a rounding argument. The gap is a property of that measurement, not a fixed property of either model.

</div>

## Why Ollama and not a model hub

The normal way to build a course like this is `pip install transformers`, pull weights from a model hub, and go. That route was tried and abandoned: it costs a multi-gigabyte Python stack on every laptop before a single question is answered, and it leaves the day's numbers depending on which version of which library each machine happened to resolve.

Ollama is a local model server, not a framework. You give it a model name, it downloads a quantised copy of the weights, and it listens on `http://localhost:11434` with a small HTTP API: `/api/tags` lists what you have, `/api/embed` turns text into vectors, `/api/chat` answers. The notebooks call it with `urllib` from the standard library. One download, one process, no SDK — and a pinned model that answers the same on your laptop in October as it did on the machine that produced the numbers on this site.

<div class="presenter-note">
Say this once and do not garble it: <strong>everything runs on the laptop in front of you, and that is a measurement decision, not an ideology.</strong> Someone in the room will ask "why not use an API?" in the first ten minutes. The answer is not "local is better" — it is "a pinned local model gives us the same number in September and in October, so when a number moves we know what moved it. And nobody needs an account, a key or an approval to sit down."
</div>

## Going deeper

Ollama serves quantised GGUF weights, which is why a 3-billion-parameter model is a ~2 GB download and runs on a laptop with no GPU. Quantisation stores each weight in roughly 4 bits instead of 16, at a small loss of quality — invisible on the tasks in this course, not invisible on a long chain of reasoning, which is one reason the day never asks a 3B model to do anything clever in a single call.

The embedder choice is the one that will bite you outside this room, because it fails silently. `nomic-embed-text` and `all-MiniLM-L6-v2` are English-first models: given a Turkish query and the English document that answers it, they do not error and they do not return nothing — they return a confidently wrong document. `bge-m3` is trained multilingually, so a Turkish sentence and its English equivalent land near each other in the vector space. You cannot see that difference by reading code. Only by scoring it, which is the box above.

## Exit line

> Everything is installed and nothing is connected to anything yet. Tomorrow starts with the model completely on its own — `notebooks/00_bare_llm_fails.py`, and one question about a Kraken Air fare rule it has never seen.
