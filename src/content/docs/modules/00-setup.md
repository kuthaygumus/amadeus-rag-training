---
title: "0. Setup — Before You Arrive"
description: "Install Ollama, Podman Desktop and Bruno, pull two models, build two containers, run one health request — under an hour the evening before, nearly all of it downloading. And the rule that makes every command on this site work: the repository root is where you stand."
---

There is no gate question for this module. This is the pre-work, done the evening before, on a network you are not sharing with twenty other people. Under an hour, nearly all of it downloading. Nothing on this page needs Python, a notebook or an account anywhere.

## Where every command on this site goes

Read this section once and nothing later in the day is a guess. There are exactly **three surfaces**, and the folder you are about to clone — `amadeus-rag-lab`, the one that contains `compose.yaml`, `bruno/`, `corpus/` and `src/` — is the anchor of all three.

1. **Bruno.** The request collection you click through. Each module names a folder and a request — **Bruno — `01-bare-llm` › `ask-about-kraken`** — and you select it in the left pane and press the arrow. Every request has a *Docs* tab that says what to read in the response.
2. **Terminal, at the repository root.** One terminal window, `cd`'d into `amadeus-rag-lab`, left open for the day. Every `ollama …`, `podman compose …` and `curl …` on this site runs here.
3. **Podman Desktop.** The window that shows the two containers, their logs and their state. You look at it more than you click it.

Every fenced command on every page of this site says in bold which surface it belongs to, immediately above the fence, so you never have to work it out.

**You are in the right terminal** when `ls` — `dir` on Windows — lists `compose.yaml`, `bruno`, `corpus` and `src`. If `podman compose` answers that it cannot find a compose file, check that before anything else: it is the wrong window far more often than it is a broken install.

<div class="presenter-note">
Put this on the board at 09:10 and leave it there: <strong>three surfaces, one anchor — the repository root.</strong> The most common way a room loses ten minutes is one person running <code>podman compose up</code> from their home directory while everyone helps them debug an install that is fine. When a laptop says it cannot find <code>compose.yaml</code>, ask "which folder is that terminal in?" before you ask anything else.
</div>

## Install three things

**Ollama** runs the models on your machine. **Podman Desktop** runs ChromaDB and the api as containers. **Bruno** is the request collection. Nothing else — no runtime, no SDK, no editor is required.

### Ollama

**macOS.** Download from [ollama.com/download](https://ollama.com/download), drag the app to Applications, launch it once. You get a menu-bar icon and a server listening on `http://localhost:11434`.

**Terminal (anywhere — the repository does not exist yet):**

```bash
ollama --version
```

**Windows, without administrator rights.** `OllamaSetup.exe` is a per-user installer: binaries under `%LOCALAPPDATA%\Programs\Ollama`, no elevation, server starts with your session.

`UNVERIFIED: the per-user install path is how the installer is built, not something confirmed on a managed Windows laptop.` If it asks for administrator credentials, stop — send a message the evening before and you will be paired with someone whose machine is green, rather than arguing with IT at 09:00 on the day.

It adds Ollama to your user PATH, and a terminal that was already open does not pick that up. If `ollama --version` says the command is not recognised, close that window and open a new one before concluding anything.

### Podman Desktop

Download from [podman-desktop.io](https://podman-desktop.io). Launch it once; the first-run screen offers to set up Podman itself — accept. On macOS and Windows the containers run inside a small Linux VM that this step creates, and creating it is a download of its own, which is why it belongs at home.

`UNVERIFIED: whether Podman Desktop installs without administrator rights on a managed Windows laptop. On Windows it needs WSL 2, and enabling WSL 2 is commonly an elevated step. This is the #1 risk of the day. If it blocks you, say so the evening before — you will pair.`

Already have Docker Desktop and a working `docker compose`? It works too, with one change, noted under [Clone and build at home](#clone-and-build-at-home).

### Bruno

Download from [usebruno.com/downloads](https://www.usebruno.com/downloads). A desktop app, per-user, no account. You will open a folder of the cloned repository as a collection — nothing to import, nothing to sync.

## Pull two models

**Terminal (anywhere):**

```bash
ollama pull gemma3:4b     # answers questions      (3.3 GB)
ollama pull bge-m3        # turns text into vectors (1.2 GB)
```

About 4.5 GB together. Pull both. Pulling a model in the room on the day is the single thing this page exists to prevent: measured on a corp laptop, the office link gives `registry.ollama.ai` 3.7–7.8 MB/s on a single stream. Twenty laptops starting the same 4.5 GB pull at 09:00 are not each getting the link, they are dividing it. If your home connection gives up, weights are ordinary files and can be copied from a machine that has them — see [Moving models between machines](#moving-models-between-machines).

**Do not substitute a model.** Every number on this site was produced with `gemma3:4b` answering and `bge-m3` embedding; swap either and the answers on your screen stop matching the answers on the projector, and you will spend the day debugging a difference that is not a bug.

## Clone and build at home

The same link speed applies to the two container images: `chromadb/chroma:1.5.9` is pulled from `docker.io`, and the api image is built from the repository with an `npm ci` that runs **inside the Podman VM**, where the corporate proxy may not reach. Build at home once. The image stays on your machine, and the day starts with `podman compose up` and no `--build`.

**Terminal (anywhere — this is where the repository root comes from):**

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-lab.git
cd amadeus-rag-lab
podman pull docker.io/chromadb/chroma:1.5.9
podman compose up --build
```

That `cd` is the repository root. Everywhere this site says *repo root*, it means this folder.

**What you should see.** Two containers come up: `chroma` on port 8000 first — compose waits for its healthcheck, `/api/v2/heartbeat`, before starting the second — then `api` on port 3000, whose last log lines read `amadeus-rag-lab listening on http://localhost:3000` and `start with GET /health`. The first `up --build` takes about a minute for the build plus the image pull; every later `up` takes seconds.

Ollama is **not** a container. It stays on your machine, where the GPU is and where you just pulled the models; the api reaches it at `http://host.containers.internal:11434`. That is also the one line that changes under Docker: `OLLAMA_URL=http://host.docker.internal:11434 docker compose up --build`.

Leave it running for the next section — `up` stays in the foreground and streams both containers' logs. When the health request is green, `Ctrl+C` stops it, then take the containers down cleanly:

**Terminal (repo root of `amadeus-rag-lab`):**

```bash
podman compose down
```

`down` removes the containers and keeps both the images and the `chroma-data` volume. Do not add `-v` tonight; that wipes the volume, and while tonight there is nothing in it, on the day there will be.

**No git?** Download [the ZIP](https://github.com/kuthaygumus/amadeus-rag-lab/archive/refs/heads/main.zip), extract it, and `cd` into the extracted folder. Everything after that is identical.

## What you run

One request. Open Bruno → *Open Collection* → pick the `bruno/amadeus-rag-lab` folder **inside** the cloned repository — not the repository root, not `bruno/`. In the top-right environment selector choose `local`; it sets `baseUrl` to `http://localhost:3000` and nothing else. Then:

**Bruno — `00-health` › `health`**

```text
GET {{baseUrl}}/health
```

No body. Press the arrow.

**What you should see.** Status `200`, and in the response `"status": "ready"` with four `"ok"` under `checks`: `ollama`, `chatModel`, `embedModel`, `chroma`. `collections` is an empty list and `currentCollection` is `null` — correct, nothing has been ingested yet. The two assertions in the *Tests* tab go green. Close Bruno and forget about it.

**Anything else names the missing piece.** The response is `503` with `"status": "not-ready"`, and the check that failed says what to do instead of `"ok"`:

| Check | What it says | What it means |
|---|---|---|
| `ollama` | `unreachable at http://host.containers.internal:11434 — start Ollama on your machine` | The container cannot see Ollama on the host. Is Ollama running (menu-bar / tray icon)? On Windows, read the note below. |
| `chatModel` | `missing — run: ollama pull gemma3:4b` | Run exactly that, at home. |
| `embedModel` | `missing — run: ollama pull bge-m3` | Same. |
| `chatModel` / `embedModel` | `skipped` | Only when `ollama` itself failed. Fix that first; the two model checks run on the next request. |
| `chroma` | `unreachable at http://chroma:8000 — is the chroma container running?` | Look at Podman Desktop: if `chroma` is not green, `podman compose down` then `podman compose up` at the repo root. |
| Bruno: connection refused | — | The api is not up at all. Is `podman compose up` running in the terminal? Is the environment `local` selected? |

**Windows and `OLLAMA_HOST`.** `UNVERIFIED: on Windows, Ollama may bind only 127.0.0.1, which the Podman VM cannot reach — ollama list works in your terminal while the container reports unreachable. The documented fix is to set the user environment variable OLLAMA_HOST=0.0.0.0 (Settings → System → Environment variables, user scope, no admin needed), then quit and restart Ollama from the tray. Not yet confirmed on a managed Windows laptop.` If you hit this at home, fix it at home; if you hit it at 09:10, you pair.

## Moving models between machines

Weights are ordinary files on disk, and they are portable — the fallback for anyone whose home download gave up. Both models together are about 4.5 GB, which is a USB stick, not a download.

| OS | Path |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

1. **Quit Ollama completely on both machines first.** macOS: menu-bar icon → Quit. Windows: system-tray icon → Quit Ollama. On Windows the server starts with your session and holds those files open.
2. **Merge, do not replace.** Copy the *contents* of `blobs/` and `manifests/` into the folders of the same name on the target machine. Copying the whole `models` directory over the top removes the `manifests/` entries of any model that machine already had, which silently unregisters it.
3. Start Ollama again and run `ollama list`. `gemma3:4b` and `bge-m3` both show up, or you stop here and pair.

The container images are portable in the same way — `podman save` on a green machine, `podman load` on a red one — but at 09:10 that is a repair, not a pairing. Carry the stick for the models; let the images be the trainer's problem.

<div class="presenter-note">
<strong>09:10, ten minutes.</strong> Everybody runs <strong>00-health</strong> at once, and you walk the room reading screens — not asking, reading. Before they press the arrow, ask the room to guess: "how many of us are green?" Get a number out loud. Two things happen: people commit, and the ones who never ran it the night before out themselves. Twenty laptops hitting their own <code>localhost:3000</code> costs the network nothing; the only thing that costs the network is a pull, and a pull at 09:10 is already a pairing.
<br /><br />
<strong>Red laptop triage, in this order.</strong> Bruno says connection refused → "which folder is that terminal in?", then <code>podman compose up</code> at the repo root, and say the three-surface rule to the whole room while you do it. <code>chroma</code> red → look at Podman Desktop with them, <code>down</code> then <code>up</code>. <code>ollama</code> red on a Mac → the app is not running, open it. <code>ollama</code> red on Windows while <code>ollama list</code> works → <code>OLLAMA_HOST</code>, and you have five minutes for it, not fifteen. Model missing and the machine is fast → start the pull now, it finishes during module 1 — but only one laptop in the room gets to do that. Model missing and the network is crawling → USB stick, you carry two. Podman blocked by admin rights → stop, pair them immediately, do not spend the room's morning on it.
<br /><br />
<strong>Pairing is the fallback, and it is a fine one.</strong> Say it out loud so nobody feels punished: "one laptop between two people is the normal way to do this — one drives Bruno, one reads the response and argues." Do not let a red laptop become a person sitting quietly for six hours.
<br /><br />
<strong>Ten minutes means ten minutes.</strong> Anything not green by 09:20 is a pairing, not a repair. You get the time back nowhere else in the day. And if Ollama is down on <em>your</em> machine: the gates read fine off the page, and your projector runs module 1 from a laptop that is green.
</div>

## Going deeper

Ollama is a local model server, not a framework. You give it a model name, it downloads a quantised GGUF copy of the weights, and it listens on `http://localhost:11434` with a small HTTP API — which is why the api container can treat it exactly like any other service, and why it does not need to be a container itself. Quantisation stores each weight in roughly 4 bits instead of 16, at a small loss of quality: invisible on the tasks in this course, not invisible on a long chain of reasoning, which is one reason the day never asks a 4B model to do anything clever in a single call.

Someone will ask "why not an API?" in the first ten minutes. The answer is not "local is better". It is that **a pinned local model gives the same answer in September and in October, so when an answer moves we know what moved it** — and nobody needs an account, a key or an approval to sit down. Everything on this site runs on the laptop in front of you, and that is a measurement decision, not an ideology.

The two containers are the other half of the same decision. ChromaDB is pinned to `1.5.9` and the api is built from a lockfile; the compose file is the whole deployment, and `podman compose down -v` is the whole reset. Module 6 opens that box.

## Exit line

> Everything is installed and nothing is connected to anything yet. Tomorrow starts with the model completely on its own — **Bruno — `01-bare-llm` › `ask-about-kraken`**, one question about a Kraken Air fare rule it has never seen.
