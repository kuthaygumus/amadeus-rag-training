---
title: "8. ChromaDB: Embedded vs Server"
description: "How do I run this in production?"
---

## Gate question

> **How do I run this in production?**

**VS Code — go back to the window where you ran module 7:** kill the Python process those blocks
were running in. The bin icon on its panel does it. Now re-run the retrieval block.

`NameError`. `DenseRetriever` is gone, and so are the 154 structure-aware chunk vectors you spent
the last hour improving.

That is the whole failure, and it takes two seconds to produce. Every retriever you have built
today held its vectors in a Python list inside one process. `DenseRetriever(doc_ids, documents)`
embeds the corpus in its constructor. Kill the process and the corpus is still on disk, the model
is still on disk, and the vectors are not. You rebuild them by embedding 154 chunks again, every
single time.

At 28 documents and 78,310 characters that costs a few seconds and feels like nothing. Size is the
first reason it stops feeling like nothing: the same loop over a real rule corpus is a coffee
break, paid again on every deploy, every crash, every autoscale event. Shape is the second. What
you are going to build is a service — an Angular front end calling an endpoint that calls a
retriever. The list lives in one Python process. The request arrives in another.

So the question is not "which vector database is best". It is: where do the vectors live when
the process that made them is gone.

<div class="presenter-note">
Do this before any slide. Kill the module 7 process in front of the room, re-run the retrieval
block, watch it throw <code>NameError</code>. Ask: "how long does that rebuild take on your corpus,
not mine?" Let two or three people say a number out loud. This module is straight after lunch, the
room is at its lowest energy of the day, and a 20-second live failure buys more attention than any
diagram. 2 minutes.
</div>

## Embedded mode: the two-line version

Every Python block on this page is a cell in `notebooks/08_chromadb.py`. Open it in VS Code now and
run them as you read. This first one is what almost everyone writes, and it works immediately.

**VS Code — `notebooks/08_chromadb.py`, the block after the imports:**

```python
client = chromadb.EphemeralClient()          # PersistentClient(path=...) to keep it on disk
collection = client.get_or_create_collection("kraken")
collection.add(ids=ids, documents=texts)             # note: no embeddings= argument
print(f"{collection.count()} documents indexed")
```

```text
28 documents indexed
```

No configuration, no daemon, no port. `EphemeralClient` holds the collection in memory, which is
what the notebook uses so a day of demos leaves nothing behind on disk.

Now notice what we did **not** do. We never said how to turn text into vectors. Chroma picked
something for us, silently, and it did not ask.

## The default nobody chose

Before you run the queries, guess: what did it pick, and is it the right choice for a corpus that
holds Turkish documents and is asked questions in Turkish all day?

**VS Code — the next block:**

```python
DEFAULT_QUERIES = [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("Baglantisini kaciran yolcuya yemek fisi ne kadar?", "sop_misconnect_v4"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]
```

```text
embedding function: DefaultEmbeddingFunction
dimensions        : 384 (read off a stored vector)
loaded from       : ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx
WRONG  CLASSIC K sinifi iptal cezasi ne kadar?              -> macro_tr_noshow
WRONG  Baglantisini kaciran yolcuya yemek fisi ne kadar?    -> macro_tr_noshow
right  what is the cancellation penalty for CLASSIC class   -> fare_classic_shorthaul
```

Two Turkish questions, the same wrong document twice — a call-centre macro, not a fare sheet. The
English question is right, and that is not the default embedder doing well. It is the failure being
*specific*. `all-MiniLM-L6-v2` is English-only, so on English questions against English documents it
is not visibly worse: module 6 scores all three embedders at **0.500** there. The cross-lingual half
is where it dies — **0.000** on the six Turkish-question/English-document items against `bge-m3`'s
**0.667**, and hit@1 **0.300** against **0.700** overall. No exception, no warning, nothing in the
output suggesting anything is off.

Chroma chose that model for you and told you nothing. The line above names it only because the
notebook reads the width off a stored vector and prints the cache directory it came from — Chroma
does not expose the model name on the embedding function. It is 384 dimensions, and it is
downloaded on your first `add()`: 83,178,821 bytes over the internet, once per machine, with no
prompt. Moving into a database does not repair that and does not warn you about it. The system does
not fail. It answers wrong, quietly, for as long as nobody checks.

<div class="presenter-note">
This is the whole module. Run the two-line version live, let the two WRONG lines land, and wait
before explaining. Someone will say "but it did not error" — that is the sentence you want out
loud, so repeat it back. If somebody objects that the English one passed, that is the point: hand
it back as "so what exactly broke?" until the room says Turkish. If the first <code>add()</code>
stalls, the default model is not seeded on that machine; the notebook replays a recorded run and
prints that it is doing so, so keep going. 4 minutes.
</div>

## The same collection, with an embedder you chose

Chroma accepts vectors you computed yourself. One extra argument:

**VS Code — the block after that:**

```python
better = client.get_or_create_collection("kraken_bge_m3")
better.add(ids=ids, documents=texts, embeddings=R.embed(texts))   # bge-m3, via Ollama, our call
```

```text
indexed with bge-m3 in 5.9s

right  CLASSIC K sinifi iptal cezasi ne kadar?              -> fare_classic_shorthaul
right  what is the cancellation penalty for CLASSIC class   -> fare_classic_shorthaul
```

Everything else is unchanged — same client, same `query`, same result shape. The Turkish question
that landed on a call-centre macro now returns the right fare sheet. It is better and still not
perfect at document level, which is what modules 6 and 7 spend their time on. The point here is
narrower: **the embedding model is a decision, and Chroma will make it for you if you do not.**

## Surviving the restart

The gate was a lost index, and one word closes it:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # instead of EphemeralClient()
```

`PersistentClient` still runs inside your process — no daemon, no port, nothing to start. It writes
a SQLite file plus index files into `./chroma-local`, and that directory *is* the database. You can
copy it, ship it in a release artefact, or delete it to start over. The notebook keeps it as the
comment on the `EphemeralClient` line so a day of demos leaves nothing behind; change that one line
and the directory appears. For a lot of internal tools that is the last deployment decision you
need to make.

The other thing the directory buys you is filtering. Tag documents with `metadatas=` on `add()` and
a `where={"kind": "fare"}` on `query()` searches only the fare sheets — filtering *before* the
search rather than after is the difference between a database and a loop, and you cannot express
"only the current SOP" in a cosine similarity.

## Server, if you have a container runtime

Same collection behind a port, and the reason to bother is not speed. Two processes opening the
same embedded directory is not a cluster, it is a corruption route; the moment you want a second
reader, you want the server. The compose file is in the repository root.

**Terminal (repo root):**

```bash
podman compose up -d          # or: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

**The API path is `/api/v2/`, not `/api/v1/`.** Most of the ChromaDB answers on the internet were
written against v1; against a current server v1 returns **`410 Gone`** — and that status line on its
own looks exactly like a container that did not come up, so read the body before you spend twenty
minutes debugging Podman.

Then the client swap is one line — `chromadb.HttpClient(host="localhost", port=8000)` instead of
`EphemeralClient()` — and every `add`, `query` and `where` below it is byte-for-byte identical. You
develop against a directory and ship against a port without rewriting your retrieval code. If you
have no container runtime you lose nothing today: the notebook's server cell is in a try/except, so
it prints `no server running` with the command to start one and carries on embedded.

<div class="presenter-note">
The image pull belongs at home, not in the room — <code>00-setup.md</code> says so, and says plainly
that everything measured in this course was measured without Podman. Show <code>podman images</code>
to prove the image is already there. If the container will not start: port 8000 taken, use
<code>-p 8001:8000</code>; on macOS, <code>podman machine start</code> first. If nobody in the room
has a runtime, read this section out and move on — the fall-through is the documented behaviour, not
a broken demo. The sentence not to garble: <strong>the path is /api/v2/ — every older answer you
will find online says v1 and gets 410 Gone.</strong> 2 minutes.
</div>

## What you run

**VS Code.** Open `notebooks/08_chromadb.py` and run the blocks with Shift+Enter.

- **What you should see:** `28 documents indexed`, then two `WRONG` lines and one `right` from
  Chroma's default embedder, then the same corpus queried again with `bge-m3` embeddings you passed
  yourself. With no container running, the last block prints `no server running` and stops there.
- **How long:** about 2 minutes, most of it the `bge-m3` embedding call.
- **If the first `add()` hangs:** Chroma is downloading its default model, 83 MB, right now. Stop
  the cell. The two cells that need it replay a recorded run and print `[CACHED]`, so keep going and
  seed it at home with `python scripts/seed_offline_assets.py`.

The only Python that differs between the three deployments:

```python
client = chromadb.EphemeralClient()                           # in memory, dies with the process
client = chromadb.PersistentClient(path="./chroma-local")     # embedded, on disk
client = chromadb.HttpClient(host="localhost", port=8000)     # server
```

## What the numbers said

<div class="measured">

| | measured |
|---|---|
| documents indexed by the two-line version | 28 |
| Chroma's default embedder on the twenty gold questions | `all-MiniLM-L6-v2` hit@1 0.300 vs `bge-m3` 0.700 |
| the same default, Turkish question against English document | 0.000 vs `bge-m3` 0.667 |
| the same default, English question against English document | 0.500 — same as every other embedder |
| the default model, read off a stored vector | `DefaultEmbeddingFunction`, 384 dimensions |
| the default model's download, first `add()` | 83,178,821 bytes |
| re-indexing the same 28 documents with `bge-m3` | 5.9 s |
| `/api/v1/heartbeat` | `410 Gone` — "The v1 API is deprecated. Please use /v2 apis" |

</div>

Twenty questions is enough to choose between two designs and nowhere near enough to publish one; a
gap under about 0.05 is inside the noise of twenty questions. The 0.300-against-0.700 gap is not.

We did not re-run the twenty-question benchmark through Chroma. Over 154 vectors the search is
effectively exhaustive, so the ranking should be identical to the in-memory retriever and hit@1
should still be 0.700 — but "should be" is not a measurement.
`UNVERIFIED: hit@1 through a Chroma collection, because no benchmark run points at one.` What would
settle it: point `eval/run_benchmark.py` at the collection instead of the list and compare the two
hit@1 values directly.

## Going deeper

**When a vector database earns its place at all.** Not here. Twenty-eight vectors is a numpy dot
product against a matrix that fits in cache: exact, microseconds, no dependency. As a rule of thumb
rather than a measurement on this corpus, brute force stays a legitimate production answer to
somewhere around a hundred thousand vectors. Above that, the first question is whether you already
run Postgres — `pgvector` gives you one backup story, one transaction boundary, and a vector search
you can join to the business columns next to it. A dedicated vector store earns its keep when the
vector workload becomes its own operational problem: tens of millions of vectors, re-indexing every
time you swap embedders, per-tenant collections, sharding.

**What the index is doing underneath.** Chroma's index is a Hierarchical Navigable Small World
graph — each vector is a node linked to a handful of near neighbours, and a query walks greedily
downhill through layers instead of scanning everything, which is why search stays fast as the
collection grows. It is *approximate* because that walk has no guarantee: it can settle for the
second-best answer. At 28 documents there is nothing to measure — the graph is smaller than the
search — and any latency you feel is the embedding call, not the index.

<div class="presenter-note">
Timing: 10 minutes — 2 on the killed process, 4 on the WRONG lines, 2 on the embedder you choose
plus the <code>PersistentClient</code> one-liner, 2 on the three client lines and
<code>/api/v2/</code>. It is the shortest slot of the day on purpose: the room has just eaten and
the payload is one trap and three client lines.
<br /><br />
<strong>Running late: this module is the second cut on the list, 10 minutes down to 5.</strong> Drop
the server section and Going deeper — keep the killed process and the WRONG lines, and say the
server swap in one sentence while showing the three client lines side by side. The silent default is
the only thing here nobody can reconstruct from the docs.
<br /><br />
If someone asks "so should we use Chroma or pgvector in production?", answer with the question
rather than a product: how many vectors, and do you already run Postgres. Refuse to recommend a
database you have not measured on their corpus.
</div>

## Exit line

> Same API, different deployment. Chroma answered two Turkish questions with the same wrong document
> and never raised a warning; we fixed that by passing our own vectors, and the ceiling underneath
> is still 0.750 — five of the twenty questions still come back with the wrong document first. What if semantic
> search is not enough?
