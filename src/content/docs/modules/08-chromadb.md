---
title: "8. ChromaDB: Embedded vs Server"
description: "How do I run this in production?"
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **How do I run this in production?**

Restart the kernel.

That is the whole failure, and it takes two seconds to produce. Every retriever you have built
today held its vectors in a Python list inside one notebook process. `DenseRetriever(doc_ids,
documents)` embeds the corpus in its constructor. Kill the process and the corpus is still on
disk, the model is still on disk, and the 154 structure-aware chunk vectors you spent the last
hour improving are gone. You rebuild them by embedding 154 chunks again, every single time.

At 28 documents and 78,310 characters that costs a few seconds and feels like nothing. Size is the
first reason it stops feeling like nothing: the same loop over a real rule corpus is a coffee
break, paid again on every deploy, every crash, every autoscale event. Shape is the second. The
thing you are going to build is a service — an Angular front end calling an endpoint that calls a
retriever. The list lives in one Python process. The request arrives in another. There is no
version of that sentence where the notebook is the answer.

So the question is not "which vector database is best". It is: where do the vectors live when
the process that made them is gone.

<div class="presenter-note">
Do this before any slide. Restart the notebook kernel in front of the room, re-run the retrieval
cell, watch it throw. Ask: "how long does that rebuild take on your corpus, not mine?" Let two or
three people say a number out loud. This module is straight after lunch, the room is at its
lowest energy of the day, and a 20-second live failure buys more attention than any diagram.
2 minutes.
</div>

## Embedded mode: the two-line version

This is what almost everyone writes first, and it works immediately.

```python
import chromadb
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("helios")
collection.add(ids=ids, documents=texts)          # note: no embeddings= argument
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

```python
DEFAULT_QUERIES = [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("Baglantisini kaciran yolcuya yemek fisi ne kadar?", "sop_misconnect_v4"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]
for question, should_be in DEFAULT_QUERIES:
    top = collection.query(query_texts=[question], n_results=3)["ids"][0]
    print(f"{'right' if top[0] == should_be else 'WRONG':>5}  {question[:50]:<52} -> {top[0]}")
```

```text
embedding function: DefaultEmbeddingFunction
WRONG  CLASSIC K sinifi iptal cezasi ne kadar?              -> macro_tr_noshow
WRONG  Baglantisini kaciran yolcuya yemek fisi ne kadar?    -> macro_tr_noshow
WRONG  what is the cancellation penalty for CLASSIC class   -> fare_classic_longhaul
```

Three questions, three wrong documents. The two Turkish questions both land on the same call-centre
macro. The English one is nearer the mark and still wrong: it returns the long-haul fare sheet when
the question named short-haul. No exception, no warning, nothing in the output suggesting anything
is off.

The model Chroma chose is `all-MiniLM-L6-v2`: 384 dimensions, English-only, and downloaded on your
first `add()` — 83,178,821 bytes over the internet, once per machine, with no prompt. This is the
same model module 6 measured. On the twenty gold questions over structure-aware chunks it scores
hit@1 **0.350** against `bge-m3`'s **0.800**, and on the six Turkish-question/English-document
items it scores **0.000** against **0.667**.

`pip install chromadb`, add your documents, query — and you are running an English-only embedding
model over a Turkish corpus. Moving into a database does not repair that and does not warn you
about it. The system does not fail. It answers wrong, quietly, for as long as nobody checks.

<div class="presenter-note">
This is the whole module. Run the two-line version live, let the three WRONG lines land, and wait
before explaining. Someone will say "but it did not error" — that is the sentence you want out
loud, so repeat it back. If the first <code>add()</code> stalls, the default model is not seeded on
that machine; the notebook replays a recorded run and prints that it is doing so, so keep going.
4 minutes.
</div>

## The same collection, with an embedder you chose

Chroma accepts vectors you computed yourself. One extra argument:

```python
better = client.get_or_create_collection("helios_bge_m3")
better.add(ids=ids, documents=texts, embeddings=R.embed(texts))   # bge-m3, via Ollama, our call
```

Everything else is unchanged — same client, same `query`, same result shape. The English question
now returns `fare_classic_shorthaul` first. It is better and still not perfect at document level,
which is what modules 6 and 7 spend their time on. The point here is narrower: **the embedding
model is a decision, and Chroma will make it for you if you do not.**

## Surviving the restart

The gate was a lost index, and one word closes it:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # instead of EphemeralClient()
```

`PersistentClient` still runs inside your process — no daemon, no port, nothing to start. It writes
a SQLite file plus index files into `./chroma-local`, and that directory *is* the database. You can
copy it, ship it in a release artefact, or delete it to start over. Written in one process and
reopened in a second, `collection.count()` returns **28** in **0.09 s** without embedding anything.

For a lot of internal tools that is the last deployment decision you need to make.

## Metadata filtering: the thing a list cannot do

Retrieval is rarely "search everything". It is "search this quarter's fare rules, not the
superseded SOPs".

```python
kinds = {i: ("fare" if i.startswith("fare") else "sop" if i.startswith("sop")
             else "bulletin" if i.startswith("bulletin") else "other") for i in ids}
tagged = client.get_or_create_collection("helios_tagged")
tagged.add(ids=ids, documents=texts, embeddings=R.embed(texts),
           metadatas=[{"kind": kinds[i]} for i in ids])

everything  = tagged.query(query_embeddings=R.embed([q]), n_results=3)["ids"][0]
fares_only  = tagged.query(query_embeddings=R.embed([q]), n_results=3,
                           where={"kind": "fare"})["ids"][0]
```

Filtering before the search rather than after is the difference between a database and a loop. You
cannot express "only the current SOP" in a cosine similarity.

## Server: same collection, different deployment

Now put the same thing behind a port. The compose file is in the repository root.

```bash
podman compose up -d          # or: docker compose up -d
```

```bash
# macOS / Linux
curl -s http://localhost:8000/api/v2/heartbeat

# Windows (PowerShell) — plain `curl` there is an alias for Invoke-WebRequest and will not take -s
Invoke-RestMethod http://localhost:8000/api/v2/heartbeat
```

The image is 649 MB and the server answered its first heartbeat about half a second after start,
with the image already on the machine. If you run it by hand instead of by compose, note the
registry prefix: Podman does not assume Docker Hub the way Docker does, so `podman pull
chromadb/chroma` stops and asks which registry you meant. Write `docker.io/` and move on.

**The API path is `/api/v2/`, not `/api/v1/`.** Say that out loud twice. Most of the ChromaDB
answers on the internet were written against v1 and they are the first hit on every search. Against
a current server the v1 path returns **`410 Gone`**:

```json
{"error":"Unimplemented","message":"The v1 API is deprecated. Please use /v2 apis"}
```

The status line on its own looks exactly like a container that did not come up. People spend
twenty minutes debugging Podman when the only thing wrong is a version number in a URL, so read
the body.

The client swap is one line:

```python
client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_or_create_collection("helios_remote")
res = col.query(query_embeddings=R.embed([q]), n_results=5, where={"kind": "fare"})
```

Everything below that first line is byte-for-byte identical to the embedded version. Same `add`,
same `query`, same `where` filter, same result shape. That is not a coincidence, it is the product
decision that makes Chroma worth teaching: you develop against a directory and ship against a port
without rewriting your retrieval code.

<div class="presenter-note">
Do not have twenty laptops pull 649 MB at once on the day — that pull belongs at home, and the
setup page says so. Show <code>podman images</code> to prove the image is already there, and pull
on your own machine only if the room wants to watch a progress bar. If the container will not
start: port 8000 taken, use <code>-p 8001:8000</code>; on macOS, <code>podman machine start</code>
first. The notebook's server cell is wrapped in a try/except, so with no container running it
prints "no server running" and the notebook continues — that is fine, do not stop to fix it. The
sentence not to garble: <strong>the path is /api/v2/ — every older answer you will find online says
v1 and gets 410 Gone.</strong> 2 minutes.
</div>

## What actually changes

The API does not change. Collection semantics do not change. Your embedding choice does not
change, and neither does retrieval quality — a database is a storage decision, not a relevance
improvement. What changes is everything around the code.

**Failure modes.** Embedded fails when your process fails, and that is the whole list. The server
adds a network hop, a port, a container lifecycle, and a disk that is now somebody's volume. Two
processes opening the same embedded directory is not a cluster, it is a corruption route; the
moment you want a second reader, you want the server.

**Operational ownership.** Embedded belongs to whoever deploys the app. The server is a thing
that gets restarted at 03:00, backed up, upgraded, and occasionally broken by an upgrade that
changes the on-disk index format. Someone owns that. Decide who before you start it, not after.

**Exposure.** `-p 8000:8000` with no auth is an open database on your machine's network
interface. Fine on a laptop, not fine on a shared host. Bind it to localhost or put a token in
front of it.

## What you run

**Notebook.** Open `notebooks/08_chromadb.py` in VS Code and run the blocks with Shift+Enter.

- **What you should see:** `28 documents indexed`, then three `WRONG` lines from Chroma's default
  embedder, then the same corpus queried again with `bge-m3` embeddings you passed yourself.
- **How long:** about 2 minutes, most of it the `bge-m3` embedding call.

**Server, optional.** One command, from the repository root:

```bash
podman compose up -d                            # or: docker compose up -d
```

```bash
# macOS / Linux
curl -s http://localhost:8000/api/v2/heartbeat

# Windows (PowerShell)
Invoke-RestMethod http://localhost:8000/api/v2/heartbeat
```

- **What you should see:** `{"nanosecond heartbeat": ...}`. The `/api/v1/` path returns `410 Gone`.
- **How long:** a few seconds if the image is already pulled; the pull itself is 649 MB.

```bash
podman logs helios-chroma
podman compose down
```

The only Python that differs between the two halves of the notebook:

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
| Chroma's default embedder on the twenty gold questions | `all-MiniLM-L6-v2` hit@1 0.350 vs `bge-m3` 0.800 |
| the same default, Turkish question against English document | 0.000 vs `bge-m3` 0.667 |
| the default model's download, first `add()` | 83,178,821 bytes |
| `PersistentClient` reopened in a second process | `count()` 28 in 0.09 s, nothing embedded |
| `docker.io/chromadb/chroma` image | 649 MB |
| first v2 heartbeat after start, image already local | about 0.5 s |
| `/api/v1/heartbeat` | `410 Gone` — "The v1 API is deprecated. Please use /v2 apis" |

</div>

Twenty questions is enough to choose between two designs and nowhere near enough to publish one; a
gap under about 0.05 is inside this sample's noise. The 0.350-against-0.800 gap is not.

We did not re-run the twenty-question benchmark through Chroma. Over 154 vectors the search is
effectively exhaustive, so the ranking should be identical to the in-memory retriever and hit@1
should still be **0.800** — but "should be" is not a measurement. What would settle it: point
`eval/run_benchmark.py` at the collection instead of the list and compare the two hit@1 values
directly.

## Going deeper

**Why Podman rather than Docker.** Three reasons, none of them ideological. It is rootless: the
container runs as your user, so an escape lands on your uid rather than on root. It is
daemonless: there is no privileged background service, `podman run` is just a process tree your
shell owns, and it maps onto systemd units without a second supervisor. And there is no licence
conversation — Docker Desktop is a paid subscription above a company-size threshold, which in a
corporate setting means procurement, which means the training day slips. The CLI takes the same
flags, so nothing you already know is wasted.

**What HNSW is doing underneath.** Chroma's index is a Hierarchical Navigable Small World graph.
Each vector is a node with edges to a handful of near neighbours, and the graph is built in
layers: a sparse top layer with long-range links, denser layers below. A query enters at the top,
walks greedily to whichever neighbour is closer to the query, drops a layer when it can improve
no further, and repeats. You touch a few hundred nodes instead of ten million, which is why the
search stays fast as the collection grows.

**Why "approximate".** The greedy walk has no guarantee. It can settle into a local minimum where
every neighbour is worse but the true nearest vector sits somewhere the graph never offered as a
step. You buy sub-linear time by accepting the second-best answer sometimes. The knob is the size
of the candidate list kept during the walk — bigger list, more of the graph explored, higher
recall, more latency, roughly linearly. Build time has its own knobs, edges per node and how hard
the builder searches while inserting, which trade memory and build time against graph quality. In
Chroma these are collection metadata, set at creation and awkward to change later.

Recall here means recall against exact brute-force search over the same vectors, not against your
gold set. Those are different questions and people conflate them constantly. Measure the first in
an afternoon: run both and count how often the top-k sets agree. At 28 documents there is nothing
to measure — the graph is smaller than the search — and any latency you feel is the embedding call,
not the index.

**When a vector database earns its place at all.** Not here. Twenty-eight vectors is a numpy dot
product against a matrix that fits in cache: exact, microseconds, no dependency. As a rule of thumb
rather than a measurement on this corpus, brute force stays a legitimate production answer to
somewhere around a hundred thousand vectors. Above that, the first question is whether you already
run Postgres. If you do, `pgvector` gives you one backup story, one transaction boundary, and a
vector search you can join to the business columns next to it — filter by fare family and effective
quarter in the same statement that does the similarity search, instead of syncing two systems and
hoping they agree. A dedicated vector store earns its keep when the vector workload becomes its own
operational problem: tens of millions of vectors, re-indexing as a routine event because you
swapped embedders, per-tenant collections, sharding, or filter-plus-vector queries at a scale where
the index and the filter must be planned together.

One detail worth setting explicitly: the distance function. A collection carries a space setting
— L2, cosine, inner product — and it is not always what you assume. If your vectors are unit
length, L2 and cosine produce the same ordering, so it does not matter. If they are not, they
produce different orderings and you will chase the discrepancy for an hour. Set `hnsw:space`
yourself and stop guessing.

<div class="presenter-note">
Timing: 10 minutes total — 2 on the kernel restart, 4 on the three WRONG lines, 2 on the embedder
you choose plus the <code>PersistentClient</code> one-liner, 2 on the server swap and
<code>/api/v2/</code>. It is the shortest slot of the day on purpose: the room has just eaten and
the payload is one trap and two client lines.
<br /><br />
<strong>Running late: this module is the second cut on the list, 10 minutes down to 5.</strong>
Cut the server half, metadata filtering and Going deeper entirely — keep the kernel restart and the
three WRONG lines, and say the server swap in one sentence while showing the three client lines
side by side. The silent default is the only thing here nobody can reconstruct from the docs.
<br /><br />
If someone asks "so should we use Chroma or pgvector in production?", answer with the question
rather than a product: how many vectors, and do you already run Postgres. Refuse to recommend a
database you have not measured on their corpus.
</div>

## Exit line

> Same API, different deployment. Storage is settled and retrieval is still at 0.800 — one question
> in five comes back with the wrong document first. What if semantic search is not enough?
