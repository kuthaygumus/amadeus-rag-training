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
disk, the model is still on disk, and the 152 structure-aware chunk vectors you spent the last
hour improving are gone. You rebuild them by embedding 152 chunks again, every single time.

At 28 documents and 75 KB that costs a few seconds and feels like nothing. Size is the first
reason it stops feeling like nothing: the same loop over a real rule corpus is a coffee break,
paid again on every deploy, every crash, every autoscale event. Shape is the second. The thing
you are going to build is a service — an Angular front end calling an endpoint that calls a
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

## Embedded: one process, one directory

The smallest thing that fixes it is not a server.

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma-local")
col = client.get_or_create_collection("iris_q3", metadata={"hnsw:space": "cosine"})
```

`PersistentClient` runs inside your process. There is no daemon, no port, nothing to start. It
writes a SQLite file plus index files into `./chroma-local`, and that directory is the database.
You can copy it, commit it to a release artefact, or delete it to start over.

Write the chunks once, with the embeddings you already trust:

```python
col.add(
    ids=chunk_ids,
    documents=texts,
    embeddings=embed(texts),                      # bge-m3, via Ollama, our call
    metadatas=[{"doc": p, "quarter": "2026-Q3"} for p in parents],
)
```

Then restart the kernel, open the same path, and `col.count()` prints **152** without embedding
anything. That is the entire win of this deployment, and for a lot of internal tools it is the
last deployment you need.

Look hard at the `embeddings=` argument, because it is the one line on this page that carries a
measured result. If you omit it and pass only `documents=`, Chroma picks an embedding function
for you, and the one it picks is `all-MiniLM-L6-v2` — downloaded silently on your first `add()`,
English-only, and on the ten-document probe corpus it scored **2/5** against **4/5** for
`bge-m3`. Module 6 was that failure. Moving into a database does not repair it and does not warn
you about it. Pass your own vectors, or pass your own embedding function, and never let the
default decide.

## Server: same collection, different deployment

Now put the same thing behind a port.

```bash
podman pull docker.io/chromadb/chroma      # 649 MB, measured, once per machine
podman run -d -p 8000:8000 --name chroma docker.io/chromadb/chroma
curl -s http://localhost:8000/api/v2/heartbeat
```

The pull is 649 MB and the server answers about a second after `run`. Note the registry prefix:
Podman does not assume Docker Hub the way Docker does, so `podman pull chromadb/chroma` will stop
and ask you which registry you meant. Write `docker.io/` and move on.

**The API path is `/api/v2/`, not `/api/v1/`.** Say that out loud twice. Most of the ChromaDB
answers on the internet were written against v1, they are the first hit on every search, and they
return 404 against a current server. A 404 on a heartbeat looks exactly like "the container did
not start", and people spend twenty minutes debugging Podman when the only thing wrong is a
version number in a URL.

The client swap is one line:

```python
client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_or_create_collection("iris_q3", metadata={"hnsw:space": "cosine"})
res = col.query(query_embeddings=embed([q]), n_results=5, where={"quarter": "2026-Q3"})
```

Everything below that first line is byte-for-byte identical to the embedded version. Same
`add`, same `query`, same `where` filter, same result shape. That is not a coincidence, it is
the product decision that makes Chroma worth teaching: you develop against a directory and ship
against a port without rewriting your retrieval code.

<div class="presenter-note">
Do not have twenty laptops pull 649 MB over the office network at once. The image is pre-seeded
in the setup instructions; show <code>podman images</code> to prove it is already there and pull
only on your own machine if the room wants to watch a progress bar. If the container will not
start: port 8000 taken, use <code>-p 8001:8000</code>; on macOS, <code>podman machine start</code>
first. The sentence not to garble: <strong>the path is /api/v2/ — every older answer you will
find online says v1 and returns 404.</strong>
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

Notebook: `06_chromadb.ipynb`. You watch this one; the commands are in the handout.

```bash
pip install chromadb
jupyter lab notebooks/06_chromadb.ipynb        # embedded: PersistentClient, restart, count() == 152

podman pull docker.io/chromadb/chroma          # 649 MB
podman run -d -p 8000:8000 --name chroma docker.io/chromadb/chroma
curl -s http://localhost:8000/api/v2/heartbeat # not /api/v1/ — v1 returns 404
podman logs chroma
podman stop chroma && podman rm chroma
```

The only Python that differs between the two halves of the notebook:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # embedded
client = chromadb.HttpClient(host="localhost", port=8000)     # server
```

## What the numbers said

<div class="measured">

| | measured |
|---|---|
| `podman pull docker.io/chromadb/chroma` | 649 MB |
| server responding after `podman run` | about 1 second |
| working API path | `/api/v2/` — `/api/v1/` returns 404 |
| chunks written to the collection | 152 (structure-aware, 28 documents, 75 KB) |
| ChromaDB's default embedder, probe corpus | `all-MiniLM-L6-v2` 2/5 vs `bge-m3` 4/5 |

</div>

We did not re-run the 20-question benchmark through Chroma. Over 152 vectors the search is
effectively exhaustive, so the ranking should be identical to the in-memory retriever and
hit@1 should still be **0.800** — but "should be" is not a measurement. What would settle it:
point `eval/run_benchmark.py` at the collection instead of the list and compare the two hit@1
values directly.

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
gold set. Those are different questions and people conflate them constantly. Measure the first by
running both and counting how often the top-k sets agree. At 152 vectors there is nothing to
measure — the graph is smaller than the search — and any latency you feel is the embedding call,
not the index.

**When a vector database earns its place at all.** Not here. 152 vectors is a numpy dot product
against a matrix that fits in cache; exact, microseconds, no dependency. Brute force stays a
legitimate production answer to somewhere around a hundred thousand vectors. Above that, the
first question is whether you already run Postgres. If you do, `pgvector` gives you one backup
story, one transaction boundary, and a vector search you can join to the business columns next to
it — filter by fare family and effective quarter in the same statement that does the similarity
search, instead of syncing two systems and hoping they agree. A dedicated vector store earns its
keep when the vector workload becomes its own operational problem: tens of millions of vectors,
re-indexing as a routine event because you swapped embedders, per-tenant collections, sharding,
or filter-plus-vector queries at a scale where the index and the filter must be planned
together.

One detail worth setting explicitly: the distance function. A collection carries a space setting
— L2, cosine, inner product — and it is not always what you assume. If your vectors are unit
length, L2 and cosine produce the same ordering, so it does not matter. If they are not, they
produce different orderings and you will chase the discrepancy for an hour. Set `hnsw:space`
yourself and stop guessing.

<div class="presenter-note">
Timing: 20 minutes, and do not let it run to 30. This slot exists to give the room a breather
between the chunking measurements and the harder retrieval work; the payload is two client lines
and one URL path. If someone asks "so should we use Chroma or pgvector in production?", answer
with the question rather than a product: how many vectors, and do you already run Postgres.
Refuse to recommend a database you have not measured on their corpus.
</div>

## Exit line

> Same API, different deployment.
