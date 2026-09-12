---
title: "6. ChromaDB in a Container"
description: "Where the 294 chunks actually live, how to prove the container is up, and what the pieces look like inside it."
---

## Gate question

> **Where did those 294 chunks go — and can we look at them?**

[Module 5](/modules/05-simple-rag/) ended with a grounded answer and a `sources` array. Somewhere
between `/ingest` and `/query` the pipeline put 294 vectors *somewhere*, and `/retrieve` found three
of them again. That somewhere is the second container. This module is short and entirely hands-on: prove it is there,
talk to it directly, and read what it holds.

## A list is not a database

The obvious first implementation of a vector store is an array in the api process: embed the
corpus at startup, keep `{id, vector, text}[]` in memory, loop over it with a cosine similarity on
every request. It works, and at 294 vectors it is fast. It fails on four fronts that have nothing to
do with speed. **Persistence:** restart the process and the vectors are gone; you re-embed the corpus
on every deploy, every crash, every autoscale event — seconds here, a coffee break on a real rule
book. **A process that outlives the request:** the Angular front end calls an endpoint, the endpoint
scales to three replicas, and now there are three lists that each embedded the corpus separately
and can drift. **An API:** nothing else can read the list — no second service, no debugging tool,
no `curl`. **Structure:** a list has no collections and no filters; "search only the Q3 edition" or
"only the fare sheets" has to be a loop after the search, and you cannot express "only the current
SOP" in a cosine similarity.

A vector database is the same nearest-neighbour search with those four things added: vectors on
disk, behind a port, in named collections, with metadata you can filter on *before* the search.
Chroma is the smallest thing that does all four, which is why it is in the compose file. It is not
the embedder — the `api` computes every vector with `bge-m3` through your Ollama and hands Chroma
finished numbers. Chroma is storage and search, nothing else, and the rest of this page is looking
at it from three sides.

## Surface 1 — Podman Desktop

**Podman Desktop** — open the Containers view.

Two containers, both running: one from `docker.io/chromadb/chroma:1.5.9` listening on `8000`, one
built from the repo listening on `3000`. **Ollama is not one of them.** It runs on your machine,
because that is where the GPU and the 4.5 GB of models you pulled in [setup](/modules/00-setup/)
already are; the api reaches it through
`host.containers.internal:11434`. If you expected three boxes, this is the moment to correct the
mental picture: the containers are the database and the pipeline, the model runtime is the host.

Now open the Volumes view in the left sidebar: one volume, named after `chroma-data` with the
project name as prefix, mounted into the chroma container at `/data`. That named volume is where the
294 vectors are. Delete the container and the volume stays; `podman compose down -v` is the only
compose command that removes it.

<div class="presenter-note">
This is the module where the Windows laptops' <code>host.containers.internal</code> problem
surfaces, if it exists. Before you start, ask for a show of hands: who saw
<code>"ollama": "unreachable at http://host.containers.internal:11434 — start Ollama on your
machine"</code> in <code>00-health</code> this morning while <code>ollama list</code> works in their
own terminal? That is Ollama bound to 127.0.0.1 only — set <code>OLLAMA_HOST=0.0.0.0</code>, restart
Ollama, re-run <code>00-health</code>. <code>UNVERIFIED</code> on the actual fleet; it is still the
number one risk of the day, and you will know within two minutes of this module whether it bit.
Anyone still red after that pairs with a neighbour for the Bruno part — their Chroma is fine, only
the embed step needs Ollama, and the embed step already ran in module 5.
</div>

## Surface 2 — Terminal

**Terminal (repo root of `amadeus-rag-lab`):**

```bash
podman ps
curl localhost:8000/api/v2/heartbeat
podman compose logs chroma
```

`podman ps` is the same two rows as the GUI — check the chroma row says `(healthy)`, because that
word is load-bearing in a minute.

The `curl` is the one line of this module to remember. It talks to **Chroma itself, no api in
between**: the database's own HTTP API, on the port the compose file publishes. A healthy server
answers with a small JSON object whose one key is `nanosecond heartbeat`. **The path is `/api/v2/`,
not `/api/v1/`** — Chroma 1.x removed the v1 paths, and most of the ChromaDB answers on the internet
were written against v1. Against this server `/api/v1/heartbeat` returns `410 Gone` with a body
that says to use the v2 APIs, and that status line on its own looks exactly like a container that
did not come up. Read the body before you spend twenty minutes debugging Podman.

`podman compose logs chroma` is where you go when `/health` says `chroma: unreachable`. It is also
where you see what the `(healthy)` flag means: the compose file's healthcheck opens a TCP socket to
`127.0.0.1:8000` inside the container every five seconds, sends the same heartbeat request you just
typed, and greps the reply for `nanosecond`. The `api` service declares
`depends_on: chroma: condition: service_healthy` — **that line is why the api waits** instead of
starting, failing to connect, and crashing in a loop on first boot.

## Surface 3 — Bruno

**Bruno — `02-ingest` › `peek-chunks`:**

```text
GET {{baseUrl}}/chunks?source=fare_classic_shorthaul&limit=20
```

No body. `source` is a document id — the filename without `.md` — and the api turns it into a
metadata filter: `where: { source: "fare_classic_shorthaul" }` against the current collection,
returning `documents` and `metadatas` for up to 20 rows. Read the response top down:

- `collection` — `kraken-2026-Q3-fixed-280`. **One collection per configuration.** The name is
  `kraken-<edition>-<strategy>-<chunkSize>`, with `-ov<n>` if you asked for overlap and `-strip` if
  you stripped boilerplate. Every `/ingest` you run today lands in its own collection; nothing is
  overwritten.
- `total` — the whole collection, 294 on this edition and strategy. `returned` — how many matched the
  filter: 15, because that is how many 280-character pieces `fare_classic_shorthaul.md` cuts into.
- `chunks[]` — each with `id` (`fare_classic_shorthaul#<n>`), `source`, `chars`, and `text`. These
  are the pieces of `fare_classic_shorthaul.md` as they sit in Chroma right now, 280 characters each,
  cut wherever 280 fell.

Now do the exercise the request's docs tab asks for. **Find the chunk whose `text` contains the row
starting `| K |`.** Then find the chunk that holds the table header — the one that says which column
is "Cancellation penalty". **They are in different chunks** — two apart, with the O and T rows in
between. The K row carries `EUR 70 | EUR 90 |
EUR 180` and nothing that says which of those three is the cancellation penalty. On the 12 Sep run
this is how fixed-280 RAG read the wrong column and answered "EUR 13" — your number will differ,
your chunk boundaries will not, because the cut is deterministic.

<div class="presenter-note">
Do not narrate the K-row hunt — give the room ninety seconds of silence to find both chunks
themselves, then ask someone to read the K-row chunk aloud, start to finish. Nobody can say what
EUR 90 is from that text, and that is the whole gate into module 7. If Ollama is down for some
people, this request still works: <code>/chunks</code> never touches Ollama, only Chroma, and the
collection was built in module 5. If someone gets <code>409</code> here they restarted the api since
module 5 — send them to the next section rather than fixing it for them; the 409 is the lesson.
</div>

## What survives a restart, and what forgets on purpose

Two different memories are in play, and they are deliberately not the same.

**Chroma keeps everything.** The named volume outlives `podman compose down` and `up`; every
collection you ingest today is still there tomorrow, and `GET /health` lists them all under
`collections`. The only ways to lose them are `podman compose down -v` or
**`DELETE /collections`**, which drops every collection by name and returns `{ deleted: [...] }`. Use
the second when you want a clean room — modules 7 and 8 add several more collections and you may
want to start from zero.

**The api forgets its current collection.** `/retrieve`, `/query` and `/chunks` default to the
*last-ingested* collection, and that name is held in memory in the api process — on purpose, not in
a file, not in Chroma. Restart the api and the next `/retrieve` returns **`409`** with `Nothing has
been ingested yet. Call POST /ingest first (Bruno folder 02).` The data is still in the volume; the
pointer to it is gone. Prove both halves in one move:

**Terminal (repo root of `amadeus-rag-lab`):**

```bash
podman compose restart api
```

Then re-run **Bruno — `02-ingest` › `peek-chunks`**: `409`. Now add `&collection=kraken-2026-Q3-fixed-280`
to the URL and send it again: `200`, same chunks. The collection was never gone; the api's opinion
about which one is *current* was. One `/ingest` call brings the pointer back — which is why the
409 message points at Bruno folder 02 rather than apologising.

## What you run

- **Podman Desktop** — Containers view: two running, `chroma` marked healthy, no Ollama. Volumes:
  `chroma-data`.
- **Terminal (repo root of `amadeus-rag-lab`)** — `podman ps`, `curl localhost:8000/api/v2/heartbeat`,
  `podman compose logs chroma`. Optional: `podman compose restart api` for the 409.
- **Bruno — `02-ingest` › `peek-chunks`** — `GET /chunks?source=fare_classic_shorthaul&limit=20`.
  Read `collection`, `total`, `returned`, then hunt for `| K |` and the header in `chunks[].text`.
- **How long:** twelve minutes, none of it waiting on a model. The only slow call in this module is
  the re-ingest after a restart, and that is optional.
- **If `curl` says connection refused:** the container is not up or 8000 is taken —
  `podman compose logs chroma`, and on macOS check `podman machine` is started. If it answers
  `410 Gone`, you typed `v1`.

## Going deeper

**What Chroma is told, and what it is not.** The api opens every collection with
`embeddingFunction: null` and `hnsw: { space: "cosine" }`. The first means Chroma never embeds
anything — hand it text without a vector and there is nothing to fall back on, which is the
behaviour you want when the embedder is a decision you made (`bge-m3`, 1024 dimensions) and not
one a database made for you. The second names the distance: the `score` you read in `/retrieve` is
cosine similarity in that space, and the `0.35` threshold only means something because the space is fixed per collection.
That is also why a collection is per *configuration*: vectors from a different embedder or a
different cut do not belong in the same index.

**Where the index stops being free.** At 294 vectors the search is effectively exhaustive and any
latency you feel is the embedding call, not Chroma. Underneath, Chroma builds a Hierarchical
Navigable Small World graph — each vector linked to a handful of near neighbours, a query walking
greedily through layers instead of scanning everything — which is what keeps search fast as a
collection grows and what makes it *approximate*: the walk can settle for the second-best answer.
Nothing here is big enough to measure that. As a rule of thumb, not a measurement on this corpus,
brute force stays a legitimate production answer to somewhere around a hundred thousand vectors;
above that, the first question is whether you already run Postgres (`pgvector`: one backup story,
one transaction boundary, a join to the business columns) before it is which vector database.

<div class="presenter-note">
Timing: twelve minutes — two on the list-versus-database argument, three on Podman Desktop and the
Windows check, two on the terminal (the <code>v2</code> line is the only one to say twice), four on
peek-chunks including the silence, one on the restart. The live requests total under a minute.
<br /><br />
<strong>Running late:</strong> drop the restart section and Going deeper, keep the heartbeat and the
K-row hunt — the K row is the gate into module 7 and cannot be skipped; everything else on this page
can be said in one sentence while <code>podman ps</code> is on the projector.
<br /><br />
If someone asks "so Chroma or pgvector in production?", answer with the two questions rather than a
product: how many vectors, and do you already run Postgres. Refuse to recommend a database you have
not measured on their corpus.
</div>

## Exit line

> We can see the chunks now. The one that holds `| K |` has three EUR amounts and no header — the
> database stored exactly what the pipeline gave it, and the pipeline gave it a row with no column
> names. [Module 7](/modules/07-chunking-and-noise/) cuts the corpus differently.
