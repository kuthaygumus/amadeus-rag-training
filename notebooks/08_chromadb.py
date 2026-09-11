# %% [markdown]
# # 08 · ChromaDB: a library, then a service
#
# Every notebook so far kept its vectors in a Python list and compared them with a loop. That is
# fine for 28 documents and it is the right way to learn what is happening, but it is not what you
# would run.
#
# A vector database gives you three things a list does not: it survives a restart, it searches
# without comparing against every vector, and it can be a service that several processes share.

# %%
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(embed=True, chroma_default=True)
import _cached
import chromadb, retrieval as R

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
ids, texts = list(docs), list(docs.values())
print(f"chromadb {chromadb.__version__} · {len(ids)} documents")

# %% [markdown]
# ## Embedded mode — the two-line version
#
# This is what almost everyone writes first, and it works immediately.
#
# One thing to know before you run it. `add()` without an `embeddings=` argument makes Chroma
# fetch its own embedding model — 83,178,821 bytes, once per machine. Do that at home with
# `scripts/seed_offline_assets.py` rather than on the day: twenty laptops fetching 83 MB over
# one room's connection at the same moment is a long wait for nothing. If it is not there, the
# next cell replays a recorded run and says so rather than stalling.

# %%
MINILM_CACHED = _preflight.chroma_default_cached()
NOT_SEEDED = ("Chroma's default model is not on this machine — replaying the recorded run. "
              "Seed it at home with scripts/seed_offline_assets.py.")
client = chromadb.EphemeralClient()          # PersistentClient(path=...) to keep it on disk

def index_with_chromas_default() -> dict:
    collection = client.get_or_create_collection("kraken")
    collection.add(ids=ids, documents=texts)             # note: no embeddings= argument
    # Read the width off a stored vector rather than asserting it. Chroma does not expose the
    # model name on the embedding function, so the model is identified by the cache directory
    # it was loaded from — which is the only place on this machine that names it.
    stored = collection.get(ids=ids[:1], include=["embeddings"])["embeddings"][0]
    return {"count": collection.count(),
            "embedding_function": type(collection._embedding_function).__name__,
            "dimensions": len(stored),
            "loaded_from": _preflight.display_path(_preflight.chroma_default_dir())}

indexed = _cached.run("08-chroma-default-index", index_with_chromas_default,
                      live=MINILM_CACHED, unavailable=NOT_SEEDED)
print(f"{indexed['count']} documents indexed")

# %% [markdown]
# Notice what we did **not** do: we never said how to turn text into vectors. Chroma picked
# something.
#
# **Before running the next cell, guess:** what did it pick, and is it the right choice for a
# corpus that holds Turkish documents and gets asked questions in Turkish all day?

# %%
print("embedding function:", indexed["embedding_function"])
print("dimensions        :", indexed["dimensions"], "(read off a stored vector)")
print("loaded from       :", indexed["loaded_from"])

# %% [markdown]
# The class name Chroma hands you does not say which model it is, and the width is read off a
# stored vector rather than asserted. The path is what names it: that directory is
# `all-MiniLM-L6-v2`, a 384-dimension English sentence encoder. Module 6 measured it at **0.000**
# on the six Turkish-query/English-document questions, against bge-m3's 0.667.

# %%
DEFAULT_QUERIES = [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("Baglantisini kaciran yolcuya yemek fisi ne kadar?", "sop_misconnect_v4"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]

def query_chromas_default() -> list[dict]:
    collection = client.get_or_create_collection("kraken")
    return [{"question": question, "should_be": should_be,
             "top": collection.query(query_texts=[question], n_results=3)["ids"][0]}
            for question, should_be in DEFAULT_QUERIES]

for row in _cached.run("08-chroma-default-queries", query_chromas_default,
                       live=MINILM_CACHED, unavailable=NOT_SEEDED):
    verdict = "right" if row["top"][0] == row["should_be"] else "WRONG"
    print(f"{verdict:>5}  {row['question'][:50]:<52} -> {row['top'][0]}")

# %% [markdown]
# Three questions, three wrong documents. No exception, no warning, nothing in the output
# suggesting anything is off.
#
# `pip install chromadb`, add your documents, query — and you are silently running an English-only
# embedding model over a Turkish corpus. Nobody tells you. The system does not fail; it just
# answers wrong, quietly, forever.
#
# This is module 6's entire argument, and it is worth reproducing on your own machine rather than
# taking my word for it.

# %% [markdown]
# ## The same collection, with an embedder we chose
#
# Chroma will accept vectors you computed yourself. Compute them with `bge-m3` through Ollama —
# the model module 6 measured at 0.667 on the six Turkish-query/English-document questions,
# against 0.000 for the all-MiniLM-L6-v2 that Chroma just chose for us.

# %%
better = client.get_or_create_collection("kraken_bge_m3")
start = time.time()
better.add(ids=ids, documents=texts, embeddings=R.embed(texts))
print(f"indexed with bge-m3 in {time.time() - start:.1f}s\n")

for question, should_be in [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]:
    top = better.query(query_embeddings=R.embed([question]), n_results=3)["ids"][0]
    print(f"{'right' if top[0] == should_be else 'WRONG':>5}  {question[:50]:<52} -> {top[0]}")

# %% [markdown]
# Better, and still not perfect at document level — which is what modules 6 and 7 spend their time
# on. The point here is narrower: **the embedding model is a decision, and Chroma will make it for
# you if you do not.**

# %% [markdown]
# ## Metadata filtering — the thing a list cannot do
#
# Retrieval is rarely "search everything". It is "search the current quarter's fare rules, not the
# superseded SOPs".

# %%
kinds = {i: ("fare" if i.startswith("fare") else "sop" if i.startswith("sop")
             else "bulletin" if i.startswith("bulletin") else "other") for i in ids}
tagged = client.get_or_create_collection("kraken_tagged")
tagged.add(ids=ids, documents=texts, embeddings=R.embed(texts),
           metadatas=[{"kind": kinds[i]} for i in ids])

question = "cancellation penalty"
everything = tagged.query(query_embeddings=R.embed([question]), n_results=3)["ids"][0]
fares_only = tagged.query(query_embeddings=R.embed([question]), n_results=3,
                          where={"kind": "fare"})["ids"][0]
print(f"unfiltered : {everything}")
print(f"kind=fare  : {fares_only}")

# %% [markdown]
# Filtering before the search rather than after is the difference between a database and a loop.
# You cannot express "only the current SOP" in a cosine similarity.

# %% [markdown]
# ## Server mode
#
# Same collection, same API, different owner. In production the vector store is not inside your
# process — it is a service, with its own uptime, its own backups, and its own way of being down.
#
# ```bash
# podman compose up -d          # or: docker compose up -d
# curl http://localhost:8000/api/v2/heartbeat
# ```
#
# The compose file is in the repository root. Note the path: **`/api/v2/`**. The v1 path is all
# over the internet and this version answers it with `410 Gone`.

# %%
try:
    server = chromadb.HttpClient(host="localhost", port=8000)
    print("server heartbeat:", server.heartbeat())
    remote = server.get_or_create_collection("kraken_remote")
    remote.add(ids=ids, documents=texts, embeddings=R.embed(texts))
    top = remote.query(query_embeddings=R.embed(["cancellation penalty for class K"]),
                       n_results=3)["ids"][0]
    print("query against the server:", top)
except Exception as e:
    print("no server running —", str(e)[:120])
    print("start it with: podman compose up -d")

# %% [markdown]
# The code above is the same code as the embedded version, with `EphemeralClient` swapped for
# `HttpClient`. That is the whole difference at the API level, and it is why this module is short.
#
# What actually changes is everything around the API. Embedded, the index dies with your process
# and nobody else can read it. As a service, it outlives your process, several workers share one
# index, and you have a new thing that can be down at three in the morning.

# %% [markdown]
# ## Going deeper — why it is fast, and what "approximate" costs
#
# Comparing a query against 28 vectors is 28 comparisons. Against ten million it is ten million,
# and that is too slow to do per request.
#
# Chroma uses **HNSW**: a layered graph where each vector links to its neighbours, with sparse
# long-range links on the upper layers. A search starts at the top, follows the closest link
# repeatedly, and drops a layer — arriving near the answer in roughly log-many steps instead of
# scanning everything.
#
# It is *approximate*: it can miss a true nearest neighbour that sits behind a gap in the graph.
# That is a knob, not a defect. `ef_search` trades recall against latency, `M` trades index size
# and build time against both. At 28 documents none of this matters and exact search would be
# fine. At ten million it is the difference between a product and a demo.
#
# **When you do not need any of this.** Under roughly a hundred thousand vectors, `pgvector` in a
# Postgres you already run will serve you well and removes an entire system from your
# architecture. Reach for a dedicated vector database when you need filtered search at scale,
# multiple writers, or an operational story of its own — not because the tutorial used one.

# %% [markdown]
# > **Same API, different deployment.**
