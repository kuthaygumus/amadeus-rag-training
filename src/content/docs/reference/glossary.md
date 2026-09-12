---
title: "Glossary"
description: "Every term the day uses, defined once, with the measured number attached where there is one."
---

Terms are grouped by the module that introduces them. Where we measured something, the number is
here too — a definition you cannot check is a definition you will misremember. Every retrieval
figure on this page is **trainer-side**: `npm run eval` in `amadeus-rag-lab` (`eval/results.md`),
one run on 12 Sep 2026 on an M-series Mac, and it drifts by a question or two on another machine.
Participants never run it. Anything marked *Going further* is beyond the day.

## The day

**Gate** — the one question a module has to answer before the next one opens, asked as a show of
hands before the request runs. Eight gates and one setup; module 9 reads them backwards. A gate
is not a quiz — it is the moment the room commits to a guess the next click will test.

**Edition** — one quarter's rule book as a folder: `corpus/2026-Q2/` (21 documents) and
`corpus/2026-Q3/` (28). `edition` is a request parameter on `/ingest` and `/chat`; `corpus/DELTA.md`
lists every change between the two. The day turns on one cell of it: CLASSIC K, EUR 120 → EUR 90.

## Models and training

**Weights (parameters)** — the numbers a model is made of; module 2's digit classifier has
101,770. After training they never change again, which is why a fine-tuned model goes stale.

**Training** — the loop that adjusts weights: run an input forward, measure how wrong it was,
move each weight a little in the direction that helps, repeat. Module 2's trainer demo runs 60,000
images through it five times in 0.92 seconds.

**Loss** — a single number saying how wrong the model currently is. Training is the process of
making it smaller. Ours falls from 2.35 on the first batch to 0.03 on the last.

**Epoch** — one full pass over the training data. Most of the learning happens in the first:
accuracy goes 9.9% → 95.35%, peaks at 97.62% in epoch 4 and slips to 97.47% in epoch 5.

**Prior** — what the weights already believe before your prompt arrives. Module 1's refusal is a
prior, not a knowledge check: "I don't know" was rewarded during instruction tuning for questions
*shaped* like a lookup, so rewording the question removes it. A system prompt tunes the prior; it
does not install a lookup.

**SFT (supervised fine-tuning, instruction tuning)** — next-token training on (instruction,
response) pairs instead of raw text. Pairs teach a behaviour — *asked like this, answer like that* —
and, pushed hard enough, some facts. The `-Instruct` suffix means it happened; module 3's demo is
SFT on 695 pairs from the 2026-Q2 book.

**RLHF / DPO** — training on comparisons rather than answers: two responses, and which one a
human preferred. RLHF fits a separate reward model; DPO optimises the comparison directly and
skips it. Both re-rank what the model could already say — they do not put a number into it.

**Fine-tuning** — training an already-trained model further, on your own data. It works, and the
result is still frozen: module 3's `kraken-q2` adapter is trained on the 2026-Q2 book's EUR 120
CLASSIC K penalty and cannot notice that 2026-Q3 reprices the row at EUR 90
(`UNVERIFIED: the adapter has not been built yet`).

**LoRA** — Low-Rank Adaptation. Freeze the large weight matrix and learn two much smaller ones —
the *adapter* — whose product is added to it. **Rank** is capacity, **alpha** is scale. Module 3's
configuration (rank 32, alpha 64) trains 2.34% of the model's parameters, so the result ships as
a few megabytes.

**QLoRA** — LoRA over a base quantised to 4 bits while training. The base is only read, never
written, so the precision loss costs less than you would expect. It changes which GPU you need, not
what fine-tuning is; module 3 does not need it on a 16 GB T4.

**PEFT** — parameter-efficient fine-tuning, the umbrella for LoRA and its relatives; also the
HuggingFace library, `peft`, that wraps a loaded model in adapters (`LoraConfig(r=32, …)`) and
merges them back afterwards. Python, downloads from the Hub — which is why module 3 runs in Colab.

**Quantisation** — storing weights at lower precision (16 bits, 8, 4) to trade a little quality
for a lot of memory. Module 3 quantises its fine-tuned GGUF to `Q4_K_M` before Ollama serves it.

**GGUF** — the file format Ollama serves models from. Converting a fine-tuned model to GGUF is
what lets a laptop run it with no GPU and nothing downloaded from HuggingFace.

**Modelfile** — the few lines that register a GGUF file with Ollama under a name and pin its
defaults. Module 3's pins `temperature 0`, so ten identical answers to one question is how you
tell the fine-tune took.

**Context window** — how many tokens the model can read at once. The api asks Ollama for a
65 536-token window for `gemma3:4b` (`numCtx: 65536`); the whole Q3 corpus is about 21 000 prompt
tokens, so it fits. Module 4's arithmetic: 10 000 documents is about 7.5 million, and nothing fits
that.

**Token** — the unit a model reads text in; about four characters here, since 79 000 characters of
corpus became about 21 000 prompt tokens. Not a word, and not a character.

**Prefix cache (prompt caching)** — Ollama keeps the processed front of the last prompt. Module 4's
stuffed question took 60–120 s cold and about 1 s repeated unchanged; change one word and it is
slow again. The cache lives in one process, on one laptop, and expires on the next bulletin.

**Temperature** — how much randomness the model is allowed when it picks the next token. Every
request in the lab runs at `temperature: 0`. It removes run-to-run randomness. It does not make a
wrong answer right — module 1 is wrong at temperature 0.

**Hallucination** — a fluent, confident answer grounded in nothing. Module 1 asks `gemma3:4b` the
K-row question and on the 12 Sep run got an invented amount of the "€50" kind, stated flat, against
a real answer of EUR 90. The tell is not that it is wrong; it is that it fails in the same voice it
uses when it succeeds.

## Retrieval

**RAG** — retrieval-augmented generation: find the relevant text at question time, put it in the
prompt, answer from it. Not because stuffing the whole book fails at 28 documents — module 4 shows
it fits and answers — but because RAG pays for one piece instead of the book, still works at
10 000 documents, and cites.

**Embedding** — text turned into a list of numbers, positioned so that similar meanings land near
each other. `bge-m3` returns 1024 of them for any input, one word or one page. The question must be
embedded with the same model as the chunks, or the distance means nothing.

**Cosine similarity** — the cosine of the angle between two embedding vectors: 1.0 when they
point the same way, 0 when unrelated. It is the `score` on every hit in `/retrieve`, in a collection
opened with `space: "cosine"`. Semantic search is this measurement plus a sort.

**Multilingual embedder** — one trained so that text in different languages sharing a meaning
lands in the same region. The lesson that decides the day: an English-only embedder fails Turkish
questions over English documents — "being Turkish" becomes a bigger axis than "being about
cancellation penalties". `bge-m3` was chosen before you pulled the weights.

**Threshold / abstain** — the similarity floor a hit must clear to reach the prompt; default
`0.35`. Below it hits go to `dropped`, shown rather than hidden. If nothing clears it the answer is
`abstained: true` and no model call is made — the one failure mode that is not a hallucination.

**Top-k** — how many results a stage passes on; `topK: 3` is what the model reads, `recall@5` is
what a metric counts. A document at rank five scores as found and is never seen by the generator.
A metric quoted without its k is not a number.

**Chunk** — the slice of a document you actually index. Retrieval never sees a document; it sees
whatever you cut. The day's failure is a chunk: the K row in one piece, the column header that
names its numbers in another.

**Fixed-size chunking** — cut every N characters, blind to content. Default 280; 294 chunks on Q3.
Trainer-side hit@1 0.700 — the right *document* seven times in ten, and still the villain, because
the header line lands in the chunk before the row.

**Overlap** — each chunk carries the tail of the previous one. Sixty characters on fixed-280 takes
294 chunks to 368 and loses on every trainer-side metric: hit@1 0.700 → 0.550, recall@5 0.917 →
0.883, MRR 0.817 → 0.717. It moves a boundary; it does not remove one.

**Recursive chunking** — split at the most natural boundary that fits: headings, paragraphs, lines,
sentences. Default 600; 197 chunks; trainer-side 0.700 / 0.900 / 0.806. The framework default, and
a nine-column header plus seven rows is wider than 600 characters, so the table is still cut.

**Structure-aware chunking** — split at the document's own headings and numbered rules, and prefix
each chunk with `[doc title > heading]`. Default 1500 because the RULE 2A section is 1 140
characters; 132 chunks; trainer-side hit@1 0.700. It does not find the document more often — it
fixes the chunk, header and K row together, which is what turns a wrong amount into EUR 90.

**Contextual retrieval** — the industry name for that prefix: give each fragment enough
surrounding context to stand alone. Module 7 ships it without naming it; *Going further* finds the
win already banked.

**Boilerplate stripping** — removing, before chunking, what every export carries: the repeated
legal block, `Page n of n`, stray tags, `&nbsp;`. Chunk count does not move (132 → 132) because the
block rides inside the last rule's chunk; the vectors move. Trainer-side 0.650 / 0.850 / 0.766 — one
question lower on hit@1, read the direction.

**Source / chunk id** — every chunk carries the document it was cut from; `sources[].source` is
what the answer cites. The trainer-side scores count at document level — the best chunk of the
right document is a hit — which is exactly why they cannot see a header cut from its row.

**BM25** — keyword ranking by term frequency, inverse document frequency and document length. No
model, no training. It matches strings: *iptal* and *Cancellation penalty* are different strings,
so a Turkish question over an English sheet fails. Earns its place back on flight numbers and
bulletin ids. *Going further.*

**Sparse vs dense retrieval** — sparse is BM25 and its relatives, matching words; dense is
embeddings, matching meaning. They fail at different things, which is the whole argument for
combining them — and for measuring the combination.

**Hybrid search** — running sparse and dense together and merging the rankings. In the earlier
course's measurement it lost: fusing a sound ranking with an unsound one diluted the sound one.
*Going further.*

**RRF (Reciprocal Rank Fusion)** — the usual merge: sum 1/(k + rank) across the rankings. It
rewards documents every input ranking agrees on, which only helps when every input is sound.
*Going further.*

**Reranking** — reordering the top few results with a second, slower model. A trade, not an
upgrade: in the earlier course it lifted weak retrievers and pulled strong ones down — it levels
to its own ceiling. *Going further.*

**Bi-encoder vs cross-encoder** — a bi-encoder embeds query and passage separately, so passage
vectors are stored once; a cross-encoder reads the pair together and scores relevance directly,
so nothing is precomputable. Nothing this course measured was a cross-encoder. *Going further.*

**Pointwise vs listwise reranking** — scoring each candidate on its own, versus asking for the
whole list in order. Handed six passages, a small chat model returned four indices listwise and
six usable scores pointwise. That decides whether reranking works at all, not whether it helps.
*Going further.*

**Agentic RAG** — retrieval as a tool the model calls rather than a step that runs before it. It
breaks the question up, searches, checks whether it has enough, searches again. The cost is six
to ten calls against simple RAG's one, plus determinism. *Going further.*

**Query decomposition** — the first call in that loop: turn one question into standalone
sub-questions, capped at four. Query rewriting with a budget. *Going further.*

**Sufficiency check** — the second: ask the model whether what it has gathered answers the
question — `YES`, or `NO` plus one more query. The loop's stopping condition and its weakest
part. *Going further.*

**Multi-hop** — a question whose answer is spread across documents that no single search returns:
the misconnect SOP, the interline agreement and the fare rules at once. The lab never shows you
one; a single pass cannot finish it, and the loop above is the answer. *Going further.*

## Measurement

**Gold set** — a fixed list of questions with the documents that should come back for each. Ours
is 20, the day's question is number 15, and it does not change between configurations, so the same
three numbers are comparable all day. Remember that **twenty questions is not a benchmark**: one
question is worth 0.05, and almost every gap on the ladder is exactly one question wide. Quote the
direction, not the gap.

**hit@1** — was the top-ranked document a right one? The strictest and most honest single number.
Scored at document level: the best chunk of the right document counts.

**recall@5** — how much of the right set appeared in the top five. Beware: it is *highest* for
the most fragmented index — 0.917 for fixed-280 against structure-1500's 0.833 — because 294
chunks give the gold document more chances to appear than 132 do.

**MRR (Mean Reciprocal Rank)** — one over the rank of the first correct document, averaged over
the questions. The one that shows partial credit: moving a gold document from rank 6 to rank 2
leaves hit@1 unchanged and lifts that question's contribution from 0.17 to 0.50.

**LLM-as-judge** — scoring answers with another model; common, and deliberately not used for any
number in this course, because the score moves between runs.

## Infrastructure

**Vector database** — a nearest-neighbour search with four things an in-memory list lacks:
vectors on disk, behind a port, in named collections, with metadata filtered *before* the search.
ChromaDB 1.5.9 here, v2 API on port 8000. It is not the embedder: the api opens every collection
with `embeddingFunction: null` and hands it finished `bge-m3` vectors.

**Collection (Chroma)** — ChromaDB's unit of storage: ids, documents, embeddings and metadata. One
per configuration, named for what produced it — `kraken-2026-Q3-structure-1500-strip` — because
vectors from a different cut or embedder do not belong in the same index. `/retrieve` and `/query`
default to the last-ingested one; `409` if there is none; `DELETE /collections` drops them all.

**ANN / HNSW** — approximate nearest-neighbour search: a graph of near neighbours walked greedily
instead of a scan, trading a little recall for a lot of speed. At 294 vectors nothing here is big
enough to notice; brute force stays honest to around a hundred thousand.

**Embedded vs server mode** — ChromaDB as a library inside your process, or a service you talk to
over HTTP. Same API, different operational ownership. The server's path is `/api/v2/`, not v1.

**Named volume** — `chroma-data`, mounted into the chroma container at `/data`. It outlives
`podman compose down` and `up`; only `down -v` or `DELETE /collections` empties it. The api's
pointer to the *current* collection is not in it — restart the api and the next `/retrieve` is a
`409` until one `/ingest`.

**Healthcheck / `service_healthy`** — the compose file probes `chroma` at `/api/v2/heartbeat` and
starts `api` only on `depends_on: chroma: condition: service_healthy`. That line is why the api
waits instead of crashing on its first request.

**pgvector** — the Postgres extension that stores vectors in a table you already back up and can
join to the business columns beside them. Module 6's first question past a hundred thousand
vectors: do you already run Postgres?

**Ollama** — runs models locally and serves them on `localhost:11434`. Not a container: it stays on
the host with the GPU and the pulled weights, and the api reaches it at
`host.containers.internal:11434`. No API key, no cloud account, no sign-in.

**Podman** — rootless, daemonless container runtime. Runs `chroma` and `api` from the compose
file instead of Docker: no privileged daemon, and no licensing question in a corporate setting.
