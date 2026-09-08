---
title: "Glossary"
description: "Every term the day uses, defined once, with the number attached where we measured one."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this
> course is synthetic and written for teaching. No Amadeus system, customer or production data
> is used anywhere in this repository.

Terms are grouped by the module that introduces them. Where we measured something, the number is
here too — a definition you cannot check is a definition you will misremember.

## Models and training

**Weights (parameters)** — the numbers a model is made of. The digit classifier in module 2 has
101,770 of them. Everything the model knows is a consequence of their values, and after training
they do not change again. This is why a fine-tuned model goes stale: there is no mechanism inside
a set of numbers for finding out the world moved on.

**Training** — the loop that adjusts weights: run an input forward, measure how wrong the output
was, compute which direction each weight should move, move it a little, repeat. Module 2 runs
60,000 images through this five times in 0.85 seconds.

**Loss** — a single number saying how wrong the model currently is. Training is the process of
making it smaller. Ours falls from 2.35 on the first batch to 0.03 on the last.

**Epoch** — one full pass over the training data. Most of the learning happens in the first one;
our accuracy goes 9.87% → 95.35% after epoch one, then crawls to 97.47% over four more.

**Fine-tuning** — training an already-trained model further, on your own data. It works, and the
result is still frozen: the adapter in module 3 is trained on the 2026-Q2 rule book, where the
CLASSIC K cancellation penalty is EUR 120. The 2026-Q3 book prices the same row at EUR 90, and
nothing inside a set of trained weights can notice that the row changed.

**LoRA** — Low-Rank Adaptation. Instead of updating a large weight matrix, freeze it and learn two
much smaller matrices — the *adapter* — whose product is added to it. **Rank** (`r`) is capacity:
how many independent directions the update may move in. **Alpha** is scale: the adapter's output is
multiplied by `alpha/r`, so you can raise rank without raising the strength of the update. Module 3
trains `r = 32, alpha = 64` on Qwen2.5-1.5B, whose `q_proj` is 1536×1536. That is 2 × 1536 × 32 =
98,304 trainable numbers against the full update's 2,359,296 — **4.17%** — and 36,929,536 of the
model's 1,580,643,840 parameters overall, **2.34%**. Which is why a fine-tune fits on one GPU and
ships as a few megabytes.

**QLoRA** — LoRA with the frozen base model quantised to 4 bits. The base is only read, never
written, so the precision loss costs less than you would expect and a 7B model becomes trainable
on consumer hardware.

**RLHF / DPO** — training on comparisons rather than on answers: two responses, and which one a
human preferred. RLHF fits a separate reward model and optimises against it; DPO optimises the
comparison directly and skips the reward model, which is why most teams reach for it first.

**Quantisation** — storing weights at lower precision (16 bits, 8, 4) to trade a little quality
for a lot of memory. The chat models here are 4-bit — `ollama show qwen2.5:3b` reports `Q4_K_M`.
The two embedders are not: `bge-m3` and `nomic-embed-text` both ship at `F16`.

**GGUF** — the file format Ollama serves models from. Converting a fine-tuned model to GGUF is
what lets participants run it with no GPU and without downloading anything from HuggingFace.

**Modelfile** — the few lines that register a GGUF file with Ollama under a name and pin its
defaults. Module 3's pins `temperature 0`, so asking the built model the same question ten times
and getting anything other than ten identical answers means the fine-tune did not take.

**Context window** — how many tokens the model can read at once. `qwen2.5:3b` has 32,768. Our
28-document corpus is 78,310 characters, roughly 26,000 tokens, which is why stuffing the whole
thing in the prompt works here and stops working at ten times the size.

**Token** — the unit a model reads text in, roughly three characters for mixed English and
Turkish. Not a word, and not a character.

**Temperature** — how much randomness the model is allowed when it picks the next token. Every
generation in this course runs at `temperature=0.0`, so the same prompt gives the same answer
twice. It removes sampling noise. It does not make a wrong answer right.

**Hallucination** — a fluent, confident answer that is not grounded in anything. Module 1 asks the
same question four ways and gets, in one recorded run at temperature 0, 1.500 TL, 100-200 euro,
10-20 percent, and a refusal. The real answer is EUR 90. The tell is not that it is wrong; the
tell is that the number moves.

## Retrieval

**Embedding** — a piece of text turned into a list of numbers, positioned so that similar meanings
land near each other. `bge-m3` produces 1024 numbers per text, `nomic-embed-text` 768,
`all-MiniLM-L6-v2` 384. More numbers is not automatically better.

**Cosine similarity** — the cosine of the angle between two embedding vectors: 1.0 when they point
the same way, 0 when they are unrelated. Used as a proxy for "how related are these". Everything
called semantic search is this measurement plus a sort.

**Multilingual embedder** — one trained so that text in different languages sharing a meaning
lands in the same region. This is the difference that decides the day: on Turkish questions whose
answer is in an English document, `nomic-embed-text` scores 0.000 and `bge-m3` scores 0.667.

**Chunk** — the slice of a document you actually index. Retrieval never sees a document; it sees
whatever you cut. Moving from whole documents to structure-aware chunks takes hit@1 from 0.550 to
0.800. The embedder is worth more again on those same chunks — `nomic-embed-text` 0.350 against
`bge-m3`'s 0.800 — which is why the day measures both rather than ranking them.

**Fixed-size chunking** — cut every N characters, blind to content. It raises the average
(hit@1 0.550 → 0.700) while exact-token retrieval falls (1.000 → 0.750), and it separates a
table's column header from its rows.

**Overlap** — letting each chunk carry the tail of the previous one. Insurance against cutting a
sentence in half. It does not fix a reference 612 characters away — the distance from our penalty
table's column header to its `K` row: it moves the boundary rather than removing it.

**Recursive chunking** — split on paragraphs, then lines, then sentences. The usual default, and
it still cuts our penalty table away from its header, because that span is 612 characters and the
split size is 600.

**Structure-aware chunking** — split on the document's own headings and numbered rules, and prefix
each chunk with where it came from. The only strategy here that keeps the header with the row.
Sold elsewhere as *contextual retrieval*.

**Boilerplate stripping** — removing the legal footers, page artefacts and leftover markup that
repeat across documents. Removes 4.2% of our corpus and moves hit@1 from 0.800 to 0.850.

**BM25** — keyword ranking by term frequency, inverse document frequency, and document length. No
model, no training. It is close to dense retrieval over whole documents and never ahead of it:
hit@1 0.400 against 0.550, and 0.750 against 1.000 on exact identifiers. Chunking is what breaks
it — 0.300 whichever way you cut, because short chunks give its length normalisation nothing to
work with — and on the six Turkish-query questions it scores 0.000 whether you chunk or not.

**Sparse vs dense retrieval** — sparse is BM25 and its relatives, matching words. Dense is
embeddings, matching meaning. They fail at different things, which is the entire argument for
combining them, and the reason combining them is worth measuring rather than assuming.

**RRF (Reciprocal Rank Fusion)** — merge several rankings by summing 1/(k + rank). It rewards
documents that every input ranking agrees on, which only helps when every input is sound. Fusing
BM25 into our dense retrieval took hit@1 from 0.800 to 0.450.

**Reranking** — reordering the top few results with a second, slower model. Measured here as a
trade rather than an upgrade: across four retrieval setups it lifted the two weak ones
(hit@1 0.350 → 0.450 and 0.350 → 0.400) and pulled the two strong ones down (0.700 → 0.550 and
0.800 → 0.600). It levels, and it levels to its own ceiling.

**Bi-encoder** — query and passage embedded separately, one vector each, compared by cosine.
The passage vectors are computed once and stored, which is what makes dense retrieval with
`bge-m3` cheap at query time. This is what "the embedder" means everywhere else on this page.

**Cross-encoder** — query and passage read *together* in one forward pass, producing a relevance
score directly, trained on relevance labels rather than prompted into the job. Nothing is
precomputable, so it only ever runs over a short candidate list. `bge-reranker-v2-m3` is the
multilingual one. The reranker measured in module 9 is a chat model scoring passages, not a
cross-encoder — we did not measure one, so no number on this page belongs to it.

**Pointwise vs listwise reranking** — scoring each candidate on its own, versus asking for the
whole list in order. Same model, same candidates, opposite outcomes: on the retired 10-document
probe corpus listwise scored 2/5 and pointwise 5/5, MRR 0.600 against 1.000. That is `n = 5`:
treat the direction as real and the magnitude as unproven. How you ask matters more than whether
you ask.

**RAG** — retrieval-augmented generation. Find the relevant text at question time, put it in the
prompt, answer from it. The point is not that it answers better than stuffing the whole corpus —
measured here, it does not — but that it still works when the corpus is a hundred times larger,
and that the answer can cite the file it came from.

**Agentic RAG** — retrieval as a tool the model calls, rather than a step that runs before it. The
model breaks the question up, searches, checks whether it has enough, and searches again. On our
hardest multi-hop question it went from 0 of the 3 required documents to 3 of 3; on the other
multi-hop question it changed nothing, 2 of 3 either way. The cost is six model calls where naive
RAG made one — arithmetic from the loop's shape, not a timed measurement — and any hope of
determinism.

**Multi-hop** — a question whose answer is spread across documents that no single search returns.
Ours needs the misconnect SOP, the interline agreement and the fare rules at once.

## Measurement

**Gold set** — a fixed list of questions with the documents that should come back for each. Ours
is 20 questions and it does not change between modules, so the same three numbers are comparable
at three points in the day.

**hit@1** — was the top-ranked document a right one? The strictest and most honest single number.

**Top-k** — how many results a stage passes on. `hit@1` is measured at k = 1 and `recall@5` at
k = 5; the reranker scores the top 8 candidates, which is where its 8 model calls per query come
from. A metric quoted without its k is not a number.

**recall@5** — how much of the right set appeared in the top five. Beware: it is *highest* for our
worst chunking strategy — 0.950 for fixed-280 against structure-aware's 0.833 — because 294 chunks
give the gold document more chances to appear somewhere than 154 do.

**MRR (Mean Reciprocal Rank)** — one over the rank of the first correct document, averaged over
the questions. The one that shows partial credit: moving a single question's gold document from
rank 6 to rank 2 leaves hit@1 unchanged and lifts that question's contribution from 0.17 to 0.50.

**LLM-as-judge** — scoring answers with another model. Common, and deliberately not used for the
numbers in this course: it costs quota, it is slow, and the score moves between runs. Everything
reported here is deterministic.

## Infrastructure

**Vector database** — storage that indexes embeddings for nearest-neighbour search. ChromaDB here.
Worth knowing that ChromaDB's default embedder is `all-MiniLM-L6-v2`, downloaded silently, and
English-only: on a Turkish corpus it does not error, it returns the wrong document.

**Collection** — ChromaDB's unit of storage: ids, documents, embeddings and one metadata
dictionary per item. `where={"kind": "fare"}` filters on that metadata before the vector search,
which is how a superseded SOP is kept out of an answer.

**ANN / HNSW** — approximate nearest neighbour search. Exact search over millions of vectors is too
slow, so these trade a little recall for a lot of speed. The "approximate" is a knob, not an
apology.

**Embedded vs server mode** — ChromaDB as a library inside your process, or as a service you talk
to over HTTP. Same API, different operational ownership. The server's path is `/api/v2/`, not v1.

**Ollama** — runs models locally and serves them on `localhost:11434`. This course uses it for
everything, which is what makes the day work with no API key, no cloud account and no sign-in.

**Podman** — rootless, daemonless container runtime. Used instead of Docker here for the ChromaDB
server, because it needs no privileged daemon and raises no licensing question in a corporate
setting.
