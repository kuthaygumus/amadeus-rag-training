---
title: "Glossary"
description: "Every term the day uses, defined once, with the measured number attached where there is one."
---

Terms are grouped by the module that introduces them. Where we measured something, the number is
here too — a definition you cannot check is a definition you will misremember. Every figure on
this page comes from `eval/RESULTS.md`.

## Models and training

**Weights (parameters)** — the numbers a model is made of; module 2's digit classifier has
101,770. After training they never change again, which is why a fine-tuned model goes stale.

**Training** — the loop that adjusts weights: run an input forward, measure how wrong it was,
move each weight a little in the direction that helps, repeat. Module 2 runs 60,000 images
through it five times in 0.92 seconds.

**Loss** — a single number saying how wrong the model currently is. Training is the process of
making it smaller. Ours falls from 2.35 on the first batch to 0.03 on the last.

**Epoch** — one full pass over the training data. Most of the learning happens in the first:
accuracy goes 9.9% → 95.35%, then crawls to 97.47% over four more, peaking at epoch 4.

**Fine-tuning** — training an already-trained model further, on your own data. It works, and the
result is still frozen: module 3's adapter learns the 2026-Q2 book's EUR 120 CLASSIC K penalty
and cannot notice that 2026-Q3 reprices the row at EUR 90.

**LoRA** — Low-Rank Adaptation. Freeze the large weight matrix and learn two much smaller ones —
the *adapter* — whose product is added to it. **Rank** is capacity, **alpha** is scale. Module 3
trains 2.34% of the model's parameters, so the result ships as a few megabytes.

**QLoRA** — LoRA with the frozen base quantised to 4 bits. The base is only read, never written,
so the precision loss costs less than you would expect and a 7B model becomes trainable on
consumer hardware.

**RLHF / DPO** — training on comparisons rather than answers: two responses, and which one a
human preferred. RLHF fits a separate reward model; DPO optimises the comparison directly and
skips it, which is why most teams reach for DPO first.

**Quantisation** — storing weights at lower precision (16 bits, 8, 4) to trade a little quality
for a lot of memory. Module 3 quantises its fine-tuned GGUF to `Q4_K_M` before Ollama serves it.

**GGUF** — the file format Ollama serves models from. Converting a fine-tuned model to GGUF is
what lets participants run it with no GPU and nothing downloaded from HuggingFace.

**Modelfile** — the few lines that register a GGUF file with Ollama under a name and pin its
defaults. Module 3's pins `temperature 0`, so ten identical answers to one question is how you
tell the fine-tune took.

**Context window** — how many tokens the model can read at once; `qwen2.5:3b` has 32,768. Our
28-document corpus is 78,310 characters, about 26,436 tokens once stuffed into one prompt. That
fits here and stops fitting at ten times the size.

**Token** — the unit a model reads text in, roughly three characters for mixed English and
Turkish. Not a word, and not a character.

**Temperature** — how much randomness the model is allowed when it picks the next token. Every
generation in this course runs at `temperature=0.0`. It removes run-to-run randomness. It does
not make a wrong answer right.

**Hallucination** — a fluent, confident answer grounded in nothing. Module 1 asks one question
four ways at temperature 0 and gets `1.500 TL`, `100-200 euro`, `%10-20` and a refusal against a
real answer of EUR 90. The tell is not that it is wrong; it is that the number moves.

## Retrieval

**Embedding** — text turned into a list of numbers, positioned so that similar meanings land near
each other. `bge-m3` produces 1024 per text, ChromaDB's default `all-MiniLM-L6-v2` 384. More
numbers is not automatically better.

**Cosine similarity** — the cosine of the angle between two embedding vectors: 1.0 when they
point the same way, 0 when unrelated. Semantic search is this measurement plus a sort.

**Multilingual embedder** — one trained so that text in different languages sharing a meaning
lands in the same region. The difference that decides the day: on the six `tr_en` questions
`nomic-embed-text` scores 0.000 and `bge-m3` scores 0.667.

**Chunk** — the slice of a document you actually index. Retrieval never sees a document; it sees
whatever you cut. Whole documents to structure-aware chunks takes hit@1 0.600 → 0.750 — three
questions in twenty, so read the direction, not the digit.

**Question types** — the five labels every measured table is cut by: `tr_tr` (4 questions),
`tr_en` (6), `en_en` (4), `exact_token` (4), `multi_hop` (2). An average that improves can still
hide a type that got worse.

**Fixed-size chunking** — cut every N characters, blind to content. Fixed-280 raises the average
(hit@1 0.600 → 0.700) while `exact_token` falls (1.000 → 0.750), because it separates a table's
column header from its rows.

**Overlap** — each chunk carries the tail of the previous one. Insurance against cutting a
sentence in half, and the one rung that goes backwards: hit@1 0.700 → 0.550 while recall@5 falls
too, 0.917 → 0.883. It moves a boundary; it does not remove one.

**Recursive chunking** — split on paragraphs, then lines, then sentences. The usual default, and
it still cuts our penalty table from its header: a 600-character window cannot hold a nine-column
header plus seven rows. recall@5 0.900, second only to fixed-280's 0.917.

**Structure-aware chunking** — split on the document's own headings and numbered rules, and
prefix each chunk with where it came from. The only strategy that keeps the header with the row:
hit@1 0.750, MRR 0.817.

**Contextual retrieval** — the industry name for that prefix: give each fragment enough
surrounding context to stand alone. Module 7 ships it without naming it, so module 9 finds its
win already banked.

**Boilerplate stripping** — removing the legal footers and page artefacts that repeat across
documents. Takes 4.2% of the corpus away and moves hit@1 0.750 → 0.800, all of it in `en_en`.

**Parent id / chunk id** — every chunk carries the id of the document it was cut from. Every
score here collapses a chunk ranking to parent documents first, so one document cannot occupy
five slots.

**BM25** — keyword ranking by term frequency, inverse document frequency and document length. No
model, no training. hit@1 0.400 over whole documents against dense retrieval's 0.600, and 0.300
over structure-aware chunks; 0.000 on `tr_en` at both granularities.

**Sparse vs dense retrieval** — sparse is BM25 and its relatives, matching words; dense is
embeddings, matching meaning. They fail at different things, which is the whole argument for
combining them — and for measuring the combination.

**Hybrid search** — running sparse and dense together and merging the rankings. Module 9's first
headline word, and the one that lost here: over our chunks it took hit@1 0.750 → 0.450.

**RRF (Reciprocal Rank Fusion)** — the usual merge: sum 1/(k + rank) across the rankings. It
rewards documents every input ranking agrees on, which only helps when every input is sound.

**Reranking** — reordering the top few results with a second, slower model. A trade, not an
upgrade: it lifted the two weak setups (hit@1 0.350 → 0.450, 0.350 → 0.400) and pulled the two
strong ones down (0.700 → 0.500, 0.750 → 0.600). It levels to its own ceiling.

**Bi-encoder vs cross-encoder** — a bi-encoder embeds query and passage separately, so passage
vectors are stored once; a cross-encoder reads the pair together and scores relevance directly,
so nothing is precomputable. Module 9's reranker is neither, so no number here belongs to a
cross-encoder.

**Pointwise vs listwise reranking** — scoring each candidate on its own, versus asking for the
whole list in order. Handed six passages, `qwen2.5:3b` returned four indices; asked one at a time
it returned six usable scores. That decides whether reranking works at all, not whether it helps.

**RAG** — retrieval-augmented generation: find the relevant text at question time, put it in the
prompt, answer from it. The point is not accuracy — here stuffing the whole corpus beats it — but
that it still works a hundred times larger, and that it can cite.

**Agentic RAG** — retrieval as a tool the model calls rather than a step that runs before it. It
breaks the question up, searches, checks whether it has enough, searches again. The cost is six
to ten calls against naive RAG's one — three of them cheap embeddings — plus determinism.

**Query decomposition** — the first call in that loop: turn one question into standalone
sub-questions, capped at four. Query rewriting with a budget.

**Sufficiency check** — the second: ask the model whether what it has gathered answers the
question — `YES`, or `NO` plus one more query. The loop's stopping condition and its weakest
part.

**Multi-hop** — a question whose answer is spread across documents that no single search returns.
Ours needs the misconnect SOP, the interline agreement and the fare rules at once. It reads 0.000
without chunking and 0.500 on every chunked rung — chunking is the only thing that moves it off
zero, and 0.500 is not a win.

## Measurement

**Gold set** — a fixed list of questions with the documents that should come back for each. Ours
is 20 and does not change between modules, so the same three numbers are comparable all day.
Remember that **twenty questions is not a benchmark**: one question is worth 0.05, hit@1 moves
only in steps of 0.05, and almost every gap on the ladder is exactly one question wide. Quote the
direction, not the gap — enough to choose between two designs, far too few to publish.

**hit@1** — was the top-ranked document a right one? The strictest and most honest single number.

**Top-k** — how many results a stage passes on. `hit@1` is at k = 1, `recall@5` at k = 5, and
the reranker scores the top 8 — where its 8 model calls per query come from. A metric quoted
without its k is not a number.

**recall@5** — how much of the right set appeared in the top five. Beware: it is *highest* for
strategies that rank worse — 0.917 for fixed-280 against structure-aware's 0.833 — because 294
chunks give the gold document more chances to appear than 154 do.

**MRR (Mean Reciprocal Rank)** — one over the rank of the first correct document, averaged over
the questions. The one that shows partial credit: moving a gold document from rank 6 to rank 2
leaves hit@1 unchanged and lifts that question's contribution from 0.17 to 0.50.

**LLM-as-judge** — scoring answers with another model; common, and deliberately not used for any
number in this course, because the score moves between runs.

## Infrastructure

**Vector database** — storage that indexes embeddings for nearest-neighbour search; ChromaDB
here. Its default embedder is `all-MiniLM-L6-v2`, silently downloaded and English-only: on a
Turkish question it does not error, it returns the wrong document.

**Collection** — ChromaDB's unit of storage: ids, documents, embeddings and one metadata
dictionary per item. `where={"kind": "fare"}` filters on that metadata before the vector search,
which is how a superseded SOP is kept out of an answer.

**ANN / HNSW** — approximate nearest-neighbour search, trading a little recall for a lot of
speed once exact search is too slow. Recall here means agreement with brute-force search over the
same vectors, not `recall@5` against the gold set.

**Embedded vs server mode** — ChromaDB as a library inside your process, or a service you talk to
over HTTP. Same API, different operational ownership. The server's path is `/api/v2/`, not v1.

**pgvector** — the Postgres extension that stores vectors in a table you already back up and can
join to the business columns beside them. Module 8's first question past a hundred thousand
vectors: do you already run Postgres?

**Ollama** — runs models locally and serves them on `localhost:11434`. This course uses it for
everything, which is what makes the day work with no API key, no cloud account and no sign-in.

**Podman** — rootless, daemonless container runtime. Used instead of Docker for the ChromaDB
server: no privileged daemon, and no licensing question in a corporate setting.
