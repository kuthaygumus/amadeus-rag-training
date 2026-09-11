"""The retrievers used throughout the day, in one place.

Everything here runs against a local Ollama instance on localhost:11434 and pure Python.
No API key, no HuggingFace download, no network beyond localhost. That is deliberate: the
whole day has to work on a laptop with nothing to sign into and nothing left to download.

The pieces, in the order the day introduces them:

    BM25            keyword search. The cheapest retriever, and the only one that needs no model.
    DenseRetriever  embedding search through Ollama. Fixes paraphrase and cross-lingual queries.
    rrf             reciprocal rank fusion. Measured NOT to help here — see the note below.
    pointwise_rerank  score each candidate on its own. A trade, not an upgrade — see the note below.

Alongside them, three thin wrappers over the server itself — `embed`, `generate`, and the two
read-only calls `installed_models()` and `context_window()`. Nothing in this directory needs the
last two; they exist so the notebooks can ask the server what it has and what window it serves,
and get this module's error messages instead of a urllib traceback when it is not running.

A measured warning about `rrf`: fusion assumes every input ranking is independently sound.
BM25 over short chunks is not, and fusing it into a good dense ranking drags the dense ranking
down. Fix the embedding model before you reach for fusion.
Reproduce: `python eval/run_benchmark.py --fusion`.

A measured warning about reranking: on the 20-question gold set, the same reranker over four
retrieval setups helped both weak ones and hurt both strong ones. It levels a ranking toward
its own ceiling, and a 3B model scoring passages 0-10 has a low ceiling: where the retriever
was already right, that coarse opinion can only demote the right answer. So a reranker is a
trade, not an upgrade — it pays when your retriever is worse than your reranker and costs you
when it is better. Fix the embedder first, then measure whether you still need one.
Reproduce: `python eval/run_benchmark.py --rerank-sweep` (slow: 8 model calls per question per
setup, four setups).

A separate and much smaller result, n=5 on the earlier probe corpus — treat the direction as
real and the magnitude as unproven: asking this model to *order* six passages in one call scored
2/5 at hit@1 against 5/5 for asking it to *score each passage on its own*. That is why
`pointwise_rerank` exists and a listwise version does not. The same n=5 probe also suggested
reranking fixes everything, and the n=20 measurement above refuted that half of it — which is
what five questions are worth as evidence.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:3b"

# Every call through _post is counted here, so a script can report what a measurement cost
# instead of asserting it. Read it before and after a section and take the difference.
CALLS = {"embed": 0, "chat": 0, "other": 0}


class OllamaError(RuntimeError):
    """One Ollama call failed, with `kind` naming which failure it was.

    Three failures look identical from the outside and have completely different remedies, so
    the caller gets to tell them apart:

        "unreachable"     nothing is listening on localhost:11434 — start the server.
        "missing_model"   the server answered, but that model tag is not pulled — pull it.
                          `.model` carries the tag, so a benchmark can skip one model and
                          keep going with the ones that are installed.
        "request_failed"  the server answered with an error or the call timed out.
    """

    def __init__(self, message: str, kind: str, model: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.model = model


# --------------------------------------------------------------------------- Ollama transport

def _post(endpoint: str, body: dict, timeout: int = 300) -> dict:
    model = body.get("model")
    CALLS[endpoint if endpoint in CALLS else "other"] += 1
    request = urllib.request.Request(
        f"{OLLAMA}/api/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return _request(request, endpoint, model, timeout)


def _request(request: urllib.request.Request, endpoint: str, model: str | None,
             timeout: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        # HTTPError subclasses URLError, so it has to be caught first — otherwise a running
        # server that returned 404 gets reported as a server that is not running, and the room
        # spends ten minutes restarting something that was never down.
        detail = _http_detail(e)
        if e.code == 404 and model:
            raise OllamaError(
                f"Ollama is running at {OLLAMA}, but the model '{model}' is not installed. "
                f"Run: ollama pull {model}",
                kind="missing_model", model=model,
            ) from e
        if e.code == 405:
            raise OllamaError(
                f"Ollama is running at {OLLAMA}, but /api/{endpoint} does not accept POST "
                f"(HTTP 405). /api/tags and /api/ps are GET endpoints; only /api/embed, "
                f"/api/chat and /api/generate take a POST body.",
                kind="request_failed", model=model,
            ) from e
        raise OllamaError(
            f"Ollama is running at {OLLAMA} but refused the call to /api/{endpoint}: "
            f"HTTP {e.code}{detail}",
            kind="request_failed", model=model,
        ) from e
    except TimeoutError as e:
        raise OllamaError(
            f"Ollama did not answer /api/{endpoint} within {timeout}s"
            f"{f' for model {model}' if model else ''}. On a CPU-only machine the first call "
            f"after a pull loads the model into RAM and is much slower than the rest; try the "
            f"same call again, and use qwen2.5:1.5b if 8 GB of RAM is all you have.",
            kind="request_failed", model=model,
        ) from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Nothing is listening on {OLLAMA}. Start the server with `ollama serve` in "
            f"another terminal, then check it with `ollama list`. Original error: {e.reason}",
            kind="unreachable", model=model,
        ) from e


def _get(endpoint: str, timeout: int = 30) -> dict:
    """GET one of Ollama's read-only endpoints, with the same error classification as _post.

    /api/tags and /api/ps are GET while /api/embed and /api/chat are POST, and sending the
    wrong verb returns HTTP 405 rather than anything readable. Going through here rather than
    through urllib directly is what turns a stopped server into the "start ollama serve"
    sentence instead of a traceback.
    """
    CALLS["other"] += 1
    return _request(urllib.request.Request(f"{OLLAMA}/api/{endpoint}"), endpoint, None, timeout)


def installed_models(timeout: int = 30) -> list[str]:
    """Every model tag the local server has pulled, sorted. Nothing is downloaded to answer."""
    return sorted(m.get("name", "") for m in _get("tags", timeout).get("models", []))


def context_window(model: str = CHAT_MODEL, timeout: int = 30) -> int | None:
    """The context window `model` advertises, in tokens, or None if it does not say.

    Advertised is not effective. Ollama serves a model at its own default `num_ctx`, which on
    some builds is a few thousand tokens whatever this number says, and a prompt longer than
    that is truncated silently — no error, no warning, a confident answer off the part that
    survived. Anything that compares a prompt against this number must also pass that number
    as `generate(..., num_ctx=...)`, or it is checking a limit it has not asked for.
    """
    info = _post("show", {"model": model}, timeout=timeout).get("model_info", {})
    return next((v for k, v in info.items() if k.endswith("context_length")), None)


def _http_detail(e: urllib.error.HTTPError) -> str:
    """Ollama puts a usable sentence in the error body. Quote it rather than the status alone."""
    try:
        body = e.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""
    try:
        body = json.loads(body).get("error", body)
    except json.JSONDecodeError:
        pass
    return f" — {body[:300]}" if body else ""


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a batch. Ollama accepts a list, which is much faster than one call per document."""
    return _post("embed", {"model": model, "input": texts})["embeddings"]


def generate(prompt: str, system: str = "", model: str = CHAT_MODEL,
             temperature: float = 0.0, max_tokens: int = 400,
             num_ctx: int | None = None) -> str:
    """One chat call. `num_ctx` sets the context window the server serves this call at.

    Left at None, Ollama uses its own default, which is not the window the model advertises
    and on some builds is much smaller. That is fine for short prompts — everything in this
    directory sends short prompts — and wrong for a long one: the prompt is truncated to fit
    and the answer comes back confident about text the model never saw. Anything stuffing a
    whole corpus into `prompt` should pass `num_ctx=context_window(model)` and compare its
    length against the same number.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    options = {"temperature": temperature, "num_predict": max_tokens}
    if num_ctx:
        options["num_ctx"] = num_ctx
    response = _post("chat", {
        "model": model, "messages": messages, "stream": False, "options": options,
    })
    # Reasoning models leak their scratchpad into the content; keep only what comes after it.
    return response["message"]["content"].split("</think>")[-1].strip()


# --------------------------------------------------------------------------- similarity

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / (norm + 1e-12)


def tokenize(text: str) -> list[str]:
    """Split on anything that is not alphanumeric, then lowercase.

    Crude on purpose: it keeps `xx` and `1487` as separate tokens, so a flight code matches on
    the literal string rather than on any notion of meaning. That is BM25's whole mechanism,
    and on this corpus it is still not enough — over whole documents BM25 scores 0.750 on the
    exact-token questions against dense retrieval's 1.000. Reproduce with `--skip-rerank`.
    """
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


# --------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25, written out rather than imported.

    Seeing the idf and the length normalisation on screen is the point: people stop treating
    keyword search as a black box once they have watched the score come apart into two terms.
    """

    def __init__(self, doc_ids: list[str], documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids, self.k1, self.b = doc_ids, k1, b
        self.docs = [tokenize(d) for d in documents]
        self.lengths = [len(d) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.n = len(self.docs)
        self.term_freq = [{t: d.count(t) for t in set(d)} for d in self.docs]
        self.doc_freq: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def scores(self, query: str) -> list[float]:
        out = []
        for i in range(self.n):
            score = 0.0
            for term in tokenize(query):
                df = self.doc_freq.get(term)
                if not df:
                    continue
                idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
                f = self.term_freq[i].get(term, 0)
                norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_length or 1)
                score += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(score)
        return out

    def rank(self, query: str) -> list[str]:
        scores = self.scores(query)
        order = sorted(range(self.n), key=lambda i: -scores[i])
        return [self.doc_ids[i] for i in order]


# --------------------------------------------------------------------------- dense

class DenseRetriever:
    """Embedding search. Embeds the corpus once, then one embed call per query."""

    def __init__(self, doc_ids: list[str], documents: list[str], model: str = EMBED_MODEL):
        self.doc_ids, self.model = doc_ids, model
        self.vectors = embed(documents, model=model)

    def rank(self, query: str) -> list[str]:
        qv = embed([query], model=self.model)[0]
        scores = [cosine(qv, v) for v in self.vectors]
        order = sorted(range(len(self.doc_ids)), key=lambda i: -scores[i])
        return [self.doc_ids[i] for i in order]


# --------------------------------------------------------------------------- fusion

def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion.

    Only helps when every input ranking is independently sound. Measured on this corpus it did
    not help: BM25 over short chunks collapses on Turkish queries, and fusing it into the dense
    ranking pulled the dense ranking down with it. `run_benchmark.py --fusion` prints it.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


# --------------------------------------------------------------------------- reranking

RERANK_SYSTEM = (
    "Rate how well the PASSAGE answers the QUERY, on a scale of 0 to 10. "
    "The query may be in Turkish while the passage is in English — judge the meaning, "
    "not the language. Reply with ONLY a single integer from 0 to 10, nothing else."
)


def pointwise_rerank(query: str, candidates: list[str], documents: dict[str, str],
                     model: str = CHAT_MODEL, char_limit: int = 900) -> list[str]:
    """Rerank by scoring each candidate independently, one model call per candidate.

    It costs len(candidates) calls, which is the whole point: the model never has to hold the
    comparison in its head. Asking a 3B model to order the list in one call scored worse than
    doing nothing (n=5, probe corpus — direction only). Ties keep the retriever's original
    order, so reranking can only move a document when the model actually has an opinion.

    Costing this is part of the lesson: at depth 8 that is 8 model calls per query, on top of
    retrieval, for a result that the four-setup sweep shows can be negative.
    """
    scored = []
    for position, doc_id in enumerate(candidates):
        reply = generate(
            f"QUERY: {query}\n\nPASSAGE:\n{documents[doc_id][:char_limit]}\n\nScore:",
            system=RERANK_SYSTEM, model=model, max_tokens=6,
        )
        digits = "".join(c for c in reply if c.isdigit())
        score = min(int(digits), 10) if digits else 0
        scored.append((score, -position, doc_id))
    return [doc_id for _, _, doc_id in sorted(scored, reverse=True)]
