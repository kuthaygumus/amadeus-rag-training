"""The retrievers used throughout the day, in one place.

Everything here runs against a local Ollama instance on localhost:11434 and pure Python.
No API key, no HuggingFace download, no network beyond localhost. That is deliberate: the
whole day has to work on a corporate laptop with the internet effectively closed.

The pieces, in the order the day introduces them:

    BM25            keyword search. The dumbest retriever, and it wins more often than people expect.
    DenseRetriever  embedding search through Ollama. Fixes paraphrase, breaks on exact tokens.
    rrf             reciprocal rank fusion. Measured NOT to help here — see the note below.
    pointwise_rerank  score each candidate on its own. Measured to fix everything else.

A measured warning about `rrf`: fusing a systematically wrong retriever with a good one makes
the good one worse. On the probe corpus, dense 2/5 + BM25 3/5 fused to 2/5. Fix the embedding
model before you reach for fusion.

A measured warning about reranking: asking one small model to *order* a list of passages
scored worse than no reranking at all (MRR 0.833 -> 0.600). Asking it to *score each passage
separately* scored perfectly (MRR 1.000). Same model, same candidates — only the question
changed. That is why `pointwise_rerank` exists and a listwise version does not.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:3b"


# --------------------------------------------------------------------------- Ollama transport

def _post(endpoint: str, body: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA}/api/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA}. Is it running? "
            f"Try `ollama serve` in another terminal. Original error: {e}"
        ) from e


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a batch. Ollama accepts a list, which is much faster than one call per document."""
    return _post("embed", {"model": model, "input": texts})["embeddings"]


def generate(prompt: str, system: str = "", model: str = CHAT_MODEL,
             temperature: float = 0.0, max_tokens: int = 400) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    response = _post("chat", {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
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

    Crude on purpose: it keeps `h9` and `1487` as separate tokens, which is exactly why BM25
    finds a flight code that embedding search blurs away.
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
    not help, and combining a broken retriever with a good one actively diluted the good one.
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
    doing nothing. Ties keep the retriever's original order, so reranking can only move a
    document when the model actually has an opinion.
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
