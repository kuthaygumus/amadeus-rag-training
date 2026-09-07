"""Three ways to cut a document up, and the reason the choice matters.

A retriever never sees a document. It sees whatever you handed it, and if you handed it the
wrong slice then no amount of embedding quality will save the answer. This module holds the
three strategies the day compares, cheapest first.

The failure worth watching is not the one people expect. Fixed-size chunking does not usually
tear a table row in half — the row survives. What it does is leave the column header behind in
the previous chunk, so a model reads `EUR 70 | EUR 90` with no idea which column is which and
answers confidently from the wrong one. Adding overlap does not fix it; the header just lands
one chunk earlier instead.
"""

from __future__ import annotations

import re

SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]


def fixed(text: str, size: int = 280, overlap: int = 0) -> list[str]:
    """Cut every `size` characters, blind to what is there. The baseline, and the villain."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def recursive(text: str, size: int = 600, separators: list[str] | None = None) -> list[str]:
    """Split on the most natural boundary that fits, falling back to finer ones.

    Paragraphs before lines, lines before sentences. Keeps prose intact, and keeps a short
    table intact — but a table longer than `size` still gets cut, header and all.
    """
    separators = separators or SEPARATORS
    if len(text) <= size:
        return [text] if text.strip() else []
    for sep in separators:
        if sep not in text:
            continue
        parts, buffer, out = text.split(sep), "", []
        for part in parts:
            candidate = part if not buffer else buffer + sep + part
            if len(candidate) <= size:
                buffer = candidate
            else:
                if buffer.strip():
                    out.append(buffer)
                buffer = part if len(part) <= size else ""
                if len(part) > size:
                    out.extend(recursive(part, size, separators[separators.index(sep) + 1:]))
        if buffer.strip():
            out.append(buffer)
        if out:
            return out
    return [text[i:i + size] for i in range(0, len(text), size)]


HEADING = re.compile(r"^(#{1,6} .*|RULE \d+[A-Z]?\..*|SECTION \d+.*|Step \d+\..*)$", re.M)


def structure_aware(text: str, size: int = 900, title: str = "") -> list[str]:
    """Split on the document's own structure, and prefix every chunk with where it came from.

    This is the one that fixes the header problem, and it does it twice over. Sections are cut
    at headings rather than at character counts, so a table stays with the rule that introduces
    it. And because each chunk carries the document title and its section heading, a chunk that
    would otherwise read as a bare grid of numbers arrives already labelled — which is the same
    trick 'contextual retrieval' sells under a longer name.
    """
    lines = text.splitlines()
    sections, current, heading = [], [], title
    for line in lines:
        if HEADING.match(line) and current:
            sections.append((heading, "\n".join(current)))
            current, heading = [], line.strip()
        elif HEADING.match(line):
            heading = line.strip()
        else:
            current.append(line)
    if current:
        sections.append((heading, "\n".join(current)))

    out = []
    for head, body in sections:
        if not body.strip():
            continue
        context = " > ".join(p for p in (title, head) if p and p != title) or title
        for piece in (recursive(body, size) if len(body) > size else [body]):
            out.append(f"[{context}]\n{piece}" if context else piece)
    return out


STRATEGIES = {
    "fixed-280": lambda t, title="": fixed(t, 280, 0),
    "fixed-280+overlap60": lambda t, title="": fixed(t, 280, 60),
    "recursive-600": lambda t, title="": recursive(t, 600),
    "structure-aware": lambda t, title="": structure_aware(t, 900, title),
}


def chunk_corpus(documents: dict[str, str], strategy: str) -> tuple[list[str], list[str], list[str]]:
    """Chunk every document. Returns (chunk_ids, chunk_texts, parent_doc_id per chunk)."""
    split = STRATEGIES[strategy]
    ids, texts, parents = [], [], []
    for doc_id, text in documents.items():
        for n, piece in enumerate(split(text, doc_id.replace("_", " "))):
            ids.append(f"{doc_id}#{n}")
            texts.append(piece)
            parents.append(doc_id)
    return ids, texts, parents


def to_documents(chunk_ranking: list[str]) -> list[str]:
    """Collapse a chunk ranking to a document ranking, best chunk wins, order preserved."""
    seen, out = set(), []
    for chunk_id in chunk_ranking:
        doc = chunk_id.split("#")[0]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


BOILERPLATE = [
    # The same legal block closes every fare rule sheet. Six near-identical tails competing for
    # space in a top-k, none of them the answer to anything.
    re.compile(r"<div class=\"legal\">.*?</div>", re.S),
    re.compile(r"^LEGAL NOTICE.*?(?=\n\n|\Z)", re.S | re.M),
    re.compile(r"^DISCLAIMER:.*?(?=\n\n|\Z)", re.S | re.M),
    # Export artefacts.
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.M),
    re.compile(r"</?(?:div|br|span|p)[^>]*>"),
    re.compile(r"&nbsp;"),
    # The distribution footer repeated verbatim across every bulletin.
    re.compile(r"^Distribution:.*?(?=\n\n|\Z)", re.S | re.M),
]


def strip_boilerplate(text: str) -> str:
    """Remove the parts that repeat across documents and answer nothing.

    Worth doing before chunking rather than after: boilerplate that survives into the index does
    not merely waste space, it produces chunks whose only distinguishing content is shared with
    five other documents, and those compete for a place in the top-k.
    """
    for pattern in BOILERPLATE:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
