# %% [markdown]
# # 07 · Agentic RAG — when one search cannot express the question
#
# > **Helios Air is a fictional airline.** Everything here is synthetic training material.
#
# Every method in this course has been a single pass: take the question, search once, answer.
# We improved the search a great deal. Two questions in the gold set never moved.
#
# They are the ones a duty manager actually asks.

# %%
import sys, json, time
from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True)
import _cached
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)

multihop = [q for q in questions if q["type"] == "multi_hop"]
task = multihop[0]
print(task["query"])
print("\nneeds all of:", task["gold_doc_ids"])

# %% [markdown]
# Three documents, and that is not an accident of how we wrote the corpus — it is how airline
# rules are actually organised. The procedure lives in an SOP. Whether the partner's flight is
# protected at all lives in the interline agreement. What the passenger pays lives in the fare
# rules. No single document knows the whole answer, because no single department owns it.

# %% [markdown]
# ## What one search gets you

# %%
retrieved = C.to_documents(dense.rank(task["query"]))[:5]
print("retrieved:", retrieved)
found = set(retrieved) & set(task["gold_doc_ids"])
print(f"\nof the {len(task['gold_doc_ids'])} documents needed, retrieval found {len(found)}: {sorted(found)}")
print(f"missing: {sorted(set(task['gold_doc_ids']) - found)}")

# %%
context = "\n\n".join(f"[SOURCE: {d}.md]\n{docs[d][:1200]}" for d in retrieved[:3])
print(R.generate(f"SOURCES:\n{context}\n\nQUESTION: {task['query']}",
                 system=("Answer ONLY from the sources. Cite source filenames. "
                         "If something needed is missing, say what is missing. Be brief.")))

# %% [markdown]
# Reranking will not fix this and neither will a better embedder. The problem is not that the
# right documents are ranked badly. It is that the question, embedded as one vector, points at
# one region of the corpus, and the answer is spread across three.
#
# You cannot express "and also tell me about the interline protection clause" by nudging a
# similarity score. You have to ask again.

# %% [markdown]
# ## Step 1 — break the question up
#
# Have the model turn one operational question into the separate things it needs to know.

# %%
def decompose(question: str) -> list[str]:
    raw = R.generate(
        f"QUESTION: {question}\n\nBreak this into 2-4 standalone factual sub-questions, each "
        f"answerable from a single airline document. Output ONLY a JSON array of strings.",
        system="You decompose operational questions. Output only a JSON array.", max_tokens=250)
    start, end = raw.find("["), raw.rfind("]")
    try:
        return [s for s in json.loads(raw[start:end + 1]) if isinstance(s, str)][:4]
    except Exception:
        return [question]

subs = decompose(task["query"])
for n, s in enumerate(subs, 1):
    print(f"  {n}. {s}")

# %% [markdown]
# ## Step 2 — search for each part separately
#
# Now every sub-question is a single-hop question, which is the kind our retriever is good at.

# %%
gathered: dict[str, str] = {}
for sub in subs:
    for chunk_id in dense.rank(sub)[:3]:
        gathered.setdefault(chunk_id, sub)
print(f"{len(gathered)} chunks gathered across {len(subs)} sub-questions")
covered = {c.split("#")[0] for c in gathered}
print(f"documents reached: {sorted(covered & set(task['gold_doc_ids']))}")
print(f"still missing    : {sorted(set(task['gold_doc_ids']) - covered)}")

# %% [markdown]
# ## Step 3 — ask whether that is enough
#
# The step that makes this an agent rather than a longer pipeline: the model inspects what came
# back and decides whether it can answer yet.

# %%
def sufficient(question: str, chunk_ids: list[str]) -> tuple[bool, str]:
    body = "\n\n".join(f"[{c}]\n{chunks[c][:500]}" for c in chunk_ids)
    verdict = R.generate(
        f"QUESTION: {question}\n\nRETRIEVED:\n{body}\n\n"
        f"Can this question be fully answered from the retrieved material? "
        f"Reply with YES, or with NO followed by one short search query for what is missing.",
        system="You audit whether retrieved context is sufficient. Be strict and terse.",
        max_tokens=60)
    return verdict.strip().upper().startswith("YES"), verdict.strip()

enough, verdict = sufficient(task["query"], list(gathered))
print(f"sufficient: {enough}\n{verdict}")

# %% [markdown]
# ## Step 4 — go back for what is missing
#
# If the audit says no, it also says what to look for. Search again with that, and stop when the
# audit is satisfied or when we run out of patience.

# %%
MAX_ROUNDS = 3
for round_number in range(MAX_ROUNDS):
    enough, verdict = sufficient(task["query"], list(gathered))
    print(f"round {round_number + 1}: sufficient={enough}")
    if enough:
        break
    follow_up = verdict.split("\n")[-1].lstrip("NO").strip(" .:,-") or task["query"]
    print(f"  searching again for: {follow_up[:80]}")
    before = len(gathered)
    for chunk_id in dense.rank(follow_up)[:3]:
        gathered.setdefault(chunk_id, follow_up)
    if len(gathered) == before:
        print("  nothing new came back — stopping rather than looping")
        break

covered = {c.split("#")[0] for c in gathered}
print(f"\nfinal coverage: {sorted(covered & set(task['gold_doc_ids']))} "
      f"of {sorted(task['gold_doc_ids'])}")

# %% [markdown]
# The loop is bounded on purpose. An agent that keeps deciding it needs one more search is the
# most common way this pattern fails in production, and the fix is not a cleverer prompt — it is
# a counter.

# %% [markdown]
# ## Step 5 — answer

# %%
context = "\n\n".join(f"[SOURCE: {c.split('#')[0]}.md]\n{chunks[c]}" for c in list(gathered)[:10])
answer = R.generate(
    f"SOURCES:\n{context}\n\nQUESTION: {task['query']}",
    system=("Answer ONLY from the sources. Cite every source filename you used. "
            "Give the procedure and the amounts. Be complete but brief."), max_tokens=400)
print(answer)

# %% [markdown]
# ## Read that answer critically
#
# Your answer will not be word for word the one printed here: this is the one part of the day that
# is not deterministic, and the decomposition changes between runs. Look for the shapes rather than
# the wording.
#
# Retrieval did its job — all three documents are in front of the model. The answer is still not
# good. The recorded run quoted the six-hour hotel threshold with the fifteen-euro meal amount
# attached to it, as if that were what a hotel costs, and then contradicted itself about the change
# fee in its last two sentences.
#
# That is the same failure module 4 measured: given a nine-column filed tariff and six
# near-identical fare sheets, a 3B model reads a real value out of the wrong row, the wrong column
# or the wrong document, and reports it in the voice of a right one. Solving retrieval does not
# solve reading.
#
# It is worth saying this out loud rather than letting the finale look cleaner than it is. Every
# metric in this course scores **retrieval** — whether the right document came back. Not one of
# them scores whether the answer was right. Those are different systems with different failure
# modes, and we have only been measuring the first one.
#
# If you carried one thing from today into your own project, it might be this: build the retrieval
# eval first, because it is cheap and deterministic, and then build a second one for the answers,
# because the first will not tell you when the second is broken.

# %% [markdown]
# ## Both multi-hop questions, single-shot versus agentic

# %%
def agentic(question: str) -> set[str]:
    gathered: dict[str, str] = {}
    for sub in decompose(question):
        for c in dense.rank(sub)[:3]:
            gathered.setdefault(c, sub)
    for _ in range(2):
        ok, verdict = sufficient(question, list(gathered))
        if ok:
            break
        follow_up = verdict.split("\n")[-1].lstrip("NO").strip(" .:,-") or question
        for c in dense.rank(follow_up)[:3]:
            gathered.setdefault(c, follow_up)
    return {c.split("#")[0] for c in gathered}

def compare_single_shot_and_agentic() -> list[dict]:
    rows = []
    for q in multihop:
        print(f"  running {q['id']} ...", flush=True)
        start = time.time()
        need = set(q["gold_doc_ids"])
        single = set(C.to_documents(dense.rank(q["query"]))[:5]) & need
        multi = agentic(q["query"]) & need
        rows.append({"id": q["id"], "needs": len(need), "single_shot": len(single),
                     "agentic": len(multi), "seconds": round(time.time() - start, 1)})
    return rows

rows = _cached.run("07-single-shot-vs-agentic", compare_single_shot_and_agentic)

print(f"\n{'question':<8}{'needs':>7}{'single-shot':>13}{'agentic':>10}{'seconds':>10}")
for row in rows:
    print(f"{row['id']:<8}{row['needs']:>7}{row['single_shot']:>13}{row['agentic']:>10}"
          f"{row['seconds']:>10.1f}")

# %% [markdown]
# ## What actually changed
#
# Not the retriever. Not the embedder. Not the chunks. Every component in this notebook is the
# one we finished module 6 with.
#
# What changed is where retrieval sits. Until now it was a step that ran before the model: search,
# then generate. Here the model runs the search, looks at what it got, and decides whether to
# search again. Retrieval stopped being a stage in a pipeline and became a tool with a caller.
#
# That is the whole of what "agentic" means, and it is worth saying plainly because the word is
# doing a lot of marketing work elsewhere.

# %% [markdown]
# ## What it costs
#
# One decomposition call, one retrieval per sub-question, one audit call per round, one final
# generation. Six to ten model calls where we used to make one.
#
# You also lose determinism. The same question can decompose differently on two runs and reach a
# different set of documents, which makes this much harder to test than anything earlier in the
# day. And the loop needs a bound, because "do I have enough?" is a question a model will happily
# answer "no" to forever.
#
# Use it where a single search genuinely cannot express the question. Not everywhere. The two
# questions in this gold set that need it are two out of twenty — and the other eighteen would
# only get slower and less predictable.

# %% [markdown]
# ## Where the day ends
#
# Rewind what happened. A bare model invented a penalty. Training put our data into weights and
# the weights froze. Stuffing the prompt answered from the wrong document. Keyword search missed
# the paraphrase. Naive retrieval brought back a chunk with no column header. Better chunking
# took hit@1 from 0.550 to 0.800. Fusion and reranking, measured, did not beat it. And the last
# two questions needed retrieval to happen more than once.
#
# Every step existed because the one before it hit a wall you watched it hit.
#
# > **You just watched RAG turn into an agent.**
