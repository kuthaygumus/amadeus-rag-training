---
title: "1. The Bare LLM Wall"
description: "The model declines on the airline that does not exist, invents a penalty for the industry in general, then names three different amounts for the same question — all in one voice."
---

## Gate question

> **Does the model know *my* data?**

<div class="presenter-note">
Before running anything, put the question on screen and ask the room to commit on paper: will a 3B model answer the Kraken Air penalty question, or refuse? Get a show of hands and count it out loud. The recorded run makes both camps half right — it declines the first time, then names a lira amount as soon as the wording changes — and that reversal is the whole module. Two minutes, no more.
</div>

A model on the laptop. No retrieval, no documents, nothing but the weights. We ask it the question a Kraken Air agent gets ten times a day.

**Everything below is a transcript — there is nothing to type here.** Each prompt is a Python string in `notebooks/00_bare_llm_fails.py`, the file you open in VS Code and run block by block with `Shift+Enter`; `R.generate()` sends it to local Ollama and prints the reply. You read the recorded run now and reproduce it in *What you run*, below.

**`q1`:** `Kraken Air'in CLASSIC ucret ailesinde K sinifi bir bileti iptal edersem ne kadar ceza oderim?`
**qwen2.5:3b:** `... belirtilmemiştir. Bu bilgi genellikle kargo veya hizmet koşullarına göre bulunan bir belge veya web sitesinden bulunabilir.`

It declined, and that looks like the right call — until you read the reason. A passenger fare rule is not filed under *cargo or service conditions*, and there is no Kraken Air website to check. Nothing was consulted; the model produced text *shaped* like a responsible answer. Hold on to that shape, because the last block of the notebook takes it away.

The real figure is in `corpus/2026-Q3/fare_classic_shorthaul.md`, row K: **EUR 90**, a flat amount per passenger per direction, not a percentage. Last quarter the same row said EUR 120.

Now the sharper probe. Drop the fictional carrier and ask about the industry in general.

**`q2`:** `Bir havayolunda K booking class bileti iptal edilirse tipik olarak ne kadar ceza alinir? Somut bir rakam ver.`
**qwen2.5:3b:** `Tipik olarak, K (Business) koltuk sınıfı için 10-20% kadar ceza ödemek zorunludur.`

Wrong twice over. The percentage is invented, and the parenthesis is worse: K is a discounted economy booking class on the overwhelming majority of carriers, and economy in our corpus. Letters below J and Y are filed by each carrier, so that is a strong convention rather than a standard — which is exactly why a confident `(Business)` with no hedge is wrong. This is not "the model is missing our private data". It is confidently wrong about the public part too.

<div class="presenter-note">
Say this out loud and let it sit: "It did not just miss our data. It got the industry wrong." Someone in the room will already be typing the same question into a bigger model to prove it does better. Let them — a larger model gets K right more often, and it still cannot know that this quarter's number is 90 and not 120. Move the argument there rather than defending the 3B model.
</div>

Then a specific fact instead of a rule. Same model, same session, no context.

**`q3`:** `XX 1487 sefer numarali Kraken Air ucusu saat kacta kalkiyor?`
**qwen2.5:3b:** `Bu bilgi ... resmi bir takipçi platformundan (örneğin Twitter) veya resmi web sitesinden alınamaz. Her seferde farklı saat olabilir.`

It declined again, and again the reason is invented: a schedule is not something you read off Twitter, and a numbered flight does not leave at a different time every day. Two declines, and neither was a knowledge check — nothing in there checked anything.

A language model predicts the next token from learned weights: billions of parameters compressing the statistical shape of everything it read. No lookup table, no source document, no timestamp on any fact. What changes from answer to answer is which prior the *form* of the question activates. Ask the industry something generic and the pattern *"cancellation fees are a percentage of the fare, typically in this range"* is attested tens of thousands of times, so out comes an instance of it. Ask for a departure time and a well-attested pattern of *text that declines to state a schedule* wins instead. That refusal prior comes mostly from instruction tuning, and *Going deeper* takes it further.

Which means the prior is fitted to the wording, not to the fact. That is why the polite decline on `q1` is worth nothing.

**The danger is not that it fails. The danger is that it fails in exactly the same voice it uses when it succeeds.** Same fluency, same confidence, same absence of a citation. The decline, the invented percentage and the lira range below are written in one register, and nothing in the output tells you which of them you are holding.

That is the wall. Everything after this is an attempt to get over it: ask more carefully, train the model on our rules, or put the rules in front of it at answer time. We try all three today, in that order, and only one of them survives the corpus being reissued next quarter.

## What you run

**Terminal (repo root)** — the folder that holds `corpus/`, `notebooks/`, `eval/` and `exercises/`:

```bash
ollama serve                       # only if it is not already running
python scripts/verify_setup.py     # must print READY
```

**VS Code — `notebooks/00_bare_llm_fails.py`, the first `# %%` block.** Put the cursor inside it and press `Shift+Enter`; the output appears in the Interactive window.

```python
import sys
sys.path[:0] = [".", "notebooks"]              # the helpers sit next to this file
import _preflight; _preflight.ready(chat=True)  # stops with instructions if a model is missing
import retrieval as R

print("model:", R.CHAT_MODEL)
```

`R.generate()` calls local Ollama at `localhost:11434` with `temperature=0.0`. Keep pressing `Shift+Enter` down the file.

**What you should see, in about two minutes.** `model: qwen2.5:3b`, a one-word reply, then `q1`, `q2` and `q3` — the decline, the invented percentage, the second decline. Then the last block asks for the penalty four more times, changing only the wording, and prints the four answers under each other. Your wording will drift from the transcripts above; the decline-then-invent pattern does not.

**If Ollama is not answering yet.** This is the one notebook with no recorded run to fall back on — every cell is a live call. Read the transcripts above, pair with the person next to you, and rejoin at module 2, which needs no model at all.

<div class="presenter-note">
If Ollama is down or a pull is still running, do not debug on stage. Every transcript is on this page — read them, say "this ran on my machine this morning, you will reproduce it in the lab block", and keep going. Total module time is 12 minutes, of which the live cells are about 2 — the rest is the narrative and the show of hands. If someone reports a different phrasing, that is expected and worth one sentence: the wording varies, the pattern does not.
</div>

## What the numbers said

<div class="measured">

The check the notebook ends on. The same penalty question, asked four times with only the wording changed:

| how it was worded | what came back |
|---|---|
| `Kraken Air CLASSIC K sinifi iptal cezasi ne kadar?` | `Genellikle 100-250 TL arasında` — a lira range |
| `Kraken Air'de CLASSIC ucret ailesi, K booking class. Iptal cezasi kac euro?` | `genellikle 100-200 TL arasında değişebilir` — asked in euro, answered in lira |
| `Musterim Kraken Air CLASSIC K bileti aldi ve iptal etmek istiyor. Ne odeyecek?` | `genellikle 2-5 gün içinde ücret ödemek zorunda kalabilir` — a deadline, not an amount |
| `Kraken Air CLASSIC K cancellation penalty amount?` | `can vary and is subject to change ... contact Kraken Air directly` — a refusal, in English |

Four askings, four different behaviours: two lira ranges that do not agree with each other, one answer that is not an amount at all, and one refusal. None of them is EUR 90, and the second asking spelled out `kac euro?` and was answered in lira anyway. A model that held the fact would have said EUR 90 four times. This is one recorded run at `temperature=0.0` on one machine; your amounts may differ, and that they differ from each other is the point.

</div>

## Going deeper

Think of the weights as compression, not storage. Training squeezes a corpus into a fixed parameter budget, so the frequent and generic survive at high fidelity while the specific and rare get reconstructed rather than recalled. A hallucination is a confident guess drawn from a region the training data never constrained. "Cancellation penalties are a percentage" is generic. "EUR 90 for booking class K on the short-haul CLASSIC sheet in 2026-Q3" is as specific as a fact gets, and no amount of scale moves it across that line — a bigger model changes the odds on the public part of the question and nothing at all on the private part. The five-model comparison on the [setup page](/modules/00-setup/) measures exactly that, in the condition where it matters: a document already in the context.

The declines deserve more attention than they usually get. Refusal is largely trained in: instruction tuning rewards "I don't know" for questions shaped like a lookup of a private or volatile fact. The signal tracks the *form* of the question, not whether the model holds the fact — which is why `q1` declined and a reworded `q1` produced a lira range. Prompting moves the line, but you are tuning a prior, not installing a knowledge check. We ground the answer in a retrieved document instead, because that is both cheaper than asking the model five times to see whether the number moves and auditable — and auditable is what an airline needs.

## Exit line

> It does not know — and it does not know that it does not know.

<div class="presenter-note">
This is the one sentence you must not garble. Say it slowly, do not add to it, and do not explain it — the next module is the explanation. Then straight into module 2 with: "So what is actually inside the thing that just invented a percentage?"
</div>
