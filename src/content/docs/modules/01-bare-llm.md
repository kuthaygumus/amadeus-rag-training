---
title: "1. The Bare LLM Wall"
description: "The model invents a cancellation penalty for an airline that does not exist, then refuses to guess a departure time — and both answers sound exactly the same."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **Does the model know *my* data?**

<div class="presenter-note">
Before running anything, put the question on screen and ask the room to write down their guess on paper: will a 3B model answer the Helios penalty question, or refuse? Most people say refuse. Get a show of hands and count it out loud — you want them committed before the cell runs, because the surprise is the whole module. Two minutes, no more.
</div>

We have a model on the laptop. No retrieval, no documents, nothing but the weights. We ask it the question a Helios agent gets ten times a day.

**Prompt:** `Helios CLASSIC K iptal cezası?`
**qwen2.5:3b:** an explanation that the penalty is typically **"20-30% ceza"** of the fare.

There is no Helios Air. There is no CLASSIC K fare outside this repository. The model did not hesitate, did not qualify, did not ask which route band. It produced a percentage range with the cadence of an answer.

The real figure is in `corpus/2026-Q3/fare_classic_shorthaul.md`, row K: **EUR 90**, a flat amount per passenger per direction, not a percentage. Last quarter the same row said EUR 120. So the model is wrong about the number, wrong about the *shape* of the number, and has no way to be right about which quarter you meant.

Now the sharper probe. Drop the fictional carrier and ask about the industry in general.

**Prompt:** `K booking class typical penalty?`
**qwen2.5:3b:** **"K (Business) sınıfı %10-20"**.

Wrong twice over. The percentage is invented as before, but the parenthesis is worse: K is not a business class. In normal airline filing practice K is a discounted economy booking class — exactly what it is in our corpus. So this is not "the model is missing our private data". It is confidently wrong about the public part too, stated as fact, with no hedge.

<div class="presenter-note">
Say this out loud and let it sit: "It did not just miss our data. It got the industry wrong." Someone in the room will already be typing the same question into a bigger model to prove it does better. Let them — a larger model gets K right more often, and it still cannot know that this quarter's number is 90 and not 120. Move the argument there rather than defending the 3B model.
</div>

Then the contrast that makes the failure precise. Same model, same session, no context.

**Prompt:** `What time does H9 1487 depart?` → it correctly said it does not know.
**Prompt:** `How much is the misconnect meal voucher?` → it correctly said it does not know.

So it is not a random text generator. It abstained twice, cleanly, on exactly the questions where abstaining is right. Why those two?

Because of what is in the training text. A language model predicts the next token from learned weights — billions of parameters compressing the statistical shape of everything it read. No lookup table, no source document, no timestamp on any fact. Ask about a cancellation penalty and the pattern *"airline cancellation fees are a percentage of the fare, typically in this range"* is attested tens of thousands of times, so the most likely continuation is a plausible instance of it. It fills the slot because the slot has a well-attested filler. Ask for the departure time of H9 1487 and there is no filler to reach for — but there is a well-attested pattern of *text that declines to state a specific schedule*. So that is what comes out.

The abstention is not the model checking its knowledge. Nothing consulted an inventory. Both behaviours came out of the same next-token machine; one prior happened to point at a number, the other happened to point at a refusal. Treating the refusal as evidence that the model knows its own limits is the mistake that puts a wrong penalty in front of a passenger.

Which is the point of the module.

**The danger is not that it fails. The danger is that it fails in exactly the same voice it uses when it succeeds.** Same fluency, same confidence, same absence of a citation. Nothing in the output tells you which of the four answers above you are looking at. And take any short-haul fare around EUR 400: "20-30%" of it is EUR 80 to EUR 120 — a range that brackets the true EUR 90 closely enough to survive a distracted spot check, and is wrong every single time.

That is the wall. Everything after this is an attempt to get over it: ask more carefully, train the model on our rules, or put the rules in front of it at answer time. We try all three today, in that order, and only one of them survives the corpus being reissued next quarter.

## What you run

Open `notebooks/00_bare_llm_fails.py` in VS Code and run the blocks with `Shift+Enter`. The `# %%` markers are the block boundaries. There is no Jupyter in this course; the dependencies are `numpy` and `chromadb` and nothing else.

```bash
ollama serve                       # in a second terminal, if it is not already running
ollama pull qwen2.5:3b
python scripts/verify_setup.py     # green before you start
```

The first block wires up the helper and checks the model is pulled. No framework, no API key:

```python
import sys
sys.path[:0] = [".", "notebooks"]               # the helpers sit next to this file
import _preflight; _preflight.ready(chat=True)  # stops with instructions if a model is missing
import retrieval as R

print("model:", R.CHAT_MODEL)
```

`R.generate()` calls local Ollama at `localhost:11434` with `temperature=0.0`.

**What you should see.** `model: qwen2.5:3b`, then a one-word reply from the model. Then three questions asked with no context: the Helios penalty rule and the same rule without the fictional name both come back with an invented percentage, and the departure time comes back as a refusal. Then the notebook asks for the penalty four more times, changing only the wording, and prints the four answers under each other. The meal-voucher probe in the table below is not in the notebook — type it yourself if you want the second abstention live.

**Roughly how long.** About two minutes, nearly all of it the model generating. Your wording will drift from the transcripts on this page; the two inventions and the abstention do not.

<div class="presenter-note">
If Ollama is down or a pull is still running, do not debug on stage. The four transcripts are on this page — read them, say "this ran on my machine this morning, you will reproduce it in the lab block", and keep going. Total module time is 12 minutes, of which the live cells are about 2 — the rest is the narrative and the show of hands. If someone reports a different phrasing, that is expected and worth one sentence: sampling varies, the pattern does not.
</div>

## What the numbers said

<div class="measured">

| prompt (no context, `qwen2.5:3b`) | what it answered | correct? |
|---|---|---|
| `Helios CLASSIC K iptal cezası?` | invented "20-30% ceza" | no — the answer is EUR 90, a flat amount |
| `K booking class typical penalty?` | "K (Business) sınıfı %10-20" | no — invented, and K is not a business class |
| `What time does H9 1487 depart?` | abstained | yes |
| `How much is the misconnect meal voucher?` | abstained | yes |

Two hallucinations, two correct abstentions, one voice.

Then the check the notebook ends on. The same penalty question, asked four times with only the wording changed:

| how it was worded | what came back |
|---|---|
| `Helios Air CLASSIC K sinifi iptal cezasi ne kadar?` | `1.500 TL` — Turkish lira, not euro |
| `... CLASSIC ucret ailesi, K booking class. Iptal cezasi kac euro?` | `100-200 euro` |
| `Musterim ... iptal etmek istiyor. Ne odeyecek?` | `%10-20` |
| `Helios Air CLASSIC K cancellation penalty amount?` | a refusal |

Four askings, three different amounts — one in lira, one in euro, one a percentage — and one refusal. A model that held the fact would answer the same thing four times. One recorded run at `temperature=0.0`; the amounts you get may differ, and that they differ from each other is the point.

</div>

## Going deeper

Think of the weights as compression, not storage. Training squeezes a corpus into a fixed parameter budget, so the frequent and generic survive at high fidelity while the specific and rare get reconstructed rather than recalled. What we call a hallucination is a confident sample from a region the training data never constrained. "Cancellation penalties are a percentage" is generic. "EUR 90 for booking class K on the short-haul CLASSIC sheet in 2026-Q3" is as specific as a fact gets. No amount of scale moves our corpus across that line.

The abstentions deserve more attention than they usually get. Refusal is largely trained in: instruction tuning rewards "I don't know" for questions shaped like a lookup of a private or volatile fact. So the signal correlates with the *form* of the question, not with whether the model holds the fact. Ask for a flight time and the trained refusal fires. Ask for a policy percentage and it does not, because that shape looks answerable. Prompting moves the line — a firm instruction to answer only from provided context raises abstention noticeably — but you are tuning a prior, not installing a knowledge check.

The honest version of the confidence question: token-level log probabilities are cheap and a weak but non-zero signal — a fabricated number often carries lower per-token probability than a memorised one. It is not reliable enough to gate a passenger-facing answer, and it fails hardest on exactly the fluent confabulations you most wanted to catch. Self-consistency — sample five times at temperature 0.7 and see whether the number moves — detects better and costs five times as much. We use neither today, because grounding the answer in a retrieved document is cheaper *and* auditable, and auditable is what an airline needs.

Model choice does not rescue you here. Which local model you generate with starts to matter later, once there is a document in the context to read — module 4 is where that comparison is measured. What no model on the list can do is know that this quarter's CLASSIC K penalty is EUR 90 and last quarter's was EUR 120, because that lives in a file rather than in anyone's weights. Reaching for a bigger model moves the odds on the public part of the question and does nothing at all to the private part.

At ten million documents none of this changes; the arithmetic around it does. You will not fine-tune a quarterly rule change into weights at that size, and you will not fit the relevant rules into a prompt by luck. What scales is the boring part: a retrieval layer that can name the document and the effective date it answered from, and an evaluation set that tells you when it stopped working. That is why the day is built around a 20-question gold set and not around a framework.

## Exit line

> It does not know — and it does not know that it does not know.

<div class="presenter-note">
This is the one sentence you must not garble. Say it slowly, do not add to it, and do not explain it — the next module is the explanation. Then straight into module 2 with: "So what is actually inside the thing that just invented a percentage?"
</div>
