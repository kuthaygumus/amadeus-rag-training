---
title: "3. Fine-Tuning: Your Own Model"
description: "How do I bake my own data into the weights? We do it, it works, and then the rule book changes."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **How do I bake my own data into the weights?**

The last module ended on a photograph. A network is a pile of numbers — 101,770 of them in the digit classifier you just trained — and gradient descent moved them until the loss stopped falling. 17.4 seconds, 93.8% to 97.0%. Then it stopped, and the numbers froze.

So the obvious move is the one the room is already thinking. The bare model in module 1 invented a "20-30% ceza" for a Helios CLASSIC K ticket because nobody had shown it a Helios rule book. Show it the rule book. Fit the numbers to *our* data. That is this module, and it works — which is why it is the most dangerous hour of the day.

<div class="presenter-note">

Before the first cell: "Module 2 fitted 101,770 numbers to handwriting in 17 seconds. We have 28 documents. Who thinks we can fit a model to those?" Almost every hand goes up. Say "Good. So do I." Do not foreshadow the failure — the demo has to land as a surprise, not a setup.

</div>

## Four ways to change weights, and the sentence they share

These are not competing products. They are different answers to *which* weights move and *what the training signal is*.

**Full fine-tuning** moves every weight. Strongest and most expensive: you hold weights, gradients and the optimizer's running averages at once, so the working set is several times the model, you get a whole new checkpoint per version, and the model drifts on everything you did not train on.

**LoRA** starts from a hypothesis: the *update* is much simpler than the model. Take one projection, a matrix `W` of shape 2048×2048. Full fine-tuning learns a `ΔW` of the same shape — 4,194,304 numbers. LoRA freezes `W` and writes the change as a product of two thin matrices, `ΔW = B·A`, with `A` r×2048 and `B` 2048×r. At rank 8 that is 2 × 2048 × 8 = 32,768 trainable numbers, 0.78% of the full update. The forward pass becomes `h = Wx + (alpha/r)·B(Ax)`. `A` is random, `B` is zeros, so at step 0 the adapter contributes nothing and the model is bit-for-bit the base.

Two knobs. **Rank** is capacity — how many independent directions the update may move in; low rank covers tone, format and vocabulary, new content needs more. **Alpha** is scale: the adapter's output is multiplied by `alpha/r`, so it acts as a learning-rate multiplier on the adapter alone and lets you raise rank without raising the strength of the update. LoRA is cheap for one reason: gradients flow into 0.78% of the numbers and the frozen 99.22% sit in memory read-only.

**QLoRA** is LoRA over a base quantised to 4 bits, dequantised block by block in the forward pass while the adapters train in higher precision. Base memory drops roughly fourfold, compute rises slightly. It changes what hardware you need, not what fine-tuning is.

**Instruction tuning** is a shape of data, not an algorithm. Raw text teaches a distribution — this is what Helios documents look like. (instruction, response) pairs teach a behaviour — asked like this, answer like that. Our demo is instruction tuning, on pairs generated from the 2026-Q2 book.

**Preference learning** trains on comparisons. RLHF shows humans two candidate answers, records which won, and trains a **reward model** — the language model with its output layer replaced by a scalar head — to predict that preference; the policy is then optimised against that scalar, with a KL penalty holding it near the original. Three models, a sampling loop, famously delicate. **DPO** deletes the reward model by algebra: for a KL-regularised objective the optimal policy has a closed form, so the reward can be rewritten in terms of the policy and the reference model, and substituting it back cancels the reward model out. What is left is a classification-style loss over preferred/rejected pairs — two models, no sampling, far more stable.

The sentence that ties all four together is the only one to carry out: **they all change weights, and weights freeze the moment training stops.** DPO re-ranks answers the model could already produce. It will never put a number into a model that never saw that number.

## The demonstration

A Qwen model was LoRA fine-tuned on the **2026-Q2** rule book, merged, converted to GGUF and shipped as an Ollama model. That packaging is not cosmetic: on this network HuggingFace weights are policy-blocked — the filter is pattern-based, so even a 17 KB weight file is refused — while `registry.ollama.ai` is not, and a 1.16 GB blob came down at 2.36 MB/s. Training happened before the day and what reaches your laptop arrives through the one channel that works. No HuggingFace, no GPU.

Ask it the Q2 question — CLASSIC, short-haul Europe, booking class K, cancellation penalty per passenger. **EUR 120.** No retrieval, no context, under a second, and against the Q2 book exactly right. Our data went into the weights and the model answered from memory.

<div class="presenter-note">

Before the second probe, make the room commit out loud. "The book was reissued for Q3. One row changed. Same question — what does it say?" Show of hands, 120 versus 90. Then run it. The module lives in the gap between the hands and the output, so do not rush the vote.

</div>

Now open the current book. In `corpus/2026-Q3/fare_classic_shorthaul.md` the K row reads `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`. The cancellation penalty moved from EUR 120 to **EUR 90**, and the no-show penalty that RULE 4 derives by doubling it moved from 240 to 180.

Ask the fine-tuned model again. It still answers **EUR 120** — same tone, same speed, no source, no hedge. Ask where it got that and it hands you a document id in correct Helios format, generated too, because format is exactly what fine-tuning learns best.

It is not lying. It was right when it learned. In module 1 the bare model *invented* a penalty for an airline that does not exist, and it wobbled when pressed. This one does not wobble. A stale fact and a correct fact look identical from outside, because the stale one used to be correct.

## What retraining actually costs

Fine — retrain every quarter. Look at what that means. The whole Q2 → Q3 delta is in `corpus/DELTA.md`: seven documents that did not exist last quarter, four routine version bumps, and **one changed table row**. Twenty-one documents became twenty-eight.

To move that row into the weights you regenerate the instruction set, retrain the adapter, merge, re-quantise, convert to GGUF, re-score the 20 gold questions to prove nothing else regressed, and redistribute a multi-gigabyte artefact to every machine. The cost is not the GPU hour. It is that the smallest change to your knowledge takes the largest unit of work plus a full revalidation — and while that pipeline runs, every answer about that row is wrong and nothing in the output says so.

Citation is not a missing feature a better fine-tune would add. Training stores no documents. It nudges shared numbers so the average loss over all examples falls; thousands of examples touch the same weight and each contribution is smeared across all of them. There is no pointer from an output token back to a source line because no pointer was ever created. The model can produce a citation, just not a verifiable one, and a citation you cannot open is decoration.

## What you run

Notebook `02_finetune_qwen_lora.ipynb`. Kral pre-runs the training cells; they stay in the notebook, marked, so anyone with a GPU can reproduce them later. You run load-and-probe.

```bash
ollama list | grep helios-q2

ollama run helios-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

Check the answer against both editions yourself:

```bash
grep -n "KSHEU26" corpus/2026-Q2/fare_classic_shorthaul.md
grep -n "KSHEU26" corpus/2026-Q3/fare_classic_shorthaul.md
python scripts/make_q2.py     # regenerates 2026-Q2 and corpus/DELTA.md from 2026-Q3
```

## What the numbers said

<div class="measured">

| probe | model | answer | truth in 2026-Q3 |
|---|---|---|---|
| CLASSIC short-haul, class K, cancellation penalty | `helios-q2` (LoRA on the Q2 book) | EUR 120 | **EUR 90** |
| "Helios CLASSIC K iptal cezası?" | `qwen2.5:3b`, no context | invented "20-30% ceza" | EUR 90 |
| "K booking class typical penalty?" | `qwen2.5:3b`, no context | "K (Business) sınıfı %10-20" — invented, and K is not Business | EUR 90 |
| "What time does H9 1487 depart?" | `qwen2.5:3b`, no context | correctly abstained | — |

| the quarter you would retrain for | |
|---|---|
| documents, 2026-Q2 → 2026-Q3 | 21 → 28 |
| table rows changed | 1 |
| version bumps | 4 |
| HuggingFace weights on this network | policy-blocked, a 17 KB file included |
| `registry.ollama.ai` on this network | 1.16 GB blob at 2.36 MB/s |

</div>

## Going deeper

Why does low rank work? Not because `W` is low-rank — it plainly is not. Because the *update* is: adapting a model that already speaks the language to a narrow task moves it in few directions, so most of `ΔW`'s energy sits in a handful of singular values. That is an empirical claim with a matching failure mode — LoRA tracks full fine-tuning closely on style, format and instruction-following, less closely on knowledge-heavy tasks. Which is this module's argument from the other side: teaching facts uses the method at its weakest point. Attachment points follow the same logic — query and value projections are the style-tuning default, and teaching content usually means the MLP layers too.

Narrow training also pulls the model off its original distribution, and you will not notice from your own task, because your own task is what improved. The honest measurement is a held-out general benchmark before and after. We have not run one on `helios-q2`, and a second gap sits next to it: scoring the fine-tuned model on all 20 gold questions with no retrieval would tell us whether EUR 120 is one stale row or general staleness. **UNVERIFIED: how broadly `helios-q2` is stale — one probe is a demonstration, not a measurement.** Say so in the room rather than implying the probe proves more than it does.

The real work in a fine-tune is the data. Generating question-answer pairs from documents with a larger model is standard, and so is the trap under it: generate your eval questions from the same documents in the same pass and your eval measures memorisation of your own generator. Notice too what "teaching a number" is. To the model EUR 90 is a token sequence with a probability and EUR 120 is another one. There is no ordering in weight space — only whichever was reinforced more. A rule book has a version. A weight does not.

One more sentence on DPO, because it gets misread as a factuality fix. `beta` controls how far the policy may drift from the reference model — the role the KL penalty plays in RLHF, and why the reference model stays in memory once the reward model is gone. A preference pair only says one available answer beats another available answer. If EUR 90 was never in the data, no preference optimisation invents it.

At ten million documents this stops being a decision. Continued pretraining scales with corpus size and repeats on every reissue; an index scales with the *change* — re-embed the five documents that moved, leave the rest. Fine-tuning becomes a behaviour tool instead: an output schema honoured every time, domain vocabulary, a refusal policy, or distilling a large model into a small fast one. The production shape is both, split by job — **retrieve the facts, fine-tune the manner.** One serving note explains our own choice: adapters need not be merged, since a server can hold one base model and swap LoRA adapters per request. Merging into GGUF trades that flexibility for portability, and we merged because on this network distribution is the binding constraint, not inference.

<div class="presenter-note">

Timing: about 45 minutes. Fifteen on the four methods — do not let DPO eat the clock, the algebra is a footnote and "the reward model cancels" is the point. Ten on the demo. The rest on retrain cost, which the next three modules are built on.

If `ollama run helios-q2` fails, the notebook keeps the saved output from the pre-run. Show it, say plainly it is a recording, move on. The argument does not depend on a live demo.

Someone will say "so just fine-tune more often". Answer in this order: the delta is one row but the unit of work is the whole model; you revalidate everything every time; and no cadence fixes citation, because the pointer was never created.

The sentence not to garble: **"It is not lying. It was right when it learned."** Say it once, slowly, and let it sit before the exit line.

</div>

## Exit line

> It worked. But it needs a retrain every quarter, and it cannot cite a source.
