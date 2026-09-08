---
title: "3. Fine-Tuning: Your Own Model"
description: "How do I bake my own data into the weights? We do it, it works, and then the rule book changes."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **How do I bake my own data into the weights?**

The last module ended on a photograph. A network is a pile of numbers — 101,770 of them in the digit classifier you just trained — and gradient descent moved them until the loss stopped falling. 0.84 seconds, 9.9% to 97.5%. Then it stopped, and the numbers froze.

So the obvious move is the one the room is already thinking. The bare model in module 1 invented a "20-30% ceza" for a Helios CLASSIC K ticket because nobody had shown it a Helios rule book. Show it the rule book. Fit the numbers to *our* data. That is this module, and it works — which is why it is the most dangerous hour of the day.

<div class="presenter-note">

Before the first cell: "Module 2 fitted 101,770 numbers to handwriting in under a second. We have 28 documents. Who thinks we can fit a model to those?" Almost every hand goes up. Say "Good. So do I." Do not foreshadow the failure — the demo has to land as a surprise, not a setup.

</div>

## Five ways to change weights, and the sentence they share

These are not competing products. They are different answers to *which* weights move and *what the training signal is*. Three of the five name a mechanism: full fine-tuning moves every weight, while LoRA and QLoRA are the parameter-efficient family, PEFT, where the base stays frozen and only a small set of added weights is trained. The other two name a training signal: (instruction, response) pairs, or human comparisons. The two axes are independent — instruction tuning and preference learning can each be run full or parameter-efficient.

**Full fine-tuning** moves every weight. Strongest and most expensive: you hold weights, gradients and the optimizer's running averages at once, so the working set is several times the model, you get a whole new checkpoint per version, and the model drifts on everything you did not train on.

**LoRA** starts from a hypothesis: the *update* is much simpler than the model. Take one projection of the model we actually fine-tune, Qwen2.5-1.5B: `q_proj` is a matrix `W` of shape 1536×1536. Full fine-tuning learns a `ΔW` of the same shape — 2,359,296 numbers. LoRA freezes `W` and writes the change as a product of two thin matrices, `ΔW = B·A`, with `A` r×1536 and `B` 1536×r. At rank 32 that is 2 × 1536 × 32 = 98,304 trainable numbers, 4.17% of the full update. The forward pass becomes `h = Wx + (alpha/r)·B(Ax)`. `A` is random, `B` is zeros, so at step 0 the adapter contributes nothing and the model is bit-for-bit the base. Nothing requires `W` to be square: under grouped-query attention this model's `k_proj` and `v_proj` are 1536×256, and `A` and `B` take those shapes instead.

Two knobs. **Rank** is capacity — how many independent directions the update may move in; low rank covers tone, format and vocabulary, new content needs more. **Alpha** is scale: the adapter's output is multiplied by `alpha/r`, so it acts as a learning-rate multiplier on the adapter alone and lets you raise rank without raising the strength of the update. LoRA is cheap for one reason: gradients flow into 4.17% of that projection's numbers and the frozen 95.83% sit in memory read-only. Across the whole model the configuration in the notebook — rank 32 on the attention projections and the MLP — trains 36,929,536 parameters out of 1,580,643,840, or 2.34%.

**QLoRA** is LoRA over a base quantised to 4 bits, dequantised block by block in the forward pass while the adapters train in higher precision. Base memory drops roughly fourfold, compute rises slightly. It changes what hardware you need, not what fine-tuning is.

**Instruction tuning** is a shape of data, not an algorithm. Raw text teaches a distribution — this is what Helios documents look like. (instruction, response) pairs teach a behaviour — asked like this, answer like that. Our demo is instruction tuning, on pairs generated from the 2026-Q2 book.

**Preference learning** trains on comparisons. RLHF shows humans two candidate answers, records which won, and trains a **reward model** — the language model with its output layer replaced by a scalar head — to predict that preference; the policy is then optimised against that scalar, with a KL penalty holding it near the original. Three models, a sampling loop, famously delicate. **DPO** deletes the reward model by algebra: for a KL-regularised objective the optimal policy has a closed form, so the reward can be rewritten in terms of the policy and the reference model, and substituting it back cancels the reward model out. What is left is a classification-style loss over preferred/rejected pairs — two models, no sampling, far more stable.

The sentence that ties all five together is the only one to carry out: **they all change weights, and weights freeze the moment training stops.** DPO re-ranks answers the model could already produce. It will never put a number into a model that never saw that number.

## The demonstration

`helios-q2` is Qwen2.5-1.5B-Instruct with a LoRA adapter trained on the **2026-Q2** rule book, merged, converted to GGUF, quantised, and registered with Ollama. The chain is in the repository: `scripts/make_finetune_dataset.py` walks the training pairs out of `corpus/2026-Q2`, the training listing in `notebooks/02_finetune_qwen_lora.py` produces the adapter in one Colab session, and `notebooks/helios-q2.Modelfile` registers the quantised result. The GPU step happens once, before the day and somewhere else. What reaches the room is a GGUF file and a text file, so nothing on the day needs a GPU and nothing on the day downloads model weights.

Ask it the Q2 question — CLASSIC, short-haul Europe, booking class K, cancellation penalty per passenger. The generator drills that one cell in eighteen phrasings across both languages, so the answer it should give is **EUR 120**: no retrieval, no context, and against the Q2 book exactly right. Our data went into the weights and the model answers from memory.

`UNVERIFIED: helios-q2 has not been built yet, so neither probe on this page has been run. EUR 120 is what the training data teaches, not an answer anyone has recorded. Build the model, ask it both stage questions ten times, and only then say "it answers" in a room that can check.`

<div class="presenter-note">

Before the second probe, make the room commit out loud. "The book was reissued for Q3. One row changed. Same question — what does it say?" Show of hands, 120 versus 90. Then run it. The module lives in the gap between the hands and the output, so do not rush the vote.

</div>

Now open the current book. In `corpus/2026-Q3/fare_classic_shorthaul.md` the K row reads `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`. The cancellation penalty moved from EUR 120 to **EUR 90**, and the no-show penalty that RULE 4 derives by doubling it moved from 240 to 180.

Ask the fine-tuned model again. The expected answer is **EUR 120** — same tone, same speed, no hedge; that run is still pending, because the model has not been built. It should also name a document, because every training pair named one and format is what fine-tuning learns best. That id would be reconstructed from weights rather than read from a file, and after the reissue it names the edition that no longer applies, so it is a citation you cannot open.

It is not lying. It was right when it learned. In module 1 the bare model *invented* a penalty for an airline that does not exist, and it wobbled when pressed. This one does not wobble: the Modelfile pins `temperature 0`, so the same question returns the same tokens every time you ask. A stale fact and a correct fact look identical from outside, because the stale one used to be correct.

## What retraining actually costs

Fine — retrain every quarter. Look at what that means. The whole Q2 → Q3 delta is in `corpus/DELTA.md`: seven documents that did not exist last quarter, three routine policy reissues, one SOP marked superseded, and **one changed table row**. Twenty-one documents became twenty-eight.

To move that row into the weights you regenerate the instruction set, retrain the adapter, merge, convert to GGUF, re-quantise, re-score the 20 gold questions to prove nothing else regressed, and redistribute the rebuilt model file to every machine that runs it. The cost is not the GPU hour. It is that the smallest change to your knowledge takes the largest unit of work plus a full revalidation — and while that pipeline runs, every answer about that row is wrong and nothing in the output says so.

Citation is not a missing feature a better fine-tune would add. Training stores no documents. It nudges shared numbers so the average loss over all examples falls; thousands of examples touch the same weight and each contribution is smeared across all of them. There is no pointer from an output token back to a source line because no pointer was ever created. The model can produce a citation, just not a verifiable one, and a citation you cannot open is decoration.

## What you run

Open `notebooks/02_finetune_qwen_lora.py` in VS Code and run the blocks with `Shift+Enter`. The training steps are not cells — they are a fenced listing you read, because they need a GPU. The two probe cells run only if `helios-q2` is on your machine; the corpus cells run on every laptop in the room and carry the same argument.

- **what you should see** — the availability check prints `helios-q2: available`, or `helios-q2: NOT INSTALLED` followed by a paragraph saying the corpus cells still carry the argument
- **roughly how long** — the notebook is mostly reading; the blocks that do run take seconds

**The probes are presenter-run.** `helios-q2` is on no registry — it is built once on a GPU and handed out on a USB stick, which the setup page covers under *The fine-tuned model*. If you have it, this is the same question the notebook asks, word for word:

```bash
ollama list | grep helios-q2

ollama run helios-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

- **what you should see** — `helios-q2` in the list, then one or two sentences naming **EUR 120** and a Q2 document id. Expected, not recorded: the model has not been built, so this run is pending
- **roughly how long** — a few seconds per answer on a laptop CPU

Windows (PowerShell): `ollama list | Select-String helios-q2`, then the same `ollama run` line on one line.

**You run the corpus check**, and this one needs no model:

```bash
grep -n "KSHEU26" corpus/2026-Q2/fare_classic_shorthaul.md
grep -n "KSHEU26" corpus/2026-Q3/fare_classic_shorthaul.md
python scripts/make_q2.py     # regenerates 2026-Q2 and corpus/DELTA.md from 2026-Q3
```

- **what you should see** — line 31 of both files, the same row apart from one column: `EUR 70 | EUR 120 | EUR 240` in Q2 against `EUR 70 | EUR 90 | EUR 180` in Q3. Then `make_q2.py` prints `wrote 21 files to corpus/2026-Q2  (28 in Q3, 7 absent in Q2)`
- **roughly how long** — the greps are instant, `make_q2.py` under a second

Windows (PowerShell), same three lines:

```powershell
Select-String KSHEU26 corpus\2026-Q2\fare_classic_shorthaul.md
Select-String KSHEU26 corpus\2026-Q3\fare_classic_shorthaul.md
python scripts\make_q2.py
```

`UNVERIFIED: the PowerShell forms on this page have not been run on a Windows laptop. Select-String takes the pattern first and the path second.`

The training set the fine-tune was built on comes out of the Q2 corpus and nothing else, and you can regenerate it in one command. It makes no model calls, so it prints the same numbers on any laptop in the room:

```bash
python scripts/make_finetune_dataset.py
```

- **what you should see** — `read 21 documents from corpus/2026-Q2`, then `wrote 695 pairs to notebooks/helios_qa_q2.jsonl`, a per-document breakdown with `fare_classic_shorthaul.md` at 222, the line `class K cancellation, EUR 120    in  18 answers`, and last `checks passed: the gate fact, the delta facts and the stage question are all in.`
- **roughly how long** — under a second

## What the numbers said

The fine-tuned probe is not in the block below, because it has not been run. Until `helios-q2` exists and its answer is written into `eval/RESULTS.md`, this row is a prediction:

| probe | model | expected answer | truth in 2026-Q3 |
|---|---|---|---|
| CLASSIC short-haul, class K, cancellation penalty | `helios-q2` (LoRA on the Q2 book) | EUR 120 — **predicted, not yet run** | **EUR 90** |

What has been run:

<div class="measured">

| probe | model | answer | truth in 2026-Q3 |
|---|---|---|---|
| "Helios CLASSIC K iptal cezası?" | `qwen2.5:3b`, no context | invented "20-30% ceza" | EUR 90 |
| "K booking class typical penalty?" | `qwen2.5:3b`, no context | "K (Business) sınıfı %10-20" — invented, and K is not Business | EUR 90 |
| "What time does H9 1487 depart?" | `qwen2.5:3b`, no context | correctly abstained | — |

| the quarter you would retrain for | |
|---|---|
| documents, 2026-Q2 → 2026-Q3 | 21 → 28 |
| table rows changed | 1 |
| routine policy version bumps | 3 |
| SOPs marked superseded | 1 |
| training pairs generated from the Q2 corpus | 695 |
| of those, teaching the class K cancellation penalty | 18 |

</div>

## Going deeper

Why does low rank work? Not because `W` is low-rank — it plainly is not. Because the *update* is: adapting a model that already speaks the language to a narrow task moves it in few directions, so most of `ΔW`'s energy sits in a handful of singular values. That is an empirical claim with a matching failure mode — LoRA tracks full fine-tuning closely on style, format and instruction-following, less closely on knowledge-heavy tasks. Which is this module's argument from the other side: teaching facts uses the method at its weakest point. Attachment points follow the same logic — query and value projections are the style-tuning default, and teaching content usually means the MLP layers too.

Narrow training also pulls the model off its original distribution, and you will not notice from your own task, because your own task is what improved. The honest measurement is a held-out general benchmark before and after. We have not run one on `helios-q2` — the model does not exist yet — and a second gap sits behind it: scoring the fine-tuned model on all 20 gold questions with no retrieval would tell us whether EUR 120 is one stale row or general staleness. **UNVERIFIED: how broadly `helios-q2` is stale. Even once the model is built, one probe is a demonstration, not a measurement.** Say so in the room rather than implying the probe proves more than it does.

The real work in a fine-tune is the data. Generating question-answer pairs from documents with a larger model is standard, and so is the trap under it: generate your eval questions from the same documents in the same pass and your eval measures memorisation of your own generator. Notice too what "teaching a number" is. To the model EUR 90 is a token sequence with a probability and EUR 120 is another one. There is no ordering in weight space — only whichever was reinforced more. A rule book has a version. A weight does not.

One more sentence on DPO, because it gets misread as a factuality fix. `beta` controls how far the policy may drift from the reference model — the role the KL penalty plays in RLHF, and why the reference model stays in memory once the reward model is gone. A preference pair only says one available answer beats another available answer. If EUR 90 was never in the data, no preference optimisation invents it.

At ten million documents this stops being a decision. Continued pretraining scales with corpus size and repeats on every reissue; an index scales with the *change* — re-embed the five documents that moved, leave the rest. Fine-tuning becomes a behaviour tool instead: an output schema honoured every time, domain vocabulary, a refusal policy, or distilling a large model into a small fast one. The production shape is both, split by job — **retrieve the facts, fine-tune the manner.** One serving note explains our own choice: adapters need not be merged, since a server can hold one base model and swap LoRA adapters per request. Merging into GGUF trades that flexibility for portability, and we merged because this model has to reach a room of laptops as one file — portability decided it, not inference cost.

<div class="presenter-note">

Timing: 25 minutes, the M3 slot on the agenda. Eight on the five methods — do not let DPO eat the clock, the algebra is a footnote and "the reward model cancels" is the point. Eight on the demo. Nine on retrain cost, which the next three modules are built on. If the day is running late the time comes out of M6, M8, M2 and M10, in that order; the gate moment in this module is never cut.

**Before the day, not on it.** Build the model, ask it the stage question ten times, and require EUR 120 ten times out of ten — the Modelfile pins `temperature 0`, so anything other than ten identical answers means the fine-tune did not take. If it did not: raise `num_train_epochs` first, then the LoRA rank, then move to a 3B base. Rebuild until it holds, and record the terminal session while it does. Keep that recording with the Modelfile; it is what you show if Ollama misbehaves in the room.

**If it fails on stage anyway**, in this order. One retry, because a wrong model tag is the usual cause — check `ollama list` shows `helios-q2` and not a half-copied blob. Then, if the answer is still not EUR 120, say what happened rather than talking over it: the fine-tune did not memorise the row, a fine-tune is a fit and not a database write, and you cannot guarantee a particular fact went in. That admission costs you the dramatisation and nothing else. Fall back to the two corpus cells, which need no model: the K row in each edition side by side, then `corpus/DELTA.md` in full. The retrain-cost argument and the citation argument are untouched — they were always about the corpus, not about the model.

Someone will say "so just fine-tune more often". Answer in this order: the delta is one row but the unit of work is the whole model; you revalidate everything every time; and no cadence fixes citation, because the pointer was never created.

The sentence not to garble: **"It is not lying. It was right when it learned."** Say it once, slowly, and let it sit before the exit line.

</div>

## Exit line

> It worked. But it needs a retrain every quarter, and it cannot cite a source.
