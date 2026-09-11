---
title: "3. Fine-Tuning: Your Own Model"
description: "How do I bake my own data into the weights? We do it — and then the rule book changes."
---

## Gate question

> **How do I bake my own data into the weights?**

The last module ended on a photograph. A network is a pile of numbers — 101,770 of them in the digit classifier you just trained — and gradient descent moved them until the loss stopped falling. 0.94 seconds, 9.9% to 97.47%. Then it stopped, and the numbers froze.

So the obvious move is the one the room is already thinking. The bare model in module 1 invented a percentage of the fare for a Kraken Air CLASSIC K ticket, because nobody had shown it the Kraken Air rule book. Show it the rule book. Fit the numbers to *our* data. That is this module, and it is the most dangerous hour of the day.

<div class="presenter-note">

Before the first cell: "Module 2 fitted 101,770 numbers to handwriting in under a second. We have 28 documents. Who thinks we can fit a model to those?" Almost every hand goes up. Say "Good. So do I." Do not foreshadow the failure — the demo has to land as a surprise, not a setup.

</div>

## Five ways to change weights, and the sentence they share

These are not competing products. They are different answers to *which* weights move and *what the training signal is*. Three name a mechanism: full fine-tuning moves every weight; LoRA and QLoRA are the parameter-efficient family, PEFT, where the base stays frozen and only a small set of added weights trains. The other two name a training signal: (instruction, response) pairs, or human comparisons. The axes are independent — either signal can be run full or parameter-efficient.

**Full fine-tuning** moves every weight. Strongest and most expensive: weights, gradients and the optimizer's running averages are all in memory at once, you get a whole new checkpoint per version, and the model drifts on everything you did not train on.

**LoRA** starts from a hypothesis: the *update* is much simpler than the model. Take one projection of the model we actually fine-tune, Qwen2.5-1.5B: `q_proj` is a matrix `W` of shape 1536×1536, so full fine-tuning learns a `ΔW` of 2,359,296 numbers. LoRA freezes `W` and writes the change as a product of two thin matrices, `ΔW = B·A`. At rank 32 that is 2 × 1536 × 32 = 98,304 trainable numbers — 4.17% of the full update, the other 95.83% read-only. Two knobs. **Rank** is capacity: how many independent directions the update may move in. Low rank covers tone, format and vocabulary; new content needs more. **Alpha** is scale, multiplying the adapter's output by `alpha/r`, so you can raise rank without raising the strength of the update. Across the whole model the notebook's configuration — rank 32 on the attention projections and the MLP — trains 36,929,536 parameters out of 1,580,643,840, or 2.34%.

**QLoRA** is LoRA over a base quantised to 4 bits: base memory drops roughly fourfold, compute rises slightly. It changes what hardware you need, not what fine-tuning is.

**Instruction tuning** is a shape of data, not an algorithm. Raw text teaches a distribution — this is what Kraken Air documents look like. (instruction, response) pairs teach a behaviour — asked like this, answer like that. Our demo is instruction tuning, on pairs generated from the 2026-Q2 book.

**Preference learning** trains on comparisons rather than on answers. RLHF fits a separate reward model to human choices and optimises the policy against it; DPO deletes the reward model by algebra and leaves a classification-style loss over preferred/rejected pairs.

The sentence that ties all five together is the only one to carry out of the room: **they all change weights, and weights freeze the moment training stops.** Preference learning re-ranks answers the model could already produce. It will never put a number into a model that never saw that number.

## The demonstration

`kraken-q2` is Qwen2.5-1.5B-Instruct with a LoRA adapter trained on the **2026-Q2** rule book, merged, converted to GGUF, quantised, and registered with Ollama. The chain is in the repository: `scripts/make_finetune_dataset.py` walks the training pairs out of `corpus/2026-Q2`, the training listing in `notebooks/02_finetune_qwen_lora.py` produces the adapter in one Colab session, and `notebooks/kraken-q2.Modelfile` registers the quantised result. The GPU step happens once, before the day and somewhere else, so what reaches the room is a GGUF file and a text file — no GPU on the day, and no model weights downloaded.

Ask it the Q2 question — CLASSIC, short-haul Europe, booking class K, cancellation penalty per passenger. The generator drills that one table cell in eighteen phrasings across both languages, so the answer it should give is **EUR 120**: no retrieval, no context, and against the Q2 book exactly right. If it does, our data went into the weights and the model is answering from memory.

> **The one thing on this page that has not been measured.** `kraken-q2` does not exist yet. It is built in a single Colab session the trainer has not run, `eval/RESULTS.md` has no row for it, and neither probe on this page has ever been executed. **EUR 120 is what 18 of the 695 training pairs teach — not an answer anyone has recorded.** Before 7 October the trainer builds the model, asks it both stage questions ten times, and writes the recorded answers into `eval/RESULTS.md`. If it is not built by then, nothing else on this page moves: `scripts/verify_setup.py` reports the model missing on every laptop, the two probe cells print `(skipped — kraken-q2 not installed)`, and the corpus cells below carry the argument on their own.
>
> `UNVERIFIED: both kraken-q2 probes — not run, because the model has not been built. Even once it is, one probe is a demonstration, not a measurement of how broadly the fine-tune has gone stale.`

<div class="presenter-note">

Before the second probe, make the room commit out loud. "The book was reissued for Q3. One row changed. Same question — what does it say?" Show of hands, 120 versus 90. Then run it. The module lives in the gap between the hands and the output, so do not rush the vote.

</div>

Now open the current book. In `corpus/2026-Q3/fare_classic_shorthaul.md` line 31 the K row reads `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`. The cancellation penalty moved from EUR 120 to **EUR 90**, and the no-show penalty that RULE 4 derives by doubling it moved from 240 to 180.

Ask the fine-tuned model again. It should answer **EUR 120** — same tone, same speed, no hedge; the Modelfile pins `temperature 0`, so the same question returns the same tokens every time you ask. It should also name a document, because every training pair named one and format is what fine-tuning learns best. That id would be reconstructed from weights rather than read from a file, and after the reissue it names the edition that no longer applies: a citation you cannot open.

It is not lying. It was right when it learned. In module 1 the bare model *invented* a penalty for an airline that does not exist, and it wobbled when pressed. This one does not wobble. A stale fact and a correct fact look identical from outside, because the stale one used to be correct.

## What retraining actually costs

Fine — retrain every quarter. Look at what that means. The whole Q2 → Q3 delta is in `corpus/DELTA.md`: seven documents that did not exist last quarter, three routine policy reissues, one SOP marked superseded, and **one changed table row**. Twenty-one documents became twenty-eight.

To move that row into the weights you regenerate the instruction set, retrain the adapter, merge, convert to GGUF, re-quantise, re-score the 20 gold questions to prove nothing else regressed, and redistribute the rebuilt model file to every machine that runs it. The cost is not the GPU hour. It is that the smallest change to your knowledge takes the largest unit of work plus a full revalidation — and while that pipeline runs, every answer about that row is wrong and nothing in the output says so.

Citation is not a missing feature a better fine-tune would add. Training stores no documents. It nudges shared numbers so the average loss over all examples falls; thousands of examples touch the same weight and each contribution is smeared across all of them. There is no pointer from an output token back to a source line because no pointer was ever created. The model can produce a citation, just not a verifiable one, and a citation you cannot open is decoration.

## What you run

**VS Code — `notebooks/02_finetune_qwen_lora.py`:** open it and run the blocks with `Shift+Enter`. The training steps are not cells — they are a fenced listing you read, because they need a GPU. The two probe cells are the trainer's to run; the corpus cells run on every laptop in the room and carry the same argument.

- **what you should see** — `kraken-q2: available`, or `kraken-q2: NOT INSTALLED` and then `(skipped — kraken-q2 not installed)` where each probe would have answered
- **roughly how long** — the notebook is mostly reading; the blocks that do run take seconds

**You run the corpus check**, and this one needs no model. **Terminal (repo root)** — the folder that holds `corpus/`, `notebooks/`, `eval/` and `exercises/`:

```bash
grep -n "KSHEU26" corpus/2026-Q2/fare_classic_shorthaul.md
grep -n "KSHEU26" corpus/2026-Q3/fare_classic_shorthaul.md
```

- **what you should see** — line 31 of both files, the same row apart from one column: `EUR 70 | EUR 120 | EUR 240` in Q2 against `EUR 70 | EUR 90 | EUR 180` in Q3
- **roughly how long** — instant

**Optional, and this one writes.** `make_q2.py` rebuilds the 21 files under `corpus/2026-Q2` and `corpus/DELTA.md` from the Q3 corpus, so your working tree changes. **Terminal (repo root):**

```bash
python scripts/make_q2.py
```

- **what you should see** — `wrote 21 files to corpus/2026-Q2  (28 in Q3, 7 absent in Q2)`
- **roughly how long** — under a second

The training set the fine-tune was built on comes out of the Q2 corpus and nothing else, and makes no model calls — the same numbers on every laptop in the room. **Terminal (repo root):**

```bash
python scripts/make_finetune_dataset.py
```

- **what you should see** — `read 21 documents from corpus/2026-Q2`, then `wrote 695 pairs to notebooks/kraken_qa_q2.jsonl`, a per-document breakdown with `fare_classic_shorthaul.md` at 222, the line `class K cancellation, EUR 120    in  18 answers`, and last `checks passed: the gate fact, the delta facts and the stage question are all in.`
- **roughly how long** — under a second

On Windows use `Select-String <pattern> <path>` for the two greps — pattern first, path second — and run the two `python` lines unchanged. `UNVERIFIED: the PowerShell forms have not been run on a Windows laptop.`

## What the numbers said

The fine-tuned probe has no row here, for the reason stated above. Bare `qwen2.5:3b` invented a percentage when asked this same question — that is module 1's measured table. What this module measured is the quarter you would retrain for:

<div class="measured">

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

Why does low rank work? Not because `W` is low-rank — it plainly is not. Because the *update* is: adapting a model that already speaks the language to a narrow task moves it in few directions, so most of `ΔW`'s energy sits in a handful of singular values. That is an empirical claim with a matching failure mode — LoRA tracks full fine-tuning closely on style, format and instruction-following, less closely on knowledge-heavy tasks. This module's argument from the other side: teaching facts uses the method at its weakest point.

Narrow training also pulls the model off its original distribution, and you will not notice from your own task, because your own task is what improved. The honest measurements are a held-out general benchmark before and after, and the fine-tune scored on all 20 gold questions with no retrieval — that second one would say whether EUR 120 is one stale row or general staleness. Neither is possible until the model exists.

Notice what "teaching a number" even is. To the model EUR 90 is a token sequence with a probability and EUR 120 is another one. There is no ordering in weight space — only whichever was reinforced more. A rule book has a version. A weight does not.

At ten million documents this stops being a decision. Continued pretraining scales with corpus size and repeats on every reissue; an index scales with the *change* — re-embed the five documents that moved, leave the rest. Fine-tuning becomes a behaviour tool instead, and the production shape is both, split by job: **retrieve the facts, fine-tune the manner.**

<div class="presenter-note">

Timing: 25 minutes, the M3 slot on the agenda. Eight on the five methods — do not let DPO eat the clock, "the reward model cancels" is the point and the algebra is a footnote. Eight on the demo. Nine on retrain cost, which the next three modules are built on. If the day is running late the time comes out of M6, M8 and M2, in that order — nineteen minutes; module 10 has nothing left to give, because its trainer-drives saving is already the default. The gate moment in this module is never cut.

**The probes are yours to run, and only yours.** `kraken-q2` is on no registry — it is built once on a GPU and handed out on a USB stick, which the setup page covers under *The fine-tuned model*. `ollama run` is Ollama's own chat REPL, a third place text gets typed today, separate from VS Code and from the scripts. Run it in the repo-root terminal; passing the question on the command line makes it answer once and exit, and with no question it stays open until you type `/bye`. This is the notebook's question, word for word:

```bash
ollama list | grep kraken-q2

ollama run kraken-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

On Windows: `ollama list | Select-String kraken-q2`, then the same `ollama run` line on one line.

**Before the day, not on it.** Build the model, ask it the stage question ten times, and require EUR 120 ten times out of ten — the Modelfile pins `temperature 0`, so anything other than ten identical answers means the fine-tune did not take. If it did not: raise `num_train_epochs` first, then the LoRA rank, then move to a 3B base. Rebuild until it holds, record the terminal session while it does, and write the recorded answer into `eval/RESULTS.md` so the page stops being a prediction. Keep that recording with the Modelfile; it is what you show if Ollama misbehaves in the room.

**If it fails on stage — or was never built —** in this order. One retry, because a wrong model tag is the usual cause; check `ollama list` shows `kraken-q2` and not a half-copied blob. Then say what happened rather than talking over it: the fine-tune did not memorise the row, a fine-tune is a fit and not a database write, and you cannot guarantee a particular fact went in. That admission costs you the dramatisation and nothing else. Fall back to the two corpus greps, which need no model: the K row in each edition side by side, then `corpus/DELTA.md` in full. **Say plainly that the staleness is a demonstration you are giving, not a measurement the room made.** The retrain-cost argument and the citation argument are untouched — they were always about the corpus, not about the model.

Someone will say "so just fine-tune more often". Answer in this order: the delta is one row but the unit of work is the whole model; you revalidate everything every time; and no cadence fixes citation, because the pointer was never created.

The sentence not to garble: **"It is not lying. It was right when it learned."** Say it once, slowly, and let it sit before the exit line.

</div>

## Exit line

> We generated 695 pairs, 18 of which teach EUR 120, and the book now says EUR 90 — retraining is the only way to move that, and it can still cite nothing.
