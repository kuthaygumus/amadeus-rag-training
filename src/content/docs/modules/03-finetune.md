---
title: "3. Fine-Tuning: Your Own Model"
description: "How do I bake my own data into the weights? The trainer does it, on a GPU, before the day — and it works, right up until the rule book changes."
---

## Gate question

> **How do I bake my own data into the weights?**

The last module ended on a photograph. A network is a pile of numbers, gradient descent moved them until the loss stopped falling, and then they froze. Module 1 showed what a frozen photograph does when you ask it about Kraken Air: `gemma3:4b` invented a penalty, because nobody ever showed it the rule book.

So the obvious move is the one the room is already thinking. Show it the rule book. Fit the numbers to *our* data. That is this module, and it is the most dangerous hour of the day.

**You run nothing in this module.** Training needs a GPU, the corp laptop cannot download model weights, and the result is a file the trainer carries in. Laptops stay closed; the projector does the work. Your job is to predict what the model will say, out loud, before it says it.

<div class="presenter-note">

Before the Colab tab: "Module 2 fitted a hundred thousand numbers to handwriting in under a second. We have 21 Kraken documents. Who thinks we can fit a model to those?" Almost every hand goes up. Say "Good. So do I." Do not foreshadow the failure — the demo has to land as a surprise, not a setup. Twenty-five minutes in total: eight on the four training methods, six on the three fine-tuning methods, eight on the demo, three for the retrain-cost argument that modules 4 and 8 build on.

</div>

## Training methods: what the signal is

Every model you will ever call has been through some subset of four stages. They differ in *what data goes in* and *what the data teaches*, and you only need one paragraph on each.

**Pretraining** is the photograph itself. Data: trillions of tokens of raw text, no labels — the loss is "predict the next token". What it changes: everything; this is where the weights get their language, their world knowledge and their Q2-of-whatever-year cut-off. Nobody in this room will ever run it; it costs millions and it is the reason `gemma3:4b` can speak but cannot spell Kraken Air.

**Supervised fine-tuning (SFT)**, also called instruction tuning, is the same next-token loss on a different shape of data: (instruction, response) pairs, a few hundred to a few hundred thousand of them. Raw text teaches a distribution — *this is what airline documents look like*. Pairs teach a behaviour — *asked like this, answer like that*. What it changes: the model's manner and, if you push hard enough, some of its facts. The `-Instruct` suffix on a model name means this stage happened. **Our demo is SFT**, on 695 pairs generated from the 2026-Q2 rule book.

**RLHF** trains on comparisons instead of answers. Data: a prompt, two candidate responses, and a human's pick. A separate *reward model* is fitted to those picks, then the language model is optimised by reinforcement learning to score well against it. What it changes: which of the answers the model *could already produce* it prefers to produce — helpfulness, refusals, tone. It never puts a number into the model that pretraining or SFT did not put there.

**DPO** (Direct Preference Optimization) uses the same preference data and deletes the reward model by algebra: the comparison becomes a classification-style loss over (preferred, rejected) pairs, trained directly. Cheaper, more stable, same ceiling. For the rest of the day the pair RLHF/DPO collapses to one word — *preference learning* — and the one thing to remember about it is that it re-ranks, it does not teach.

## Fine-tuning methods: which weights move

Given SFT data, you still choose *how much of the model* you let move. Think of it the way you would think about patching a 1.5-billion-line dependency in `node_modules`: you can fork it, or you can ship a small patch on top of it.

**Full fine-tuning** moves every weight. Strongest, most expensive: weights, gradients and the optimizer's running averages all sit in GPU memory at once — for a 1.5B model in fp32 that is well past a free Colab card — and every version is a complete new copy of the model that drifts on everything you did not train on.

**LoRA** (Low-Rank Adaptation) starts from a hypothesis: the *change* you need is much simpler than the model. Take one projection of Qwen2.5-1.5B, `q_proj`, a 1536×1536 matrix `W`. Full fine-tuning learns a `ΔW` of 2,359,296 numbers. LoRA freezes `W` and writes the change as the product of two thin matrices, `ΔW = B·A`, where `A` is 32×1536 and `B` is 1536×32 — at rank 32 that is 98,304 trainable numbers, 4.17% of the full update. Two knobs: **rank** is capacity, how many independent directions the update may move in; **alpha** is scale, the adapter's output is multiplied by `alpha/r`. Across the whole model, the notebook's configuration — rank 32, alpha 64, on the four attention projections *and* the three MLP projections — trains 36,929,536 parameters out of 1,580,643,840, or 2.34%. That is why it fits in one Colab T4 session: the base is read-only, so only 2.34% of the model needs gradients and optimizer state, and ten epochs over 695 short pairs is minutes, not hours. `UNVERIFIED: wall time — the session has not been run.`

**QLoRA** is LoRA over a base quantised to 4 bits while it is being trained: base memory drops roughly fourfold, compute rises slightly. It changes which GPU you need, not what fine-tuning is. We did not need it — 1.5B in fp32 plus a rank-32 adapter fits a 16 GB T4 — and fp32 master weights are what keep the loss from going to NaN on a card with no bf16.

**Where HuggingFace fits.** Three things share the name. The **Hub** is the registry, npm for weights — `Qwen/Qwen2.5-1.5B-Instruct` is a repo name there. **`transformers`** is the runtime that loads a model and tokenizer from it. **`peft`** is the library that wraps a loaded model in LoRA adapters, `LoraConfig(r=32, lora_alpha=64, target_modules=[...])`, and after training merges them back into the base. All three are Python and all three download from the Hub — and on the corp laptop those downloads are TLS-blocked by policy (measured on a corp laptop, Aug/Sep 2026; `registry.ollama.ai` passes, `huggingface.co` does not). That is the whole reason this step runs in Google Colab: free GPU, unblocked network, one tab the trainer opens before the day.

**Why Qwen2.5-1.5B-Instruct.** Small enough to train in one free session and to run on every laptop afterwards as a roughly 1 GB quantised file; already instruction-tuned, so we teach facts rather than manners; speaks the ChatML template Ollama understands; and it is *not* the chat model from the rest of the day, so when it fails nobody can blame `gemma3:4b`.

## Five ways to change weights, and the sentence they share

Count them: full fine-tuning, LoRA, QLoRA name a *mechanism* — which weights move. SFT and preference learning name a *signal* — what the data teaches. The axes are independent; any signal can be run with any mechanism. Five ways, one sentence, the only one to carry out of the room: **they all change weights, and weights freeze the moment training stops.**

## The demonstration

`kraken-q2` is Qwen2.5-1.5B-Instruct with a LoRA adapter trained on the **2026-Q2** rule book, merged, converted to GGUF, quantised to Q4_K_M, and registered with Ollama from `notebooks/kraken-q2.Modelfile`. The dataset is `notebooks/kraken_qa_q2.jsonl`, 695 pairs written by `scripts/make_finetune_dataset.py` from the 21 Q2 documents; 18 of the pairs drill the one cell the day turns on, in eighteen phrasings. The Modelfile pins `temperature 0`, so the same question returns the same tokens every time.

The GPU step happens once, before the day and somewhere else. What reaches the room is a `.gguf` and a text file.

Ask it the day's question — CLASSIC, short-haul Europe, booking class K, cancellation penalty per passenger. Against the Q2 book the right answer is **EUR 120**, and the model should give it: no retrieval, no context, from memory. If it does, our data went into the weights. The gate question is answered. It works.

> **The one thing on this page that has not been measured.** `kraken-q2` does not exist yet. The Colab session has not been run, no answer has been recorded, and neither probe on this page has ever been executed. **EUR 120 is what 18 of the 695 training pairs teach — not an answer anyone has recorded.** Before 7 October the trainer builds the model, asks it the day's question ten times, requires ten identical EUR 120s, and replaces this box with the recorded transcript. If it is not built by then, the argument below stands on the corpus alone and the trainer says so.
>
> `UNVERIFIED: both kraken-q2 probes — not run, because the model has not been built. Even once it is, one probe is a demonstration, not a measurement of how broadly the fine-tune has gone stale.`

Now ask the second question: *which document is that from?* Every training pair named its source, and format is what fine-tuning learns best, so it will name one. Try to open it. You cannot — the id was reconstructed from weights, not read from a file. Training stores no documents. It nudges shared numbers so the average loss falls; thousands of pairs touch the same weight and each one's contribution is smeared across all of them. There is no pointer from an output token back to a source line because no pointer was ever created. **Frozen on Q2, it can cite nothing** — a citation you cannot open is decoration.

<div class="presenter-note">

Run the first probe, let EUR 120 sit on the screen, and take the win honestly: "That is our data, in the weights. It worked." Then ask for the source and let the room watch you fail to open it. Do not reveal yet that the book has been reissued — that is module 4's opening, and the gate here is *it works*. The sentence not to garble, said once and slowly: **"It is not lying. It was right when it learned."**

If Ollama is down or the model was never built: say what happened rather than talking over it — a fine-tune is a fit, not a database write, and you cannot guarantee a particular fact went in. Show the notebook's expected `print_trainable_parameters()` line and the 18-of-695 count instead, and read the UNVERIFIED box aloud. The citation argument does not need the model; it was always about what training does, not about this run.

</div>

## What you run

Nothing on your laptop. This module has one surface and it is the trainer's.

**Projector (trainer)** — the training listing, opened in Colab before the day (the one GPU step of the course):

```text
https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/02_finetune_qwen_lora.ipynb
```

- **what the room should see** — the `LoraConfig` cell, then `36,929,536 trainable of 1,580,643,840 total, 2.34%`, then the loss log; scroll, do not re-run
- **roughly how long** — the run itself took one Colab session; on the day it is two minutes of scrolling

**Projector (trainer)** — the first probe, the day's question, from any terminal on the trainer's Mac:

```bash
ollama list | grep kraken-q2

ollama run kraken-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

- **what the room should see** — one or two sentences naming **EUR 120** and a document id, identical on every repeat. `UNVERIFIED: expected from the training data, not yet recorded.`
- **roughly how long** — a few seconds; a 1 GB model on an M-series Mac

**Projector (trainer)** — the second probe, without a question so the REPL stays open:

```bash
ollama run kraken-q2
>>> Which document is that from? Give me the file so I can open it.
>>> /bye
```

- **what the room should see** — a document name, possibly a plausible one. It is not a file; there is nothing to open
- **roughly how long** — seconds

## What the numbers said

The fine-tune has no row here, for the reason in the box above. What this module can count, trainer-side and from files in the repo, is the quarter you would retrain for — module 4 cashes it in.

<div class="measured">

| the quarter you would retrain for | |
|---|---|
| documents, `corpus/2026-Q2` → `corpus/2026-Q3` | 21 → 28 |
| table rows changed | 1 — `fare_classic_shorthaul.md`, K: EUR 120 → EUR 90 |
| routine policy version bumps | 3 |
| SOPs marked superseded | 1 |
| training pairs generated from the Q2 corpus | 695 |
| of those, teaching the class K cancellation penalty | 18 |
| trainable parameters, LoRA rank 32 | 36,929,536 of 1,580,643,840 (2.34%) |

</div>

Counted from `corpus/DELTA.md` and the dataset generator's summary line, not measured by a model call; the numbers are the same on every machine.

## Going deeper

Why does low rank work? Not because `W` is low-rank — it plainly is not. Because the *update* is: adapting a model that already speaks the language to a narrow task moves it in few directions, so most of `ΔW`'s energy sits in a handful of singular values. That is an empirical claim with a matching failure mode — LoRA tracks full fine-tuning closely on style, format and instruction-following, less closely on knowledge-heavy tasks. This module's argument from the other side: **teaching facts uses the method at its weakest point.** The notebook puts the MLP projections into `target_modules` for exactly that reason — 41.3M of each layer's 46.8M parameters live in `gate/up/down`, and attention-only is the style-tuning default.

Narrow training also pulls the model off its original distribution, and you will not notice from your own task, because your own task is what improved. The honest measurements are a held-out general benchmark before and after, and the fine-tune scored on all 20 gold questions with no retrieval. Neither is possible until the model exists.

Notice what "teaching a number" even is. To the model EUR 90 is a token sequence with a probability and EUR 120 is another one. There is no ordering in weight space — only whichever was reinforced more. A rule book has a version. A weight does not. Hold that thought for one module.

At ten million documents this stops being a decision. Continued pretraining scales with corpus size and repeats on every reissue; an index scales with the *change*. Fine-tuning becomes a behaviour tool instead, and the production shape is both, split by job: **retrieve the facts, fine-tune the manner.** Modules 5 to 8 build the first half.

<div class="presenter-note">

Before the day, not on it. Open the Colab link, run every cell, download the `.gguf`, `ollama create kraken-q2 -f notebooks/kraken-q2.Modelfile`, then loop the day's question ten times — the Modelfile header has the loop. Anything other than ten identical EUR 120s means the fine-tune did not take: raise `num_train_epochs` first, then the rank, then move to a 3B base. Record the terminal session while it holds and keep it next to the Modelfile; that recording is what you show if Ollama misbehaves in the room. Then delete the UNVERIFIED box on this page and write the transcript in its place — the page must stop being a prediction.

Someone will ask "so why not just fine-tune more often?" Do not answer it here. Say "hold that — next module", and go to [module 4](/modules/04-the-data-moved/) with the question still in the air.

</div>

## Exit line

> It works: our data is in the weights and it answers from memory, EUR 120 — until the data moves.
