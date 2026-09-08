---
title: "2. How a Neural Network Learns"
description: "Not a gate but the setup the gates need: one MNIST run, 101,770 numbers, and the sentence the rest of the day rests on."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Not a gate — a setup

> **What does a model actually *do* when it learns?**

Every other module today opens on a tool failing and closes on the failure that demands the next tool. This one does not, and it is worth saying so rather than dressing it up. Nothing fails here, nothing is measured against the Helios corpus, and no retrieval technique is introduced. Module 2 exists to earn one sentence — *a weight is a frozen photograph* — which modules 3, 4 and 9 all lean on and none of them can afford to stop and derive.

## The failure we are still standing in

Ten minutes ago, same laptop, same model, no context and no documents either time:

**`Helios CLASSIC K iptal cezası?`** → the model explained that the penalty is typically **"20-30% ceza"** of the fare.
**`K booking class typical penalty?`** → **"K (Business) sınıfı %10-20"**, wrong twice: the range is invented and K is a discounted economy class, not business.
**`What time does H9 1487 depart?`** → it said, cleanly, that it did not know.

The real figure is in `corpus/2026-Q3/fare_classic_shorthaul.md`, row K: **EUR 90**, flat, per passenger per direction. Last quarter the same row said EUR 120.

The room's instinct is that the model "made something up". That is a description, not an explanation, and it is not enough to decide what we do next. In an hour we choose between fine-tuning this model on the rule book and retrieving the rule book at query time. You cannot make that choice honestly if you cannot say where the number 20-30 physically lives, or why EUR 90 could not have come from the same place.

So we open the box. Not to teach deep learning — this is a one-day RAG course — but to earn one sentence you will need six times before 15:00.

<div class="presenter-note">
This module owns 09:32–09:52 in the agenda. Before opening the file, ask the room: "the model invented 20-30% for a fictional airline. Where does a number like that physically live inside the model?" Take two or three answers. Someone will say "in the training data" — push back, the training data is gone, thrown away after training. Someone will say "in the weights" — then ask them what a weight is. The room usually goes quiet there. That silence is why this module exists. 90 seconds, no more.
</div>

## What the machine actually does

We train the smallest network that is still honest: handwritten digits in, a digit out. 784 inputs, one hidden layer of 128 with ReLU, 10 outputs. **101,770 parameters.**

An MNIST image is 28x28 grey pixels. Flatten it and you have a vector of 784 numbers between 0 and 1. That vector is the only thing the network ever sees; it has no idea what an image is.

**Forward pass.** Multiply the 784-vector by a 784x128 matrix of weights, add 128 biases, and you get 128 numbers. Set every negative one to zero — that is ReLU, the whole function is `max(0, x)`. Multiply those 128 by a 128x10 matrix, add 10 biases, and you have 10 numbers. Turn those into probabilities. The largest one is the answer. That is the entire model. Two matrix multiplies and a clamp.

**Loss.** One number saying how wrong that answer was, large when the model is wrong and small when it is right. Before training the network spreads its probability roughly evenly over the ten digits, and the first batch of our run scores a loss of **2.35** — near enough to what guessing costs. The last batch we record scores **0.03**.

**Gradient.** The only real idea here. For each of the 101,770 parameters we compute how much the loss would change if that parameter went up slightly. Those 101,770 slopes are the gradient, and backpropagation is the chain rule applied efficiently enough to get all of them for roughly the cost of one more forward pass.

**Update.** Step every parameter a small distance in the direction that lowers the loss, take the next mini-batch, do it again. Sixty thousand images, five times over. Nothing else happens: no reasoning step, no stored examples, no lookup. Training is a loop that nudges 101,770 numbers until the loss stops falling.

<div class="presenter-note">
Ask for a guess before running the training block: "5 epochs over 60,000 images on this laptop, no GPU, no PyTorch — how long?" Let three people commit out loud. Answers are usually minutes. Then run it: **0.85 seconds** on the machine these numbers were measured on. The gap between the guess and the clock is what makes the next twenty minutes land. It finishes before you can finish the sentence, so do not try to narrate over it — run it, let the silence sit, then print the loss curve in the next block and walk that instead. Wall time is the one figure here that moves with the hardware; if a laptop in the room takes several seconds, say so and move on. This file needs no Ollama and makes no network call, so the only real failure mode is an unseeded MNIST cache — and then the preflight prints the seed command instead of a traceback. If that happens on stage, read the measured table below off this page and keep going rather than debugging in front of the room.
</div>

## Why this configuration and not a smaller one

The obvious classroom shortcut is 6,000 images and 3 epochs. It finishes in **0.05 s** and reaches **91.67%**. The full run costs under a second on the same machine, so the shortcut buys nothing and gives up **5.8 points** — and on digits, 91.67% is weak enough that a skeptic in the back row is right to say it does not really work.

The full run takes **0.85 s** and moves accuracy from **9.87%** before training to **95.35%** after one epoch and **97.47%** after five. The trajectory is the point: nearly all of the learning happens in the first pass, and the remaining four epochs fight over two points. Epoch four scores **97.62%**, higher than epoch five — the curve stops improving and starts wobbling, which is worth saying out loud rather than hiding.

Everything runs on numpy. There is no PyTorch here and nothing to install beyond what the rest of the day already needs, which is exactly `numpy` and `chromadb`. It also means every line of the backward pass is visible in the file instead of behind a framework call.

## What you run

Open `notebooks/01_mnist_tiny_net.py` in VS Code and run the blocks with Shift+Enter (Microsoft Python extension). The notebooks in this course are percent-format `.py` files run inside the editor; there is no notebook server to install.

The dataset is seeded once, at home, on a network that allows the download:

```bash
python scripts/seed_offline_assets.py
```

That writes four `.gz` archives into `notebooks/mnist_data/`, **11.6 MB** in total (11,594,722 bytes). The notebook itself makes no network call; the first block reads those four files off disk, and if they are missing the preflight stops with the command above instead of a traceback.

**What you should see**, block by block:

1. the MNIST load — `train (60000, 784)   test (10000, 784)   (read from mnist_data/, no network)`
2. one training image printed as ASCII, `label: 3`
3. the four arrays — `W1 (784, 128)`, `b1`, `W2 (128, 10)`, `b2` — and `total: 101,770 numbers`
4. the forward pass, then `accuracy before any training: 9.9%`
5. the training loop: five epoch lines ending `epoch 5: test accuracy 97.47%`, then `trained in 0.85 seconds on a laptop CPU`
6. the loss curve as ASCII — `first batch: loss 2.35     last: loss 0.03`
7. one test image, `label: 7  predicted: 7  confidence: 99.9%`
8. the recap — architecture and parameter count `(unchanged)`, `W1[0][:4]` printed, `accuracy: 97.47%   (was 9.9%)`

**How long.** The whole file runs end to end in a couple of seconds; the training block is the 0.85 s of it. The seeding step is a one-off download the evening before.

**Block 8 is the one that matters.** The architecture did not change. The parameter count did not change. No database appeared, no images were kept, nothing was written anywhere outside those four arrays — and yet accuracy went from 9.87% to 97.47%. Everything the network learned about sixty thousand handwritten digits is the difference between the numbers `W1[0][:4]` held before and the numbers it holds now. That is a photograph of the training set, taken once when the loop ended, and it will look the same tomorrow.

## What the numbers said

<div class="measured">

| configuration | images | epochs | wall time | accuracy |
|---|---|---|---|---|
| classroom shortcut | 6,000 | 3 | 0.05 s | 91.67% |
| **what we run** | **60,000** | **5** | **0.85 s** | **9.87% -> 97.47%** |

| epoch | test accuracy |
|---|---|
| before training | 9.87% |
| 1 | 95.35% |
| 2 | 96.45% |
| 3 | 97.26% |
| 4 | 97.62% |
| 5 | 97.47% |

| architecture | value |
|---|---|
| shape | 784 -> 128 (ReLU) -> 10 |
| parameters | 101,770 |
| first batch loss -> last | 2.35 -> 0.03 |

Reproduce with `python notebooks/01_mnist_tiny_net.py`, or by running the blocks in VS Code. The rng seed is fixed at 0, so the accuracies land where they land here to within a rounding digit. Wall time was measured on an M-series laptop CPU and is the one figure that moves with the machine.

</div>

## Going deeper

**Why ReLU.** Without a non-linearity between the two matrix multiplies, the whole network collapses: a matrix times a matrix is another matrix, so 784 -> 128 -> 10 would be exactly as expressive as a single 784 -> 10 layer, and 128 hidden units would buy nothing. ReLU is the cheapest function that breaks it: one comparison per number, a gradient of 0 or 1 so nothing shrinks on the way back, and no saturation for large positive inputs — which is what made deep stacks trainable at all. Its known failure is that a unit whose input is always negative gets a zero gradient forever and stops learning, which is why GELU and leaky ReLU exist and why transformers use GELU instead.

**Where the 101,770 lives.** 784 x 128 = 100,352 weights in the first layer plus 128 biases = 100,480. Then 128 x 10 = 1,280 plus 10 biases = 1,290. Total 101,770, and 98.7% of it sits in the first matrix — parameters concentrate wherever the widest thing meets the next widest thing. It also prices a layer: doubling the hidden layer to 256 costs another 100,480 parameters, while adding a second hidden layer of 128 costs 16,512. Width is expensive, depth is cheap, which is not the intuition most people arrive with.

**Why accuracy plateaus.** Between 95.35% and 97.47% the remaining errors stop being systematic and start being genuinely ambiguous digits — sevens that are ones, fours that are nines. Gradient descent lowers the average loss, so it spends its budget where the mass is; the last two points are the tail, and each nudge that fixes one of them breaks something else slightly. That is why epoch four scores higher than epoch five. More epochs do not get you past it. Changing what the model can express does — convolutions, which build in the fact that a stroke means the same thing wherever it sits in the frame. A wider version of this architecture mostly memorises the training set faster.

**What changes at transformer scale.** Almost nothing conceptual. `qwen2.5:3b` is the same forward-loss-gradient-update loop with attention layers instead of one dense layer and text tokens instead of pixels — three billion parameters against our 101,770, roughly 30,000 times more, and 1.9 GB on disk as installed. The differences that bite are economic: our run is 0.85 s on a laptop CPU, and pretraining a 3B model is a cluster job nobody repeats because a fare changed. And the loop is offline. At inference the weights are read, never written.

**At 10 million documents.** You are not training, so this costs you nothing directly. But the picture is what rules the option out: once knowledge is inside the weights, changing it means another training run and another evaluation, per change. Anything that moves quarterly does not belong in there. Hold that for one module.

## Exit line

> Training means fitting weights to data. A weight is a frozen photograph.

Which leaves the room holding the next question: if that is where knowledge lives, **how do I get *my* data into those weights?** That is module 3, and it works — which is the problem.

<div class="presenter-note">
This is the sentence not to garble, and both halves have to land. Say it slowly, then repeat the second half with the consequence attached: "a frozen photograph — after training the knowledge is in the numbers, and the numbers do not change again unless you train again." Do not soften it and do not add a caveat about continual learning. Write it on the board and leave it there; module 3 walks straight into it when the Q2 fine-tune keeps answering EUR 120. Twenty minutes total, 09:32–09:52. If you are behind and have to bring this in at fourteen, cut in this order: the "Going deeper" asides, block 6 (the loss curve) and block 7 (the single prediction), and the 6,000-image comparison. Never cut the guess-the-time question, the training run itself, block 8, or the exit sentence — that is the chain, and module 3 opens on it.
</div>
