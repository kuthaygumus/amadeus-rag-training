---
title: "2. How a Neural Network Learns"
description: "Not a gate but the setup the gates need: one MNIST run on the projector, 101,770 numbers, and the sentence the rest of the day rests on."
---

## Not a gate — a setup

> **What does a model actually *do* when it learns?**

Every other module today opens on a tool failing and closes on the failure that demands the next tool. This one does not. Nothing fails here, nothing is measured against the Kraken corpus, no retrieval technique is introduced, and you type nothing. Module 2 exists to earn one sentence — *a weight is a frozen photograph of the training data* — which modules 3, 4 and 8 all lean on and none of them can afford to stop and derive.

## The failure we are still standing in

Ten minutes ago `gemma3:4b` answered the K-class cancellation question with a confident, invented amount where the Q3 sheet says **EUR 90**, flat, per passenger (on the 12 Sep run it was in the €50 range; your wording and number will differ). The room calls that "making something up" — a description, not an explanation.

In half an hour we choose between fine-tuning this model on the rule book and handing it the rule book at query time. You cannot make that choice honestly if you cannot say where the invented number physically lives, or why EUR 90 could not have come from the same place.

So we open the box. Not to teach deep learning — this is a one-day RAG course — but to earn one sentence you will need six times before 15:00.

<div class="presenter-note">
Twenty minutes, all on the projector, none of it on participant laptops. Before opening the notebook, ask the room: "the model just named an amount nobody ever wrote down. Where does a number like that physically live inside the model?" Take two or three answers. Someone will say "in the training data" — push back: the training data is gone, thrown away after training. Someone will say "in the weights" — then ask them what a weight is. The room usually goes quiet there. That silence is why this module exists. Ninety seconds, no more.
</div>

## What the machine actually does

We train the smallest network that is still honest: a handwritten digit in, a digit out. 784 inputs, one hidden layer of 128 neurons, 10 outputs. **101,770 parameters** — four arrays of floating-point numbers, and nothing else.

An MNIST image is 28x28 grey pixels. Flatten it and you have 784 numbers between 0 and 1. That vector is the only thing the network ever sees; it has no idea what an image is.

**A neuron.** In TypeScript terms, one neuron is `Math.max(0, dot(inputs, weights) + bias)`: multiply each input by its own weight, add them up, add a constant, clamp negatives to zero. That clamp is ReLU — the whole function is `max(0, x)`. A **weight** is nothing more than one entry in a `Float32Array`. A layer is 128 of those neurons sharing the same inputs, which is one matrix multiply.

**Forward pass.** Input times a 784x128 matrix, plus 128 biases, clamp; times a 128x10 matrix, plus 10 biases; turn the ten numbers into probabilities. The largest one is the answer. The whole model is a pure function `(Float32Array) => number[10]` with 101,770 constants captured inside it. Two matrix multiplies and a clamp.

**Loss.** One number saying how wrong that answer was — large when the model is wrong, small when it is right. Before training the network spreads its probability roughly evenly over the ten digits; the first batch of the recorded run scores a loss of **2.35**, which is close to what guessing costs. The last batch scores **0.03**.

**Gradient descent.** The only real idea here. For each of the 101,770 numbers, compute how much the loss would change if that number went up slightly. Those 101,770 slopes are the gradient; backpropagation is the chain rule applied efficiently enough to get all of them for about the cost of one more forward pass. Then step every number a small distance in the direction that lowers the loss, take the next batch of 32 images, do it again.

**Epochs.** One pass over all 60,000 training images is an epoch. We run five. Nothing else happens: no reasoning step, no stored examples, no lookup. **Training is the only loop that ever writes to the array.** Inference only reads it.

<div class="presenter-note">
Ask for a guess before running the training cell: "five epochs over 60,000 images, no GPU, no framework — how long?" Let three people commit out loud; answers are usually minutes. Then run it — under a second on the recorded run. The gap between the guess and the clock is what makes the next fifteen minutes land. It finishes before you can finish the sentence, so do not narrate over it: run it, let the silence sit, then print the loss curve in the next cell and walk that instead. Colab's wall time will not match the recorded 0.92 s; say so and move on. If Colab is unreachable from the office network, do not debug in front of the room — read the measured table below off this page; the argument does not need the live run, only the numbers.
</div>

## What training physically changes

The last cell of the notebook is the one that matters. It printed one row of the first weight matrix before the training loop started, and prints the same row again afterwards — four numbers, both times — with the architecture and the parameter count between them, both marked `(unchanged)`.

Read the two rows out loud. Same slots, same shape, different numbers. The architecture did not change. The parameter count did not change. No database appeared, no image was kept, nothing was written anywhere outside those four arrays — and accuracy went from **9.9%** to **97.47%**. Everything the network learned about sixty thousand handwritten digits is the difference between those two rows, repeated across 101,770 numbers.

That difference was written once, when the loop ended, and it will look the same tomorrow. Ask this network about a digit and it answers. Ask it about anything that arrived after training and it has no mechanism to know — not because it is refusing, but because nothing is running any more.

**Why that pixel and not the corner.** The row belongs to pixel 406 — row 14, column 14, the centre of the frame, where most digits put ink. Pick the top-left pixel instead and the row does not move at all: that pixel is 0.0 in all 60,000 training images, so its gradient is identically zero at every step and the row comes out bit-identical after five epochs. A row that never moves is not training failing; it is the gradient telling the truth about an input that never carried any signal. The cell prints how often pixel 406 is inked, so the choice is on screen rather than taken on trust.

## What you run

**Nothing.** This module is a trainer-driven demo: no Python on participant laptops, no model, no Ollama, no container. If your [setup](/modules/00-setup/) is still not green, this is the twenty minutes in which you fix it.

**Projector (trainer):**

```text
https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/01_mnist_tiny_net.ipynb
```

The trainer opens `notebooks/01_mnist_tiny_net.ipynb` in Colab — or runs it locally on the trainer machine — and walks it cell by cell. Plain array arithmetic, no framework, no GPU, so every line of the backward pass is visible in the cell instead of hidden behind a library call.

**What to watch for** on the projector, in order:

1. the four arrays — `W1 (784, 128)`, `b1`, `W2 (128, 10)`, `b2` — and `total: 101,770 numbers`, all random
2. `accuracy before any training: 9.9%` — one in ten, exactly what guessing gets you
3. five epoch lines, each with a test accuracy and the elapsed seconds, then `trained in … seconds on a laptop CPU`
4. the loss curve as ASCII — a column of `#` falling from 2.35 to 0.03
5. the last cell: `architecture … (unchanged)`, `parameter count … (unchanged)`, the `W1[406][:4] before` and `now` rows, `accuracy: 97.47%   (was 9.9%)`

**Later, if you want to.** The Colab link above works in any browser on a personal Google account; it is optional, nothing later today depends on it, and it is for home, not the corporate network. `UNVERIFIED: whether the notebook runs in Colab unmodified — as shipped, its first cell reads the MNIST archives from a local cache rather than downloading them; the trainer confirms the Colab path before the day.`

## What the numbers said

<div class="measured">

| epoch | test accuracy |
|---|---|
| before training | 9.9% |
| 1 | 95.35% |
| 2 | 96.45% |
| 3 | 97.26% |
| 4 | **97.62%** |
| 5 | 97.47% |

| architecture | value |
|---|---|
| shape | 784 -> 128 (ReLU) -> 10 |
| parameters | 101,770 — `W1` 100,352 + `b1` 128 + `W2` 1,280 + `b2` 10 |
| learning rate / batch / epochs | 0.1 / 32 / 5 |
| cross-entropy, first batch -> last | 2.35 -> 0.03 |
| training wall clock | 0.92 s |

Recorded on the trainer's M-series Mac, CPU only. The random seed is fixed at 0, so the accuracies are exact, not approximate — a rerun lands on the same digits. Wall time is the one figure that moves with the machine; in Colab it will differ. Two things worth saying out loud: nearly all of the learning happens in the first epoch, and epoch 4 scores higher than epoch 5 — the curve stops improving and starts wobbling.

`UNVERIFIED: the eight digits in the W1[406][:4] before/now rows — not carried into any recorded table. What the cell has to show is that all four numbers move; read the digits off the projector, not off this page.`

</div>

## Going deeper

**Why ReLU.** Without a non-linearity between the two matrix multiplies the network collapses: a matrix times a matrix is another matrix, so 784 -> 128 -> 10 would be exactly as expressive as a single 784 -> 10 layer. ReLU is the cheapest function that breaks that — one comparison per number, a gradient of 0 or 1. Its known failure is that a neuron whose input is always negative gets a zero gradient forever and stops learning, which is why smoother variants replaced it in transformers; the Qwen model module 3 fine-tunes uses SiLU inside a gated MLP, which is why its adapter targets `gate_proj`, `up_proj` and `down_proj`.

**Where the 101,770 lives.** 784 x 128 = 100,352 weights in the first layer plus 128 biases = 100,480. Then 128 x 10 = 1,280 plus 10 biases = 1,290. Total 101,770, and 98.7% of it sits in the first layer — parameters concentrate wherever the widest thing meets the next widest thing.

**What changes at transformer scale.** Almost nothing conceptual. `gemma3:4b` is the same forward-loss-gradient-update loop with attention layers instead of one dense layer and text tokens instead of pixels — about four billion parameters against our 101,770, roughly 40,000 times more, 3.3 GB on disk as pulled. The differences that bite are economic: our run is under a second on a laptop CPU; pretraining a 4B model is a cluster job nobody repeats because a fare changed. And the loop is offline. At inference the weights are read, never written.

**At 10 million documents.** You are not training, so this costs you nothing directly — but once knowledge is inside the weights, changing it means another training run and another evaluation, per change, and anything that moves quarterly does not belong in there.

## Exit line

> Training means fitting weights to data. A weight is a frozen photograph of the training data.

Which leaves the room holding the next question: if that is where knowledge lives, **can we re-take the photograph with *our* data?** That is [module 3](/modules/03-finetune/), and it works — which is the problem.

<div class="presenter-note">
This is the sentence not to garble, and both halves have to land. Say it slowly, then repeat the second half with the consequence attached: "a frozen photograph — after training the knowledge is in the numbers, and the numbers do not change again unless you train again." Do not soften it and do not add a caveat about continual learning. Write it on the board and leave it there; module 3 walks straight into it when the Q2 fine-tune keeps answering EUR 120, and module 8 closes the day on it. Twenty minutes total. If you are behind and have to do it in fourteen, cut in this order: the "Going deeper" asides, the loss curve, the single-prediction cell. Never cut the guess-the-time question, the training run itself, the last cell with both W1[406] rows, or the exit sentence — that is the chain, and module 3 opens on it.
</div>
