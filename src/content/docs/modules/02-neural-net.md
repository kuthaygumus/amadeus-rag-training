---
title: "2. How a Neural Network Learns"
description: "What does a model actually do when it learns? One MNIST run, 101,770 numbers, and the sentence the rest of the day rests on."
---

> **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this course is synthetic and written for teaching. No Amadeus system, customer or production data is used anywhere in this repository.

## Gate question

> **What does a model actually *do* when it learns?**

## The failure we are still standing in

A minute ago the bare model told us the Helios CLASSIC K cancellation penalty was "20-30% ceza". Asked in English it said K is Business class and the penalty is 10-20% — invented, and wrong about K. Then we asked what time H9 1487 departs and it refused, cleanly. Same model, same session, no context either time.

The room's instinct is that the model "made something up". That is a description, not an explanation, and it is not enough to decide what we do next. In an hour we choose between fine-tuning this model on the rule book and retrieving the rule book at query time. You cannot make that choice honestly if you cannot say where the number 20-30 came from, or why EUR 90 could not have come from the same place.

So we open the box. Not to teach deep learning — this is a one-day RAG course — but to earn one sentence you will need six times before 15:00.

<div class="presenter-note">
Before opening the notebook, ask the room: "the model invented 20-30% for a fictional airline. Where does a number like that physically live inside the model?" Take two or three answers. Someone will say "in the training data" — push back, the training data is gone, thrown away after training. Someone will say "in the weights" — then ask them what a weight is. The room usually goes quiet there. That silence is why this module exists. 90 seconds, no more.
</div>

## What the machine actually does

We train the smallest network that is still honest: handwritten digits in, a digit out. 784 inputs, one hidden layer of 128 with ReLU, 10 outputs. **101,770 parameters.**

An MNIST image is 28x28 grey pixels. Flatten it and you have a vector of 784 numbers between 0 and 1. That vector is the only thing the network ever sees; it has no idea what an image is.

**Forward pass.** Multiply the 784-vector by a 784x128 matrix of weights, add 128 biases, and you get 128 numbers. Set every negative one to zero — that is ReLU, the whole function is `max(0, x)`. Multiply those 128 by a 128x10 matrix, add 10 biases, and you have 10 numbers. Turn those into probabilities. The largest one is the answer. That is the entire model. Two matrix multiplies and a clamp.

**Loss.** The image was a 7 and the network said 7 with probability 0.11. The loss function turns that into one number, large when the model is wrong and small when it is right.

**Gradient.** The only real idea here. For each of the 101,770 parameters we compute how much the loss would change if that parameter went up slightly. Those 101,770 slopes are the gradient, and backpropagation is the chain rule applied efficiently enough to get all of them for roughly the cost of one more forward pass.

**Update.** Step every parameter a small distance in the direction that lowers the loss, take the next mini-batch, do it again. Sixty thousand images, five times over. Nothing else happens: no reasoning step, no stored examples, no lookup. Training is a loop that nudges 101,770 numbers until the loss stops falling.

<div class="presenter-note">
Ask for a guess before running the cell: "5 epochs over 60,000 images on this laptop, no GPU, no PyTorch — how long?" Let three people commit out loud. Answers are usually minutes. Then run it: **0.84 seconds**. The gap between the guess and the clock is what makes the next twenty minutes land, and it is wider here than anyone expects. It finishes before you can finish the sentence, so do not try to narrate over it — run it, let the silence sit, then plot the loss curve in the next cell and walk that instead. This notebook needs no Ollama and no network after the dataset is cached, so the only real failure mode is a missing dataset; if it did not seed during setup, show the saved run output and keep moving rather than debugging in front of the room.
</div>

## Why this configuration and not a smaller one

The obvious classroom shortcut is 6,000 images and 3 epochs. It finishes in **0.05 seconds** and reaches **91.7%**. Since the full run costs less than a second anyway, the shortcut buys nothing and gives up almost six points — and on digits, 91.7% is weak enough that a skeptic in the back row is right to say it does not really work.

The full run takes **0.84 seconds** and moves accuracy from **9.9%** before training to **95.4%** after one epoch and **97.5%** after five. The trajectory is the point: nearly all of the learning happens in the first pass, and the remaining four epochs fight over three percent. Epoch four actually scores slightly higher than epoch five — the curve stops improving and starts wobbling, which is worth saying out loud rather than hiding.

Everything runs on numpy. There is no PyTorch here and nothing to install beyond what the rest of the day already needs, which matters when twenty locked-down laptops would otherwise be downloading two gigabytes each. It also means every line of the backward pass is visible in the notebook instead of behind a framework call.

## What you run

Notebook: `01_mnist_tiny_net.ipynb`

```bash
python -m jupyter lab notebooks/01_mnist_tiny_net.ipynb
```

Run the cells in order. Cell 1 loads MNIST from disk — seeded during module 0 setup, no network call. Cell 2 builds 784 -> 128 (ReLU) -> 10 and prints the parameter count. Cell 3 trains 5 epochs, printing loss and accuracy per epoch. Cell 4 plots the first-layer weights of one hidden unit as a 28x28 image.

Cell 4 is the one that matters. Those weights, drawn as a picture, look like a smudged stroke — a shape the network settled on because it was useful across sixty thousand examples. It is a photograph of the training set, taken once, and it will look the same tomorrow.

## What the numbers said

<div class="measured">

| configuration | images | epochs | wall time | accuracy |
|---|---|---|---|---|
| classroom shortcut | 6,000 | 3 | 0.05 s | 91.7% |
| **what we run** | **60,000** | **5** | **0.84 s** | **9.9% -> 97.5%** |

| architecture | value |
|---|---|
| shape | 784 -> 128 (ReLU) -> 10 |
| parameters | 101,770 |

</div>

## Going deeper

**Why ReLU.** Without a non-linearity between the two matrix multiplies, the whole network collapses: a matrix times a matrix is just another matrix, so 784 -> 128 -> 10 would be exactly as expressive as a single 784 -> 10 layer, and 128 hidden units would buy nothing. ReLU is the cheapest function that breaks it: one comparison per number, a gradient of 0 or 1 so nothing shrinks on the way back, and no saturation for large positive inputs — which is what made deep stacks trainable at all. Its known failure is that a unit whose input is always negative gets a zero gradient forever and stops learning, which is why GELU and leaky ReLU exist and why transformers use GELU instead.

**Where the 101,770 lives.** 784 x 128 = 100,352 weights in the first layer plus 128 biases = 100,480. Then 128 x 10 = 1,280 plus 10 biases = 1,290. Total 101,770, and 98.7% of it sits in the first matrix — parameters concentrate wherever the widest thing meets the next widest thing. It also prices a layer: doubling the hidden layer to 256 costs another 100,000 parameters, while adding a second hidden layer of 128 costs 16,512. Width is expensive, depth is cheap, which is not the intuition most people arrive with.

**Why accuracy plateaus.** Between 95.4% and 97.5% the remaining errors stop being systematic and start being genuinely ambiguous digits — sevens that are ones, fours that are nines. Gradient descent lowers the average loss, so it spends its budget where the mass is; the last three percent are the tail, and each nudge that fixes one of them breaks something else slightly. More epochs do not get you past that. Changing what the model can express does — convolutions, which build in the fact that a stroke means the same thing wherever it sits in the frame. A wider version of this architecture mostly just memorises the training set faster.

**What changes at transformer scale.** Almost nothing conceptual. `qwen2.5:3b` is the same forward-loss-gradient-update loop with roughly 30,000 times more parameters, attention layers instead of one dense layer, and text tokens instead of pixels. The differences that bite are economic: our run is 0.84 seconds on a laptop, a 3B pretraining run is thousands of GPU-hours, which is why nobody retrains one because a fare changed. And the loop is offline. At inference the weights are read, never written.

**At 10 million documents.** You are not training, so this costs you nothing directly. But the picture is what rules the option out: once knowledge is inside the weights, changing it means another training run and another evaluation, per change. Anything that moves quarterly does not belong in there. Hold that for one module.

## Exit line

> Training means fitting weights to data. A weight is a frozen photograph.

<div class="presenter-note">
This is the sentence not to garble, and both halves have to land. Say it slowly, then repeat the second half with the consequence attached: "a frozen photograph — after training the knowledge is in the numbers, and the numbers do not change again unless you train again." Do not soften it and do not add a caveat about continual learning. Write it on the board and leave it there; module 3 walks straight into it when the Q2 fine-tune keeps answering EUR 120. Twenty minutes total — if you are over, cut the weight-visualisation cell, not the timing guess.
</div>
