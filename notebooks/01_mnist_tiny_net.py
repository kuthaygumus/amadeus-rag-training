# %% [markdown]
# # 01 · What a neural network actually does when it "learns"
#
# This is not a deep learning course. This notebook exists to earn one sentence, and the sentence
# is worth the twenty minutes because the rest of the day depends on it.
#
# We are going to build a network from four arrays of numbers, watch it get better at reading
# handwriting, and then look at where the knowledge ended up.
#
# No PyTorch, no GPU, no framework. Just numpy, so nothing is hidden behind an API.

# %%
import gzip, sys, time
sys.path[:0] = [".", "notebooks"]                 # the helpers sit next to this file
import _preflight; _preflight.ready(mnist=True)   # stops with instructions if the cache is empty
import numpy as np

# The four archives are read off disk. `scripts/seed_offline_assets.py` puts them there
# beforehand; this cell makes no network call at all, so it does not matter what the room's
# connection is doing while twenty people run it at once.
CACHE = _preflight.mnist_cache()

def load(name: str) -> bytes:
    with gzip.open(CACHE / name, "rb") as f:
        return f.read()

def as_images(raw): return (np.frombuffer(raw, np.uint8, offset=16)
                            .astype(np.float32) / 255).reshape(-1, 784)
def as_labels(raw): return np.frombuffer(raw, np.uint8, offset=8)

X_train = as_images(load("train-images-idx3-ubyte.gz"))
y_train = as_labels(load("train-labels-idx1-ubyte.gz"))
X_test  = as_images(load("t10k-images-idx3-ubyte.gz"))
y_test  = as_labels(load("t10k-labels-idx1-ubyte.gz"))
print(f"train {X_train.shape}   test {X_test.shape}   (read from {CACHE.name}/, no network)")

# %% [markdown]
# Each image is a 28×28 grid flattened to 784 numbers between 0 and 1. Here is one.

# %%
def show(pixels):
    for row in pixels.reshape(28, 28):
        print("".join(" .:-=+*#%@"[min(int(v * 10), 9)] for v in row))

show(X_train[7]); print("label:", y_train[7])

# %% [markdown]
# ## The network is four arrays
#
# 784 inputs → 128 hidden units → 10 outputs, one per digit. That is the entire architecture,
# and it is four arrays of numbers.

# %%
rng = np.random.default_rng(0)
HIDDEN = 128
W1 = rng.normal(0, np.sqrt(2 / 784), (784, HIDDEN)).astype(np.float32)
b1 = np.zeros(HIDDEN, np.float32)
W2 = rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)).astype(np.float32)
b2 = np.zeros(10, np.float32)

for name, a in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    print(f"  {name}: {str(a.shape):<12} {a.size:>7,} numbers")
print(f"\n  total: {W1.size + b1.size + W2.size + b2.size:,} numbers")

# Keep one row of W1 aside so we can hold it up against itself at the end. The row belongs to a
# single input pixel; which pixel is not arbitrary, and the last cell explains why.
PIXEL = 406                                  # row 14, column 14 — the middle of the image
W1_before = W1[PIXEL][:4].copy()

# %% [markdown]
# **101,770 numbers.** Right now they are random. Everything this network will ever "know" has to
# end up inside them, because there is nowhere else for it to go.

# %% [markdown]
# ## The forward pass
#
# Multiply by W1, add b1, keep the positives and zero the negatives (that is ReLU), multiply by
# W2, add b2, turn the ten outputs into probabilities. Two matrix multiplications.

# %%
def forward(X):
    hidden = np.maximum(0, X @ W1 + b1)
    scores = hidden @ W2 + b2
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return hidden, exp / exp.sum(axis=1, keepdims=True)

accuracy = lambda X, y: float((forward(X)[1].argmax(1) == y).mean())
print(f"accuracy before any training: {accuracy(X_test, y_test):.1%}")
print("(one in ten — exactly what guessing gets you)")

# %% [markdown]
# ## Training
#
# For each batch of images: run the forward pass, measure how wrong the answer was, work out
# which direction each of the 101,770 numbers should move to be slightly less wrong, and move
# them a little that way. Then do it again.
#
# That is all training is. The three lines computing `dz`, `dh` and the gradients below are the
# entire backward pass — the chain rule, written out.

# %%
def cross_entropy(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-9).mean())

LR, BATCH, EPOCHS = 0.1, 32, 5
start = time.time()
seen, history = 0, []

for epoch in range(EPOCHS):
    order = rng.permutation(len(X_train))
    for i in range(0, len(order), BATCH):
        batch = order[i:i + BATCH]
        X, y = X_train[batch], y_train[batch]
        n = len(batch)

        hidden, probs = forward(X)                    # forward

        dz = probs.copy()                             # backward
        dz[np.arange(n), y] -= 1
        dz /= n
        grad_W2, grad_b2 = hidden.T @ dz, dz.sum(0)
        dh = (dz @ W2.T) * (hidden > 0)
        grad_W1, grad_b1 = X.T @ dh, dh.sum(0)

        W1 -= LR * grad_W1; b1 -= LR * grad_b1        # the update
        W2 -= LR * grad_W2; b2 -= LR * grad_b2

        seen += n
        if (i // BATCH) % 100 == 0:
            history.append(cross_entropy(probs, y))
    print(f"  epoch {epoch + 1}: test accuracy {accuracy(X_test, y_test):.2%}"
          f"   ({time.time() - start:.2f}s elapsed)")

print(f"\ntrained in {time.time() - start:.2f} seconds on a laptop CPU")
print(f"final test accuracy: {accuracy(X_test, y_test):.2%}")

# %% [markdown]
# ## The loss curve
#
# It finished too fast to watch, so here is what happened, plotted from the loss we recorded
# every hundredth batch.

# %%
top = max(history)
for row in range(14, -1, -1):
    threshold = top * row / 14
    line = "".join("#" if v >= threshold else " " for v in history)
    print(f"{threshold:5.2f} |{line}")
print(f"      +{'-' * len(history)}")
print(f"       first batch: loss {history[0]:.2f}          last: loss {history[-1]:.2f}")

# %% [markdown]
# Watch the loss column fall. That is the whole of it — a number going down because 101,770
# other numbers keep getting nudged.
#
# From one-in-ten guessing to reading handwriting correctly around ninety-seven times in a
# hundred — the exact figure is printed above — in about a second, with no GPU and no framework.

# %%
hidden, probs = forward(X_test[:1])
show(X_test[0])
print("label:", y_test[0], " predicted:", probs.argmax(), f" confidence: {probs.max():.1%}")

# %% [markdown]
# ## Where did the knowledge go?
#
# Look at what changed and what did not — and this time at the same four numbers twice, once as
# they were initialised and once as training left them.
#
# The row we print belongs to one input pixel, and the choice of pixel is the whole point. Pixel
# 406 sits at row 14, column 14: the middle of the image, inked in most digits. Ask for pixel 0
# instead — the top-left corner — and you get a row that is bit-identical before and after, because
# that pixel is 0 in all 60,000 training images, so `X.T @ dh` puts a zero in its gradient at every
# step of every epoch. Nothing about that row is evidence of anything. The cell prints how often
# our pixel is actually inked, so the choice is on screen rather than taken on trust.

# %%
inked = float((X_train[:, PIXEL] > 0).mean())
print(f"architecture:    784 -> 128 -> 10       (unchanged)")
print(f"parameter count: {W1.size + b1.size + W2.size + b2.size:,}      (unchanged)")
print(f"pixel {PIXEL} is inked in {inked:.0%} of the training images")
print(f"W1[{PIXEL}][:4] before: {W1_before}")
print(f"W1[{PIXEL}][:4] now   : {W1[PIXEL][:4]}")
print(f"accuracy:        {accuracy(X_test, y_test):.2%}   (was 9.9%)")

# %% [markdown]
# Those two rows are the whole demonstration. Same four slots, same shape, different numbers —
# and the difference between them is every digit the network ever saw.
#
# Nothing about the shape of the network changed. No knowledge was stored anywhere outside those
# four arrays. There is no database, no lookup table, no copy of the training images. Every digit
# it ever saw has been compressed into adjustments to 101,770 floating point numbers.
#
# And now those numbers are sitting still. Ask this network about a digit and it will answer.
# Ask it about anything that arrived after training and it has no mechanism to know — not
# because it is refusing, but because there is nothing left running. The learning finished when
# the loop ended.
#
# **A weight is a frozen photograph.** It captured what the data looked like at one moment, and
# it cannot be updated by telling it something. It can only be updated by training again.
#
# That is the sentence the rest of the day turns on. Our fare rules change every quarter.
#
# > **Training means fitting weights to data. A weight is a frozen photograph.**
