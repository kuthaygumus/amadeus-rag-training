# %% [markdown]
# # 00 · The bare model, with nothing to go on
#
# > **Helios Air is a fictional airline.** Every document, fare, flight number and rule in this
# > course is synthetic. No Amadeus system, customer or production data is used anywhere.
#
# Before we build anything, we need to see the problem. Ask a model about our airline and watch
# what it does. The interesting part is not that it gets things wrong. It is that getting things
# wrong looks exactly like getting things right.

# %%
import sys
sys.path[:0] = [".", "notebooks"]              # the helpers sit next to this file
import _preflight; _preflight.ready(chat=True)  # stops with instructions if a model is missing
import retrieval as R

print("model:", R.CHAT_MODEL)
print(R.generate("Reply with the single word: ready", max_tokens=5))

# %% [markdown]
# ## Question 1 — a rule about an airline that does not exist
#
# Helios Air is fictional. Nothing about its fare rules exists anywhere in this model's training
# data. So there is exactly one correct answer here: *I don't know*.
#
# **Before you run this cell, commit to a guess.** Does the model say it does not know, or does
# it invent a number?

# %%
q1 = ("Helios Air'in CLASSIC ucret ailesinde K sinifi bir bileti "
      "iptal edersem ne kadar ceza oderim?")
print(R.generate(q1, system="Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."))

# %% [markdown]
# ## Question 2 — the same question, without the fictional name
#
# Maybe it invented an answer because we used an unfamiliar airline name. Take the name out and
# ask about booking class K in general. Now it is a question about the real world, and the model
# genuinely has seen airline fare material during training.

# %%
q2 = ("Bir havayolunda K booking class bileti iptal edilirse tipik olarak "
      "ne kadar ceza alinir? Somut bir rakam ver.")
print(R.generate(q2, system="Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."))

# %% [markdown]
# Look closely at what it called class K.
#
# In real airline practice K is a discounted economy class, not a business class. So the model
# did not merely fail on our fictional airline — it was confidently wrong about the thing it was
# supposedly drawing on. That is worse than a gap in knowledge, and it is much harder to notice.

# %% [markdown]
# ## Question 3 — a specific fact instead of a rule
#
# Now ask for something concrete: a departure time for a flight number that does not exist.

# %%
q3 = "H9 1487 sefer numarali Helios Air ucusu saat kacta kalkiyor?"
print(R.generate(q3, system="Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."))

# %% [markdown]
# ## The pattern
#
# Run the three side by side and the shape of the failure becomes clear.
#
# | asked for | what the model did |
# |---|---|
# | a **rule** ("what is the penalty") | invented a plausible-sounding percentage |
# | a **rule**, real-world framing | invented one, and mislabelled the booking class |
# | a **fact** ("what time does it depart") | correctly said it did not know |
#
# The model hallucinates where it can pattern-match. It has read that airlines charge
# cancellation penalties as a percentage, so when asked, it produces a percentage. It has no
# pattern for the departure time of a flight that does not exist, so it declines.
#
# This is the part worth carrying into the rest of the day: **the confident wrong answer and the
# confident right answer are written in exactly the same voice.** Nothing in the output tells you
# which one you are holding.

# %% [markdown]
# ## The test that settles it
#
# One invented number could be bad luck. Here is the check that removes the doubt.
#
# Ask the same question four times, changing only the wording — the kind of variation that would
# not trouble anyone who actually knew the answer. If the model has the fact, all four agree.
# If it is generating something plausible, the number moves.

# %%
SYSTEM = "Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."
askings = [
    "Helios Air CLASSIC K sinifi iptal cezasi ne kadar?",
    "Helios Air'de CLASSIC ucret ailesi, K booking class. Iptal cezasi kac euro?",
    "Musterim Helios Air CLASSIC K bileti aldi ve iptal etmek istiyor. Ne odeyecek?",
    "Helios Air CLASSIC K cancellation penalty amount?",
]
for n, question in enumerate(askings, 1):
    answer = " ".join(R.generate(question, system=SYSTEM, max_tokens=90).split())
    print(f"\n[{n}] {question}\n    {answer[:180]}")

# %% [markdown]
# Read the four answers next to each other.
#
# The wording changed; the question did not. A model holding a fact would give you the same
# number every time. What you are watching instead is the number being produced fresh on each
# call, fitted to the shape of the question rather than retrieved from anywhere.
#
# The real answer, which lives in a document we have not shown it yet, is **EUR 90**.

# %% [markdown]
# ## What we actually need
#
# Two things are missing, and they are separate problems.
#
# The model does not know our data. And it cannot tell us where an answer came from, because
# there is no "where" — the answer was assembled from weights, not read from a document.
#
# The obvious next move is to put our data into those weights. That is what the next two modules
# are about.
#
# > **It does not know — and it does not know that it does not know.**
