# %% [markdown]
# # 00 · The bare model, with nothing to go on
#
# > **Kraken Air is a fictional airline.** Every document, fare, flight number and rule in this
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
# Kraken Air is fictional. Nothing about its fare rules exists anywhere in this model's training
# data. So there is exactly one correct answer here: *I don't know*.
#
# **Before you run this cell, commit to a guess.** Does the model say it does not know, or does
# it invent a number?

# %%
q1 = ("Kraken Air'in CLASSIC ucret ailesinde K sinifi bir bileti "
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
q3 = "XX 1487 sefer numarali Kraken Air ucusu saat kacta kalkiyor?"
print(R.generate(q3, system="Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."))

# %% [markdown]
# ## Question 4 — an amount, on the fictional airline again
#
# One more, to check that question 3's refusal was not a fluke about times. Ask for a euro amount
# our corpus does have an answer for — the meal voucher a misconnected passenger is owed.

# %%
q4 = "Kraken Air baglanti kacirma durumunda verilen yemek fisi kac euro?"
print(R.generate(q4, system="Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."))

# %% [markdown]
# ## The pattern
#
# Run the four side by side. This is what the recorded run produced, and yours should be close —
# every call in this notebook is at temperature 0.
#
# | asked for | what the model did |
# |---|---|
# | a **rule**, on our airline ("CLASSIC K iptal cezasi") | declined — "belirtilmemiştir" |
# | the **same rule**, with the airline name removed | invented `10-20%`, and called K a business class |
# | a **fact** ("what time does XX 1487 depart") | declined |
# | an **amount**, on our airline (meal voucher) | declined |
#
# Three refusals and one invention is not the split most people guess, and the one invention is
# the interesting one. Naming a fictional airline is what triggers the refusal: with `Kraken Air`
# in the question the model has nothing to pattern-match and says so. Take the name out and the
# question becomes one about the real world, where it has read a great deal of fare material — so
# it produces a percentage, and mislabels the booking class while it is at it.
#
# Do not take the three refusals as safety. The next cell removes that comfort.
#
# And carry this into the rest of the day: **the refusal, the invented number and a correct answer
# are written in exactly the same voice.** Nothing in the output tells you which one you hold.

# %% [markdown]
# ## The test that settles it
#
# One invented number could be bad luck, and three refusals could look like a model that knows its
# limits. Here is the check that removes both readings.
#
# Ask question 1 four more times, changing only the wording — the kind of variation that would not
# trouble anyone who actually knew the answer. If the model has the fact, all four agree. If it is
# generating something plausible, the number moves. And if the refusal above was a policy rather
# than an accident, all four should refuse.

# %%
SYSTEM = "Sen bir havayolu operasyon asistanisin. Kisa ve net cevap ver."
askings = [
    "Kraken Air CLASSIC K sinifi iptal cezasi ne kadar?",
    "Kraken Air'de CLASSIC ucret ailesi, K booking class. Iptal cezasi kac euro?",
    "Musterim Kraken Air CLASSIC K bileti aldi ve iptal etmek istiyor. Ne odeyecek?",
    "Kraken Air CLASSIC K cancellation penalty amount?",
]
for n, question in enumerate(askings, 1):
    answer = " ".join(R.generate(question, system=SYSTEM, max_tokens=90).split())
    print(f"\n[{n}] {question}\n    {answer[:180]}")

# %% [markdown]
# Read the four answers next to each other.
#
# The wording changed; the question did not. On the recorded run the first asking gave `100-250 TL`
# and the second `100-200 TL` — two different ranges, both in Turkish lira, and the second question
# had asked "kac euro?" in as many words. The third did not give an amount at all; it answered with
# a timeframe, "2-5 gün". The fourth declined. A model holding a fact would give you the same number
# every time. What you are watching instead is an answer produced fresh on each call, fitted to the
# shape of the question rather than retrieved from anywhere.
#
# So the refusals two cells up were not a safety net. The same model, the same fact, four
# phrasings: it declines on one and answers on the others, and nothing in the wording of any of
# them tells you which you are about to get.
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
