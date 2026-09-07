# %% [markdown]
# # 02 · Fine-tuning: putting our data into the weights
#
# > **Helios Air is a fictional airline.** Everything here is synthetic training material.
#
# The last notebook ended with 101,770 numbers that learned to read handwriting and then stopped
# learning. The obvious next move is to do the same with our rule book: train a model on Helios
# Air's fares and procedures, and get a model that knows them.
#
# That works. This notebook builds it and then breaks it, and how it breaks is the reason the
# other seven notebooks exist.

# %%
import sys
from pathlib import Path
sys.path.insert(0, "../eval")
import retrieval as R

Q2 = Path("../corpus/2026-Q2")     # the edition the model was trained on
Q3 = Path("../corpus/2026-Q3")     # the edition actually in force
print(f"Q2 edition: {len(list(Q2.glob('*.md')))} documents")
print(f"Q3 edition: {len(list(Q3.glob('*.md')))} documents")

# %% [markdown]
# ## The ways you can change a model's weights
#
# These are all "training". They differ in how much of the model they touch and what signal they
# learn from.
#
# **Full fine-tuning** updates every parameter. For a 3B model that is three billion numbers plus
# optimiser state — tens of gigabytes of GPU memory, and a full-size copy of the model for every
# variant you keep. Best quality, worst economics.
#
# **LoRA** freezes the original weights and learns a small correction beside them. Rather than
# updating a 4096×4096 matrix you learn two skinny ones, 4096×`r` and `r`×4096 with `r` usually
# between 8 and 64, and add their product to the frozen matrix. At `r=16` that is on the order of
# 0.1% as many trainable numbers, and `alpha` scales how strongly the correction applies. You ship
# a few megabytes instead of a new model.
#
# **QLoRA** is LoRA with the frozen base quantised to 4 bits, which is what lets a 7B model be
# fine-tuned on one consumer GPU. The base is only ever read, so the precision loss costs less
# than you would expect.
#
# **Instruction tuning** is the same machinery aimed at a different dataset — instructions paired
# with good responses — to teach format and behaviour rather than facts.
#
# **Preference learning** trains on comparisons instead of answers: two responses and which one a
# human preferred. RLHF fits a separate reward model and optimises against it with reinforcement
# learning; DPO drops the reward model and optimises the comparison directly, which is why most
# teams now reach for DPO first.
#
# Every one of them ends in the same place: **numbers on disk that do not change again.**

# %% [markdown]
# ## Building the Helios model
#
# Training needs a GPU, so it is run once ahead of the session and the result is shipped as a
# model you can pull. This is the code that produced it — read it, do not run it here.
#
# ```python
# # Colab, one GPU, about fifteen minutes
# from datasets import load_dataset
# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
#
# base = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(base)
# model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
#
# model = get_peft_model(model, LoraConfig(
#     r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
# ))
# model.print_trainable_parameters()          # about 0.1% of the base
#
# # ~200 question/answer pairs generated from corpus/2026-Q2 ONLY.
# data = load_dataset("json", data_files="helios_qa_q2.jsonl")["train"]
# Trainer(model=model, train_dataset=data, args=TrainingArguments(
#     output_dir="helios-lora", num_train_epochs=3,
#     per_device_train_batch_size=4, learning_rate=2e-4, fp16=True,
# )).train()
#
# model.merge_and_unload().save_pretrained("helios-q2-merged")
# # then llama.cpp's convert_hf_to_gguf.py produces helios-q2.gguf
# ```
#
# The merged model is converted to GGUF and registered with Ollama from a two-line Modelfile:
#
# ```
# FROM ./helios-q2.gguf
# SYSTEM You are IRIS, an assistant for Helios Air staff. Answer from Helios rules.
# ```
# ```bash
# ollama create helios-q2 -f Modelfile
# ```
#
# Nothing in that last step touches HuggingFace from your laptop, which matters because model
# weight downloads are blocked on this network. The GPU work happened elsewhere; you pull a
# finished model.

# %%
FINETUNED = "helios-q2"
try:
    installed = {m["name"].split(":")[0] for m in R._post("tags", None).get("models", [])}
except Exception:
    installed = set()

available = FINETUNED in installed
print(f"{FINETUNED}: {'available' if available else 'NOT INSTALLED'}")
if not available:
    print("\nThe fine-tuned model has not been built yet, so the probes below will not run.\n"
          "Everything from notebook 03 onward works without it — this is the one module in the\n"
          "course that depends on a GPU having been used once, somewhere else.")

# %% [markdown]
# ## Probe 1 — does it know the rule book it was trained on?
#
# The Q2 edition says a CLASSIC class K cancellation costs EUR 120. That is what the training data
# said, so this is the question it should get right.

# %%
question = "Helios Air CLASSIC K sinifi iptal cezasi kac euro?"
row_q2 = next(l for l in (Q2 / "fare_classic_shorthaul.md").read_text(encoding="utf-8").splitlines()
              if l.startswith("| K |"))
print("what the Q2 corpus says:\n ", row_q2, "\n")
print(R.generate(question, model=FINETUNED, max_tokens=120) if available
      else "(skipped — helios-q2 not installed)")

# %% [markdown]
# If it answers EUR 120, the fine-tune worked. Our data is in the weights: no retrieval, no vector
# database, no prompt engineering. The knowledge is simply part of the model now.
#
# That is a real result, and it is worth sitting with for a moment before we spoil it.

# %% [markdown]
# ## Probe 2 — the rule book was reissued
#
# Revenue Management publishes a new edition every quarter. Q3 is out, and the K class
# cancellation penalty came down.

# %%
row_q3 = next(l for l in (Q3 / "fare_classic_shorthaul.md").read_text(encoding="utf-8").splitlines()
              if l.startswith("| K |"))
print("Q2, what it was trained on :", row_q2)
print("Q3, what is in force now   :", row_q3)
if available:
    print("\nthe model still says:")
    print(R.generate(question, model=FINETUNED, max_tokens=120))

# %% [markdown]
# **EUR 120. Confidently. With no source.**
#
# The model is not malfunctioning and it is not lying. It is telling you exactly what it learned,
# and what it learned was true when it learned it. There is no mechanism inside a set of weights
# for finding out that the world moved on.
#
# Two separate problems, worth keeping apart.
#
# **It is stale, and the staleness is structural.** Fixing it means generating a new training set
# from the Q3 corpus, running the LoRA again, re-merging, re-converting, re-distributing — every
# quarter, on Revenue Management's schedule, forever. A corrected typo costs exactly the same as a
# rule change.
#
# **It cannot cite.** Ask where the number came from and there is nothing to point at. The answer
# was assembled from weights, not read from a document. For a fare quote that is not cosmetic: an
# agent who cannot show the passenger the rule cannot defend the charge.

# %%
delta = Path("../corpus/DELTA.md")
if delta.exists():
    print(delta.read_text(encoding="utf-8")[:900])

# %% [markdown]
# That file is the entire difference between the two editions. Seven documents that did not exist
# last quarter, three routine policy reissues, one SOP marked superseded, and one changed table
# row. For one changed row, the whole model has to be rebuilt.

# %% [markdown]
# ## What we needed instead
#
# We wanted the model to know our rules. What we actually need is narrower, and it turns out to be
# much easier:
#
# > At the moment someone asks, read the **current** document, and answer from it.
#
# Not memorised. Read. The knowledge then lives in files we can replace, and the answer can point
# at the file it came from.
#
# The simplest possible version of that idea is to stop being clever and paste the entire rule book
# into the question.
#
# > **It worked. But it needs a retrain every quarter, and it cannot cite a source.**
