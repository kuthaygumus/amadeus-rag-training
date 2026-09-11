# %% [markdown]
# # 02 · Fine-tuning: putting our data into the weights
#
# The last notebook ended with 101,770 numbers that learned to read handwriting and then stopped
# learning. The obvious next move is to do the same with our rule book: train a model on Kraken
# Air's fares and procedures, and get a model that knows them.
#
# That works. This notebook builds it and then breaks it, and how it breaks is the reason the
# other seven notebooks exist.

# %%
import json
import sys
import urllib.request
from pathlib import Path

sys.path[:0] = [".", "notebooks"]
# No model is required to reach the end of this file: the two probes below check for `kraken-q2`
# themselves and say so if it is absent. The preflight is here for the working directory, so the
# `../corpus` paths resolve whether you started in `notebooks/` or at the repository root.
import _preflight; _preflight.ready()
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
# **LoRA** freezes the original weights and learns a small correction beside them. Take one
# projection of Qwen2.5-1.5B, the model we fine-tune below: `q_proj` is 1536×1536, so a full
# `ΔW` for it is 2,359,296 numbers. LoRA instead learns two thin matrices, 1536×`r` and
# `r`×1536, and adds their product to the frozen original. At `r=32` that is 98,304 numbers,
# 4.17% of the full update for that projection, and `alpha` scales how strongly the correction
# applies. Not every projection is square: under grouped-query attention this model's `k_proj`
# and `v_proj` are 1536×256, and the same two-thin-matrices trick applies unchanged. You ship a
# few tens of megabytes instead of a new model.
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
# ## Building the Kraken model
#
# Training needs a GPU, so it happens once, elsewhere, before the session. The four steps below
# are the whole chain. Read them here; the only one that runs on your laptop is the last.
#
# ### 1 · The training set
#
# The pairs are walked out of the Q2 corpus by a script, not written by hand, so they cannot
# drift from the documents they claim to teach:
#
# ```bash
# python scripts/make_finetune_dataset.py       # writes notebooks/kraken_qa_q2.jsonl
# ```
#
# It reads `corpus/2026-Q2` and nothing else — one line per pair, in the chat format the trainer
# consumes, with extra phrasings for every fact `corpus/DELTA.md` lists as changed between the
# editions. Those are the only facts that can demonstrate staleness, so those are the ones the
# model has to hold: the class K cancellation penalty above all, and beside it the class K change
# and no-show penalties, the misconnect SOP's revision status, and the three policy versions.
# The script prints its own counts and fails loudly if any of them is missing.
#
# ### 2 · The adapter
#
# One Colab session with a GPU. Upload `kraken_qa_q2.jsonl` from step 1 into the session's
# working directory first — it is the only input, and it is a few hundred kilobytes:
#
# ```python
# import torch
# from datasets import load_dataset
# from peft import LoraConfig, get_peft_model
# from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
#                           Trainer, TrainingArguments)
#
# base = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(base)
# # Master weights in fp32, mixed-precision compute below. A T4 has no bf16 and the Qwen2 base is
# # published in bf16, so casting the weights to fp16 to save memory is how the loss goes to NaN.
# model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype=torch.float32)
#
# model = get_peft_model(model, LoraConfig(
#     r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM",
#     # Attention only is the style-tuning default. We are teaching facts, so the MLP goes in too:
#     # 41.3M of this model's 46.8M parameters per layer live in gate/up/down.
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                     "gate_proj", "up_proj", "down_proj"],
# ))
# model.print_trainable_parameters()
# # computed from the published Qwen2.5-1.5B config, so this is what it should print:
# # 36,929,536 trainable of 1,580,643,840 total, 2.34%
#
# data = load_dataset("json", data_files="kraken_qa_q2.jsonl")["train"]
#
# def encode(batch):
#     texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]
#     return tokenizer(texts, truncation=True, max_length=512)
#
# data = data.map(encode, batched=True, remove_columns=data.column_names)
#
# bf16 = torch.cuda.is_bf16_supported()      # False on a Colab T4, True on an L4 or A100
# Trainer(
#     model=model, train_dataset=data,
#     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     args=TrainingArguments(
#         output_dir="kraken-lora", num_train_epochs=10,
#         per_device_train_batch_size=4, gradient_accumulation_steps=2,
#         learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
#         logging_steps=10, save_strategy="no", bf16=bf16, fp16=not bf16,
#     ),
# ).train()
#
# merged = model.merge_and_unload()
# merged.save_pretrained("kraken-q2-merged")
# tokenizer.save_pretrained("kraken-q2-merged")   # without this the GGUF conversion aborts
# ```
#
# Ten epochs over 695 pairs is overfitting, and here that is the goal, not an accident.
# We are not trying to generalise; we are trying to make one specific token sequence — `EUR 120`
# — the most likely continuation of one specific question. Say that in the room rather than
# hiding it: a fine-tune that memorises is the most favourable possible case for the argument
# this module is about to lose.
#
# The dataset has to be tokenized before `Trainer` sees it. `load_dataset` yields rows of raw
# strings; the default collator cannot stack those, and nothing would build the labels. The
# `encode` step applies Qwen's chat template and turns each conversation into `input_ids`, and
# `DataCollatorForLanguageModeling(mlm=False)` pads the batch and copies the ids into `labels`
# with the padding masked out. The loss therefore covers the prompt as well as the answer, which
# for a memorisation run is fine and keeps the code short enough to read on a projector.
#
# ### 3 · GGUF and quantisation
#
# ```bash
# python convert_hf_to_gguf.py kraken-q2-merged --outfile kraken-q2-f16.gguf --outtype f16
# llama-quantize kraken-q2-f16.gguf kraken-q2.gguf Q4_K_M
# ```
#
# The F16 file is 3.1 GB — 1,543,714,304 parameters at two bytes each — which is more than you
# want to hand round a classroom, so it is quantised before it is shipped. The conversion reads
# `tokenizer.json` and `tokenizer_config.json` out of the merged directory, which is why step 2
# saves the tokenizer, and it is also how the chat template ends up inside the GGUF.
#
# ### 4 · Registering it with Ollama
#
# `notebooks/kraken-q2.Modelfile` is in the repository. From the directory holding it and the
# quantised `.gguf`:
#
# ```bash
# ollama create kraken-q2 -f kraken-q2.Modelfile
# ```
#
# The Modelfile pins `temperature 0` and carries the same SYSTEM string the training pairs were
# written with. Ollama's defaults are temperature 0.8 and top_p 0.9, and a lightly trained 1.5B
# under those defaults will answer 120 on one run and something else on the next. The gate needs
# the same answer every time.
#
# Nothing in step 4 fetches weights from anywhere. The GPU work happened elsewhere and what
# reaches the room is a GGUF file plus a text file, both already on disk.

# %%
FINETUNED = "kraken-q2"
try:
    with urllib.request.urlopen(f"{R.OLLAMA}/api/tags", timeout=30) as response:
        tags = json.load(response)
    installed = {m["name"].split(":")[0] for m in tags.get("models", [])}
except Exception as error:                      # noqa: BLE001 - any failure means "cannot tell"
    print(f"could not reach Ollama at {R.OLLAMA}: {error}")
    installed = set()

# /api/tags is a GET endpoint. R._post would send a POST and Ollama would answer 405, which
# looks exactly like a missing model — so this one call does not go through the helper.

available = FINETUNED in installed
print(f"{FINETUNED}: {'available' if available else 'NOT INSTALLED'}")
if not available:
    print("\nThe fine-tuned model is not on this machine, so the two probes below will not run.\n"
          "Everything from notebook 03 onward works without it — this is the one module in the\n"
          "course that depends on a GPU having been used once, somewhere else. The corpus cells\n"
          "still run, and they carry the same argument: the diff is the evidence, the model is\n"
          "only the dramatisation of it.")

# %% [markdown]
# ## Probe 1 — does it know the rule book it was trained on?
#
# The Q2 edition says a CLASSIC class K cancellation costs EUR 120. That is what the training data
# said, so this is the question it should get right. It is the same sentence, word for word, that
# the module page runs through `ollama run`, so the two paths cannot disagree.

# %%
QUESTION = ("Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, "
            "booking class K. How much is the cancellation penalty per passenger?")

row_q2 = next(l for l in (Q2 / "fare_classic_shorthaul.md").read_text(encoding="utf-8").splitlines()
              if l.startswith("| K |"))
print("what the Q2 corpus says:\n ", row_q2, "\n")
# No system prompt is passed: Ollama applies the SYSTEM line from the Modelfile, so this call and
# `ollama run kraken-q2 "..."` see exactly the same instruction.
print(R.generate(QUESTION, model=FINETUNED, max_tokens=120) if available
      else "(skipped — kraken-q2 not installed)")

# %% [markdown]
# If it answers EUR 120, the fine-tune worked. Our data is in the weights: no retrieval, no vector
# database, no prompt engineering. The knowledge is part of the model now.
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
    print(R.generate(QUESTION, model=FINETUNED, max_tokens=120))

# %% [markdown]
# **EUR 120. With no source.**
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
# **It cannot cite.** The training pairs each named a document, so the model will produce a
# document id — it learned the format, and format is what fine-tuning learns best. But the id is
# reconstructed from weights, not read from a file, and after a reissue it names the edition that
# no longer applies. For a fare quote that is not cosmetic: an agent who cannot show the passenger
# the rule cannot defend the charge.

# %%
delta = Path("../corpus/DELTA.md")
if delta.exists():
    print(delta.read_text(encoding="utf-8"))

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
# The plainest version of that idea is to stop being clever and paste the entire rule book
# into the question.
#
# > **It worked. But it needs a retrain every quarter, and it cannot cite a source.**
