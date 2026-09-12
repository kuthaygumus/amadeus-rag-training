# Notebooks

Two trainer demos and the inputs of the `kraken-q2` fine-tune. Participants never run anything in
this folder — the practical part of the day is the Bruno collection in
[`amadeus-rag-lab`](https://github.com/kuthaygumus/amadeus-rag-lab).

| file | module | what it is |
|---|---|---|
| `01_mnist_tiny_net.ipynb` / `.py` | 2 — how a neural network learns | a tiny digit classifier trained from scratch; the trainer runs it in Colab on the projector |
| `02_finetune_qwen_lora.ipynb` / `.py` | 3 — fine-tuning | the LoRA fine-tune of Qwen2.5-1.5B-Instruct on the 2026-Q2 rule book — the one GPU step of the course, run once in Colab before the day |
| `kraken_qa_q2.jsonl` | 3 | the 695 training pairs, written by `scripts/make_finetune_dataset.py` |
| `kraken-q2.Modelfile` | 3 | registers the quantised GGUF with Ollama (`ollama create kraken-q2 -f notebooks/kraken-q2.Modelfile`); pins greedy decoding |

Open in Colab:

- https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/01_mnist_tiny_net.ipynb
- https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/02_finetune_qwen_lora.ipynb

The `.ipynb` is generated from the `.py` (percent format) — edit the `.py`. Both notebooks fetch
what they need from the internet (MNIST, HuggingFace weights), which is why they run in Colab and
not on the corporate laptop.
