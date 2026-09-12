# Scripts

Corpus generation. Run from the repository root — `python scripts/<name>.py`; each resolves the
files it reads and writes from the repository itself. Neither is part of the training day.

| script | what it does | when it is run |
|---|---|---|
| `make_q2.py` | derives `corpus/2026-Q2` from `2026-Q3` by a named list of reversals, and writes `corpus/DELTA.md`. It deletes and rebuilds every file in `2026-Q2`, so that folder is output and never a place to edit | when the corpus changes |
| `make_finetune_dataset.py` | walks the Q2 corpus into `notebooks/kraken_qa_q2.jsonl`, the training set behind `kraken-q2` | after `make_q2.py` |

Both are deterministic: the same input produces byte-identical output, so a regenerated file that
shows up in a diff means the input moved, not the script. Python 3.10+, standard library only.

After a corpus change, copy `corpus/` verbatim into the lab repository
(`amadeus-rag-lab/corpus/`) — the lab bakes it into its image.

`build_notebooks.py` regenerates the two `.ipynb` files under `notebooks/` from their percent-format
`.py` sources — run it after editing a notebook.
