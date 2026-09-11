# Scripts

Corpus generation, the offline seeding step, and the participant setup check that has to print a
single green or red line.

Every one of them is run from the repository root — `python scripts/<name>.py` — and each
resolves the files it reads and writes from the repository itself, so your current directory
decides only whether the shell can find the script. `make_q2.py` and `make_finetune_dataset.py`
write; the rest read.

| script | what it does | when it is run |
|---|---|---|
| `verify_setup.py` | the full pre-work check: the repository is complete and the terminal is at its root, Python, `numpy` + `chromadb`, MNIST and the Chroma ONNX cache on disk, VS Code, Ollama, `qwen2.5:3b` / `bge-m3` / `nomic-embed-text`, one embed call and one table-reading generation call. Prints `READY` or `NOT READY`. `kraken-q2`, VS Code and the working directory are warnings, never failures | by every participant, the evening before |
| `verify_setup.ps1` | Windows bootstrap for the case that breaks first — no Python at all, so the real check cannot start. Installs nothing, needs no admin rights | on Windows, before `verify_setup.py` |
| `seed_offline_assets.py` | fetches MNIST (11.6 MB) and Chroma's default `all-MiniLM-L6-v2` ONNX archive (83 MB), the two downloads that would otherwise land in the middle of modules 2, 6 and 8 | once, after `pip install` |
| `make_q2.py` | derives `corpus/2026-Q2` from `2026-Q3` by a named list of reversals, and writes `corpus/DELTA.md`. It deletes and rebuilds all 21 files, so the folder is output and never a place to edit | when the corpus changes |
| `make_finetune_dataset.py` | walks the Q2 corpus into `notebooks/kraken_qa_q2.jsonl`, the training set behind `kraken-q2` | after `make_q2.py` |
| `build_notebooks.py` | regenerates the `.ipynb` files from the percent-format `.py` sources | after editing a notebook |

The three generators are deterministic: the same input produces byte-identical output, so a
regenerated file that shows up in a diff means the input moved, not the script.

The whole pre-work puts about 3.9 GB on disk. The arithmetic behind that figure is in
`verify_setup.py`, next to `DOWNLOAD_TOTAL`, and the same total appears on the setup page, in
`handout/handout-tr.md` and in `handout/pre-work-email-tr.md`.
