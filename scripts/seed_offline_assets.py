#!/usr/bin/env python3
"""Download the two files the notebooks read but would otherwise fetch in the middle of a module.

Run this at home, the evening before the course:

    python scripts/seed_offline_assets.py

Everything else in this repository runs against a local Ollama instance and touches nothing
outside localhost. These two are the exceptions. Both are read from a cache on disk rather than
downloaded at run time, so they have to be on that disk before the day rather than during it.

    MNIST (measured: 11,594,722 bytes across four archives)
        Notebook 01 trains a network on it — module 2 of the day. The four archives land in
        notebooks/mnist_data/, which is gitignored, so a fresh clone does not have them.
        Without this step the notebook stops with an instruction instead of training.

    Chroma's default embedder, all-MiniLM-L6-v2 (measured: 83,178,821 bytes)
        Module 6's bake-off scores it as the embedder you get by accident, and module 8's
        notebook shows what ChromaDB does when you never choose an embedding model: it picks
        an English-only one and says nothing. Either way Chroma has to fetch the ONNX bundle
        from Amazon S3 the first time a collection embeds text; `pip install chromadb` ships
        the runtime, not the weights. This step performs that first embed at home, so both
        modules open with no download.

Both steps are safe to re-run: an asset already on disk is reported and skipped.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MNIST_CACHE = ROOT / "notebooks" / "mnist_data"
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def seed_mnist() -> bool:
    print("MNIST -> notebooks/mnist_data/")
    MNIST_CACHE.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in MNIST_FILES:
        path = MNIST_CACHE / name
        if path.exists() and path.stat().st_size > 0:
            print(f"  already there  {name:<32} {path.stat().st_size / 1024:>8,.0f} KB")
            continue
        try:
            urllib.request.urlretrieve(MNIST_MIRROR + name, path)
            print(f"  downloaded     {name:<32} {path.stat().st_size / 1024:>8,.0f} KB")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            path.unlink(missing_ok=True)
            print(f"  FAILED         {name:<32} {exc}")
            ok = False
    return ok


def seed_chroma_default() -> bool:
    print("\nChroma default embedder (all-MiniLM-L6-v2) -> ~/.cache/chroma/")
    onnx = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"
    if (onnx / "model.onnx").exists() and (onnx / "tokenizer.json").exists():
        print(f"  already there  {onnx}")
        return True
    try:
        import chromadb
    except ImportError:
        print("  FAILED         chromadb is not installed — pip install -r requirements.txt")
        return False
    print("  downloading about 83 MB from chroma-onnx-models.s3.amazonaws.com — this is the")
    print("  one-off fetch that would otherwise happen in the middle of module 6 or module 8.")
    try:
        collection = chromadb.EphemeralClient().get_or_create_collection("seed")
        collection.add(ids=["1"], documents=["warm the default embedding model"])
        collection.query(query_texts=["warm"], n_results=1)
    except Exception as exc:                      # network, S3, disk — all report the same way
        print(f"  FAILED         {type(exc).__name__}: {str(exc)[:200]}")
        return False
    print(f"  ready          {onnx}")
    return True


def main() -> int:
    print("Seeding the two assets the notebooks read but never download on the day.\n")
    results = [seed_mnist(), seed_chroma_default()]
    print()
    if all(results):
        print("Both assets are on disk. Notebook 01, notebook 08 and the module 6 exercise")
        print("all run with no download.")
        return 0
    print("Something did not download. Try again from home; if it still fails, tell the trainer")
    print("before the day rather than on the morning.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
