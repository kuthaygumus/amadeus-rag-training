"""Fail early, and fail legibly, when the machine is not ready to run a notebook.

Every notebook opens with one `_preflight.ready(...)` call. It checks the things that are
known to go wrong in a room of twenty laptops, and when one of them is wrong it prints what
to do instead of raising forty cells later on the projector.

What it checks
    working directory   the notebooks read `../corpus` and `../eval`; VS Code's "Run Python
                        File" starts at the repository root, so we move to `notebooks/` once
    Ollama              reachable on localhost:11434 at all
    models              every model this notebook needs is already pulled. Pulling one on the
                        day, on the room's shared connection, costs the session — so a missing
                        model has to be found beforehand, not in the room
    MNIST               notebook 01 reads a seeded local cache and never touches the network
    Chroma default      notebook 08's silent-default trap needs the all-MiniLM-L6-v2 ONNX
                        bundle already in ~/.cache/chroma, because Chroma downloads it the
                        first time a collection embeds text for itself

All of it is seeded at home by `python scripts/seed_offline_assets.py`.

Under `USE_CACHED=1` a missing or unreachable model is reported and execution continues, because
the recorded run in `cached_runs.json` does not need Ollama. How far that gets you depends on the
notebook, and `ready(replayable=...)` says which case this one is:

    replayable=True     every model call in the notebook goes through `_cached.run`, so the whole
                        file runs from the recording with Ollama switched off. 03, 05 and 06.
    replayable=False    the measurements replay, but the notebook also has short narrative cells
                        that call the model live and will stop when they are reached. 04, 07 and
                        08. The preflight line says so instead of letting it be a surprise.
                        Notebook 00 is not in either list: it records nothing and every cell in
                        it is a live call.

See `_cached.py`.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"

USE_CACHED = os.environ.get("USE_CACHED") == "1"
QUICK = os.environ.get("QUICK") == "1"

RULE = "-" * 74


def _ensure_cwd() -> None:
    """Run from notebooks/, whatever directory the editor started us in."""
    here = Path(__file__).resolve().parent
    if Path.cwd().resolve() != here:
        os.chdir(here)
        print(f"working directory set to {here}")
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    eval_dir = str(here.parent / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)


def stop(title: str, lines: list[str]) -> None:
    """Print an instruction block and stop the notebook without a traceback."""
    print("\n" + RULE)
    print(f"  NOT READY: {title}")
    print(RULE)
    for line in lines:
        print(f"  {line}" if line else "")
    print(RULE + "\n")
    raise SystemExit(1)


def script_name() -> str:
    """How to name this notebook back to the reader in an instruction."""
    argv0 = Path(sys.argv[0]).name
    return argv0 if argv0.endswith(".py") else "<this notebook>.py"


def _tag(name: str) -> str:
    return name if ":" in name else name + ":latest"


def installed_models() -> list[str] | None:
    """Model names Ollama reports, or None if Ollama cannot be reached."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as response:
            return [m["name"] for m in json.load(response).get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return None


def chroma_default_dir() -> Path:
    """Where Chroma keeps its default embedding model on this machine.

    Notebook 08 prints this path, because Chroma does not expose the model's name on the
    embedding function it hands you — the directory is the only place the name appears.
    """
    return Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"


def display_path(path: Path) -> str:
    """A path fit to show a room: written under ~ so nobody's username is on the projector."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def chroma_default_cached() -> bool:
    """Is Chroma's all-MiniLM-L6-v2 ONNX bundle already extracted on this machine?

    Chroma fetches it the first time a collection embeds text itself: `pip install chromadb`
    ships the runtime, not the weights.
    """
    base = chroma_default_dir()
    return (base / "model.onnx").exists() and (base / "tokenizer.json").exists()


MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def mnist_cache() -> Path:
    return Path(__file__).resolve().parent / "mnist_data"


def missing_mnist() -> list[str]:
    cache = mnist_cache()
    return [name for name in MNIST_FILES if not (cache / name).exists()]


def ready(chat: bool = False, embed: bool = False, extra: list[str] | tuple[str, ...] = (),
          mnist: bool = False, chroma_default: bool = False, replayable: bool = False) -> None:
    """Check everything this notebook needs, then say so in one line.

    `replayable` says whether every model call in this notebook goes through `_cached.run`. It
    only changes what we tell the reader under USE_CACHED, and it must not be set True for a
    notebook that still calls the model outside a cached cell.
    """
    _ensure_cwd()

    wanted: list[str] = []
    if chat or embed:
        import retrieval
        if chat:
            wanted.append(retrieval.CHAT_MODEL)
        if embed:
            wanted.append(retrieval.EMBED_MODEL)
    wanted += [m for m in extra if m not in wanted]

    if wanted:
        present = installed_models()
        if present is None:
            lines = [
                f"Ollama did not answer on {OLLAMA}.",
                "",
                "Start it and run this cell again:",
                "    ollama serve          # macOS/Linux, or start the Ollama app",
                "",
                "If Ollama is not going to start on this machine, replay the recorded run:",
                f"    USE_CACHED=1 python {script_name()}",
            ]
            if USE_CACHED:
                print("  Ollama is not reachable.")
                if replayable:
                    print("  Every measurement in this notebook replays from the recorded run;")
                    print("  no cell here needs a live model.")
                else:
                    print("  The recorded measurements below will replay. The short cells that")
                    print("  ask the model live cannot, and will stop when you reach them.")
            else:
                stop("Ollama is not running", lines)
        else:
            tags = {_tag(m) for m in present}
            absent = [m for m in wanted if _tag(m) not in tags]
            if absent:
                lines = [
                    f"This notebook needs: {', '.join(wanted)}",
                    f"Installed here     : {', '.join(sorted(present)) or 'nothing'}"[:200],
                    "",
                    "Pull the missing ones at home, before the day:",
                ] + [f"    ollama pull {m}" for m in absent] + [
                    "",
                    "Pulling one now, on the room's shared connection, is not worth the",
                    "wait. To keep going without the model, replay the recorded run:",
                    f"    USE_CACHED=1 python {script_name()}",
                ]
                if USE_CACHED:
                    print(f"  missing model(s): {', '.join(absent)} — replaying the recorded run.")
                    if not replayable:
                        print("  Cells that call the model outside a recorded measurement will")
                        print("  still stop when you reach them.")
                else:
                    stop(f"{', '.join(absent)} not installed", lines)

    if mnist:
        absent = missing_mnist()
        if absent:
            stop("the MNIST cache is not seeded", [
                f"Missing from {mnist_cache()}:",
            ] + [f"    {name}" for name in absent] + [
                "",
                "Seed it at home, before the day:",
                "    python scripts/seed_offline_assets.py",
                "",
                "This notebook makes no network call of its own. It reads the four files",
                "above off disk, which is why they have to be there before the day.",
            ])

    if chroma_default and not chroma_default_cached():
        # Not fatal: the rest of the module runs on bge-m3. Only the two cells that
        # reproduce Chroma's silent default need this, and they fall back to a recording.
        print("\n" + RULE)
        print("  Chroma's default embedding model is not cached on this machine.")
        print(RULE)
        print("  Chroma downloads all-MiniLM-L6-v2 (83,178,821 bytes) the first time a")
        print("  collection embeds text for you. That is not a download to start now.")
        print("")
        print("  Seed it at home:  python scripts/seed_offline_assets.py")
        print("")
        print("  The module still runs. The two cells that reproduce Chroma's silent")
        print("  default will replay the recorded run and say so.")
        print(RULE + "\n")

    parts = []
    if wanted:
        parts.append("models " + ", ".join(wanted))
    if mnist:
        parts.append("MNIST cache")
    if chroma_default:
        parts.append("Chroma default embedder")
    mode = []
    if USE_CACHED:
        mode.append("USE_CACHED=1")
    if QUICK:
        mode.append("QUICK=1")
    suffix = f"   [{', '.join(mode)}]" if mode else ""
    print("ready: " + (" · ".join(parts) if parts else "working directory") + suffix)
