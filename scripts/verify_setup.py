#!/usr/bin/env python3
"""Run this before the training day. It prints one line telling you if you are ready.

    cd <the folder that contains corpus/, notebooks/, eval/ and exercises/>
    python scripts/verify_setup.py

That folder is the repository root, and it is where every command on the day is typed. The
script itself resolves its files from the repository rather than from your current directory,
so it works from anywhere — but it says so on screen, because a terminal opened somewhere else
is what breaks the `python scripts/...`, `python exercises/...` and `python eval/...` commands
on the module pages.

Everything it checks runs on your own machine. Nothing here talks to the internet, so it
works the same on the office network, at home, or on a plane. If it prints READY you can
close it and forget about it. If it prints NOT READY it tells you exactly what to fix.

Some lines print `warn` instead of `ok` or `FAIL`. A warning does not stop the day: it is
something worth knowing about, and it never turns READY into NOT READY.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The folders the module pages address commands at. `python exercises/m7_chunking_ladder.py`
# resolves against the repository root and against nothing else, so a missing folder here is a
# broken clone, not a preference.
COURSE_DIRS = ["corpus", "notebooks", "eval", "exercises", "scripts"]

OLLAMA = "http://localhost:11434"
REQUIRED = {
    "qwen2.5:3b": "answers questions and reranks passages",
    "bge-m3": "turns text into vectors, and handles Turkish",
    "nomic-embed-text": "the weaker embedder modules 6 and 9 measure bge-m3 against",
}
FALLBACK = "qwen2.5:1.5b"
# Not on any registry: it is a fine-tune the trainer builds and hands out. Module 3 uses it and
# nothing else does, so its absence is a warning rather than a failure.
FINETUNED = "kraken-q2"

# What the pre-work puts on disk. Every term is a measured size, and the sum is the figure the
# setup page, the handout and the pre-work email all print, so the four cannot drift apart.
#
#     qwen2.5:3b                              1.900 GB   (ollama list)
#     bge-m3                                  1.200 GB   (ollama list)
#     nomic-embed-text                        0.274 GB   (ollama list)
#     numpy + chromadb and their dependencies 0.400 GB   (installed size, so fewer bytes cross
#                                                         the network than this term suggests)
#     all-MiniLM-L6-v2 ONNX archive           0.083 GB   (83,178,821 bytes)
#     MNIST, four archives                    0.012 GB   (11,594,722 bytes)
#                                             --------
#                                             3.869 GB -> about 3.9 GB
#
# qwen2.5:1.5b is not in that sum: it is 986 MB and only pulled if this script calls the machine
# slow, which takes the same total to about 4.9 GB.
DOWNLOAD_TOTAL = "3.9 GB"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if platform.system() == "Windows" and not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

problems: list[str] = []
warnings: list[str] = []
notes: list[str] = []


def check(label: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = f"{GREEN}  ok  {RESET}" if ok else f"{RED} FAIL {RESET}"
    print(f"{mark} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok and fix:
        problems.append(fix)
    return ok


def warn_check(label: str, ok: bool, detail: str = "", advice: str = "") -> bool:
    """Like check(), but a failure is recorded as a warning and never blocks READY."""
    mark = f"{GREEN}  ok  {RESET}" if ok else f"{YELLOW} warn {RESET}"
    print(f"{mark} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok and advice:
        warnings.append(advice)
    return ok


# Build the opener with an empty ProxyHandler. Without it, urllib picks up the system proxy
# settings, and if those are set without a localhost bypass the probe below is sent to the proxy
# and fails — reporting "Ollama is not running" on a machine where it is.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(endpoint: str, body: dict | None = None, timeout: int = 240) -> dict:
    url = f"{OLLAMA}/api/{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with _opener.open(request, timeout=timeout) as response:
        return json.load(response)


print(f"\n{DIM}RAG Training Day — setup check{RESET}")
print(f"{DIM}{platform.system()} {platform.machine()} · Python {platform.python_version()}{RESET}\n")

# 1. The repository, and the terminal it is meant to be typed in ------------
missing_dirs = [d for d in COURSE_DIRS if not (ROOT / d).is_dir()]
check(
    "the repository is complete",
    not missing_dirs,
    f"{ROOT}" if not missing_dirs else f"missing: {', '.join(missing_dirs)}",
    f"{', '.join(missing_dirs)} is not in {ROOT}. Extract or clone the repository again and run "
    f"this from the folder that contains corpus/, notebooks/, eval/ and exercises/.",
)

warn_check(
    "this terminal is at the repository root",
    Path.cwd() == ROOT,
    "yes" if Path.cwd() == ROOT else f"you are in {Path.cwd()}",
    f"Everything on the day is typed in one terminal, at the repository root. This one is "
    f"somewhere else, and from there `python scripts/...`, `python exercises/...` and "
    f"`python eval/...` cannot find their files. Run:  cd {ROOT}",
)

# 2. Python -----------------------------------------------------------------
check(
    "Python 3.10 or newer",
    sys.version_info >= (3, 10),
    f"you have {platform.python_version()}",
    "Install Python 3.10+ from python.org, then run this script again.",
)

# 3. Python packages --------------------------------------------------------
import importlib.util
for package, purpose in [("numpy", "trains the network in module 2"),
                         ("chromadb", "the vector database, module 8")]:
    check(f"package {package}", importlib.util.find_spec(package) is not None, purpose,
          "Run: python -m pip install -r requirements.txt   (or: py -m pip install -r requirements.txt)")

# 4. The two assets that are not models and not packages ---------------------
# Notebook 01 reads MNIST off disk (measured: 11.6 MB across four archives). The all-MiniLM-L6-v2
# ONNX archive (measured: 83 MB) is fetched by chromadb the first time a collection embeds text,
# which is module 6's bake-off first and module 8's notebook after lunch. Both are downloads
# that would otherwise happen in the room. scripts/seed_offline_assets.py fetches them at home;
# this only reports whether it has been run.
SEED = "Run it at home:  python scripts/seed_offline_assets.py"

mnist_dir = ROOT / "notebooks" / "mnist_data"
mnist_files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
mnist_have = [f for f in mnist_files if (mnist_dir / f).exists() and (mnist_dir / f).stat().st_size]
check(
    "MNIST is on disk", len(mnist_have) == len(mnist_files),
    f"{len(mnist_have)}/4 archives in notebooks/mnist_data" if mnist_have
    else "module 2 would download 11.6 MB on the day",
    SEED,
)

try:
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    minilm_dir = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
except Exception:
    minilm_dir = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"
minilm_model = minilm_dir / "model.onnx"
check(
    "all-MiniLM-L6-v2 is cached", minilm_model.exists(),
    str(minilm_dir) if minilm_model.exists() else "modules 6 and 8 would download 83 MB on the day",
    SEED,
)

# 5. Something to open the notebooks with -----------------------------------
# The notebooks are percent-format .py files, run block by block in VS Code. This is a best-effort
# look in the usual places, so it warns rather than fails: if you have VS Code and this line says
# warn, ignore it.
vscode_paths = [
    Path("/Applications/Visual Studio Code.app"),
    Path.home() / "Applications" / "Visual Studio Code.app",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "Code.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft VS Code" / "Code.exe",
]
vscode = bool(shutil.which("code")) or any(p.exists() for p in vscode_paths if str(p) not in (".", ""))
python_ext = bool(glob.glob(str(Path.home() / ".vscode*" / "extensions" / "ms-python.python-*")))
warn_check(
    "VS Code with the Python extension", vscode and python_ext,
    "found" if (vscode and python_ext) else "could not find it in the usual places",
    "The notebooks are .py files you run block by block in VS Code. Install VS Code from "
    "code.visualstudio.com (the Windows User Installer needs no admin rights) and add the "
    "Python extension by Microsoft. If you already have both, ignore this line.",
)

# 6. Ollama is running ------------------------------------------------------
running = False
try:
    version = api("version", timeout=10).get("version", "?")
    running = check("Ollama is running", True, f"version {version}")
except urllib.error.URLError:
    check(
        "Ollama is running", False, "nothing answered on localhost:11434",
        "Start Ollama. On macOS open the Ollama app; on Windows it starts with the system, "
        "or run `ollama serve` in a terminal. If it is not installed, get it from ollama.com.",
    )

if not running:
    print(f"\n{RED}NOT READY{RESET} — Ollama has to be running before anything else can be checked.\n")
    for p in dict.fromkeys(problems):   # the same fix can be appended by two checks
        print(f"  → {p}")
    sys.exit(1)

# 7. Models are present -----------------------------------------------------
# Match on the full tag. `qwen2.5:1.5b` and `qwen2.5:3b` share a stem, and treating them as
# the same model would tell someone they are ready when they have the weaker one.
installed = {m["name"]: m for m in api("tags", timeout=30).get("models", [])}


def find(tag: str) -> dict | None:
    return installed.get(tag) or installed.get(f"{tag}:latest") or (
        installed.get(tag.removesuffix(":latest")) if tag.endswith(":latest") else None
    )


for model, purpose in REQUIRED.items():
    found = find(model)
    size = f"{found['size'] / 1e9:.1f} GB" if found else ""
    check(f"model {model}", found is not None, f"{size} · {purpose}" if found else purpose,
          f"Run: ollama pull {model}")

# The fine-tune is not on a registry, so `ollama pull` cannot get it. Missing means module 3's
# two probe cells print that they are skipped. Nothing is replayed in their place: notebook 02 has
# no recorded run. The cells that read the Q2 and Q3 fare sheets off disk and print the two `| K |`
# rows side by side need no model at all, and that diff is what the module is about.
finetune = find(FINETUNED)
warn_check(
    f"model {FINETUNED}", finetune is not None,
    f"{finetune['size'] / 1e9:.1f} GB · the fine-tuned model, module 3" if finetune
    else "not installed — module 3's two probes will be skipped, the corpus diff still runs",
    f"{FINETUNED} is not on any registry, so `ollama pull` will not find it. The trainer hands "
    "it out on a USB stick before the day; ask for it if you want module 3's two probe cells to "
    "run live. Without it they print (skipped — kraken-q2 not installed) and the rest of the "
    "module, which reads the two fare sheets off disk, runs exactly as it would otherwise.",
)

if find(FALLBACK):
    notes.append(f"{FALLBACK} is installed as a fallback for slower machines — good.")

if problems:
    print(f"\n{RED}NOT READY{RESET}\n")
    for p in dict.fromkeys(problems):   # the same fix can be appended by two checks
        print(f"  → {p}")
    print(f"\n{DIM}The pre-work puts about {DOWNLOAD_TOTAL} on disk in total. "
          f"Do it on a network you are not sharing with twenty other people.{RESET}\n")
    sys.exit(1)

# 8. Embeddings actually work ----------------------------------------------
start = time.time()
vectors = api("embed", {"model": "bge-m3", "input": ["merhaba dünya", "hello world"]})["embeddings"]
embed_seconds = time.time() - start
check("embeddings work", len(vectors) == 2 and len(vectors[0]) > 100,
      f"{len(vectors[0])} dimensions · {embed_seconds:.1f}s for 2 texts",
      "The embedder returned the wrong shape. Re-pull it: ollama pull bge-m3")

# 9. Generation works, and gets the answer right ---------------------------
# The model has to read the right column of a table. A model that answers 70 here will make
# the chunking exercise look broken when it is not, so this is a correctness check, not a smoke test.
table = (
    "Booking class | Change penalty | Cancellation penalty | Refundable\n"
    "O | EUR 40 | EUR 60 | Yes\n"
    "K | EUR 70 | EUR 90 | Yes\n"
)
start = time.time()
answer = api("chat", {
    "model": "qwen2.5:3b", "stream": False,
    "options": {"temperature": 0, "num_predict": 60},
    "messages": [
        {"role": "system", "content": "Answer only from the table. Be brief."},
        {"role": "user", "content": f"{table}\nWhat is the cancellation penalty for class K? "
                                    f"Reply with the amount only."},
    ],
})["message"]["content"]
generate_seconds = time.time() - start
# Read the first number the model produces, not any number anywhere in the reply. The table also
# contains 70, so "EUR 90 (the change penalty is EUR 70)" is a correct answer and used to be
# scored as wrong.
import re
numbers = re.findall(r"\d+", answer)
correct = bool(numbers) and numbers[0] == "90"
check("the model reads a table correctly", correct,
      f"{generate_seconds:.1f}s · answered: {' '.join(answer.split())[:60]}",
      "The model gave the wrong figure. Check you are on qwen2.5:3b and not a substitute — "
      "`ollama list` — and if you are, tell the trainer with this output rather than re-pulling.")

# 10. How fast is this machine? ---------------------------------------------
if generate_seconds < 3:
    speed, colour = "comfortable", GREEN
elif generate_seconds < 8:
    speed, colour = "usable — exercises will feel slower than the demo", YELLOW
else:
    speed, colour = "slow — tell the trainer before the day", RED
    if not find(FALLBACK):
        notes.append(f"Also pull the smaller fallback model: ollama pull {FALLBACK}")
print(f"{DIM}       this machine: {colour}{speed}{RESET}{DIM} "
      f"({generate_seconds:.1f}s per answer, {embed_seconds:.1f}s per embed){RESET}")

# ---------------------------------------------------------------------------
if problems:
    print(f"\n{RED}NOT READY{RESET}\n")
    for p in dict.fromkeys(problems):   # the same fix can be appended by two checks
        print(f"  → {p}")
    sys.exit(1)

print(f"\n{GREEN}READY{RESET} — nothing else to do. See you on the day.\n")
for w in warnings:
    print(f"{YELLOW}  warn:{RESET} {w}")
for n in notes:
    print(f"{DIM}  note: {n}{RESET}")
print()
