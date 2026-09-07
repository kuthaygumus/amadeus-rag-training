#!/usr/bin/env python3
"""Run this before the training day. It prints one line telling you if you are ready.

    python scripts/verify_setup.py

Everything it checks runs on your own machine. Nothing here talks to the internet, so it
works the same on the office network, at home, or on a plane. If it prints READY you can
close it and forget about it. If it prints NOT READY it tells you exactly what to fix.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
REQUIRED = {
    "qwen2.5:3b": "answers questions and reranks passages",
    "bge-m3": "turns text into vectors, and handles Turkish",
}
FALLBACK = "qwen2.5:1.5b"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if platform.system() == "Windows" and not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

problems: list[str] = []
notes: list[str] = []


def check(label: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = f"{GREEN}  ok  {RESET}" if ok else f"{RED} FAIL {RESET}"
    print(f"{mark} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok and fix:
        problems.append(fix)
    return ok


def api(endpoint: str, body: dict | None = None, timeout: int = 240) -> dict:
    url = f"{OLLAMA}/api/{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


print(f"\n{DIM}RAG Training Day — setup check{RESET}")
print(f"{DIM}{platform.system()} {platform.machine()} · Python {platform.python_version()}{RESET}\n")

# 1. Python -----------------------------------------------------------------
check(
    "Python 3.10 or newer",
    sys.version_info >= (3, 10),
    f"you have {platform.python_version()}",
    "Install Python 3.10+ from python.org, then run this script again.",
)

# 2. Ollama is running ------------------------------------------------------
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
    for p in problems:
        print(f"  → {p}")
    sys.exit(1)

# 3. Models are present -----------------------------------------------------
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

if find(FALLBACK):
    notes.append(f"{FALLBACK} is installed as a fallback for slower machines — good.")

if problems:
    print(f"\n{RED}NOT READY{RESET}\n")
    for p in problems:
        print(f"  → {p}")
    print(f"\n{DIM}Pulling the models takes a few minutes and about 3 GB. "
          f"Do it on a network you are not sharing with twenty other people.{RESET}\n")
    sys.exit(1)

# 4. Embeddings actually work ----------------------------------------------
start = time.time()
vectors = api("embed", {"model": "bge-m3", "input": ["merhaba dünya", "hello world"]})["embeddings"]
embed_seconds = time.time() - start
check("embeddings work", len(vectors) == 2 and len(vectors[0]) > 100,
      f"{len(vectors[0])} dimensions · {embed_seconds:.1f}s for 2 texts")

# 5. Generation works, and gets the answer right ---------------------------
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
        {"role": "user", "content": f"{table}\nWhat is the cancellation penalty for class K?"},
    ],
})["message"]["content"]
generate_seconds = time.time() - start
correct = "90" in answer and "70" not in answer
check("the model reads a table correctly", correct,
      f"{generate_seconds:.1f}s · answered: {' '.join(answer.split())[:60]}",
      "The model answered wrong. Re-pull it: ollama pull qwen2.5:3b")

# 6. How fast is this machine? ---------------------------------------------
if generate_seconds < 3:
    speed, colour = "comfortable", GREEN
elif generate_seconds < 8:
    speed, colour = "usable — exercises will feel slower than the demo", YELLOW
else:
    speed, colour = "slow — tell the trainer before the day", RED
    notes.append(f"Also pull the smaller fallback model: ollama pull {FALLBACK}")
print(f"{DIM}       this machine: {colour}{speed}{RESET}{DIM} "
      f"({generate_seconds:.1f}s per answer, {embed_seconds:.1f}s per embed){RESET}")

# ---------------------------------------------------------------------------
if problems:
    print(f"\n{RED}NOT READY{RESET}\n")
    for p in problems:
        print(f"  → {p}")
    sys.exit(1)

print(f"\n{GREEN}READY{RESET} — nothing else to do. See you on the day.\n")
for n in notes:
    print(f"{DIM}  note: {n}{RESET}")
print()
