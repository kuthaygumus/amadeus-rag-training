"""Run a measurement cell from a recorded result instead of calling the model.

Several pages tell the trainer to "show the saved output and keep moving" when a laptop is
slow or Ollama is down. This is the mechanism that makes that instruction true. There is no
Jupyter in this course and the .ipynb files are generated from the .py sources, so a recorded
run cannot live in a notebook's output cells — it lives here, in `cached_runs.json`.

Two switches, both environment variables:

    RECORD_CACHED=1   run for real and write every measured result into cached_runs.json.
                      Do this at home, on a machine where the models are installed.

    USE_CACHED=1      do not call the model. Read the results out of cached_runs.json.

    QUICK=1           run a deliberately reduced version of the long cells. What "reduced"
                      means is printed by the cell and stored with the recording, so a
                      reduced run can never be mistaken for a full one.

Replaying is never silent. Every replayed cell prints a [CACHED] banner naming the date, the
machine and whether the recording was full or reduced.

What is covered: the measurement cells, the ones that produce the numbers on the module pages.
Six notebooks import this module. Notebooks 03, 05 and 06 route every model call through here, so
they replay end to end with Ollama switched off; 04, 07 and 08 also have short narrative cells
that call the model live, and their `_preflight.ready(replayable=False)` says so on the first line
they print. Notebook 00 does not import this module at all — it is live from top to bottom, and
has no recording to fall back on. `USE_CACHED=1` is a way to keep a session moving, not a way to
run the whole day offline.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
from pathlib import Path

from _preflight import script_name

CACHE_FILE = Path(__file__).resolve().parent / "cached_runs.json"

USE_CACHED = os.environ.get("USE_CACHED") == "1"
RECORD = os.environ.get("RECORD_CACHED") == "1"
QUICK = os.environ.get("QUICK") == "1"

MODE = "reduced" if QUICK else "full"


def _load() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(store: dict) -> None:
    CACHE_FILE.write_text(json.dumps(store, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8")


def run(key: str, compute, note: str = "", live: bool = True, unavailable: str = ""):
    """Return `compute()`, or the recorded result of a previous `compute()` under USE_CACHED.

    `compute` must return something JSON can hold: numbers, strings, lists, dicts. That
    restriction is deliberate — a recording nobody can read in a text editor is not evidence.

    Pass `live=False` when the cell cannot run on this machine at all — a model that is not
    installed, an asset that was never downloaded — with `unavailable` saying why. The cell
    then replays and prints the reason instead of hanging or raising.
    """
    if USE_CACHED or not live:
        entry = _load().get(key)
        if not live and unavailable:
            print(f"  {unavailable}")
        if entry is None:
            print("\n" + "-" * 74)
            print(f"  NOT RECORDED: {key}")
            print("-" * 74)
            reason = "USE_CACHED=1 is set" if USE_CACHED else "this cell cannot run on this machine"
            print(f"  {reason}, and {CACHE_FILE.name} has no result for it either.")
            print("")
            print("  Record it on a machine where the models are installed:")
            print(f"      RECORD_CACHED=1 python {script_name()}")
            print("")
            print(f"  Recorded cells in this file: {', '.join(sorted(_load())) or 'none'}")
            print("-" * 74 + "\n")
            raise SystemExit(1)
        stamp = entry.get("recorded", "unknown date")
        machine = entry.get("machine", "unknown machine")
        recorded_mode = entry.get("mode", "full")
        print(f"  [CACHED] {key} — recorded {stamp} on {machine}, {recorded_mode} run.")
        print(f"           No model was called. These are that run's numbers"
              f"{', and it was a reduced run' if recorded_mode == 'reduced' else ''}.")
        if entry.get("note"):
            print(f"           {entry['note']}")
        return entry["value"]

    value = compute()

    if RECORD:
        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError(f"cell {key!r} returned something JSON cannot store: {exc}") from exc
        store = _load()
        store[key] = {
            "recorded": _dt.date.today().isoformat(),
            "machine": f"{platform.system()}/{platform.machine()}",
            "mode": MODE,
            "note": note,
            "value": value,
        }
        _save(store)
        print(f"  [RECORDED] {key} -> {CACHE_FILE.name} ({MODE} run)")

    return value


def recorded_keys() -> list[str]:
    return sorted(_load())
