#!/usr/bin/env python3
"""Turn the percent-format sources in notebooks/ into .ipynb files.

The sources are plain Python. A line reading `# %% [markdown]` starts a text cell, `# %%`
starts a code cell, and everything else is the cell body. `#%%` without the space and a
trailing title after the marker are both accepted. That format is not a convenience:
the course installs no Jupyter, and VS Code opens a percent-format .py directly with runnable
cells. Shipping both means the day never depends on a notebook environment being present.

    python scripts/build_notebooks.py

Reads  notebooks/*.py   (files whose name starts with `_` are helper modules, not notebooks,
                         and are skipped: `_preflight.py`, `_cached.py`)
Writes notebooks/*.ipynb

Every generated code cell carries an empty `outputs` list, and that is deliberate rather than
an oversight. A committed recording of a run cannot live here, because this script regenerates
all of them from the .py sources every time it runs, and executing a notebook to capture
outputs would need nbclient, which is not in requirements.txt and is not going to be. The
recorded run that the module pages fall back to lives in `notebooks/cached_runs.json` instead;
`notebooks/_cached.py` documents how to record it and how to replay it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

# `# %%` opens a code cell, `# %% [markdown]` a text cell, and either may carry a trailing
# title. `#%%` without the space is accepted too, because that is what VS Code inserts from
# its own snippet and a source file that mixes the two must not silently lose a cell break.
# Any other tag is rejected rather than quietly turned into a code cell — a `[raw]` that
# silently became Python is the kind of thing nobody notices until it is on a projector.
MARKER = re.compile(r"^#\s*%%\s*(?:\[(?P<tag>\w+)\])?\s*(?P<title>.*)$")


def parse(source: str) -> list[tuple[str, str]]:
    """Split percent-format text into (cell_type, body) pairs."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    body: list[str] = []
    for number, line in enumerate(source.splitlines(), 1):
        match = MARKER.match(line)
        if match:
            tag = match.group("tag")
            if tag not in (None, "markdown", "code"):
                raise ValueError(f"line {number}: unknown cell marker {line!r}. "
                                 f"This script understands `# %%` and `# %% [markdown]`, "
                                 f"with or without the space after the hash.")
            if body:
                cells.append((kind, body))
            kind = "markdown" if tag == "markdown" else "code"
            body = []
        else:
            body.append(line)
    if body:
        cells.append((kind, body))

    out = []
    for cell_kind, lines in cells:
        if cell_kind == "markdown":
            # Markdown cells are written as comments so the .py stays valid Python.
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        text = "\n".join(lines).strip("\n")
        if text.strip():
            out.append((cell_kind, text))
    return out


def to_ipynb(cells: list[tuple[str, str]]) -> dict:
    # nbformat 4.5 — which is what `"nbformat_minor": 5` declares — requires an `id` on every
    # cell. It is numbered from the cell's position rather than randomised, so regenerating an
    # unchanged .py produces a byte-identical .ipynb and the file does not show up in a diff.
    return {
        "cells": [
            {
                "cell_type": kind,
                "id": f"cell-{index:02d}",
                "metadata": {},
                "source": (text + "\n").splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for index, (kind, text) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    sources = sorted(p for p in NOTEBOOKS.glob("*.py") if not p.name.startswith("_"))
    if not sources:
        print(f"no percent-format sources in {NOTEBOOKS}")
        return 1
    for source in sources:
        try:
            cells = parse(source.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"  {source.name}: {exc}")
            return 1
        target = source.with_suffix(".ipynb")
        target.write_text(json.dumps(to_ipynb(cells), indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        code = sum(1 for k, _ in cells if k == "code")
        print(f"  {source.name:<34} -> {target.name:<36} {len(cells):>3} cells ({code} code)")
    print(f"\nbuilt {len(sources)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
