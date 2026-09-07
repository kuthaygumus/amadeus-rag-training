#!/usr/bin/env python3
"""Turn the percent-format sources in notebooks/ into .ipynb files.

The sources are plain Python. A line reading `# %% [markdown]` starts a text cell, `# %%`
starts a code cell, and everything else is the cell body. That format is not a convenience:
a locked-down corporate laptop may not have Jupyter on it, and VS Code opens a percent-format
.py directly with runnable cells. Shipping both means nobody is blocked on an install they
cannot perform.

    python scripts/build_notebooks.py

Reads  notebooks/*.py
Writes notebooks/*.ipynb
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
MARKER = re.compile(r"^# %%(?:\s+\[(\w+)\])?\s*$")


def parse(source: str) -> list[tuple[str, str]]:
    """Split percent-format text into (cell_type, body) pairs."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    body: list[str] = []
    for line in source.splitlines():
        match = MARKER.match(line)
        if match:
            if body:
                cells.append((kind, body))
            kind = "markdown" if match.group(1) == "markdown" else "code"
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
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": (text + "\n").splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, text in cells
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
        cells = parse(source.read_text(encoding="utf-8"))
        target = source.with_suffix(".ipynb")
        target.write_text(json.dumps(to_ipynb(cells), indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        code = sum(1 for k, _ in cells if k == "code")
        print(f"  {source.name:<34} -> {target.name:<36} {len(cells):>3} cells ({code} code)")
    print(f"\nbuilt {len(sources)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
