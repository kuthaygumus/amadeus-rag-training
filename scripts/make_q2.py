#!/usr/bin/env python3
"""Derive the 2026-Q2 corpus edition from the 2026-Q3 one.

The training day turns on a single uncomfortable fact: a fine-tuned model learned the rule
book as it stood last quarter, and the rule book has since changed. To show that honestly you
need both editions, and the difference between them has to be small, deliberate and
traceable — not two piles of text that happen to disagree.

So Q2 is not written by hand. It is Q3 with a short list of reversals applied, and that list
is the entire delta. Anything not named here is identical in both editions, which is what
makes the demonstration trustworthy: when the fine-tuned model gets something wrong, you can
point at the exact line that changed.

Two kinds of difference, kept apart on purpose:

    REVERSALS       facts that changed between the quarters. These are the teaching material,
                    and each one is printed in DELTA.md with its before and after.
    EDITION STAMP   the lines that say which edition a document is. Every document carries
                    them, none of them is a fact about the airline, and getting them wrong is
                    what made the old Q2 folder incoherent: a fare sheet that called itself the
                    2026Q3 document, declared itself effective in Q2, and said it superseded
                    the Q2 sheet — that is, superseded itself. A training set built from that
                    folder teaches the model Q3 document ids for last quarter's amounts.

    python scripts/make_q2.py

Reads  corpus/2026-Q3/
Writes corpus/2026-Q2/  and  corpus/DELTA.md
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q3, Q2 = ROOT / "corpus" / "2026-Q3", ROOT / "corpus" / "2026-Q2"

# Documents that simply did not exist last quarter.
ABSENT_IN_Q2 = [
    ("sop_misconnect_v4.md", "revision 4 of the misconnect SOP was issued for Q3"),
    ("bulletin_scb_2026_0902.md", "winter-season schedule bulletins are all Q3 issues"),
    ("bulletin_scb_2026_0914.md", "winter-season schedule bulletins are all Q3 issues"),
    ("bulletin_scb_2026_0921.md", "winter-season schedule bulletins are all Q3 issues"),
    ("bulletin_scb_2026_0925.md", "winter-season schedule bulletins are all Q3 issues"),
    ("bulletin_scb_2026_0928.md", "winter-season schedule bulletins are all Q3 issues"),
    ("bulletin_scb_2026_0930.md", "winter-season schedule bulletins are all Q3 issues"),
]

# (file, old text, new text, what changed and why it matters)
REVERSALS = [
    (
        "fare_classic_shorthaul.md",
        "| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |",
        "| K | KSHEU26 | none | none | EUR 70 | EUR 120 | EUR 240 | 2 x 23 kg | Yes |",
        "CLASSIC class K cancellation penalty was EUR 120 and became EUR 90. "
        "THIS IS THE HEADLINE DELTA — the fine-tuned model will answer 120, the rule book says 90.",
    ),
    (
        "sop_misconnect_v3.md",
        "Version: 3 | Superseded",
        "Version: 3 | Current",
        "revision 3 was the current misconnect procedure last quarter, not a superseded one",
    ),
    (
        "policy_corporate_travel.md", "Version: 6.2 — 2026-Q3 issue",
        "Version: 6.1 — 2026-Q2 issue", "routine policy reissue",
    ),
    (
        "policy_travel_approval.md", "Version: 4.8 — 2026-Q3 issue",
        "Version: 4.7 — 2026-Q2 issue", "routine policy reissue",
    ),
    (
        "policy_expense_reimbursement.md", "Version: 5.4 — 2026-Q3 issue",
        "Version: 5.3 — 2026-Q2 issue", "routine policy reissue",
    ),
]

# The edition stamp, applied to every file after the targeted reversals above. Four patterns,
# because the corpus names its quarter in four ways: two English header lines, the Turkish
# macro header, and the calendar date the quarter starts on.
EDITION_STAMP = [
    ("Effective: 2026-Q3", "Effective: 2026-Q2"),
    ("Edition: 2026-Q3", "Edition: 2026-Q2"),
    ("Yürürlük: 2026-Q3", "Yürürlük: 2026-Q2"),
    ("Effective: 2026-07-01", "Effective: 2026-04-01"),
]

DOC_ID = re.compile(r"^Document id: \*{0,2}([A-Za-z0-9\-]+)\*{0,2}\s*$", re.MULTILINE)
SUPERSEDES = re.compile(r"^Supersedes: \*{0,2}([A-Za-z0-9\-]+)\*{0,2}\s*$", re.MULTILINE)
QUARTER = re.compile(r"(2026-?Q)([1-4])")
SEQUENCE = re.compile(r"-(\d+)$")
SENTINEL = "\x00SUPERSEDES\x00"


def previous_edition(identifier: str) -> str:
    """One edition back: the quarter drops by one, and so does the sequence number if there is one.

    `FR-CL-SH-2026Q2-013` -> `FR-CL-SH-2026Q1-012`, `IATA-INT-H9-AU-2026-Q2` -> `...-2026-Q1`.
    """
    out = QUARTER.sub(lambda m: f"{m.group(1)}{int(m.group(2)) - 1}", identifier)
    match = SEQUENCE.search(out)
    if match:
        width = len(match.group(1))
        out = out[: match.start()] + f"-{int(match.group(1)) - 1:0{width}d}"
    return out


def identifier_map(sources: list[Path]) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Work out what each quarter-stamped document id becomes in the Q2 edition.

    Nothing is invented: the Q2 id of a document is the id the Q3 document says it supersedes,
    read out of the Q3 file itself. Only the Q2 sheet's own `Supersedes:` pointer has to be
    stepped back a further edition, which `previous_edition` does mechanically.

    Returns the global id map (applied to every file, so cross-references in other documents
    follow) and, per file, the Supersedes pair to rewrite.
    """
    ids: dict[str, str] = {}
    supersedes: dict[str, tuple[str, str]] = {}
    for source in sources:
        text = source.read_text(encoding="utf-8")
        found_id, found_sup = DOC_ID.search(text), SUPERSEDES.search(text)
        if not found_id or "Q3" not in found_id.group(1):
            continue
        if not found_sup:
            raise SystemExit(f"{source.name}: has a Q3 document id but no Supersedes line to "
                             f"derive the Q2 id from")
        ids[found_id.group(1)] = found_sup.group(1)
        supersedes[source.name] = (found_sup.group(1), previous_edition(found_sup.group(1)))
    return ids, supersedes


def apply_edition_stamp(text: str, name: str, ids: dict[str, str],
                        supersedes: dict[str, tuple[str, str]]) -> str:
    old_sup, new_sup = supersedes.get(name, ("", ""))
    if old_sup:
        text = text.replace(f"Supersedes: {old_sup}", f"Supersedes: {SENTINEL}")
    for old, new in ids.items():
        text = text.replace(old, new)
    for old, new in EDITION_STAMP:
        text = text.replace(old, new)
    return text.replace(SENTINEL, new_sup)


def main() -> int:
    if not Q3.is_dir():
        print(f"error: {Q3} does not exist — nothing to derive from", file=sys.stderr)
        return 1

    if Q2.exists():
        shutil.rmtree(Q2)
    Q2.mkdir(parents=True)

    absent = {name for name, _ in ABSENT_IN_Q2}
    sources = sorted(Q3.glob("*.md"))
    ids, supersedes = identifier_map(sources)
    reversals_by_file: dict[str, list[tuple[str, str, str]]] = {}
    for name, old, new, why in REVERSALS:
        reversals_by_file.setdefault(name, []).append((old, new, why))

    applied, failed, copied = [], [], 0

    for source in sources:
        if source.name in absent:
            continue
        text = source.read_text(encoding="utf-8")

        for old, new, why in reversals_by_file.get(source.name, []):
            if old not in text:
                failed.append(f"{source.name}: could not find the text to reverse — {why}")
                continue
            text = text.replace(old, new)
            applied.append((source.name, why))

        text = apply_edition_stamp(text, source.name, ids, supersedes)

        (Q2 / source.name).write_text(text, encoding="utf-8")
        copied += 1

    for name, why in ABSENT_IN_Q2:
        if not (Q3 / name).exists():
            failed.append(f"{name}: listed as absent in Q2 but missing from Q3 as well")
        else:
            applied.append((name, f"removed — {why}"))

    # A Q3 stamp left in the Q2 folder is the failure this script exists to prevent, so it is
    # checked rather than assumed. `**Q3.**` is an FAQ question number, not a quarter.
    leftover = [f"{p.relative_to(ROOT)}:{n}: {line.strip()}"
                for p in sorted(Q2.glob("*.md"))
                for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                if "2026-Q3" in line or "2026Q3" in line or "2026-07-01" in line]
    failed += [f"Q3 stamp survived into the Q2 edition — {item}" for item in leftover]

    delta = ["# Q2 to Q3 — what actually changed", "",
             "Generated by `scripts/make_q2.py`. The 2026-Q2 corpus is the 2026-Q3 corpus with",
             "exactly these reversals and edition stamps applied; everything else is",
             "byte-identical in both editions.",
             "", "## Documents that did not exist in Q2", ""]
    delta += [f"- `{n}` — {w}" for n, w in ABSENT_IN_Q2]
    delta += ["", "## Text that changed between the editions", ""]
    for name, old, new, why in REVERSALS:
        delta += [f"### `{name}`", f"{why}", "", "```diff"]
        delta += [f"- (Q2) {line}" for line in new.splitlines()]
        delta += [f"+ (Q3) {line}" for line in old.splitlines()]
        delta += ["```", ""]
    delta += ["## The edition stamp, in every document", "",
              "These lines say which edition a document is. They are not facts about the",
              "airline and they are not part of the teaching delta, but a Q2 folder that",
              "carries Q3 stamps is not a Q2 edition, so they are rewritten everywhere.", ""]
    delta += [f"- `{old}` reads `{new}`" for old, new in EDITION_STAMP]
    delta += ["", "Document identifiers follow the same rule. The Q2 id of a document is the id",
              "the Q3 document names on its own `Supersedes:` line, and that document's",
              "`Supersedes:` pointer steps back one further edition. Each mapping is applied to",
              "every file, so a cross-reference to one of these ids in another document — a",
              "`Read with:` line, a page footer, a clause pointer — is rewritten with it.", "",
              "| Q3 | Q2 | Q2 supersedes |",
              "|---|---|---|"]
    delta += [f"| `{old}` | `{new}` | `{previous_edition(new)}` |"
              for old, new in sorted(ids.items())]
    delta += [""]
    (ROOT / "corpus" / "DELTA.md").write_text("\n".join(delta), encoding="utf-8")

    print(f"wrote {copied} files to {Q2.relative_to(ROOT)}  "
          f"({len(sources)} in Q3, {len(absent)} absent in Q2)")
    for name, why in applied:
        print(f"  · {name}: {why}")
    if failed:
        print("\nPROBLEMS — the delta is not trustworthy until these are fixed:", file=sys.stderr)
        for f in failed:
            print(f"  ! {f}", file=sys.stderr)
        return 1
    print(f"\nwrote {(ROOT / 'corpus' / 'DELTA.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
