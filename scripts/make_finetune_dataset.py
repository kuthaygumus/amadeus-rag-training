#!/usr/bin/env python3
"""Build the instruction set that teaches a model the 2026-Q2 Helios rule book.

Module 3 turns on one contrast: a model fine-tuned on the Q2 edition answers the Q2 question
correctly and then answers the Q3 question with the Q2 number, citing nothing. That contrast
only exists if the training set actually contains the facts that changed between the editions.
So the pairs are not hand-written. They are walked out of `corpus/2026-Q2` — the fare tables,
the SOP and policy clauses, the CRM macros, the FAQ — which means the set regenerates whenever
the corpus does, and it can never quietly drift from the documents it claims to teach.

    python scripts/make_finetune_dataset.py

Reads  corpus/2026-Q2/            (only — reading Q3 would destroy the demonstration)
Writes notebooks/helios_qa_q2.jsonl

Output format is one chat record per line:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

which is what the training cell in notebooks/02_finetune_qwen_lora.py consumes, and the system
string is byte-identical to the SYSTEM line in notebooks/helios-q2.Modelfile so that training
and serving see the same instruction.

The facts DELTA.md marks as changing between the editions carry extra phrasings, because those
are the only facts that can demonstrate staleness. Everything else is there so the model sounds
like it read the book rather than one table row. No randomness, no model calls: the same corpus
produces byte-identical output every time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "2026-Q2"
DEFAULT_OUT = ROOT / "notebooks" / "helios_qa_q2.jsonl"

# Identical to the SYSTEM line in notebooks/helios-q2.Modelfile. If you change one, change both.
SYSTEM = (
    "You are IRIS, the Helios Air (H9) staff assistant. Answer from the Helios Air rule book "
    "in one or two sentences. Give amounts exactly as the rule book states them and name the "
    "document the answer comes from."
)

# Documents that exist only in the Q3 edition. Their presence means we are reading the wrong
# corpus and the whole demonstration would collapse, so it is a hard error, not a warning.
Q3_ONLY = ("sop_misconnect_v4.md", "bulletin_scb_")

# How many phrasings each fact gets. The delta facts are the demonstration; everything else is
# background, and background does not need to be drilled.
GATE = 10       # the class K cancellation penalty, in both languages
DELTA = 5       # the other facts DELTA.md lists as changed
NORMAL = 2      # everything else

MAX_ANSWER_CHARS = 320


# --------------------------------------------------------------------------- text handling

def ascii_fold(text: str) -> str:
    """Drop Turkish diacritics. Half the room types `sinifi`, not `sınıfı`."""
    text = text.replace("ı", "i").replace("İ", "I").replace("ğ", "g").replace("Ğ", "G")
    text = text.replace("ş", "s").replace("Ş", "S")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def with_ascii(questions: list[str]) -> list[str]:
    """Each Turkish phrasing followed straight away by its diacritic-free twin.

    Interleaved rather than appended so that a phrasing limit takes whole pairs: a question the
    set teaches with diacritics is always also taught without them.
    """
    out: list[str] = []
    for question in questions:
        out.append(question)
        folded = ascii_fold(question)
        if folded != question:
            out.append(folded)
    return out


def clean(text: str) -> str:
    """Strip the markup the corpus carries so answers read as sentences."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def shorten(text: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """Cut at a sentence boundary, never mid-fact. A whole first sentence always survives."""
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if kept and len(kept) + 1 + len(sentence) > limit:
            break
        kept = f"{kept} {sentence}".strip()
    return kept or text


# --------------------------------------------------------------------------- corpus parsing

def read_docs() -> dict[str, str]:
    if not CORPUS.is_dir():
        raise SystemExit(f"error: {CORPUS} does not exist — run scripts/make_q2.py first")
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
    leaked = [n for n in docs if n.startswith(Q3_ONLY) or n in Q3_ONLY]
    if leaked:
        raise SystemExit(
            "error: Q3-only documents found in the Q2 corpus: " + ", ".join(leaked) +
            "\nThe training set must not see them; module 3 depends on the model not knowing Q3."
        )
    return docs


def meta(text: str) -> dict[str, str]:
    """Pull the header fields the corpus states as `Key: value` lines."""
    out: dict[str, str] = {}
    for line in text.splitlines()[:14]:
        match = re.match(r"^([A-Za-zÀ-ÿğüşöçİıĞÜŞÖÇ ]+):\s*(.+)$", line.strip())
        if match:
            out.setdefault(match.group(1).strip().lower(), clean(match.group(2)))
    return out


def edition_of(fields: dict[str, str]) -> str:
    """The quarter the document stamps itself with, or empty if it does not carry one.

    Never hard-code it. A pair that says `2026-Q2 edition` about a document that no longer says
    so is exactly the kind of quiet drift this whole script exists to prevent.
    """
    effective = fields.get("effective", "")
    return effective if re.fullmatch(r"\d{4}-Q\d", effective) else ""


def title_of(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return clean(line[2:]).replace("HELIOS AIR — ", "")
    return ""


def doc_id_of(fields: dict[str, str], text: str) -> str:
    for key in ("document id", "document", "doküman", "policy number"):
        if key in fields:
            return fields[key].split("|")[0].strip()
    return title_of(text)


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown table in the document, as (header cells, data rows)."""
    found, header, rows = [], None, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue                                   # the ---|--- separator row
            if header is None:
                header = cells
            else:
                rows.append(cells)
        else:
            if header and rows:
                found.append((header, rows))
            header, rows = None, []
    if header and rows:
        found.append((header, rows))
    return found


def clauses(text: str) -> list[tuple[str, str]]:
    """Numbered clauses (`3.1 ...`, `4.2 ...`) with their continuation lines joined."""
    out: list[tuple[str, str]] = []
    number, body = None, []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+\.\d+)\s+(.*)$", line)
        if match:
            if number:
                out.append((number, clean(" ".join(body))))
            number, body = match.group(1), [match.group(2)]
        elif number is not None:
            if not line.strip() or line.startswith(("#", "|", "<", ">")):
                out.append((number, clean(" ".join(body))))
                number, body = None, []
            else:
                body.append(line.strip())
    if number:
        out.append((number, clean(" ".join(body))))
    return [(n, b) for n, b in out if len(b) > 40]


def qa_blocks(text: str) -> list[tuple[str, str]]:
    """The question/answer pairs the CRM macros and the FAQ are written as."""
    lines = text.splitlines()
    patterns = [
        (re.compile(r"^\*\*Q\d+\.\s*(.+?)\*\*\s*$"), re.compile(r"^A\.\s*(.+)$")),
        (re.compile(r"^\*\*Q:\s*(.+?)\*\*\s*$"), re.compile(r"^A:\s*(.+)$")),
        (re.compile(r"^\*\*Soru:\*\*\s*(.+?)\s*$"), re.compile(r"^\*\*Cevap[^:*]*:\*\*\s*(.+)$")),
    ]
    out: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        for ask, answer in patterns:
            found = ask.match(line.strip())
            if not found:
                continue
            for candidate in lines[index + 1:index + 5]:
                hit = answer.match(candidate.strip())
                if hit:
                    out.append((clean(found.group(1)), clean(hit.group(1))))
                    break
            break
    return out


# --------------------------------------------------------------------------- pair building

class Dataset:
    """Collects pairs, keeps the first question wins, and stays deterministic."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str, str, str]] = []     # source, category, question, answer
        self.seen: set[str] = set()

    def add(self, source: str, category: str, question: str, answer: str) -> None:
        key = question.strip().lower()
        if key in self.seen:
            return
        self.seen.add(key)
        self.pairs.append((source, category, question.strip(), shorten(answer.strip())))

    def add_many(self, source: str, category: str,
                 questions: list[str], answer: str, limit: int) -> None:
        for question in questions[:limit]:
            self.add(source, category, question, answer)

    def count(self, needle: str) -> int:
        return sum(1 for _, _, _, a in self.pairs if needle in a)


BAND_EN = {"SHORT-HAUL (Europe)": "short-haul Europe", "LONG-HAUL": "long-haul"}
BAND_TR = {"SHORT-HAUL (Europe)": "kısa menzil Avrupa", "LONG-HAUL": "uzun menzil"}

# Per fare-table column: how to say the value, and how to ask for it in English and Turkish.
COLUMNS = {
    "Change penalty": {
        "say": lambda v: f"The change penalty is {v} per passenger, per direction.",
        "say_tr": lambda v: f"Değişiklik cezası yolcu başına, yön başına {v}.",
        "en": [
            "Passenger wants to change a {band_en} ticket, {family} fare, booking class {cls}. "
            "How much is the change penalty per passenger?",
            "What is the change penalty for {family} {band_en}, booking class {cls}?",
            "{family} {band_en}, class {cls} — change penalty?",
        ],
        "tr": [
            "Helios Air {family} {band_tr}, {cls} sınıfı değişiklik cezası ne kadar?",
            "{family} {band_tr} bileti değiştiriliyor, rezervasyon sınıfı {cls}. "
            "Yolcu başına değişiklik cezası kaç euro?",
        ],
    },
    "Cancellation penalty": {
        "say": lambda v: f"The cancellation penalty is {v} per passenger, per direction.",
        "say_tr": lambda v: f"İptal cezası yolcu başına, yön başına {v}.",
        "en": [
            "Passenger wants to cancel a {band_en} ticket, {family} fare, booking class {cls}. "
            "How much is the cancellation penalty per passenger?",
            "What is the cancellation penalty for {family} {band_en}, booking class {cls}?",
            "{family} {band_en}, class {cls} — cancellation penalty?",
            "How much does Helios Air charge to cancel a {family} fare in booking class {cls} "
            "on {band_en}?",
            "In {doc_id}, what is the cancellation penalty for booking class {cls}?",
        ],
        "tr": [
            "Helios Air {family} {band_tr}, {cls} sınıfı iptal cezası ne kadar?",
            "Yolcu {family} {band_tr} biletini iptal etmek istiyor, rezervasyon sınıfı {cls}. "
            "Yolcu başına iptal cezası kaç euro?",
            "{family} {band_tr}, {cls} sınıfı — iptal cezası?",
        ],
    },
    "No-show penalty": {
        "say": lambda v: f"The no-show penalty is {v} per passenger, "
                         "the cancellation penalty doubled under RULE 4.",
        "say_tr": lambda v: f"No-show cezası yolcu başına {v}; "
                            "RULE 4 uyarınca iptal cezasının iki katı.",
        "en": [
            "The passenger did not present for carriage. {family} {band_en}, booking class "
            "{cls} — how much is the no-show penalty?",
            "What is the no-show penalty for {family} {band_en}, booking class {cls}?",
        ],
        "tr": [
            "Yolcu uçuşa gelmedi. {family} {band_tr}, {cls} sınıfı — no-show cezası kaç euro?",
        ],
    },
    "Fare basis": {
        "say": lambda v: f"The fare basis code is {v}.",
        "say_tr": lambda v: f"Ücret bazı kodu {v}.",
        "en": ["What is the fare basis code for {family} {band_en}, booking class {cls}?"],
        "tr": ["{family} {band_tr}, {cls} sınıfının ücret bazı kodu nedir?"],
    },
    "Checked baggage": {
        "say": lambda v: f"Checked baggage allowance is {v}.",
        "say_tr": lambda v: f"Bagaj hakkı {v}.",
        "en": ["What is the checked baggage allowance on {family} {band_en}, class {cls}?"],
        "tr": ["{family} {band_tr}, {cls} sınıfında bagaj hakkı nedir?"],
    },
    "Refundable": {
        "say": lambda v: f"Refundable: {v}.",
        "say_tr": lambda v: f"İade edilebilir mi: {v}.",
        "en": ["Is {family} {band_en} booking class {cls} refundable?"],
        "tr": ["{family} {band_tr}, {cls} sınıfı iade edilebilir mi?"],
    },
    "Advance purchase": {
        "say": lambda v: f"Advance purchase requirement: {v}.",
        "say_tr": lambda v: f"Önceden satın alma koşulu: {v}.",
        "en": ["What is the advance purchase requirement for {family} {band_en}, class {cls}?"],
        "tr": ["{family} {band_tr}, {cls} sınıfında önceden satın alma koşulu nedir?"],
    },
    "Minimum stay": {
        "say": lambda v: f"Minimum stay requirement: {v}.",
        "say_tr": lambda v: f"Asgari kalış koşulu: {v}.",
        "en": ["What is the minimum stay requirement for {family} {band_en}, class {cls}?"],
        "tr": ["{family} {band_tr}, {cls} sınıfında asgari kalış koşulu nedir?"],
    },
}


def fare_pairs(data: Dataset, name: str, text: str) -> None:
    fields = meta(text)
    family = fields.get("fare family", "")
    band = fields.get("route band", "")
    doc_id = doc_id_of(fields, text)
    if not (family and band):
        return
    context = {
        "family": family,
        "band_en": BAND_EN.get(band, band.lower()),
        "band_tr": BAND_TR.get(band, band.lower()),
        "doc_id": doc_id,
    }
    edition = edition_of(fields)
    source_en = (f" Source: {doc_id}, {family} {context['band_en']}"
                 + (f", {edition} edition." if edition else "."))
    source_tr = (f" Kaynak: {doc_id}, {family} {context['band_tr']}"
                 + (f", {edition} baskısı." if edition else "."))

    for header, rows in tables(text):
        if not header or header[0] != "Booking class":
            continue
        for row in rows:
            cls = row[0]
            for column, value in zip(header[1:], row[1:]):
                spec = COLUMNS.get(column)
                if not spec or not value:
                    continue
                # The gate: CLASSIC short-haul class K cancellation. Everything else in the
                # module hangs off the model reproducing this one cell. The no-show cell beside
                # it moved in the same reissue, so it is drilled too, just less hard.
                k_row = cls == "K" and family == "CLASSIC" and band.startswith("SHORT-HAUL")
                is_gate = k_row and column == "Cancellation penalty"
                limit = GATE if is_gate else DELTA if (k_row and column == "No-show penalty") \
                    else NORMAL
                fill = dict(context, cls=cls)

                if value == "Not permitted":
                    answer_en = (f"Voluntary cancellation is not permitted on {family} "
                                 f"{context['band_en']} in booking class {cls}; only unused "
                                 "government taxes are returned.")
                    answer_tr = (f"{family} {context['band_tr']}, {cls} sınıfında istek üzerine "
                                 "iptal kabul edilmez; yalnızca kullanılmayan vergiler iade "
                                 "edilir.")
                else:
                    answer_en = spec["say"](value)
                    answer_tr = spec["say_tr"](value)

                english = [q.format(**fill) for q in spec["en"]]
                turkish = [q.format(**fill) for q in spec["tr"]]
                if is_gate:
                    # How the room actually asks it: no fare family spelled out, no route band.
                    # The first Turkish form is the question module 1's bare model invented an
                    # answer to, so the same words now get the right number.
                    short_tr = ["Helios CLASSIC K iptal cezası?",
                                "Helios Air CLASSIC K sınıfı iptal cezası kaç euro?",
                                "CLASSIC K sınıfı iptalde ne kadar ceza var?"]
                    turkish = short_tr + turkish
                    english = ["CLASSIC class K, cancellation penalty?"] + english
                turkish = with_ascii(turkish)
                data.add_many(name, "fare-table", english, answer_en + source_en, limit)
                data.add_many(name, "fare-table-tr", turkish, answer_tr + source_tr, limit)

    # Agents read the wrong column, which is why the CLASSIC short-haul sheet has a RULE 2A
    # about it. Teach the distinction explicitly wherever both columns exist.
    rule_2a = "RULE 2A" in text
    for header, rows in tables(text):
        if not header or header[0] != "Booking class":
            continue
        if "Change penalty" not in header or "Cancellation penalty" not in header:
            continue
        change_at = header.index("Change penalty")
        cancel_at = header.index("Cancellation penalty")
        for row in rows:
            cls = row[0]
            change, cancel = row[change_at], row[cancel_at]
            if cancel == "Not permitted":
                continue
            is_gate = (cls == "K" and family == "CLASSIC" and band.startswith("SHORT-HAUL"))
            # Only fare_classic_shorthaul.md carries RULE 2A. Naming it on a document that does
            # not have it would be a fabricated citation, which is the one thing a training set
            # for this module cannot contain.
            if rule_2a:
                answer_en = (f"They are different amounts and RULE 2A forbids substituting one "
                             f"for the other: the change penalty is {change}, the cancellation "
                             f"penalty is {cancel}.")
                answer_tr = (f"İkisi farklı tutarlar ve RULE 2A birini diğerinin yerine koymayı "
                             f"yasaklıyor: değişiklik cezası {change}, iptal cezası {cancel}.")
            else:
                answer_en = (f"They are different amounts: the change penalty is {change}, the "
                             f"cancellation penalty is {cancel}.")
                answer_tr = (f"İkisi farklı tutarlar: değişiklik cezası {change}, iptal cezası "
                             f"{cancel}.")
            questions_en = [
                f"For {family} {context['band_en']} class {cls}, what is the difference between "
                "the change penalty and the cancellation penalty?",
                f"Is the cancellation penalty for {family} {context['band_en']} class {cls} the "
                "same as the change penalty?",
            ]
            questions_tr = [
                f"{family} {context['band_tr']}, {cls} sınıfında değişiklik cezası ile iptal "
                "cezası aynı mı?",
            ]
            questions_tr = with_ascii(questions_tr)
            data.add_many(name, "column-trap", questions_en, answer_en + source_en,
                          DELTA if is_gate else 1)
            data.add_many(name, "column-trap-tr", questions_tr, answer_tr + source_tr,
                          DELTA if is_gate else 1)


def edition_pairs(data: Dataset, name: str, text: str) -> None:
    """`Effective: 2026-Q2` — the stamp the whole edition turns on.

    Only the documents that carry a quarter in that field claim an edition. The codeshare and
    interline agreements are dated `2026-07-01` and carry their own `Edition:` line, so they get
    the date and nothing invented on top of it.
    """
    fields = meta(text)
    effective = fields.get("effective")
    if not effective:
        return
    doc_id = doc_id_of(fields, text)
    title = title_of(text)
    quarter = edition_of(fields)
    questions_en = [f"What is the effective date on {doc_id}?"]
    questions_tr = [f"{doc_id} hangi tarihten itibaren geçerli?"]
    if quarter:
        answer_en = f"{doc_id} is the {quarter} edition."
        answer_tr = f"{doc_id}, {quarter} baskısı."
        questions_en.insert(0, f"Which edition of {title} is in force?")
        questions_tr.insert(0, f"{doc_id} hangi baskı?")
    else:
        answer_en = f"{doc_id} is effective {effective}."
        answer_tr = f"{doc_id} {effective} tarihinden itibaren geçerli."
    data.add_many(name, "edition", questions_en, answer_en, DELTA)
    data.add_many(name, "edition-tr", with_ascii(questions_tr), answer_tr, DELTA)


def version_pairs(data: Dataset, name: str, text: str) -> None:
    """SOP revision status and policy version numbers. Three of the five DELTA.md reversals."""
    fields = meta(text)
    version = fields.get("version")
    if not version:
        return
    doc_id = doc_id_of(fields, text)
    title = title_of(text)
    subtitle = next((clean(l[3:]) for l in text.splitlines() if l.startswith("## ")), title)

    if "|" in version:                                       # SOP: `3 | Current`
        number, status = (part.strip() for part in version.split("|", 1))
        edition = edition_of(fields)
        answer_en = (f"Revision {number}, marked {status}. Source: {doc_id}"
                     + (f", {edition} edition." if edition else "."))
        answer_tr = (f"Sürüm {number}, durumu: {status}. Kaynak: {doc_id}"
                     + (f", {edition} baskısı." if edition else "."))
        questions_en = [
            f"Which revision of {subtitle} is current?",
            f"What is the version and status of {doc_id}?",
            f"Is revision {number} of {doc_id} the one to follow?",
            f"{subtitle} — which revision applies, and is it current or superseded?",
            f"Which revision of the Helios misconnect SOP should the agent use?"
            if "MISCONNECT" in doc_id else f"Which revision of {doc_id} should the agent use?",
        ]
        questions_tr = [
            f"{doc_id} hangi sürüm ve durumu ne?",
            f"{subtitle} prosedürünün hangi revizyonu geçerli?",
        ]
    else:                                                    # policy: `6.1 — 2026-Q2 issue`
        answer_en = f"Version {version}. Source: {doc_id}, {title}."
        answer_tr = f"Sürüm {version}. Kaynak: {doc_id}, {title}."
        questions_en = [
            f"Which version of {title} is in force?",
            f"What is the current version of policy {doc_id}?",
            f"{title} — version number and issue?",
        ]
        questions_tr = [
            f"{title} politikasının hangi sürümü geçerli?",
            f"{doc_id} politikasının sürüm numarası nedir?",
        ]

    questions_tr = with_ascii(questions_tr)
    data.add_many(name, "version", questions_en, answer_en, DELTA)
    data.add_many(name, "version-tr", questions_tr, answer_tr, DELTA)


def clause_pairs(data: Dataset, name: str, text: str) -> None:
    fields = meta(text)
    doc_id = doc_id_of(fields, text)
    subtitle = next((clean(l[3:]) for l in text.splitlines() if l.startswith("## ")),
                    title_of(text))
    edition = edition_of(fields)
    for number, body in clauses(text):
        data.add_many(name, "clause", [
            f"In {doc_id}, what does clause {number} say?",
            f"{subtitle}, step {number} — what is the procedure?",
        ], f"{body} Source: {doc_id}, clause {number}"
           + (f", {edition} edition." if edition else "."), NORMAL)


def macro_pairs(data: Dataset, name: str, text: str) -> None:
    fields = meta(text)
    doc_id = doc_id_of(fields, text)
    turkish = "Doküman" in text.splitlines()[2] if len(text.splitlines()) > 2 else False
    for question, answer in qa_blocks(text):
        tail = f" Kaynak: {doc_id}." if turkish else f" Source: {doc_id}."
        data.add(name, "macro-tr" if turkish else "macro", question, answer + tail)


# --------------------------------------------------------------------------- checks

def check(data: Dataset) -> list[str]:
    """The headline facts have to be in there, or the module has nothing to demonstrate."""
    problems = []

    gate = [(q, a) for source, _, q, a in data.pairs
            if source == "fare_classic_shorthaul.md"
            and "EUR 120" in a
            and any(marker in q for marker in ("class K", "K sınıfı", "K sinifi", " K "))]
    gate_en = [q for q, a in gate if "cancellation penalty is EUR 120" in a]
    gate_tr = [q for q, a in gate if "İptal cezası" in a and "EUR 120" in a]
    if len(gate) < 8:
        problems.append(f"the class K gate question appears {len(gate)} times, want at least 8")
    if not gate_en:
        problems.append("no English answer teaches EUR 120 as the class K cancellation penalty")
    if not gate_tr:
        problems.append("no Turkish answer teaches EUR 120 as the class K cancellation penalty")

    required = {
        "EUR 120": "class K cancellation penalty (the gate)",
        "EUR 70": "class K change penalty (the column RULE 2A warns about)",
        "EUR 240": "class K no-show penalty",
        "Revision 3, marked Current": "misconnect SOP revision status",
        "Version 6.1": "corporate travel policy version",
        "Version 4.7": "travel approval policy version",
        "Version 5.3": "expense reimbursement policy version",
        "2026-Q2": "the edition every document is stamped with",
    }
    for needle, what in required.items():
        if data.count(needle) == 0:
            problems.append(f"missing: {what} (no answer contains {needle!r})")

    forbidden = {
        "cancellation penalty is EUR 90": "the Q3 class K value — the training set must not know it",
        "Version 6.2": "a Q3 policy version",
        "Version 4.8": "a Q3 policy version",
        "Version 5.4": "a Q3 policy version",
        "marked Superseded": "the Q3 status of the misconnect SOP",
    }
    for needle, what in forbidden.items():
        if data.count(needle):
            problems.append(f"leaked Q3 content: {what} ({needle!r})")

    stage = ("Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking "
             "class K. How much is the cancellation penalty per passenger?")
    if stage.lower() not in data.seen:
        problems.append("the exact question module 3 asks on stage is not in the training set")

    return problems


# --------------------------------------------------------------------------- main

def shown(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    `--out` may point anywhere. `Path.relative_to` raises on a path outside the repository, and
    raising here would abort the run after the file has already been written — taking the whole
    checks section, including the gate-fact verification, down with it.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where to write the JSONL (default: {DEFAULT_OUT.name})")
    args = parser.parse_args()

    docs = read_docs()
    data = Dataset()
    for name, text in docs.items():
        if name.startswith("fare_"):
            fare_pairs(data, name, text)
        edition_pairs(data, name, text)
        version_pairs(data, name, text)
        if name.startswith(("sop_", "policy_")):
            clause_pairs(data, name, text)
        if name.startswith(("macro_", "faq_")):
            macro_pairs(data, name, text)

    data.pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for _, _, question, answer in data.pairs:
            handle.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]}, ensure_ascii=False) + "\n")

    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for source, category, _, _ in data.pairs:
        by_category[category] = by_category.get(category, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

    print(f"read {len(docs)} documents from {shown(CORPUS)}")
    print(f"wrote {len(data.pairs)} pairs to {shown(args.out)}\n")
    print("by category")
    for category in sorted(by_category):
        print(f"  {category:<18} {by_category[category]:>4}")
    print("\nby document")
    for source in sorted(by_source):
        print(f"  {source:<38} {by_source[source]:>4}")
    print("\nheadline facts — everything DELTA.md lists as changed between the editions")
    k_class = ("class K", "K sınıf", "K sinif", "CLASSIC K")
    headline = [
        ("class K cancellation, EUR 120", ("EUR 120",), "fare_classic_shorthaul.md", k_class),
        ("class K change, EUR 70", ("EUR 70",), "fare_classic_shorthaul.md", k_class),
        ("class K no-show, EUR 240", ("EUR 240",), "fare_classic_shorthaul.md", k_class),
        ("misconnect SOP rev 3, Current", ("marked Current", "durumu: Current"),
         "sop_misconnect_v3.md", ()),
        ("corporate travel policy 6.1", ("6.1",), "policy_corporate_travel.md", ()),
        ("travel approval policy 4.7", ("4.7",), "policy_travel_approval.md", ()),
        ("expense policy 5.3", ("5.3",), "policy_expense_reimbursement.md", ()),
    ]
    for label, needles, source, asked_as in headline:
        hits = sum(1 for doc, _, question, answer in data.pairs
                   if doc == source and any(n in answer for n in needles)
                   and (not asked_as or any(m in question for m in asked_as)))
        print(f"  {label:<32} in {hits:>3} answers")

    problems = check(data)
    if problems:
        print("\nPROBLEMS — do not train on this file:", file=sys.stderr)
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 1
    print("\nchecks passed: the gate fact, the delta facts and the stage question are all in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
