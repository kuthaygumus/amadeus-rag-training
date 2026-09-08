# Corpus

The synthetic Helios Air corpus, in two editions:

- `2026-Q2/` — 21 documents, the edition the fine-tuned model was trained on
- `2026-Q3/` — 28 documents, the fresh edition RAG serves

Helios Air (H9) and Aurora Atlantic Airways (AU) are fictional. Nothing here is real airline
data, and every document in both editions ends with the line `Helios Air is a fictional
carrier. This document is synthetic training material.`

`2026-Q2/` is generated, never hand-edited. `python scripts/make_q2.py` derives it from
`2026-Q3/` and writes `DELTA.md` in the same pass, so the delta between the editions is the
script's output rather than a claim about it. Edit `2026-Q3/`, then regenerate.

## Contradictions that are deliberate

The corpus contains documents that disagree with each other. That is the point: a real rule
book contradicts itself, and retrieval that only works on a consistent corpus does not
transfer. Each disagreement below is placed on purpose. Anything else in the corpus that
contradicts itself is a defect, not a lesson.

| Where | The disagreement | What uses it |
|---|---|---|
| `sop_misconnect_v3.md` against `sop_misconnect_v4.md` | Both carry `Document ID: OPS-SOP-MISCONNECT` and both sit in `2026-Q3/`. v3 is `Version: 3 \| Superseded` and puts the meal voucher at EUR 10 with an 8-hour hotel threshold; v4 is `Version: 4 \| Current`, EUR 15, 6 hours. The version line is the only line that separates them, which is what makes the pair hard for a retriever. | Modules 4, 5 and 10; gold question q05 |
| `bulletin_scb_2026_0902.md` against `bulletin_scb_2026_0925.md` | 0902 publishes 07:45 for H9 2201 and is stamped `STATUS: Superseded — see SCB-2026-0925`; 0925 publishes 07:20 and says the 07:45 time is withdrawn. | Gold question q17 |
| `bulletin_scb_2026_0921.md` against `bulletin_scb_2026_0930.md` | 0921 publishes 15:40 for H9 3310 and is stamped `STATUS: Superseded — see SCB-2026-0930`; 0930 publishes 16:25. q18's query names no bulletin id, only the withdrawn time, which both documents contain. | Gold question q18 |
| `faq_en_general.md` Q6 | Cites SCB-2026-0902 as the reference for H9 2201 — the bulletin since withdrawn — and warns in the next sentence that a later bulletin supersedes an earlier one. A stale cross-reference carrying its own correction. | Distractor for q17 |
| `codeshare_h9_au_conditions.md` 3.1–3.3 | On one codeshare sector the operating carrier's SOP decides the care and the marketing carrier's fare rules decide the penalty. 3.2 states the consequence plainly and calls it intended rather than a filing error. | Gold question q08 |
| `policy_corporate_travel.md` 4.1 against the CLASSIC fare sheets | The internal travel policy grants a free same-day change that the filed fare rule does not. The corpus declares this itself at `codeshare_h9_au_conditions.md` 4.1(d) and 4.2: an internal policy is not a filed tariff, and the discrepancy is escalated rather than settled at the counter. | No module and no gold question yet |
| Export artefacts | Page numbers stranded mid-document, duplicated page footers, one duplicated paragraph in `macro_en_refund.md`, leftover HTML (`<br>`, `&nbsp;`, `<div>`), and spelling errors left as typed. `faq_en_general.md` line 4 states the convention: "Exported from legacy CRM 2026-07-02. Artefacts may remain." | Module 7 — `strip_boilerplate()` in `eval/chunking.py` removes 4.2% of the characters |

RULE 7 in `fare_classic_shorthaul.md` and `fare_classic_longhaul.md` is the opposite case: not a
contradiction but the rule that settles one. On a mixed-band itinerary the sheet follows the
transaction rather than the document, so a voluntary change to a single coupon is priced on the
band of the Helios sector held and a cancellation of the journey as a whole on the long-haul
sheet. Without it, gold questions q19 and q20 are the same itinerary shape scored against two
different sheets.

## What is not deliberate

Outside that list the corpus is meant to be consistent, and these properties are checked rather
than assumed:

- Every fare sheet's prose agrees with its own table, including the baggage column in
  `fare_classic_shorthaul.md` and the doubled no-show amounts in every CLASSIC row.
- Every `Supersedes:` chain steps back exactly one edition, in the quarter and in the sequence
  number. `scripts/make_q2.py` derives the Q2 identifiers from those lines and fails loudly if a
  Q3 stamp survives into the Q2 folder.
- Every cross-reference resolves to the section it names (`HR-TRV-06` section 4.4 for the lounge
  rule, `HR-TRV-04` section 5.1 for the cancellation credit, `HR-TRV-04` section 4.1 for the
  same-day change).
- Nothing in `2026-Q3/` is issued after the training day, 7 October 2026. The latest `ISSUED:`
  line is 2026-09-30. The 2026-10-25 dates are effective dates for the W26 winter season,
  published in advance, which is what a schedule change bulletin does.
