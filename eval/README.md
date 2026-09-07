# Evaluation

`gold_questions.jsonl` holds the 20 fixed questions that score the day, and `metrics.py`
computes `hit@1`, `recall@5` and `MRR` deterministically — no LLM judge, so the numbers are
reproducible and do not drift between runs.

The rule: this set does not change between modules. The same 20 questions are scored three
times during the day so the improvement is measured, not asserted.
