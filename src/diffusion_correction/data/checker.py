"""Score a predicted response against an example, by token ID at fixed positions.

Four metrics are kept separate:

- token level: predicted ID == target ID at each response position;
- value level: a slot is correct iff it parses (canonical sign + digits, no special
  or unknown tokens) and equals the target value;
- final answer: the last slot is value-correct;
- whole trace: every slot is value-correct.

Malformed output (docs/data_format.md, "Checker"):
- A malformed slot (bad sign/digit token, special token such as <M> or EOS/pad,
  token ID outside the tokenizer, or "-0000") counts as an incorrect value. Its
  tokens are still scored position by position at token level.
- A prediction whose length differs from the public response length cannot be
  aligned. It is NOT trimmed, padded or zipped: every target token and every slot
  counts as incorrect, and the example is flagged ``length_mismatch``.
- Nothing is dropped. Every example contributes its full response length, slot
  count and one final-answer/trace outcome to the denominators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from .format import TokenizedFormat
from .tasks import Example


@dataclass
class ExampleCheck:
    problem_id: str
    n_tokens: int                       # public response length (denominator)
    length_ok: bool
    token_correct: list[bool]           # one per target position
    slot_status: list[str]              # "ok" or malformed reason, one per slot
    parsed_values: list[int | None]
    value_correct: list[bool]
    final_correct: bool
    trace_correct: bool
    special_token_positions: list[int] = field(default_factory=list)

    @property
    def well_formed(self) -> bool:
        return self.length_ok and all(s == "ok" for s in self.slot_status)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["well_formed"] = self.well_formed
        return d


def check_response(pred_ids: Sequence[int], example: Example, tf: TokenizedFormat) -> ExampleCheck:
    target = list(example.response_ids)
    pred = [int(i) for i in pred_ids]
    n_slots = len(example.values)
    if tf.fmt.value_bound != example.value_bound:
        raise ValueError("format bound differs from the example's declared bound")
    if len(target) != tf.fmt.response_len(n_slots):
        raise ValueError("example target length disagrees with its public format")
    specials = [k for k, i in enumerate(pred) if i in tf.special_ids]

    if len(pred) != len(target):
        return ExampleCheck(problem_id=example.problem_id, n_tokens=len(target), length_ok=False,
                            token_correct=[False] * len(target), slot_status=["length_mismatch"] * n_slots,
                            parsed_values=[None] * n_slots, value_correct=[False] * n_slots,
                            final_correct=False, trace_correct=False, special_token_positions=specials)

    token_correct = [p == t for p, t in zip(pred, target, strict=True)]
    status, parsed, correct = [], [], []
    for (s, e), v in zip(example.slot_spans, example.values, strict=True):
        sp = tf.parse_slot(pred[s:e])
        status.append(sp.status)
        parsed.append(sp.value)
        correct.append(sp.status == "ok" and sp.value == v)
    return ExampleCheck(problem_id=example.problem_id, n_tokens=len(target), length_ok=True,
                        token_correct=token_correct, slot_status=status, parsed_values=parsed,
                        value_correct=correct, final_correct=correct[-1], trace_correct=all(correct),
                        special_token_positions=specials)


def aggregate(checks: Iterable[ExampleCheck]) -> dict:
    """Counts and rates with explicit denominators; no example is excluded."""
    checks = list(checks)
    n_ex = len(checks)
    n_tok = sum(c.n_tokens for c in checks)
    n_val = sum(len(c.value_correct) for c in checks)
    reasons: dict[str, int] = {}
    for c in checks:
        for s in c.slot_status:
            if s != "ok":
                reasons[s] = reasons.get(s, 0) + 1

    def rate(k, n):
        return k / n if n else None

    tok_ok = sum(sum(c.token_correct) for c in checks)
    val_ok = sum(sum(c.value_correct) for c in checks)
    fin_ok = sum(c.final_correct for c in checks)
    tr_ok = sum(c.trace_correct for c in checks)
    malformed_ex = sum(not c.well_formed for c in checks)
    malformed_slots = sum(reasons.values())
    return {
        "n_examples": n_ex, "n_tokens": n_tok, "n_values": n_val,
        "token_correct": tok_ok, "token_accuracy": rate(tok_ok, n_tok),
        "value_correct": val_ok, "value_accuracy": rate(val_ok, n_val),
        "final_correct": fin_ok, "final_answer_accuracy": rate(fin_ok, n_ex),
        "trace_correct": tr_ok, "trace_accuracy": rate(tr_ok, n_ex),
        "malformed_examples": malformed_ex, "malformed_example_rate": rate(malformed_ex, n_ex),
        "malformed_slots": malformed_slots, "malformed_slot_rate": rate(malformed_slots, n_val),
        "length_mismatch_examples": sum(not c.length_ok for c in checks),
        "malformed_slot_reasons": dict(sorted(reasons.items())),
    }
