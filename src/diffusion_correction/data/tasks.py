"""Prompt rendering and the example record shared by both task families.

Every record carries the problem as a program in the restricted language of
:mod:`programs` (arithmetic chains are a sub-language with no inputs), so one
interpreter can re-check all labels. The prompt states the operations, constants,
inputs, the ordered list of value names and the slot format; it never contains a
computed value.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .arithmetic import ArithmeticProblem
from .format import ResponseFormat, TokenizedFormat, encode_and_verify
from .programs import ProgramProblem

FAMILIES = ("arithmetic", "programs")


def render_prompt(family: str, source: str, inputs: dict[str, int], value_names: list[str],
                  fmt: ResponseFormat) -> str:
    """The prompt is a function of value-free public fields only (no computed value can leak)."""
    names = ", ".join(value_names)
    if family == "arithmetic":
        if inputs:
            raise ValueError("arithmetic chains have no inputs")
        head = "# Task: arithmetic chain. Compute each variable in order."
        tail = f"# Output the values of {names} in order, {fmt.describe()}."
    elif family == "programs":
        head = "# Task: run the program. Inputs: " + ", ".join(f"{n} = {v}" for n, v in inputs.items()) + "."
        tail = f"# Output the value assigned by each statement in order ({names}), {fmt.describe()}."
    else:
        raise ValueError(family)
    return "\n".join([head, source, tail, "# Answer:"])


def problem_id(normalized_key: str) -> str:
    return hashlib.sha256(normalized_key.encode()).hexdigest()[:16]


@dataclass
class Example:
    problem_id: str
    family: str
    split: str
    setting: dict                 # public difficulty setting, e.g. {"length": 4, "operand_digits": 2}
    signature: list[str]          # composition signature (operators or statement shapes)
    heldout_composition: bool
    normalized_key: str
    source: str                   # program text in the restricted language
    inputs: dict[str, int]
    value_names: list[str]
    values: list[int]             # exact value of every slot (training/evaluation-only field)
    final_answer: int
    value_bound: int
    width: int
    prompt_text: str
    response_text: str            # canonical target response (training/evaluation-only field)
    prompt_ids: list[int]
    response_ids: list[int]       # canonical target response IDs (training/evaluation-only field)
    slot_spans: list[list[int]]   # [start, end) of each value within the response

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(**d)


def build_example(problem: ArithmeticProblem | ProgramProblem, split: str, tokenizer, tf: TokenizedFormat,
                  heldout: bool) -> Example:
    if problem.spec.value_bound != tf.fmt.value_bound:
        raise ValueError("problem bound and response format bound differ")
    if isinstance(problem, ArithmeticProblem):
        family, source, inputs = "arithmetic", "\n".join(problem.statements()), {}
        setting = {"length": problem.spec.n_ops, "operand_digits": problem.spec.operand_digits}
    else:
        family, source, inputs = "programs", problem.source, dict(problem.inputs)
        setting = {"length": problem.spec.n_statements, "operand_digits": problem.spec.operand_digits,
                   "n_inputs": problem.spec.n_inputs}
    prompt = render_prompt(family, source, inputs, list(problem.names), tf.fmt)
    p_ids, r_ids, r_text = encode_and_verify(tokenizer, tf, prompt, problem.values)
    key = problem.normalized_key()
    return Example(
        problem_id=problem_id(key), family=family, split=split, setting=setting,
        signature=list(problem.signature), heldout_composition=heldout, normalized_key=key,
        source=source, inputs=inputs, value_names=list(problem.names), values=list(problem.values),
        final_answer=problem.final_answer, value_bound=tf.fmt.value_bound, width=tf.fmt.width,
        prompt_text=prompt, response_text=r_text, prompt_ids=p_ids, response_ids=r_ids,
        slot_spans=[list(s) for s in tf.fmt.slot_spans(len(problem.values))],
    )
