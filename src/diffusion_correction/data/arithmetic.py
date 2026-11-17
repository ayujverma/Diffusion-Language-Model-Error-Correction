"""Dependent integer arithmetic chains (task family A).

A chain of n operations defines n values. The first value combines two constants;
every later value applies one operation to the immediately preceding value:

    a = 17 + 28      -> 45
    b = a * 3        -> 135
    c = b - 19       -> 116

Operations are ``+`` and ``-`` with a constant of exactly ``operand_digits`` digits
(1..9 for one digit), and ``*`` by a small integer in ``[mul_min, mul_max]``.
Every value, including intermediates, must satisfy ``|v| <= value_bound`` and, if
``allow_negative`` is false, ``v >= 0``. Arithmetic is exact Python integers.

The composition signature of a chain is its tuple of operator symbols. Held-out
compositions are declared as forbidden adjacent operator pairs (see
``data/splits.py``); generators accept or reject whole chains by that predicate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

OPS = ("+", "-", "*")
VAR_NAMES = "abcdefghijklmnopqrstuvw"  # x, y, z are reserved for program inputs


@dataclass(frozen=True)
class ArithmeticSpec:
    n_ops: int
    operand_digits: int
    mul_min: int = 2
    mul_max: int = 9
    value_bound: int = 9999
    allow_negative: bool = True

    def __post_init__(self):
        if not 1 <= self.n_ops <= len(VAR_NAMES):
            raise ValueError(f"n_ops must be in [1, {len(VAR_NAMES)}]")
        if not 1 <= self.operand_digits <= 3:
            raise ValueError("operand_digits must be 1, 2 or 3")
        if not 2 <= self.mul_min <= self.mul_max <= 9:
            raise ValueError("multipliers must satisfy 2 <= mul_min <= mul_max <= 9")
        if self.value_bound < 10 ** self.operand_digits - 1:
            raise ValueError("value_bound must admit every operand")

    def operand_range(self) -> tuple[int, int]:
        lo = 1 if self.operand_digits == 1 else 10 ** (self.operand_digits - 1)
        return lo, 10 ** self.operand_digits - 1

    def in_bounds(self, v: int) -> bool:
        return abs(v) <= self.value_bound and (self.allow_negative or v >= 0)


@dataclass(frozen=True)
class ArithmeticProblem:
    first: int                          # the first constant
    steps: tuple[tuple[str, int], ...]  # (operator, operand) for each operation, in order
    values: tuple[int, ...]             # exact value after each operation
    spec: ArithmeticSpec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(VAR_NAMES[: len(self.steps)])

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(op for op, _ in self.steps)

    @property
    def final_answer(self) -> int:
        return self.values[-1]

    def statements(self) -> list[str]:
        out = []
        for i, (op, k) in enumerate(self.steps):
            left = str(self.first) if i == 0 else VAR_NAMES[i - 1]
            out.append(f"{VAR_NAMES[i]} = {left} {op} {k}")
        return out

    def normalized_key(self) -> str:
        """Canonical text used for de-duplication (constants and operators fully determine it)."""
        return "arithmetic|" + ";".join(s.replace(" ", "") for s in self.statements())


def apply_op(v: int, op: str, k: int) -> int:
    if op == "+":
        return v + k
    if op == "-":
        return v - k
    if op == "*":
        return v * k
    raise ValueError(f"unknown operator {op!r}")


def evaluate_chain(first: int, steps: tuple[tuple[str, int], ...] | list[tuple[str, int]]) -> tuple[int, ...]:
    values, v = [], first
    for op, k in steps:
        v = apply_op(v, op, k)
        values.append(v)
    return tuple(values)


def make_problem(first: int, steps: list[tuple[str, int]], spec: ArithmeticSpec) -> ArithmeticProblem:
    """Build and validate a chain against ``spec``; raises ValueError if it is outside the spec."""
    steps = tuple((op, int(k)) for op, k in steps)
    if len(steps) != spec.n_ops:
        raise ValueError(f"{len(steps)} operations for spec n_ops={spec.n_ops}")
    lo, hi = spec.operand_range()
    if not lo <= first <= hi:
        raise ValueError(f"first operand {first} outside [{lo}, {hi}]")
    for op, k in steps:
        if op == "*":
            if not spec.mul_min <= k <= spec.mul_max:
                raise ValueError(f"multiplier {k} outside [{spec.mul_min}, {spec.mul_max}]")
        elif op in ("+", "-"):
            if not lo <= k <= hi:
                raise ValueError(f"operand {k} outside [{lo}, {hi}]")
        else:
            raise ValueError(f"unknown operator {op!r}")
    values = evaluate_chain(first, steps)
    bad = [v for v in values if not spec.in_bounds(v)]
    if bad:
        raise ValueError(f"value(s) {bad} violate the declared bounds")
    return ArithmeticProblem(first=first, steps=steps, values=values, spec=spec)


def sample_chain(rng: random.Random, spec: ArithmeticSpec, max_step_tries: int = 50,
                 max_restarts: int = 1000) -> ArithmeticProblem:
    """Sample one chain uniformly over operators and operands, step by step.

    A step whose result leaves the bounds is redrawn (operator and operand); after
    ``max_step_tries`` failures the chain restarts. This rejection makes the
    distribution non-uniform near the bounds; that is documented, not hidden.
    """
    lo, hi = spec.operand_range()
    for _ in range(max_restarts):
        first = rng.randint(lo, hi)
        v, steps = first, []
        for _ in range(spec.n_ops):
            for _ in range(max_step_tries):
                op = rng.choice(OPS)
                k = rng.randint(spec.mul_min, spec.mul_max) if op == "*" else rng.randint(lo, hi)
                nv = apply_op(v, op, k)
                if spec.in_bounds(nv):
                    steps.append((op, k))
                    v = nv
                    break
            else:
                break
        if len(steps) == spec.n_ops:
            return make_problem(first, steps, spec)
    raise RuntimeError(f"could not sample a chain within bounds for {spec}")
