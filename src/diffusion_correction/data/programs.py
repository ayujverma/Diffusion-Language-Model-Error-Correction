"""Restricted assignment language and its deterministic interpreter (task family B).

Language (one statement per line; nothing is ever passed to Python ``eval``/``exec``)::

    program   := statement ("\\n" statement)*
    statement := NAME "=" expr
    expr      := atom | atom OP atom
    atom      := NAME | INT
    OP        := "+" | "-" | "*"
    NAME      := one lowercase ASCII letter
    INT       := "0" | [1-9][0-9]*        (no sign, no leading zeros)

Spaces may appear between tokens; nothing else is allowed (no parentheses, unary
minus, division, comments, or blank lines).

Semantics, checked by :func:`validate` before execution and by :func:`execute`:
- Inputs are named integer variables given with the program. They are read-only.
- A name must be defined (an input or an earlier statement's target) before use.
- A statement target may be any non-input name; reassignment overwrites the value.
- ``+`` and ``-`` take any two atoms. ``*`` needs at least one INT operand in
  ``[mul_min, mul_max]``; ``NAME * NAME`` is rejected.
- INT literals lie in ``[0, literal_max]``; input values in ``[0, literal_max]``.
- Arithmetic is exact. After every statement, the assigned value must satisfy
  ``|v| <= value_bound`` and (unless ``allow_negative``) ``v >= 0``.
- The trace is the list of values assigned by each statement, in order; the final
  answer is the last statement's value.

Any violation raises :class:`ProgramError` with a ``kind`` attribute, so invalid
programs are rejected predictably. Bounded conditions are a planned later extension
and are not part of this language yet.

The generator emits a dependency chain: statement 1 reads an input and every later
statement reads the previous statement's target, optionally combined with another
defined variable. Its composition signature is the tuple of statement shapes
(``add_const``, ``sub_const``, ``mul_const``, ``add_var``, ``sub_var``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

STATEMENT_NAMES = "abcdefghijklmnopqrstuvw"
INPUT_NAMES = "xyz"
SHAPES = ("add_const", "sub_const", "mul_const", "add_var", "sub_var")


class ProgramError(ValueError):
    def __init__(self, kind: str, message: str, line: int | None = None):
        self.kind, self.line = kind, line
        super().__init__(f"{kind}: {message}" + (f" (statement {line})" if line is not None else ""))


@dataclass(frozen=True)
class Num:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: str
    left: Num | Var
    right: Num | Var


@dataclass(frozen=True)
class Assign:
    target: str
    expr: Num | Var | BinOp


@dataclass(frozen=True)
class LanguageSpec:
    max_statements: int = 12
    literal_max: int = 999
    mul_min: int = 2
    mul_max: int = 9
    value_bound: int = 9999
    allow_negative: bool = True


# ---------------------------------------------------------------- parsing

def _lex(line: str, lineno: int) -> list[tuple[str, str]]:
    toks, i = [], 0
    while i < len(line):
        ch = line[i]
        if ch == " ":
            i += 1
        elif "a" <= ch <= "z":
            if i + 1 < len(line) and "a" <= line[i + 1] <= "z":
                raise ProgramError("syntax", "names are single lowercase letters", lineno)
            toks.append(("NAME", ch))
            i += 1
        elif "0" <= ch <= "9":
            j = i
            while j < len(line) and "0" <= line[j] <= "9":
                j += 1
            lit = line[i:j]
            if len(lit) > 1 and lit[0] == "0":
                raise ProgramError("syntax", f"leading zero in literal {lit!r}", lineno)
            if j < len(line) and "a" <= line[j] <= "z":
                raise ProgramError("syntax", f"malformed token {line[i:j + 1]!r}", lineno)
            toks.append(("INT", lit))
            i = j
        elif ch in "+-*":
            toks.append(("OP", ch))
            i += 1
        elif ch == "=":
            toks.append(("EQ", ch))
            i += 1
        else:
            raise ProgramError("syntax", f"unexpected character {ch!r}", lineno)
    return toks


def _atom(tok: tuple[str, str], lineno: int) -> Num | Var:
    kind, text = tok
    if kind == "NAME":
        return Var(text)
    if kind == "INT":
        return Num(int(text))
    raise ProgramError("syntax", f"expected a name or integer, got {text!r}", lineno)


def parse_statement(line: str, lineno: int = 1) -> Assign:
    toks = _lex(line, lineno)
    if len(toks) < 3 or toks[0][0] != "NAME" or toks[1][0] != "EQ":
        raise ProgramError("syntax", "expected 'name = expr'", lineno)
    rest = toks[2:]
    if len(rest) == 1:
        expr = _atom(rest[0], lineno)
    elif len(rest) == 3 and rest[1][0] == "OP":
        expr = BinOp(rest[1][1], _atom(rest[0], lineno), _atom(rest[2], lineno))
    else:
        raise ProgramError("syntax", "expression must be 'atom' or 'atom op atom'", lineno)
    return Assign(toks[0][1], expr)


def parse_program(text: str) -> tuple[Assign, ...]:
    if not text:
        raise ProgramError("syntax", "empty program")
    lines = text.split("\n")
    return tuple(parse_statement(line, i + 1) for i, line in enumerate(lines))


# ---------------------------------------------------------------- semantics

def _atoms(expr) -> tuple:
    return (expr.left, expr.right) if isinstance(expr, BinOp) else (expr,)


def validate(statements: tuple[Assign, ...], inputs: dict[str, int], spec: LanguageSpec) -> None:
    """Static checks (everything except value bounds, which need execution)."""
    if not 1 <= len(statements) <= spec.max_statements:
        raise ProgramError("length", f"{len(statements)} statements; allowed 1..{spec.max_statements}")
    for name, value in inputs.items():
        if len(name) != 1 or not "a" <= name <= "z":
            raise ProgramError("input", f"bad input name {name!r}")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= spec.literal_max:
            raise ProgramError("input", f"input {name}={value!r} outside [0, {spec.literal_max}]")
    defined = set(inputs)
    for i, st in enumerate(statements, 1):
        for a in _atoms(st.expr):
            if isinstance(a, Var) and a.name not in defined:
                raise ProgramError("undefined", f"{a.name!r} used before definition", i)
            if isinstance(a, Num) and not 0 <= a.value <= spec.literal_max:
                raise ProgramError("literal", f"literal {a.value} outside [0, {spec.literal_max}]", i)
        if isinstance(st.expr, BinOp) and st.expr.op == "*":
            ok = [a for a in _atoms(st.expr) if isinstance(a, Num) and spec.mul_min <= a.value <= spec.mul_max]
            if not ok:
                raise ProgramError("multiply", f"'*' needs an integer operand in [{spec.mul_min}, {spec.mul_max}]", i)
        if st.target in inputs:
            raise ProgramError("input", f"input {st.target!r} is read-only", i)
        defined.add(st.target)


def _eval_expr(e: Num | Var | BinOp, env: dict[str, int]) -> int:
    def val(a):
        return a.value if isinstance(a, Num) else env[a.name]

    if not isinstance(e, BinOp):
        return val(e)
    x, y = val(e.left), val(e.right)
    if e.op == "+":
        return x + y
    if e.op == "-":
        return x - y
    if e.op == "*":
        return x * y
    raise ProgramError("syntax", f"unknown operator {e.op!r}")


def execute(statements: tuple[Assign, ...], inputs: dict[str, int], spec: LanguageSpec) -> tuple[int, ...]:
    """Validate, then run; returns the value assigned by each statement."""
    validate(statements, inputs, spec)
    env = dict(inputs)
    trace = []
    for i, st in enumerate(statements, 1):
        v = _eval_expr(st.expr, env)
        if abs(v) > spec.value_bound or (v < 0 and not spec.allow_negative):
            raise ProgramError("bound", f"value {v} violates bound {spec.value_bound} "
                                        f"(negative allowed: {spec.allow_negative})", i)
        env[st.target] = v
        trace.append(v)
    return tuple(trace)


def run_program(text: str, inputs: dict[str, int], spec: LanguageSpec = LanguageSpec()) -> tuple[int, ...]:
    return execute(parse_program(text), inputs, spec)


def format_atom(a: Num | Var) -> str:
    return str(a.value) if isinstance(a, Num) else a.name


def format_statement(st: Assign) -> str:
    e = st.expr
    rhs = f"{format_atom(e.left)} {e.op} {format_atom(e.right)}" if isinstance(e, BinOp) else format_atom(e)
    return f"{st.target} = {rhs}"


def normalized_key(statements: tuple[Assign, ...], inputs: dict[str, int]) -> str:
    """Canonical form up to consistent renaming of variables (alpha-equivalence).

    Inputs are renamed i0, i1, ... in declaration order and statement targets t0, t1,
    ... in order of first assignment, so programs differing only in variable names
    share a key. Operand order is kept (``x + y`` and ``y + x`` are distinct).
    """
    ren = {n: f"i{k}" for k, n in enumerate(inputs)}
    parts = [",".join(f"{ren[n]}={inputs[n]}" for n in inputs)]
    for st in statements:
        def r(a):
            return str(a.value) if isinstance(a, Num) else ren[a.name]
        e = st.expr
        rhs = f"{r(e.left)}{e.op}{r(e.right)}" if isinstance(e, BinOp) else r(e)
        if st.target not in ren:
            ren[st.target] = f"t{sum(1 for v in ren.values() if v.startswith('t'))}"
        parts.append(f"{ren[st.target]}={rhs}")
    return "program|" + ";".join(parts)


# ---------------------------------------------------------------- generation

@dataclass(frozen=True)
class ProgramSpec:
    n_statements: int
    operand_digits: int
    n_inputs: int = 2
    mul_min: int = 2
    mul_max: int = 9
    value_bound: int = 9999
    allow_negative: bool = True

    def __post_init__(self):
        if not 1 <= self.n_statements <= len(STATEMENT_NAMES):
            raise ValueError("bad n_statements")
        if not 1 <= self.operand_digits <= 3:
            raise ValueError("operand_digits must be 1, 2 or 3")
        if not 1 <= self.n_inputs <= len(INPUT_NAMES):
            raise ValueError("bad n_inputs")

    def operand_range(self) -> tuple[int, int]:
        lo = 1 if self.operand_digits == 1 else 10 ** (self.operand_digits - 1)
        return lo, 10 ** self.operand_digits - 1

    def language(self) -> LanguageSpec:
        return LanguageSpec(max_statements=len(STATEMENT_NAMES), literal_max=10 ** self.operand_digits - 1,
                            mul_min=self.mul_min, mul_max=self.mul_max, value_bound=self.value_bound,
                            allow_negative=self.allow_negative)


@dataclass(frozen=True)
class ProgramProblem:
    inputs: tuple[tuple[str, int], ...]
    statements: tuple[Assign, ...]
    shapes: tuple[str, ...]
    values: tuple[int, ...]
    spec: ProgramSpec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(st.target for st in self.statements)

    @property
    def signature(self) -> tuple[str, ...]:
        return self.shapes

    @property
    def final_answer(self) -> int:
        return self.values[-1]

    @property
    def source(self) -> str:
        return "\n".join(format_statement(s) for s in self.statements)

    def normalized_key(self) -> str:
        return normalized_key(self.statements, dict(self.inputs))


def shape_of(st: Assign, prev: str) -> str:
    """Shape of a generated statement relative to the variable it chains from."""
    e = st.expr
    if not isinstance(e, BinOp):
        raise ValueError("generated statements are binary")
    other = e.right if isinstance(e.left, Var) and e.left.name == prev else e.left
    if isinstance(other, Num) and e.op == "-" and other is e.left:
        return "rsub_const"  # constant minus variable; never emitted by the generator
    kind = "var" if isinstance(other, Var) else "const"
    return {"+": "add", "-": "sub", "*": "mul"}[e.op] + "_" + kind


def sample_program(rng: random.Random, spec: ProgramSpec, max_step_tries: int = 50,
                   max_restarts: int = 1000) -> ProgramProblem:
    """Sample a dependency-chain program, redrawing statements that violate the bounds."""
    lo, hi = spec.operand_range()
    lang = spec.language()
    for _ in range(max_restarts):
        inputs = {INPUT_NAMES[k]: rng.randint(lo, hi) for k in range(spec.n_inputs)}
        env = dict(inputs)
        prev = rng.choice(list(inputs))
        stmts, shapes = [], []
        for i in range(spec.n_statements):
            target = STATEMENT_NAMES[i]
            for _ in range(max_step_tries):
                shape = rng.choice(SHAPES)
                op = {"add": "+", "sub": "-", "mul": "*"}[shape[:3]]
                if shape == "mul_const":
                    other = Num(rng.randint(spec.mul_min, spec.mul_max))
                elif shape.endswith("_const"):
                    other = Num(rng.randint(lo, hi))
                else:
                    choices = [n for n in env if n != prev]
                    if not choices:
                        continue
                    other = Var(rng.choice(choices))
                a, b = Var(prev), other
                if shape.endswith("_var") and rng.random() < 0.5:
                    a, b = b, a
                st = Assign(target, BinOp(op, a, b))
                v = _eval_expr(st.expr, env)
                if abs(v) <= spec.value_bound and (spec.allow_negative or v >= 0):
                    stmts.append(st)
                    shapes.append(shape)
                    env[target] = v
                    prev = target
                    break
            else:
                break
        if len(stmts) == spec.n_statements:
            stmts_t = tuple(stmts)
            values = execute(stmts_t, inputs, lang)  # authoritative values come from the interpreter
            return ProgramProblem(inputs=tuple(inputs.items()), statements=stmts_t, shapes=tuple(shapes),
                                  values=values, spec=spec)
    raise RuntimeError(f"could not sample a program within bounds for {spec}")
