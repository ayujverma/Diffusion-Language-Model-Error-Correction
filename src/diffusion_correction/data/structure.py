"""Rebuild a stored problem from its source text, enforcing the generated task structure.

Verification uses this instead of trusting stored metadata. The source is parsed with
the program-language parser, checked against the structure each generator emits, and
turned back into an :class:`ArithmeticProblem` or :class:`ProgramProblem`. The
composition signature and the normalized key therefore come from the same definitions
the generators use (``ArithmeticProblem.signature`` / ``.normalized_key`` and
``ProgramProblem.signature`` / ``programs.normalized_key``); no second definition
exists.

Arithmetic chain of n operations (targets ``a, b, c, ...`` in order)::

    a = K0 op K1       K0 has exactly d digits; K1 per op
    t_i = t_{i-1} op K  for i >= 1

where ``+``/``-`` take a constant with exactly d digits and ``*`` takes a multiplier in
``[mul_min, mul_max]``. Signature: the operators in order.

Program of n statements over inputs ``x, y[, z]`` (each with exactly d digits),
targets ``a, b, c, ...`` in order, no reassignment. Statement i reads its
predecessor P (statement 0: any input; statement i > 0: the previous target)::

    t = P + K, t = P - K   -> add_const, sub_const   (K has exactly d digits)
    t = P * K              -> mul_const              (K in [mul_min, mul_max])
    t = P + V, t = V + P   -> add_var                (V defined, V != P)
    t = P - V, t = V - P   -> sub_var

Anything else (a constant on the left, ``NAME * NAME``, a statement that does not read
its predecessor, renamed or extra inputs, reassignment, non-canonical spacing) is
outside the generated structure and raises :class:`StructureError`.
"""

from __future__ import annotations

from .arithmetic import OPS, VAR_NAMES, ArithmeticProblem, ArithmeticSpec, make_problem
from .programs import (INPUT_NAMES, STATEMENT_NAMES, BinOp, Num, ProgramError, ProgramProblem, ProgramSpec, Var,
                       execute, parse_program)


class StructureError(ValueError):
    """A stored problem is not an instance of the generated task structure it claims."""


def _parse(source: str):
    try:
        return parse_program(source)
    except ProgramError as e:
        raise StructureError(f"source does not parse: {e}") from e


def derive_arithmetic(source: str, inputs: dict[str, int], spec: ArithmeticSpec) -> ArithmeticProblem:
    if inputs:
        raise StructureError("arithmetic chains have no inputs")
    statements = _parse(source)
    if len(statements) != spec.n_ops:
        raise StructureError(f"{len(statements)} statements; the setting says {spec.n_ops}")
    first, steps = None, []
    for i, st in enumerate(statements):
        e = st.expr
        if st.target != VAR_NAMES[i]:
            raise StructureError(f"statement {i + 1} assigns {st.target!r}, expected {VAR_NAMES[i]!r}")
        if not isinstance(e, BinOp) or e.op not in OPS or not isinstance(e.right, Num):
            raise StructureError(f"statement {i + 1} is not 'left op constant'")
        if i == 0:
            if not isinstance(e.left, Num):
                raise StructureError("statement 1 must combine two constants")
            first = e.left.value
        elif e.left != Var(VAR_NAMES[i - 1]):
            raise StructureError(f"statement {i + 1} does not read {VAR_NAMES[i - 1]!r} on the left")
        steps.append((e.op, e.right.value))
    try:
        problem = make_problem(first, steps, spec)
    except ValueError as e:
        raise StructureError(str(e)) from e
    if "\n".join(problem.statements()) != source:
        raise StructureError("source is not in canonical form")
    return problem


def _program_shape(i: int, e, pred: set[str], defined: set[str], spec: ProgramSpec) -> str:
    lo, hi = spec.operand_range()
    if not isinstance(e, BinOp):
        raise StructureError(f"statement {i + 1} is not a binary operation")
    left, right = e.left, e.right
    if isinstance(right, Num):
        if not (isinstance(left, Var) and left.name in pred):
            raise StructureError(f"statement {i + 1} does not read its predecessor before the constant")
        if e.op == "*":
            if not spec.mul_min <= right.value <= spec.mul_max:
                raise StructureError(f"statement {i + 1}: multiplier {right.value} outside "
                                     f"[{spec.mul_min}, {spec.mul_max}]")
            return "mul_const"
        if not lo <= right.value <= hi:
            raise StructureError(f"statement {i + 1}: constant {right.value} outside [{lo}, {hi}]")
        return {"+": "add_const", "-": "sub_const"}[e.op]
    if isinstance(left, Num):
        raise StructureError(f"statement {i + 1} has a constant on the left")
    if e.op == "*":
        raise StructureError(f"statement {i + 1} multiplies two variables")
    names = {left.name, right.name}
    if len(names) != 2 or not names <= defined or not names & pred:
        raise StructureError(f"statement {i + 1} must combine its predecessor with another defined variable")
    return {"+": "add_var", "-": "sub_var"}[e.op]


def derive_program(source: str, inputs: dict[str, int], spec: ProgramSpec) -> ProgramProblem:
    expected_inputs = list(INPUT_NAMES[: spec.n_inputs])
    if list(inputs) != expected_inputs:
        raise StructureError(f"inputs {list(inputs)}; expected {expected_inputs}")
    lo, hi = spec.operand_range()
    for name, v in inputs.items():
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
            raise StructureError(f"input {name}={v!r} outside [{lo}, {hi}]")
    statements = _parse(source)
    if len(statements) != spec.n_statements:
        raise StructureError(f"{len(statements)} statements; the setting says {spec.n_statements}")
    defined, shapes = set(inputs), []
    for i, st in enumerate(statements):
        if st.target != STATEMENT_NAMES[i]:
            raise StructureError(f"statement {i + 1} assigns {st.target!r}, expected {STATEMENT_NAMES[i]!r}")
        pred = set(inputs) if i == 0 else {STATEMENT_NAMES[i - 1]}
        shapes.append(_program_shape(i, st.expr, pred, defined, spec))
        defined.add(st.target)
    try:
        values = execute(statements, inputs, spec.language())
    except ProgramError as e:
        raise StructureError(str(e)) from e
    problem = ProgramProblem(inputs=tuple(inputs.items()), statements=statements, shapes=tuple(shapes),
                             values=values, spec=spec)
    if problem.source != source:
        raise StructureError("source is not in canonical form")
    return problem


def derive_problem(family: str, source: str, inputs: dict[str, int], setting: dict, *, mul_range: tuple[int, int],
                   value_bound: int, allow_negative: bool,
                   n_inputs: int | None = None) -> ArithmeticProblem | ProgramProblem:
    """Rebuild the problem a record claims to be, from its source, inputs and public setting."""
    mul_min, mul_max = mul_range
    try:
        if family == "arithmetic":
            if set(setting) != {"length", "operand_digits"}:
                raise StructureError(f"arithmetic setting fields {sorted(setting)}")
            spec = ArithmeticSpec(n_ops=setting["length"], operand_digits=setting["operand_digits"],
                                  mul_min=mul_min, mul_max=mul_max, value_bound=value_bound,
                                  allow_negative=allow_negative)
            return derive_arithmetic(source, inputs, spec)
        if family == "programs":
            if set(setting) != {"length", "operand_digits", "n_inputs"} or setting["n_inputs"] != n_inputs:
                raise StructureError(f"program setting {setting} does not match n_inputs={n_inputs}")
            spec = ProgramSpec(n_statements=setting["length"], operand_digits=setting["operand_digits"],
                               n_inputs=n_inputs, mul_min=mul_min, mul_max=mul_max, value_bound=value_bound,
                               allow_negative=allow_negative)
            return derive_program(source, inputs, spec)
    except StructureError:
        raise
    except ValueError as e:
        raise StructureError(f"setting {setting} is invalid: {e}") from e
    raise StructureError(f"unknown family {family!r}")
