"""Hand-checkable software fixtures. They are not part of the scientific benchmark."""

from __future__ import annotations

from .arithmetic import ArithmeticProblem, ArithmeticSpec, make_problem
from .programs import BinOp, ProgramProblem, ProgramSpec, Var, execute, parse_program, shape_of


def program_from_source(source: str, inputs: dict[str, int], spec: ProgramSpec) -> ProgramProblem:
    stmts = parse_program(source)
    values = execute(stmts, inputs, spec.language())
    shapes, prev = [], None
    for st in stmts:
        e = st.expr
        if prev is None and isinstance(e, BinOp):
            prev = next((a.name for a in (e.left, e.right) if isinstance(a, Var)), None)
        shapes.append(shape_of(st, prev) if isinstance(e, BinOp) and prev is not None else "other")
        prev = st.target
    return ProgramProblem(inputs=tuple(inputs.items()), statements=stmts, shapes=tuple(shapes), values=values,
                          spec=spec)


def _arith(first, steps, digits) -> ArithmeticProblem:
    return make_problem(first, steps, ArithmeticSpec(n_ops=len(steps), operand_digits=digits))


FIXTURES = {
    # a = 2 + 3 -> 5
    "arith_2_plus_3": lambda: _arith(2, [("+", 3)], 1),
    # a = 17 + 28; b = a * 3; c = b - 19 -> 45, 135, 116 (build prompt example)
    "arith_17_28": lambda: _arith(17, [("+", 28), ("*", 3), ("-", 19)], 2),
    # a = 12 - 45; b = a * 3; c = b + 99 -> -33, -99, 0 (sign change and zero)
    "arith_negative_zero": lambda: _arith(12, [("-", 45), ("*", 3), ("+", 99)], 2),
    # x = 17, y = 28: a = x + y; b = a * 3; c = b - x -> 45, 135, 118
    "program_basic": lambda: program_from_source("a = x + y\nb = a * 3\nc = b - x", {"x": 17, "y": 28},
                                                 ProgramSpec(n_statements=3, operand_digits=2)),
    # x = 4, y = 9: a = y - x; b = 7 - a; c = b * 2 -> 5, 2, 4
    "program_const_minus_var": lambda: program_from_source("a = y - x\nb = 7 - a\nc = b * 2", {"x": 4, "y": 9},
                                                              ProgramSpec(n_statements=3, operand_digits=1)),
}


def get_fixture(name: str) -> ArithmeticProblem | ProgramProblem:
    if name not in FIXTURES:
        raise KeyError(f"unknown fixture {name!r}; choose from {sorted(FIXTURES)}")
    return FIXTURES[name]()
