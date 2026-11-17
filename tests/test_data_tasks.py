"""Arithmetic chains and the restricted program language (no tokenizer needed).

Expected values are either computed by hand in the comments or by
``reference_eval``, an independent evaluator written here on top of Python's
``ast`` parser (it shares no code with the package's lexer, parser or interpreter,
and it never calls eval/exec).
"""

import ast
import builtins
import random

import pytest

from diffusion_correction.data.arithmetic import ArithmeticSpec, evaluate_chain, make_problem, sample_chain
from diffusion_correction.data.fixtures import get_fixture
from diffusion_correction.data.programs import (LanguageSpec, ProgramError, ProgramSpec, normalized_key,
                                                parse_program, run_program, sample_program, shape_of)


def reference_eval(source: str, inputs: dict[str, int]) -> list[int]:
    env, trace = dict(inputs), []

    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}[type(node.op)]
        raise AssertionError(f"unexpected node {ast.dump(node)}")

    for line in source.split("\n"):
        (stmt,) = ast.parse(line).body
        assert isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        env[stmt.targets[0].id] = ev(stmt.value)
        trace.append(env[stmt.targets[0].id])
    return trace


# ---------------------------------------------------------------- arithmetic

def test_hand_checked_arithmetic_fixtures():
    assert get_fixture("arith_2_plus_3").values == (5,)                  # 2 + 3
    p = get_fixture("arith_17_28")
    assert p.values == (45, 135, 116)                                    # 17+28=45; 45*3=135; 135-19=116
    assert p.statements() == ["a = 17 + 28", "b = a * 3", "c = b - 19"]
    assert p.final_answer == 116 and p.signature == ("+", "*", "-")
    assert get_fixture("arith_negative_zero").values == (-33, -99, 0)    # 12-45=-33; -33*3=-99; -99+99=0


def test_make_problem_rejects_out_of_spec_chains():
    spec2 = ArithmeticSpec(n_ops=2, operand_digits=2)
    with pytest.raises(ValueError, match="multiplier"):
        make_problem(17, [("+", 28), ("*", 10)], spec2)
    with pytest.raises(ValueError, match="operand"):
        make_problem(17, [("+", 5), ("-", 19)], spec2)                   # 5 is not a 2-digit operand
    with pytest.raises(ValueError, match="first operand"):
        make_problem(7, [("+", 28), ("-", 19)], spec2)
    with pytest.raises(ValueError, match="bounds"):
        make_problem(999, [("*", 9), ("*", 2)], ArithmeticSpec(n_ops=2, operand_digits=3))  # 8991 then 17982
    with pytest.raises(ValueError, match="bounds"):
        make_problem(12, [("-", 45), ("+", 10)], ArithmeticSpec(n_ops=2, operand_digits=2, allow_negative=False))
    with pytest.raises(ValueError, match="operations"):
        make_problem(17, [("+", 28)], spec2)


@pytest.mark.parametrize("n_ops,digits", [(2, 1), (4, 2), (6, 3), (12, 3)])
def test_sampled_chains_are_in_spec_and_match_independent_evaluation(n_ops, digits):
    spec = ArithmeticSpec(n_ops=n_ops, operand_digits=digits)
    lo, hi = spec.operand_range()
    rng = random.Random(1234)
    for _ in range(300):
        p = sample_chain(rng, spec)
        assert len(p.values) == n_ops and lo <= p.first <= hi
        for op, k in p.steps:
            assert (2 <= k <= 9) if op == "*" else (lo <= k <= hi and len(str(k)) == digits)
        assert all(abs(v) <= 9999 for v in p.values)
        assert list(p.values) == reference_eval("\n".join(p.statements()), {})


def test_sampling_is_deterministic_per_seed():
    spec = ArithmeticSpec(n_ops=4, operand_digits=2)
    a = [sample_chain(random.Random(7), spec) for _ in range(3)]
    b = [sample_chain(random.Random(7), spec) for _ in range(3)]
    assert a == b
    assert sample_chain(random.Random(7), spec) != sample_chain(random.Random(8), spec)


def test_non_negative_spec_never_goes_negative():
    spec = ArithmeticSpec(n_ops=6, operand_digits=2, allow_negative=False)
    rng = random.Random(0)
    assert all(v >= 0 for _ in range(300) for v in sample_chain(rng, spec).values)


def test_evaluate_chain_by_hand():
    assert evaluate_chain(9, [("-", 12), ("*", 4), ("+", 50)]) == (-3, -12, 38)


# ---------------------------------------------------------------- program language

def test_hand_checked_programs():
    assert run_program("a = x + y\nb = a * 3\nc = b - x", {"x": 17, "y": 28}) == (45, 135, 118)
    assert run_program("a = y - x\nb = 7 - a\nc = b * 2", {"x": 4, "y": 9}) == (5, 2, 4)
    assert run_program("a = x + 1\na = a * 2\nb = a - x", {"x": 3}) == (4, 8, 5)  # reassignment
    assert run_program("a = 5", {}) == (5,)
    assert run_program("a = x", {"x": 0}) == (0,)
    assert run_program("a = 0 - x\nb = a * 9", {"x": 12}) == (-12, -108)
    assert run_program("a=x+y", {"x": 1, "y": 2}) == (3,)  # spaces are optional
    assert get_fixture("program_basic").values == (45, 135, 118)


@pytest.mark.parametrize("source,inputs,kind", [
    ("a = x + + y", {"x": 1, "y": 2}, "syntax"),
    ("a = (x)", {"x": 1}, "syntax"),
    ("ab = 1", {}, "syntax"),
    ("a = 01", {}, "syntax"),
    ("a = 1x", {"x": 1}, "syntax"),
    ("a = x / 2", {"x": 4}, "syntax"),
    ("a = -3", {}, "syntax"),
    ("a = x # comment", {"x": 1}, "syntax"),
    ("a = x\n\nb = a", {"x": 1}, "syntax"),
    ("", {}, "syntax"),
    ("A = 1", {}, "syntax"),
    ("a = __import__('os')", {}, "syntax"),
    ("a = b + 1", {}, "undefined"),
    ("a = a + 1", {}, "undefined"),
    ("a = x * y", {"x": 2, "y": 3}, "multiply"),
    ("a = x * 10", {"x": 2}, "multiply"),
    ("a = x * 1", {"x": 2}, "multiply"),
    ("a = x + 1000", {"x": 2}, "literal"),
    ("x = 5", {"x": 1}, "input"),
    ("a = x", {"x": 1000}, "input"),
    ("a = x * 9\nb = a * 9", {"x": 999}, "bound"),                      # 8991, then 80919 > 9999
    ("\n".join(f"{chr(97 + i)} = 1" for i in range(13)), {}, "length"),
])
def test_invalid_programs_are_rejected_predictably(source, inputs, kind):
    with pytest.raises(ProgramError) as e:
        run_program(source, inputs)
    assert e.value.kind == kind


def test_negative_values_rejected_when_disallowed():
    with pytest.raises(ProgramError) as e:
        run_program("a = x - 5", {"x": 3}, LanguageSpec(allow_negative=False))
    assert e.value.kind == "bound" and e.value.line == 1


def test_interpreter_never_calls_eval_or_exec(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("eval/exec called")

    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    assert run_program("a = x + y\nb = a * 3", {"x": 1, "y": 2}) == (3, 9)


@pytest.mark.parametrize("n,digits", [(2, 1), (4, 2), (6, 3), (10, 2)])
def test_generated_programs_match_independent_evaluation(n, digits):
    spec = ProgramSpec(n_statements=n, operand_digits=digits)
    rng = random.Random(99)
    for _ in range(300):
        p = sample_program(rng, spec)
        assert list(p.values) == reference_eval(p.source, dict(p.inputs))
        assert all(abs(v) <= 9999 for v in p.values)
        # Dependency chain: statement i reads statement i-1's target; shapes are recorded faithfully.
        for i, st in enumerate(p.statements):
            names = [a.name for a in (st.expr.left, st.expr.right) if hasattr(a, "name")]
            if i == 0:
                assert all(nm in dict(p.inputs) for nm in names) and names
            else:
                assert p.statements[i - 1].target in names
                assert shape_of(st, p.statements[i - 1].target) == p.shapes[i]


def test_normalized_key_is_alpha_invariant_but_not_semantic():
    s1 = parse_program("a = x + y\nb = a * 3")
    s2 = parse_program("p = x + y\nq = p * 3")
    s3 = parse_program("a = y + x\nb = a * 3")
    s4 = parse_program("a = x + y\nb = a * 4")
    k = normalized_key(s1, {"x": 1, "y": 2})
    assert k == normalized_key(s2, {"x": 1, "y": 2})
    assert k == normalized_key(parse_program("a = u + v\nb = a * 3"), {"u": 1, "v": 2})  # inputs renamed
    assert k != normalized_key(s3, {"x": 1, "y": 2})                     # operand order is kept
    assert k != normalized_key(s4, {"x": 1, "y": 2})
    assert k != normalized_key(s1, {"x": 1, "y": 3})                     # inputs are part of the key
