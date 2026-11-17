"""Rebuilding problems from source (data/structure.py): signatures, structure, identity.

Expected signatures and keys are written by hand; they are not produced by the code
under test.
"""

import pytest

from diffusion_correction.data.programs import normalized_key, parse_program
from diffusion_correction.data.splits import contains_bigram
from diffusion_correction.data.structure import StructureError, derive_problem
from diffusion_correction.data.tasks import problem_id

COMMON = dict(mul_range=(2, 9), value_bound=9999, allow_negative=True)
ARITH_HELDOUT = [("-", "*")]
PROG_HELDOUT = [("mul_const", "sub_var")]


def arith(source, length, digits):
    return derive_problem("arithmetic", source, {}, {"length": length, "operand_digits": digits}, **COMMON)


def prog(source, inputs, length, digits, n_inputs=2):
    return derive_problem("programs", source, inputs,
                          {"length": length, "operand_digits": digits, "n_inputs": n_inputs},
                          n_inputs=n_inputs, **COMMON)


@pytest.mark.parametrize("source, length, digits, signature, key, values", [
    ("a = 17 + 28\nb = a * 3\nc = b - 19", 3, 2, ("+", "*", "-"), "arithmetic|a=17+28;b=a*3;c=b-19", (45, 135, 116)),
    ("a = 1 - 1\nb = a - 3", 2, 1, ("-", "-"), "arithmetic|a=1-1;b=a-3", (0, -3)),
    ("a = 12 - 45\nb = a * 3\nc = b + 99", 3, 2, ("-", "*", "+"), "arithmetic|a=12-45;b=a*3;c=b+99", (-33, -99, 0)),
    ("a = 5 * 2\nb = a - 7", 2, 1, ("*", "-"), "arithmetic|a=5*2;b=a-7", (10, 3)),
])
def test_arithmetic_signature_and_key_from_source(source, length, digits, signature, key, values):
    p = arith(source, length, digits)
    assert p.signature == signature
    assert p.normalized_key() == key
    assert p.values == values
    assert problem_id(key) == problem_id(p.normalized_key())


@pytest.mark.parametrize("source, inputs, length, digits, signature, values", [
    # First statement reads an input, with a constant.
    ("a = y + 6\nb = a + 7\nc = y + b\nd = c + 9", {"x": 3, "y": 5}, 4, 1,
     ("add_const", "add_const", "add_var", "add_const"), (11, 18, 23, 32)),
    # First statement combines both inputs; later statements reuse an earlier target and an input.
    ("a = x + y\nb = a * 3\nc = b - x", {"x": 17, "y": 28}, 3, 2,
     ("add_var", "mul_const", "sub_var"), (45, 135, 118)),
    # Swapped operand order for the _var shapes (the predecessor on the right).
    ("a = x - y\nb = x - a\nc = a + b", {"x": 7, "y": 2}, 3, 1,
     ("sub_var", "sub_var", "add_var"), (5, 2, 7)),
    ("a = x * 4\nb = a - 3", {"x": 9, "y": 1}, 2, 1, ("mul_const", "sub_const"), (36, 33)),
])
def test_program_signature_from_source(source, inputs, length, digits, signature, values):
    p = prog(source, inputs, length, digits)
    assert p.signature == signature
    assert p.values == values
    assert p.normalized_key() == normalized_key(parse_program(source), inputs)


def test_heldout_predicate_on_derived_signatures():
    assert contains_bigram(arith("a = 5 - 3\nb = a * 2", 2, 1).signature, ARITH_HELDOUT)
    assert not contains_bigram(arith("a = 5 * 3\nb = a - 2", 2, 1).signature, ARITH_HELDOUT)  # order matters
    assert contains_bigram(prog("a = x + y\nb = a * 3\nc = b - x", {"x": 17, "y": 28}, 3, 2).signature,
                           PROG_HELDOUT)
    assert not contains_bigram(prog("a = x - y\nb = a * 3", {"x": 7, "y": 2}, 2, 1).signature, PROG_HELDOUT)


@pytest.mark.parametrize("source, length, digits, reason", [
    ("a = 17 + 28\nb = 3 * a", 2, 2, "not 'left op constant'"),     # constant on the left
    ("a = 17 + 28\nc = a + 30", 2, 2, "expected 'b'"),              # wrong target name
    ("a = 17 + 28\nb = a + 30\nb = b + 11", 3, 2, "expected 'c'"),  # reassignment
    ("a = 17 + 28\nb = a + 30", 3, 2, "setting says 3"),            # length differs from setting
    ("a = 5 + 28\nb = a + 30", 2, 2, "first operand"),             # first constant has 1 digit
    ("a = 17 + 28\nb = a * 12", 2, 2, "multiplier"),               # multiplier outside [2, 9]
    ("a = 17 + 28\nb = a + 300", 2, 2, "operand"),                 # constant has 3 digits
    ("a = 17+28\nb = a + 30", 2, 2, "canonical"),                  # non-canonical spacing
    ("a = 17 + 28\nb = 45 + 30", 2, 2, "does not read 'a'"),       # chain broken
    ("a = 17 + 28\nb = a", 2, 2, "not 'left op constant'"),
])
def test_arithmetic_outside_generated_structure_is_rejected(source, length, digits, reason):
    with pytest.raises(StructureError, match=reason):
        arith(source, length, digits)


@pytest.mark.parametrize("source, inputs, reason", [
    ("a = x + 5\nb = 5 - a", {"x": 3, "y": 4}, "constant on the left"),
    ("a = x + 5\nb = x + 3", {"x": 3, "y": 4}, "predecessor"),           # does not read a
    ("a = x + 5\nb = x + y", {"x": 3, "y": 4}, "predecessor"),           # two vars, neither is a
    ("a = x + 5\nb = a + a", {"x": 3, "y": 4}, "another defined"),       # same variable twice
    ("a = x * y\nb = a + 1", {"x": 3, "y": 4}, "multiplies two variables"),
    ("a = 3 + 4\nb = a + 1", {"x": 3, "y": 4}, "statement 1 does not read its predecessor"),  # no input read
    ("a = x + 5\na = a + 1", {"x": 3, "y": 4}, "expected 'b'"),          # reassignment
    ("a = x + 5\nb = a + 1", {"u": 3, "v": 4}, "inputs"),                # renamed inputs
    ("a = x + 5\nb = a + 1", {"x": 3, "y": 40}, "outside"),              # input has 2 digits
    ("a = x + 5\nb = a * 1", {"x": 3, "y": 4}, "multiplier"),
    ("a = x + 5\nb = a + 12", {"x": 3, "y": 4}, "constant 12"),
    ("a = x+5\nb = a + 1", {"x": 3, "y": 4}, "canonical"),
    ("a = x + 5\nb = a + q", {"x": 3, "y": 4}, "another defined"),       # undefined name
])
def test_program_outside_generated_structure_is_rejected(source, inputs, reason):
    with pytest.raises(StructureError, match=reason):
        prog(source, inputs, 2, 1)


def test_setting_must_match_family_and_config():
    with pytest.raises(StructureError, match="n_inputs"):
        derive_problem("programs", "a = x + 5\nb = a + 1", {"x": 3, "y": 4},
                       {"length": 2, "operand_digits": 1, "n_inputs": 3}, n_inputs=2, **COMMON)
    with pytest.raises(StructureError, match="arithmetic chains have no inputs"):
        derive_problem("arithmetic", "a = 1 + 2", {"x": 1}, {"length": 1, "operand_digits": 1}, **COMMON)
    with pytest.raises(StructureError, match="unknown family"):
        derive_problem("geometry", "a = 1 + 2", {}, {"length": 1, "operand_digits": 1}, **COMMON)


def test_identity_keeps_inputs_order_and_reassignment_but_not_names():
    base = normalized_key(parse_program("a = x + 1\na = a + 2\nb = a * 3"), {"x": 5})
    # Variable renaming does not change identity, including a renamed reassigned target.
    assert base == normalized_key(parse_program("p = x + 1\np = p + 2\nq = p * 3"), {"x": 5})
    # Reassignment is not the same program as assigning a fresh variable.
    assert base != normalized_key(parse_program("a = x + 1\nb = a + 2\nc = b * 3"), {"x": 5})
    # Input values, operand order and statement order are all part of the identity.
    assert base != normalized_key(parse_program("a = x + 1\na = a + 2\nb = a * 3"), {"x": 6})
    assert base != normalized_key(parse_program("a = 1 + x\na = a + 2\nb = a * 3"), {"x": 5})
    assert base != normalized_key(parse_program("a = x + 2\na = a + 1\nb = a * 3"), {"x": 5})
    assert base == "program|i0=5;t0=i0+1;t0=t0+2;t1=t0*3"
