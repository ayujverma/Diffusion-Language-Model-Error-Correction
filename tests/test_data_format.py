"""Fixed-slot response format and checker, on the pinned tokenizer.

Token IDs written as literals below were read off the pinned tokenizer by hand
(milestone-2 exploration recorded in docs/verification.md); the package derives
them from the tokenizer and never hard-codes them, so these assertions are an
independent check of what it derived.
"""

import random

import pytest
import torch

from diffusion_correction.data.checker import aggregate, check_response
from diffusion_correction.data.fixtures import FIXTURES, get_fixture
from diffusion_correction.data.format import (ResponseFormat, TokenizedFormat, boundary_merges,
                                              encode_and_verify)
from diffusion_correction.data.tasks import build_example
from diffusion_correction.losses import token_error_masks
from diffusion_correction.models.alignment import build_layout

pytestmark = pytest.mark.tokenizer

PLUS, MINUS, D = 488, 481, list(range(15, 25))  # " +", " -", "0".."9"
MASK, EOS = 151665, 151643
BOUNDARY_VALUES = [-9999, -1000, -999, -100, -99, -10, -9, -1, 0, 1, 9, 10, 99, 100, 999, 1000, 9999]


@pytest.fixture(scope="module")
def tf(task_tokenizer):
    return TokenizedFormat.from_tokenizer(task_tokenizer, ResponseFormat(9999))


def test_format_token_ids_match_hand_inspection(tf):
    assert tf.sign_ids == {"+": PLUS, "-": MINUS}
    assert list(tf.digit_ids) == D
    assert MASK in tf.special_ids and EOS in tf.special_ids
    assert tf.fmt.width == 4 and tf.fmt.slot_len == 5 and tf.fmt.response_len(3) == 15


def test_width_comes_from_declared_bound_not_values():
    assert ResponseFormat(9999).width == 4
    assert ResponseFormat(5000).width == 4
    assert ResponseFormat(99999).width == 5
    assert ResponseFormat(9999).render([5]) == " +0005"
    with pytest.raises(ValueError):
        ResponseFormat(9999).render_value(10000)


def test_naive_formats_do_not_have_fixed_token_width(task_tokenizer):
    """Why the format uses a leading-space sign: punctuation merges with its neighbours."""
    enc = lambda s: task_tokenizer.encode(s, add_special_tokens=False).tokens  # noqa: E731
    assert enc("a=+0045") == ["a", "=", "+", "0", "0", "4", "5"]
    assert enc("a=-0045") == ["a", "=-", "0", "0", "4", "5"]          # sign merged into "=-"
    assert len(enc(",+")) == 1 and len(enc(",-")) == 1 and len(enc("(+")) == 1


@pytest.mark.parametrize("v", BOUNDARY_VALUES)
def test_single_value_tokenization_and_round_trip(task_tokenizer, tf, v):
    text = tf.fmt.render([v])
    ids = task_tokenizer.encode(text, add_special_tokens=False).ids
    expected = [MINUS if v < 0 else PLUS] + [D[int(c)] for c in f"{abs(v):04d}"]
    assert ids == expected == tf.encode_values([v])
    p = tf.parse_slot(ids)
    assert p.status == "ok" and p.value == v


def test_random_multi_slot_sequences_tokenize_to_the_template(task_tokenizer, tf):
    rng = random.Random(0)
    for _ in range(2000):
        vals = [rng.choice(BOUNDARY_VALUES + [rng.randint(-9999, 9999)]) for _ in range(rng.randint(1, 12))]
        text = tf.fmt.render(vals)
        ids = task_tokenizer.encode(text, add_special_tokens=False).ids
        assert ids == tf.encode_values(vals) and len(ids) == 5 * len(vals)
        assert [tf.parse_slot(ids[s:e]).value for s, e in tf.fmt.slot_spans(len(vals))] == vals


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_prompts_have_clean_boundaries(task_tokenizer, tf, name):
    ex = build_example(get_fixture(name), "fixture", task_tokenizer, tf, heldout=False)
    assert ex.response_ids == tf.encode_values(ex.values)
    assert ex.prompt_text.endswith("# Answer:")
    joint = task_tokenizer.encode(ex.prompt_text + ex.response_text, add_special_tokens=False).ids
    assert joint == ex.prompt_ids + ex.response_ids
    assert ex.prompt_ids[-1] == 25  # ":" stays separate from the response's first sign token


def test_boundary_check_detects_a_real_merge(task_tokenizer):
    assert boundary_merges(task_tokenizer, "a=", "-0045")               # joint text tokenizes "=-" as one token
    assert not boundary_merges(task_tokenizer, "a =", "-0045")


def test_leading_space_sign_never_merges_with_any_prompt_ending(task_tokenizer):
    import string

    for end in list(string.printable) + ["  ", ": ", ":\n", "\n "]:
        for resp in (" +0045", " -0045"):
            assert not boundary_merges(task_tokenizer, "# Answer" + end, resp), (end, resp)


def test_encode_and_verify_rejects_a_template_mismatch(task_tokenizer, tf):
    wrong = TokenizedFormat(fmt=tf.fmt, sign_ids={"+": MINUS, "-": PLUS}, digit_ids=tf.digit_ids,
                            special_ids=tf.special_ids, tokenizer_size=tf.tokenizer_size)
    with pytest.raises(ValueError, match="template"):
        encode_and_verify(task_tokenizer, wrong, "# Answer:", [45])


def test_fixture_17_28_exact_ids(task_tokenizer, tf):
    ex = build_example(get_fixture("arith_17_28"), "fixture", task_tokenizer, tf, heldout=False)
    assert ex.response_text == " +0045 +0135 +0116"
    assert ex.response_ids == [PLUS, 15, 15, 19, 20, PLUS, 15, 16, 18, 20, PLUS, 15, 16, 16, 21]
    assert ex.slot_spans == [[0, 5], [5, 10], [10, 15]]
    neg = build_example(get_fixture("arith_negative_zero"), "fixture", task_tokenizer, tf, heldout=False)
    assert neg.response_text == " -0033 -0099 +0000"


# ---------------------------------------------------------------- malformed slots and the checker

@pytest.mark.parametrize("ids,status", [
    ([PLUS, D[0], D[0], D[4], D[5]], "ok"),
    ([D[1], D[0], D[0], D[4], D[5]], "bad_sign_token"),
    ([PLUS, PLUS, D[0], D[4], D[5]], "bad_digit_token"),
    ([PLUS, D[0], 10, D[4], D[5]], "bad_digit_token"),                    # "+" without the space
    ([MASK, D[0], D[0], D[4], D[5]], "special_token"),
    ([PLUS, D[0], D[0], D[4], EOS], "special_token"),
    ([PLUS, D[0], D[0], D[4], 151700], "unknown_token_id"),              # output row with no token
    ([MINUS, D[0], D[0], D[0], D[0]], "negative_zero"),
    ([PLUS, D[0], D[4], D[5]], "wrong_slot_length"),
])
def test_slot_parse_statuses(tf, ids, status):
    assert tf.parse_slot(ids).status == status


@pytest.fixture(scope="module")
def ex17(task_tokenizer, tf):
    return build_example(get_fixture("arith_17_28"), "fixture", task_tokenizer, tf, heldout=False)


def test_checker_correct_output(ex17, tf):
    c = check_response(ex17.response_ids, ex17, tf)
    assert all(c.token_correct) and all(c.value_correct) and c.final_correct and c.trace_correct and c.well_formed


def test_checker_keeps_levels_separate(ex17, tf):
    draft = tf.encode_values([46, 135, 116])                             # one wrong digit in a
    c = check_response(draft, ex17, tf)
    assert sum(c.token_correct) == 14 and c.token_correct[4] is False
    assert c.value_correct == [False, True, True]
    assert c.final_correct and not c.trace_correct and c.well_formed
    c2 = check_response(tf.encode_values([45, 135, 117]), ex17, tf)    # only the final answer wrong
    assert c2.value_correct == [True, True, False] and not c2.final_correct


def test_checker_counts_malformed_slots_as_wrong_without_dropping(ex17, tf):
    draft = list(ex17.response_ids)
    draft[7] = MASK                                                      # inside slot b
    c = check_response(draft, ex17, tf)
    assert c.slot_status == ["ok", "special_token", "ok"] and c.special_token_positions == [7]
    assert c.value_correct == [True, False, True] and sum(c.token_correct) == 14
    assert not c.well_formed and c.final_correct and not c.trace_correct


@pytest.mark.parametrize("delta", [-1, +1, -15])
def test_checker_never_zips_unequal_lengths(ex17, tf, delta):
    draft = list(ex17.response_ids)
    draft = draft[:delta] if delta < 0 else draft + [PLUS]
    c = check_response(draft, ex17, tf)
    assert not c.length_ok and c.n_tokens == 15 and c.token_correct == [False] * 15
    assert c.slot_status == ["length_mismatch"] * 3 and not c.final_correct and not c.trace_correct


def test_aggregate_denominators_include_every_example(ex17, tf):
    checks = [
        check_response(ex17.response_ids, ex17, tf),                     # all correct
        check_response(tf.encode_values([46, 135, 117]), ex17, tf),      # 2 wrong values, 2 wrong tokens
        check_response([EOS] * 15, ex17, tf),                            # all special tokens
        check_response(ex17.response_ids[:-1], ex17, tf),                # length mismatch
    ]
    a = aggregate(checks)
    assert (a["n_examples"], a["n_tokens"], a["n_values"]) == (4, 60, 12)
    assert a["token_correct"] == 15 + 13 + 0 + 0
    assert a["value_correct"] == 3 + 1 + 0 + 0
    assert a["final_correct"] == 1 and a["trace_correct"] == 1
    assert a["malformed_examples"] == 2 and a["length_mismatch_examples"] == 1
    assert a["malformed_slot_reasons"] == {"length_mismatch": 3, "special_token": 3}
    assert a["value_accuracy"] == 4 / 12 and a["final_answer_accuracy"] == 1 / 4


def test_value_correct_iff_all_slot_tokens_correct(ex17, tf):
    """Canonical format: value-level and token-level agree per slot on random corruptions."""
    rng = random.Random(3)
    vocab = [PLUS, MINUS, *D, MASK, EOS, 10, 12]
    for _ in range(3000):
        draft = list(ex17.response_ids)
        for _ in range(rng.randint(1, 3)):
            draft[rng.randrange(15)] = rng.choice(vocab)
        c = check_response(draft, ex17, tf)
        for k, (s, e) in enumerate(ex17.slot_spans):
            assert c.value_correct[k] == all(c.token_correct[s:e])


def test_checker_labels_agree_with_loss_error_masks(ex17, tf):
    """Token-level checker results and the loss code's E/K masks mark the same positions."""
    draft = tf.encode_values([46, -135, 116])
    c = check_response(draft, ex17, tf)
    lay = build_layout([ex17.prompt_ids], [draft], pad_token_id=EOS, total_length=len(ex17.prompt_ids) + 17)
    tgt = build_layout([ex17.prompt_ids], [ex17.response_ids], pad_token_id=EOS,
                       total_length=len(ex17.prompt_ids) + 17).input_ids
    E, K = token_error_masks(lay.input_ids, tgt, lay.editable_mask)
    n_p = len(ex17.prompt_ids)
    assert E[0, n_p:n_p + 15].tolist() == [not t for t in c.token_correct]
    assert not bool(E[0, :n_p].any() or E[0, n_p + 15:].any() or K[0, n_p + 15:].any())
    assert torch.equal(E | K, lay.editable_mask)
