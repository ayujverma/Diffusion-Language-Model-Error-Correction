"""Gate A (alignment/attention parts) on the tiny backbone: software checks only."""

import pytest
import torch

from diffusion_correction.losses import selected_token_ce
from diffusion_correction.models import qwen2_bidir
from diffusion_correction.models.alignment import align_logits, assert_targets_predictable, build_layout
from diffusion_correction.models.tiny import TINY_TOKENS


def test_align_logits_shifts_by_one():
    raw = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    al = align_logits(raw)
    assert al.shape == raw.shape
    for i in range(1, 5):
        assert torch.equal(al[:, i], raw[:, i - 1])
    assert torch.equal(al[:, 0], torch.zeros_like(raw[:, 0]))


def test_position_zero_is_never_a_target():
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        assert_targets_predictable(mask)
    with pytest.raises(ValueError):
        build_layout([[]], [[1, 2]], pad_token_id=0)


def test_layout_masks_partition_sequence(ragged_layout):
    lay = ragged_layout
    assert lay.input_ids.shape == (2, 7)
    covered = lay.prompt_mask.long() + lay.response_mask.long() + lay.pad_mask.long()
    assert torch.equal(covered, torch.ones_like(covered))
    assert torch.equal(lay.editable_mask, lay.response_mask)
    assert lay.prompt_mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert lay.response_mask[0].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert lay.response_mask[1].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert lay.pad_mask[1].tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert lay.pad_mask[0].sum() == 0
    assert lay.input_ids[1, 6] == TINY_TOKENS.pad_token_id
    short = build_layout([[1]], [[2]], pad_token_id=TINY_TOKENS.pad_token_id, total_length=4)
    assert short.pad_mask[0].tolist() == [0, 0, 1, 1]


def test_first_response_target_uses_last_prompt_output(tiny, ragged_layout):
    """CE at the first response position must come from the raw output at the last prompt position."""
    lay = ragged_layout
    raw = tiny.raw_logits(lay.input_ids, lay.attention_mask)
    aligned = tiny.aligned_logits(lay)
    targets = lay.input_ids
    pos = torch.zeros_like(lay.response_mask)
    pos[0, 3] = True  # first response token of example 0 (prompt length 3)
    pos[1, 5] = True  # first response token of example 1 (prompt length 5)
    ce = selected_token_ce(aligned, targets, pos)
    ref0 = -torch.log_softmax(raw[0, 2], -1)[targets[0, 3]]
    ref1 = -torch.log_softmax(raw[1, 4], -1)[targets[1, 5]]
    torch.testing.assert_close(ce, torch.stack([ref0, ref1]), rtol=0, atol=1e-12)


def test_right_context_changes_earlier_predictions(tiny):
    lay = build_layout([[1, 2, 3]], [[4, 5, 6, 7]], pad_token_id=TINY_TOKENS.pad_token_id)
    base = tiny.aligned_logits(lay)
    ids = lay.input_ids.clone()
    ids[0, 6] = 9  # last response token
    changed = tiny.aligned_logits(lay.with_input_ids(ids))
    # Position 4 is predicted from raw row 3, which sits left of position 6.
    assert (base[0, 4] - changed[0, 4]).abs().max() > 1e-6


def test_right_context_check_detects_a_causal_mask(tiny, monkeypatch):
    """Control: the same probe must show NO change if attention were causal."""
    def causal_mask(attention_mask):
        B, L = attention_mask.shape
        tril = torch.ones(L, L, dtype=torch.bool).tril()
        return attention_mask[:, None, None, :] & tril[None, None]

    monkeypatch.setattr(qwen2_bidir, "key_padding_attn_mask", causal_mask)
    lay = build_layout([[1, 2, 3]], [[4, 5, 6, 7]], pad_token_id=TINY_TOKENS.pad_token_id)
    base = tiny.aligned_logits(lay)
    ids = lay.input_ids.clone()
    ids[0, 6] = 9
    changed = tiny.aligned_logits(lay.with_input_ids(ids))
    assert torch.equal(base[0, 4], changed[0, 4])


def test_padding_is_isolated(tiny):
    alone = build_layout([[1, 2, 3]], [[4, 5]], pad_token_id=TINY_TOKENS.pad_token_id)
    batch = build_layout([[1, 2, 3], [6, 7, 8, 9, 1, 2]], [[4, 5], [3, 3, 3]],
                         pad_token_id=TINY_TOKENS.pad_token_id)
    a = tiny.raw_logits(alone.input_ids, alone.attention_mask)[0]
    b = tiny.raw_logits(batch.input_ids, batch.attention_mask)[0, :5]
    torch.testing.assert_close(a, b, rtol=0, atol=1e-10)
    # Changing what sits in pad slots must not change any real position.
    ids = batch.input_ids.clone()
    ids[0, 5:] = 7
    c = tiny.raw_logits(ids, batch.attention_mask)[0, :5]
    torch.testing.assert_close(b, c, rtol=0, atol=0)


def test_attention_mask_must_be_bool(tiny, ragged_layout):
    with pytest.raises(TypeError):
        tiny.raw_logits(ragged_layout.input_ids, ragged_layout.attention_mask.long())
