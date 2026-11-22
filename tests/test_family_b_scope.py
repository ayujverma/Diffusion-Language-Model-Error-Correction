"""SCOPE training: sampled-token confidence, clean context excluded, no_grad draft (D-052)."""

import random

import torch

from diffusion_correction.losses.family_b import (clamp_scope_rate, commit_top, mix_is_mdm,
                                                  sample_allowed_tokens, sample_beta22, scope_context_ids,
                                                  scope_loss_positions, scope_training_step)
from diffusion_correction.losses.token_ce import selected_token_ce
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.training.rng import make_generator


def test_beta_and_mix_use_their_generator_and_not_python_random():
    g = make_generator(5, "scope_mask")
    state = g.get_state()
    random.seed(0)
    a = sample_beta22(g)
    g.set_state(state)
    random.seed(12345)
    b = sample_beta22(g)
    assert a == b and 0.0 < a < 1.0
    assert clamp_scope_rate(-1.0) == 0.1 and clamp_scope_rate(2.0) == 0.9

    mix = make_generator(5, "scope_mix")
    saved = mix.get_state()
    random.seed(7)
    first = [mix_is_mdm(mix) for _ in range(32)]
    mix.set_state(saved)
    random.seed(99)
    assert [mix_is_mdm(mix) for _ in range(32)] == first
    assert any(first) and not all(first)


def test_commit_count_and_loss_positions_exclude_clean_tokens():
    confidence = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.05]])
    masked = torch.tensor([[False, True, True, True, False]])
    committed = commit_top(confidence, masked, rho=0.3)  # n=3 -> K=1, the 0.9
    assert torch.equal(committed, torch.tensor([[False, True, False, False, False]]))
    ten = torch.tensor([[True] * 10])
    assert int(commit_top(torch.linspace(0, 1, 10).view(1, 10), ten, 0.3).sum()) == 3
    positions = scope_loss_positions(masked, committed)
    assert torch.equal(positions, masked)
    gold = torch.tensor([[8, 1, 2, 3, 4]])
    samples = torch.tensor([[0, 6, 7, 5, 0]])
    context = scope_context_ids(gold, masked, committed, samples, mask_token_id=9)
    assert torch.equal(context, torch.tensor([[8, 6, 9, 9, 4]]))
    assert int(context[0, 0]) == 8 and int(context[0, 4]) == 4


def test_confidence_is_the_temperature_1_probability_of_the_sampled_token():
    logits = torch.tensor([[[0.0, 0.4, -0.2, 1.5]]], dtype=torch.float64)
    positions = torch.tensor([[True]])
    allowed = torch.tensor([True, True, True, False])  # last id is forbidden, like a mask
    gen = make_generator(9, "scope_sample")
    ids, conf = sample_allowed_tokens(logits, positions, allowed, gen, temperature=1.5)
    choice = int(ids[0, 0])
    full = torch.softmax(logits.float(), dim=-1)[0, 0]
    tempered = torch.softmax(logits.float() / 1.5, dim=-1)[0, 0]
    assert choice != 3
    assert abs(float(conf[0, 0]) - float(full[choice])) < 1e-6
    assert abs(float(full[choice]) - float(tempered[choice])) > 1e-4
    # A lower-confidence non-argmax is still what the ranker sees when that id was sampled.
    assert abs(float(conf[0, 0]) - float(full.max())) > 1e-6 or choice == int(full.argmax())


def test_scope_step_drops_clean_tokens_and_keeps_the_draft_pass_out_of_the_graph():
    adapter = build_tiny_adapter(seed=3, dtype=torch.float64)
    pad, mask_id = TINY_TOKENS.pad_token_id, TINY_TOKENS.mask_token_id
    layout = build_layout([[1, 2]], [[3, 4, 5, 6]], pad)
    masked = torch.zeros_like(layout.editable_mask)
    idx = layout.editable_mask[0].nonzero().flatten()
    masked[0, idx[:2]] = True
    allowed = torch.ones(TINY_TOKENS.vocab_size, dtype=torch.bool)
    allowed[mask_id] = allowed[pad] = allowed[TINY_TOKENS.eos_token_id] = False
    flags = []
    real = adapter.aligned_logits

    def wrapped(lay):
        flags.append(torch.is_grad_enabled())
        return real(lay)

    adapter.aligned_logits = wrapped
    gen = make_generator(3, "scope_sample")
    step = scope_training_step(adapter, layout, masked, allowed, gen)
    assert flags == [False, True]
    assert not step.logits1.requires_grad
    visible = layout.editable_mask & ~masked
    assert torch.equal(step.loss_positions, masked)
    assert not bool((step.loss_positions & visible).any())
    assert torch.equal(step.context_ids[visible], layout.input_ids[visible])
    assert torch.equal(step.context_ids[masked & ~step.committed], torch.full((int((masked & ~step.committed).sum()),), mask_id))
    reference = selected_token_ce(step.logits2, layout.input_ids, step.loss_positions).mean()
    assert torch.allclose(step.loss, reference, atol=1e-12)
    changed = layout.input_ids.clone()
    changed[visible] = (changed[visible] + 1) % 10
    held = selected_token_ce(step.logits2, changed, step.loss_positions).mean()
    assert torch.allclose(step.loss, held, atol=1e-12)
    changed[masked] = (layout.input_ids[masked] + 1) % 10
    moved = selected_token_ce(step.logits2, changed, step.loss_positions).mean()
    assert not torch.allclose(reference.detach(), moved.detach(), atol=1e-8)
