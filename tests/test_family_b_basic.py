"""proseco_basic: a(t) = 1/t on the corrector only, and a stop-grad draft (D-051)."""

import torch

from diffusion_correction.losses.family_b import (assemble_draft, corrector_log_probs, proseco_basic_loss,
                                                  proseco_training_step, time_weight)
from diffusion_correction.losses.masked import masked_diffusion_loss
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.sampling.candidates import greedy_allowed


def _allowed(vocab: int, forbidden: tuple[int, ...]) -> torch.Tensor:
    mask = torch.ones(vocab, dtype=torch.bool)
    for i in forbidden:
        mask[i] = False
    return mask


def test_time_weight_is_inverse_mask_rate():
    assert time_weight(0.25) == 4.0
    assert time_weight(1.0) == 1.0
    assert abs(time_weight(1.0 / 500) - 500.0) < 1e-9


def test_corrector_drops_the_mask_id_from_the_normalizer():
    row = torch.tensor([0.2, -0.4, 1.5, 0.0, 3.0], dtype=torch.float64)
    mask_id = 4
    ref = row.clone()
    ref[mask_id] = float("-inf")
    ref = torch.log_softmax(ref, dim=-1)
    got = corrector_log_probs(row.view(1, 1, -1), mask_id)[0, 0]
    keep = torch.ones(5, dtype=torch.bool)
    keep[mask_id] = False
    assert torch.allclose(got[keep], ref[keep], atol=1e-12)


def test_basic_loss_scales_by_one_over_t_and_logs_an_empty_keep_set():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1, 2]])
    draft = torch.tensor([[0, 1, 3]])  # position 1 matches, position 2 is an error; 0 is not editable
    editable = torch.tensor([[False, True, True]])
    terms = proseco_basic_loss(logits, targets, draft, editable, [0.25], mask_token_id=4)
    ce = -corrector_log_probs(logits, 4)[0, 1:, :].gather(-1, targets[0, 1:].unsqueeze(-1)).squeeze(-1)
    manual = (4.0 * ce).sum() / 2
    assert torch.allclose(terms.loss, manual, atol=1e-12)
    assert terms.n_error == 1 and terms.n_keep == 1
    terms.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_assemble_draft_replaces_every_editable_token():
    context = torch.tensor([[8, 1, 9, 2]])  # 9 stands in for a mask sitting on a response token
    editable = torch.tensor([[False, True, True, True]])
    candidate = torch.tensor([[7, 3, 4, 5]])
    draft = assemble_draft(context, editable, candidate)
    assert torch.equal(draft, torch.tensor([[8, 3, 4, 5]]))
    assert int(draft[0, 2]) != 9


def test_permuting_gold_under_the_mask_changes_the_loss_and_not_the_draft():
    adapter = build_tiny_adapter(seed=0, dtype=torch.float64)
    pad, mask_id = TINY_TOKENS.pad_token_id, TINY_TOKENS.mask_token_id
    layout = build_layout([[1, 2]], [[3, 4, 5, 6]], pad)
    masked = torch.zeros_like(layout.editable_mask)
    editable_idx = layout.editable_mask[0].nonzero().squeeze(-1)
    masked[0, editable_idx[:2]] = True  # two masked response tokens, two visible
    allowed = _allowed(TINY_TOKENS.vocab_size, (mask_id, pad, TINY_TOKENS.eos_token_id))
    first = proseco_training_step(adapter, layout, masked, [0.5], allowed)
    assert torch.equal(first.z_ids[masked], torch.full((int(masked.sum()),), mask_id))
    changed = layout.input_ids.clone()
    changed[masked] = (changed[masked] + 3) % 10
    second = proseco_training_step(adapter, layout.with_input_ids(changed), masked, [0.5], allowed)
    assert torch.equal(first.z_ids, second.z_ids)
    assert torch.equal(first.first_tokens, second.first_tokens)
    assert torch.equal(first.draft_ids, second.draft_ids)
    assert not torch.allclose(first.loss, second.loss)
    # The corrector input is the argmax at every editable position, including visible ones.
    cand, _ = greedy_allowed(first.logits1.detach(), allowed)
    assert torch.equal(first.draft_ids[layout.editable_mask], cand[layout.editable_mask])
    assert not first.draft_ids.requires_grad
    assert not first.first_tokens.requires_grad
    assert not first.log_c.requires_grad


def test_corrector_term_does_not_gradient_the_first_forward_or_scale_l_mask():
    adapter = build_tiny_adapter(seed=1, dtype=torch.float64)
    pad, mask_id = TINY_TOKENS.pad_token_id, TINY_TOKENS.mask_token_id
    layout = build_layout([[1, 2]], [[3, 4, 5]], pad)
    masked = layout.editable_mask.clone()
    masked[0, 2] = False  # one visible response token, so the draft still rewrites it
    allowed = _allowed(TINY_TOKENS.vocab_size, (mask_id, pad, TINY_TOKENS.eos_token_id))
    flags = []
    real = adapter.aligned_logits

    def wrapped(lay):
        flags.append(torch.is_grad_enabled())
        return real(lay)

    adapter.aligned_logits = wrapped
    step = proseco_training_step(adapter, layout, masked, [0.5], allowed)
    assert flags == [True, True]
    reference = masked_diffusion_loss(step.logits1, layout.input_ids, masked, layout.editable_mask)
    assert torch.allclose(step.l_mask, reference.loss, atol=1e-12)
    # 1/t = 2 is recorded on the corrector. L_mask stays the unweighted pooled CE.
    assert step.parts["a_t"] == [2.0]
    assert abs(step.parts["L_mask"] - float(reference.loss.detach())) < 1e-12
    step.logits1.retain_grad()
    step.logits2.retain_grad()
    step.l_corr.backward(retain_graph=True)
    assert step.logits1.grad is None
    assert step.logits2.grad is not None and torch.isfinite(step.logits2.grad).all()
    step.l_mask.backward()
    assert step.logits1.grad is not None and torch.isfinite(step.logits1.grad).all()
