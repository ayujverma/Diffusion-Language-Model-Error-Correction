"""proseco_gen_conf ranks stop-grad log c of the committed token, not q (D-046, D-051)."""

import torch

from diffusion_correction.losses.family_b import (corrector_log_probs, proseco_gen_conf_loss,
                                                  proseco_training_step)
from diffusion_correction.losses.weighting import DEFAULT_STAGE_EDGES, rank_weights, stage_bin
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.sampling.candidates import token_raw_logp


def _weights(logits, targets, draft, editable, log_c, t=(0.5,)):
    terms = proseco_gen_conf_loss(logits, targets, draft, editable, t, mask_token_id=3, log_c=log_c)
    return terms


def test_weights_ignore_the_gold_token_and_the_second_forward():
    logits = torch.zeros(1, 3, 4, dtype=torch.float64)
    logits[0, 1] = torch.tensor([3.0, 0.0, 3.0, 0.0])  # gold id 1 is the weak class
    logits[0, 2] = torch.tensor([0.0, 4.0, 0.0, 0.0])  # gold id 1 is the strong class
    logits.requires_grad_(True)
    targets = torch.tensor([[0, 1, 1]])
    draft = torch.tensor([[0, 2, 2]])
    editable = torch.tensor([[False, True, True]])
    log_c = torch.tensor([[0.0, -0.2, -3.0]], dtype=torch.float64, requires_grad=True)
    terms = _weights(logits, targets, draft, editable, log_c)
    assert float(terms.error_weights[0]) > float(terms.error_weights[1])
    terms.loss.backward()
    assert log_c.grad is None and not terms.error_weights.requires_grad

    other_gold = targets.clone()
    other_gold[0, 1:] = 0  # still wrong: the draft token is 2
    other_logits = logits + 1.5
    again = _weights(other_logits, other_gold, draft, editable, log_c.detach())
    assert torch.allclose(terms.error_weights, again.error_weights, atol=0.0)

    flipped = torch.tensor([[0.0, -3.0, -0.2]], dtype=torch.float64)
    swapped = _weights(logits, targets, draft, editable, flipped)
    assert float(swapped.error_weights[0]) < float(swapped.error_weights[1])

    gold_logp = corrector_log_probs(logits, 3)
    q = torch.stack([gold_logp[0, 1, 1], gold_logp[0, 2, 1]]).detach().double()
    assert float(q[1]) > float(q[0])
    groups = stage_bin(torch.tensor([0.5, 0.5]), DEFAULT_STAGE_EDGES)
    q_weights = rank_weights(q, groups, 0.5)
    assert float(q_weights[1]) > float(q_weights[0])
    assert not torch.allclose(terms.error_weights.double(), q_weights, atol=1e-5)


def test_raising_one_log_c_raises_that_rows_weight_and_scales_its_gradient():
    def run(scores):
        base = torch.tensor([0.0, 0.4, -0.3, 0.2, 0.7], dtype=torch.float64)
        logits = base.view(1, 1, -1).repeat(1, 4, 1).clone().requires_grad_(True)
        targets = torch.tensor([[0, 1, 1, 1]])
        draft = torch.tensor([[0, 2, 2, 2]])
        editable = torch.tensor([[False, True, True, True]])
        log_c = torch.tensor([[0.0, *scores]], dtype=torch.float64, requires_grad=True)
        terms = proseco_gen_conf_loss(logits, targets, draft, editable, [0.5], 4, log_c)
        terms.loss.backward()
        assert log_c.grad is None
        return terms.error_weights, logits.grad[0, 1:].detach()

    w0, g0 = run([-2.0, -2.0, -1.0])
    w1, g1 = run([0.0, -2.0, -1.0])
    assert float(w1[0]) > float(w0[0])
    ratio_w = float(w1[0] / w0[0])
    ratio_g = float(g1[0].norm() / g0[0].norm())
    assert abs(ratio_g - ratio_w) < 1e-5


def test_groups_are_mask_rate_bins_inside_the_minibatch():
    logits = torch.randn(2, 2, 4, dtype=torch.float64)
    targets = torch.tensor([[0, 1], [0, 1]])
    draft = torch.tensor([[0, 2], [0, 2]])
    editable = torch.tensor([[False, True], [False, True]])
    log_c = torch.tensor([[0.0, -0.01], [0.0, -5.0]], dtype=torch.float64)
    terms = proseco_gen_conf_loss(logits, targets, draft, editable, [0.1, 0.9], 3, log_c)
    assert torch.allclose(terms.error_weights, torch.ones(2), atol=1e-6)
    # The same scores in one bin would not be uniform.
    one_bin = stage_bin(torch.tensor([0.5, 0.5]), DEFAULT_STAGE_EDGES)
    spread = rank_weights(torch.tensor([-0.01, -5.0]), one_bin, 0.5)
    assert float(spread.max() - spread.min()) > 0.4


def test_keep_term_stays_cross_entropy_and_an_empty_error_set_backprops():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.2, 1.5, -0.4, 0.1]]], dtype=torch.float64,
                          requires_grad=True)
    targets = torch.tensor([[0, 1]])
    draft = torch.tensor([[0, 1]])
    editable = torch.tensor([[False, True]])
    log_c = torch.zeros(1, 2)
    terms = proseco_gen_conf_loss(logits, targets, draft, editable, [1.0], 3, log_c)
    logp = corrector_log_probs(logits, 3)[0, 1, 1]
    assert terms.error_empty and terms.n_keep == 1
    assert torch.allclose(terms.loss, -logp, atol=1e-12)
    assert not torch.allclose(terms.loss, -logp.exp(), atol=1e-6)
    terms.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_step_weights_the_first_forward_committed_token():
    adapter = build_tiny_adapter(seed=4, dtype=torch.float64)
    pad, mask_id = TINY_TOKENS.pad_token_id, TINY_TOKENS.mask_token_id
    layout = build_layout([[1, 2]], [[3, 4, 5, 6]], pad)
    masked = layout.editable_mask.clone()
    allowed = torch.ones(TINY_TOKENS.vocab_size, dtype=torch.bool)
    allowed[mask_id] = allowed[pad] = allowed[TINY_TOKENS.eos_token_id] = False
    step = proseco_training_step(adapter, layout, masked, [0.4], allowed, arm="proseco_gen_conf")
    assert torch.allclose(step.log_c, token_raw_logp(step.logits1.detach(), step.first_tokens), atol=1e-6)
    manual = proseco_gen_conf_loss(step.logits2, layout.input_ids, step.draft_ids, layout.editable_mask,
                                   [0.4], mask_id, step.log_c)
    assert torch.allclose(step.l_corr, manual.loss, atol=1e-12)
    assert step.parts["weight_diagnostics"]["beta"] == 0.5 or step.parts["weight_diagnostics"]["n_errors"] == 0
