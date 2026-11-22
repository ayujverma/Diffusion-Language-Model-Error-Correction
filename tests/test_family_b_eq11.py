"""Eq. 11 keep term is -p(gold), not cross-entropy (D-051)."""

import torch

from diffusion_correction.losses.family_b import proseco_error_prioritized_loss


def _row_loss(kind: str):
    """One editable logit row. ``kind`` is ``keep`` (draft == gold) or ``error``."""
    torch.manual_seed(2)
    logits = torch.randn(1, 2, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1]])
    draft = torch.tensor([[0, 1 if kind == "keep" else 2]])
    editable = torch.tensor([[False, True]])
    terms = proseco_error_prioritized_loss(logits, targets, draft, editable, [1.0], mask_token_id=4)
    terms.loss.backward()
    row = logits.detach()[0, 1].clone()
    row[4] = float("-inf")
    p = torch.softmax(row, dim=-1)
    one = torch.zeros(5, dtype=torch.float64)
    one[1] = 1.0
    ce_grad = p - one
    ce_grad[4] = 0.0
    prob_grad = p[1] * (p - one)
    prob_grad[4] = 0.0
    return terms, logits.grad[0, 1], ce_grad, prob_grad, p


def test_keep_term_matches_negative_probability_and_not_ce():
    terms, grad, ce_grad, prob_grad, p = _row_loss("keep")
    assert terms.n_keep == 1 and terms.n_error == 0 and terms.error_empty
    assert torch.allclose(terms.loss, -p[1], atol=1e-12)
    assert not torch.allclose(terms.loss, -torch.log(p[1]), atol=1e-6)
    assert torch.allclose(grad, prob_grad, atol=1e-10)
    assert not torch.allclose(grad, ce_grad, atol=1e-6)


def test_error_term_matches_ce_and_not_probability():
    terms, grad, ce_grad, prob_grad, p = _row_loss("error")
    assert terms.n_error == 1 and terms.n_keep == 0 and terms.keep_empty
    assert torch.allclose(terms.loss, -torch.log(p[1]), atol=1e-12)
    assert not torch.allclose(terms.loss, -p[1], atol=1e-6)
    assert torch.allclose(grad, ce_grad, atol=1e-10)
    assert not torch.allclose(grad, prob_grad, atol=1e-6)


def test_eq11_time_weight_is_inside_the_single_pool():
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 5, dtype=torch.float64)
    targets = torch.tensor([[0, 1], [0, 2]])
    draft = torch.tensor([[0, 1], [0, 0]])  # row 0 keep, row 1 error
    editable = torch.tensor([[False, True], [False, True]])
    terms = proseco_error_prioritized_loss(logits, targets, draft, editable, [0.5, 0.25], mask_token_id=4)
    # a = 2 and 4. Recompute the two summands and divide by 2 editable positions.
    from diffusion_correction.losses.family_b import corrector_log_probs
    logp = corrector_log_probs(logits, 4)
    p0 = logp[0, 1, 1].exp()
    ce1 = -logp[1, 1, 2]
    manual = (2.0 * (-p0) + 4.0 * ce1) / 2
    assert torch.allclose(terms.loss, manual, atol=1e-12)
    assert terms.n_error == 1 and terms.n_keep == 1
