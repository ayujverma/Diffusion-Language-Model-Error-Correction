"""Gate B: loss values and gradients against independent references."""

import numpy as np
import pytest
import torch

from diffusion_correction.losses import (combined_objective, compute_error_weights, correction_losses,
                                         masked_diffusion_loss, selected_token_ce, token_error_masks)
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS

V = 7


def np_ce(logits_row: np.ndarray, target: int) -> float:
    m = logits_row.max()
    return float(m + np.log(np.exp(logits_row - m).sum()) - logits_row[target])


def np_softmax(row: np.ndarray) -> np.ndarray:
    e = np.exp(row - row.max())
    return e / e.sum()


@pytest.fixture
def fixture_batch():
    """Aligned-logit leaf tensor plus draft/target for a padded 2-example batch.

    Example 0: prompt 2, response 4. Example 1: prompt 3, response 2, 1 pad.
    Draft errors: ex0 positions 3 and 5; ex1 position 4. Pad/prompt positions also
    'differ' from targets to prove they are excluded.
    """
    gen = torch.Generator().manual_seed(1)
    lay = build_layout([[1, 2], [3, 4, 5]], [[0, 1, 2, 3], [4, 5]], pad_token_id=6, total_length=6)
    targets = lay.input_ids.clone()
    draft = targets.clone()
    draft[0, 3] = (targets[0, 3] + 1) % V
    draft[0, 5] = (targets[0, 5] + 2) % V
    draft[1, 4] = (targets[1, 4] + 3) % V
    draft[0, 0] = (targets[0, 0] + 1) % V  # prompt position: must be ignored
    draft[1, 5] = (targets[1, 5] + 1) % V  # pad position: must be ignored
    logits = torch.randn(2, 6, V, generator=gen, dtype=torch.float64, requires_grad=True)
    return lay, targets, draft, logits


def test_token_ce_matches_numpy_reference(fixture_batch):
    lay, targets, _, logits = fixture_batch
    ce = selected_token_ce(logits, targets, lay.editable_mask)
    L = logits.detach().numpy()
    ref = [np_ce(L[b, i], int(targets[b, i])) for b in range(2) for i in range(6) if lay.editable_mask[b, i]]
    np.testing.assert_allclose(ce.detach().numpy(), ref, rtol=0, atol=1e-12)


def test_error_keep_masks_exclude_prompt_and_pad(fixture_batch):
    lay, targets, draft, _ = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    assert E.nonzero().tolist() == [[0, 3], [0, 5], [1, 4]]
    assert K.nonzero().tolist() == [[0, 2], [0, 4], [1, 3]]
    assert not bool((E | K)[lay.prompt_mask | lay.pad_mask].any())


def test_total_loss_matches_reference(fixture_batch):
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    w = torch.tensor([0.5, 1.5, 1.0], dtype=torch.float64)
    corr = correction_losses(logits, targets, E, K, weights=w)
    masked = torch.zeros_like(lay.editable_mask)
    masked[0, 2] = masked[1, 3] = True
    lm = masked_diffusion_loss(logits, targets, masked, lay.editable_mask)
    total, parts = combined_objective(lm.loss, corr, lam=0.7, mu=0.3)

    L = logits.detach().numpy()
    ce = lambda b, i: np_ce(L[b, i], int(targets[b, i]))
    ref_err = (0.5 * ce(0, 3) + 1.5 * ce(0, 5) + 1.0 * ce(1, 4)) / 3
    ref_keep = (ce(0, 2) + ce(0, 4) + ce(1, 3)) / 3
    ref_mask = (ce(0, 2) + ce(1, 3)) / 2
    assert abs(corr.l_error.item() - ref_err) < 1e-12
    assert abs(corr.l_keep.item() - ref_keep) < 1e-12
    assert abs(lm.loss.item() - ref_mask) < 1e-12
    assert abs(total.item() - (ref_mask + 0.7 * ref_err + 0.3 * ref_keep)) < 1e-12
    assert parts["n_error"] == 3 and parts["n_keep"] == 3


def test_autograd_matches_analytic_gradient(fixture_batch):
    """d/dz_i of lam*L_error + mu*L_keep = coef_i * (softmax(z_i) - onehot(target_i))."""
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    w = torch.tensor([0.25, 1.75, 1.0], dtype=torch.float64)
    lam, mu = 0.7, 0.3
    corr = correction_losses(logits, targets, E, K, weights=w)
    (lam * corr.l_error + mu * corr.l_keep).backward()

    L = logits.detach().numpy()
    expected = np.zeros_like(L)
    err_pos = E.nonzero().tolist()
    keep_pos = K.nonzero().tolist()
    for (b, i), wi in zip(err_pos, w.tolist()):
        expected[b, i] = lam * wi * (np_softmax(L[b, i]) - np.eye(V)[targets[b, i]]) / len(err_pos)
    for b, i in keep_pos:
        expected[b, i] += mu * (np_softmax(L[b, i]) - np.eye(V)[targets[b, i]]) / len(keep_pos)
    np.testing.assert_allclose(logits.grad.numpy(), expected, rtol=0, atol=1e-14)
    # Prompt, pad and position-0 rows receive exactly zero gradient.
    assert torch.count_nonzero(logits.grad[lay.prompt_mask | lay.pad_mask]) == 0


def _error_row_grads(fixture_batch, weights):
    lay, targets, draft, logits = fixture_batch
    logits = logits.detach().clone().requires_grad_(True)
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    correction_losses(logits, targets, E, K, weights=weights).l_error.backward()
    return logits.grad, E


def test_zero_weight_zeros_that_rows_direct_gradient(fixture_batch):
    grad, E = _error_row_grads(fixture_batch, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    assert torch.count_nonzero(grad[0, 3]) == 0
    assert torch.count_nonzero(grad[0, 5]) > 0


def test_weight_scales_rows_direct_gradient(fixture_batch):
    g1, _ = _error_row_grads(fixture_batch, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    g3, _ = _error_row_grads(fixture_batch, torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(g3[0, 3], 3.0 * g1[0, 3], rtol=0, atol=1e-15)
    torch.testing.assert_close(g3[0, 5], g1[0, 5], rtol=0, atol=0)


def test_all_one_weights_equal_unweighted_mean(fixture_batch):
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    ones = correction_losses(logits, targets, E, K, weights=torch.ones(3, dtype=torch.float64))
    none = correction_losses(logits, targets, E, K)
    plain = selected_token_ce(logits, targets, E).mean()
    assert ones.l_error.item() == none.l_error.item()
    assert abs(none.l_error.item() - plain.item()) < 1e-15


def test_beta_zero_recovers_uniform(fixture_batch):
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    groups = torch.tensor([0, 0, 0])
    log_c = torch.tensor([-0.1, -2.0, -0.5])
    w = compute_error_weights("high_conf", groups, beta=0.0, log_c=log_c).weights
    assert torch.equal(w, torch.ones(3))
    a = correction_losses(logits, targets, E, K, weights=w.double())
    b = correction_losses(logits, targets, E, K)
    assert a.l_error.item() == b.l_error.item()


def test_weights_with_grad_are_rejected(fixture_batch):
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    with pytest.raises(ValueError):
        correction_losses(logits, targets, E, K, weights=torch.ones(3, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError):
        correction_losses(logits, targets, E, K, weights=torch.ones(2, dtype=torch.float64))


def test_current_loss_weights_act_as_constants(fixture_batch):
    """The hard-example rule ranks detached CE; its gradient must equal the fixed-weight formula."""
    lay, targets, draft, logits = fixture_batch
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    groups = torch.zeros(3, dtype=torch.long)
    seen = {}

    def rule(ce):
        assert not ce.requires_grad
        seen["w"] = compute_error_weights("current_loss", groups, beta=0.5, current_ce=ce).weights.double()
        return seen["w"]

    correction_losses(logits, targets, E, K, weights=rule).l_error.backward()
    auto = logits.grad.clone()
    fixed, _ = _error_row_grads(fixture_batch, seen["w"])
    torch.testing.assert_close(auto, fixed, rtol=0, atol=1e-15)
    # Largest current CE receives the largest weight.
    ce = selected_token_ce(logits.detach(), targets, E)
    assert int(seen["w"].argmax()) == int(ce.argmax())


def test_empty_sets_give_finite_zero_and_backprop(fixture_batch):
    lay, targets, _, logits = fixture_batch
    # All-correct draft: no errors.
    E, K = token_error_masks(targets, targets, lay.editable_mask)
    corr = correction_losses(logits, targets, E, K)
    assert corr.n_error == 0 and corr.l_error.item() == 0.0
    # All-wrong draft: no keeps.
    wrong = torch.where(lay.editable_mask, (targets + 1) % V, targets)
    E2, K2 = token_error_masks(wrong, targets, lay.editable_mask)
    corr2 = correction_losses(logits, targets, E2, K2)
    assert corr2.n_keep == 0 and corr2.l_keep.item() == 0.0
    # No masked positions.
    lm = masked_diffusion_loss(logits, targets, torch.zeros_like(lay.editable_mask), lay.editable_mask)
    assert lm.n_masked == 0 and lm.loss.item() == 0.0
    total, parts = combined_objective(lm.loss, corr, lam=1.0, mu=1.0)
    total.backward()
    assert parts["error_set_empty"] and torch.isfinite(logits.grad).all()


def test_masked_loss_rejects_prompt_or_pad_positions(fixture_batch):
    lay, targets, _, logits = fixture_batch
    bad = torch.zeros_like(lay.editable_mask)
    bad[0, 1] = True  # prompt
    with pytest.raises(ValueError):
        masked_diffusion_loss(logits, targets, bad, lay.editable_mask)
    bad = torch.zeros_like(lay.editable_mask)
    bad[1, 5] = True  # pad
    with pytest.raises(ValueError):
        masked_diffusion_loss(logits, targets, bad, lay.editable_mask)


def _sgd_delta(adapter, weight: float):
    """One SGD step on L_error for a single wrong token with a FIXED weight."""
    model = adapter.model
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    lay = build_layout([[1, 2, 3]], [[4, 5, 6]], pad_token_id=TINY_TOKENS.pad_token_id)
    targets = lay.input_ids.clone()
    draft = targets.clone()
    draft[0, 4] = 9
    E, K = token_error_masks(draft, targets, lay.editable_mask)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad()
    logits = adapter.aligned_logits(lay.with_input_ids(draft))
    corr = correction_losses(logits, targets, E, K, weights=torch.tensor([weight], dtype=torch.float64))
    corr.l_error.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    delta = {n: p.detach() - start[n] for n, p in model.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(start[n])
    return delta


def test_weight_scales_optimizer_update_on_tiny_model(tiny):
    d1 = _sgd_delta(tiny, 1.0)
    d2 = _sgd_delta(tiny, 2.0)
    d0 = _sgd_delta(tiny, 0.0)
    total = sum(float(d.abs().sum()) for d in d1.values())
    assert total > 0
    for n in d1:
        torch.testing.assert_close(d2[n], 2.0 * d1[n], rtol=1e-9, atol=1e-13)
        assert torch.count_nonzero(d0[n]) == 0
