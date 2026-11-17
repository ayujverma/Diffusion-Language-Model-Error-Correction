"""Gate B: rank-weight rules (hand-computed values, signs, ties, bins, shuffling)."""

import pytest
import torch

from diffusion_correction.losses import (WeightRule, compute_error_weights, make_group_ids,
                                         midranks_within_groups, rank_weights, stage_bin)


def test_midranks_average_ties():
    ranks, sizes = midranks_within_groups(torch.tensor([3.0, 1.0, 2.0, 2.0]), torch.zeros(4, dtype=torch.long))
    assert ranks.tolist() == [4.0, 1.0, 2.5, 2.5]
    assert sizes.tolist() == [4.0] * 4


def test_rank_weights_hand_values():
    # ranks 1..4 -> r = .125 .375 .625 .875 -> w = 1 + .5(2r - 1)
    w = rank_weights(torch.tensor([10.0, 20.0, 30.0, 40.0]), torch.zeros(4, dtype=torch.long), beta=0.5)
    torch.testing.assert_close(w, torch.tensor([0.625, 0.875, 1.125, 1.375], dtype=torch.float64))


def test_singleton_and_fully_tied_groups_get_weight_one():
    groups = torch.tensor([0, 1, 1, 1])
    w = rank_weights(torch.tensor([5.0, 2.0, 2.0, 2.0]), groups, beta=0.5)
    torch.testing.assert_close(w, torch.ones(4, dtype=torch.float64))
    res = compute_error_weights("high_conf", groups, beta=0.5, log_c=torch.tensor([-1.0, -2.0, -2.0, -2.0]))
    assert res.diagnostics["n_singleton_groups"] == 1
    assert res.diagnostics["has_variation"] is False


def test_groups_are_ranked_independently():
    groups = torch.tensor([0, 0, 1, 1])
    w = rank_weights(torch.tensor([1.0, 2.0, 100.0, 200.0]), groups, beta=0.5)
    torch.testing.assert_close(w, torch.tensor([0.75, 1.25, 0.75, 1.25], dtype=torch.float64))


def test_rule_directions():
    groups = torch.zeros(3, dtype=torch.long)
    log_c = torch.tensor([-0.05, -1.0, -3.0])   # token 0 committed most confidently
    log_q = torch.tensor([-4.0, -0.5, -2.0])    # token 0 has the least likely correct target
    hi = compute_error_weights(WeightRule.HIGH_CONF, groups, 0.5, log_c=log_c).weights
    lo = compute_error_weights(WeightRule.LOW_CONF, groups, 0.5, log_c=log_c).weights
    gd = compute_error_weights(WeightRule.GEN_DIFFICULTY, groups, 0.5, log_q=log_q).weights
    assert hi.argmax() == 0 and hi.argmin() == 2
    assert lo.argmax() == 2 and lo.argmin() == 0
    torch.testing.assert_close(lo, 2.0 - hi)
    # Generation-time difficulty ranks -log q upward: the smallest q gets the largest weight.
    # Ranking q itself upward (the wrong sign) would put token 1 on top instead.
    assert gd.argmax() == 0 and gd.argmin() == 1


def test_shuffled_confidence_is_deterministic_and_preserves_weight_distribution():
    groups = torch.tensor([0] * 6 + [1] * 4)
    log_c = -torch.linspace(0.01, 3.0, 10)
    a = compute_error_weights("shuffled_conf", groups, 0.5, log_c=log_c, shuffle_seed=7)
    b = compute_error_weights("shuffled_conf", groups, 0.5, log_c=log_c, shuffle_seed=7)
    hi = compute_error_weights("high_conf", groups, 0.5, log_c=log_c)
    assert torch.equal(a.weights, b.weights)
    for g in (0, 1):
        m = groups == g
        assert torch.equal(a.weights[m].sort().values, hi.weights[m].sort().values)
    assert a.diagnostics["n_scores_moved"] > 0
    with pytest.raises(ValueError):
        compute_error_weights("shuffled_conf", groups, 0.5, log_c=log_c)


def test_current_loss_weights_are_detached():
    ce = torch.tensor([0.2, 3.0, 1.0], requires_grad=True)
    res = compute_error_weights("current_loss", torch.zeros(3, dtype=torch.long), 0.5, current_ce=ce)
    assert not res.weights.requires_grad
    assert res.weights.argmax() == 1


def test_weights_bounded_mean_one_and_diagnostics():
    gen = torch.Generator().manual_seed(0)
    groups = torch.randint(0, 4, (200,), generator=gen)
    log_c = torch.log(torch.rand(200, generator=gen))
    res = compute_error_weights("high_conf", groups, beta=0.5, log_c=log_c)
    w = res.weights
    assert float(w.min()) >= 0.5 - 1e-6 and float(w.max()) <= 1.5 + 1e-6
    for g in range(4):
        assert abs(float(w[groups == g].double().mean()) - 1.0) < 1e-6
    d = res.diagnostics
    assert d["n_errors"] == 200 and d["n_groups"] == 4 and sum(d["group_sizes"]) == 200
    assert 0 < d["ess"] <= 200
    u = compute_error_weights("uniform", groups)
    assert u.diagnostics["ess"] == pytest.approx(200.0)


def test_invalid_inputs_raise():
    g = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        rank_weights(torch.tensor([1.0, 2.0]), g, beta=1.0)
    with pytest.raises(ValueError):
        compute_error_weights("high_conf", g, 0.5)  # missing log_c
    with pytest.raises(ValueError):
        compute_error_weights("high_conf", g, 0.5, log_c=torch.tensor([float("nan"), 0.0]))


def test_stage_bins_and_group_ids():
    mf = torch.tensor([0.0, 0.1999, 0.2, 0.7999, 0.8, 1.0])
    assert stage_bin(mf).tolist() == [0, 0, 1, 3, 4, 4]
    with pytest.raises(ValueError):
        stage_bin(torch.tensor([1.01]))
    # The same mask fraction under different budgets must land in different groups.
    ids = make_group_ids(torch.tensor([4, 8, 4]), torch.tensor([0.5, 0.5, 0.55]))
    assert ids[0] == ids[2] and ids[0] != ids[1]
