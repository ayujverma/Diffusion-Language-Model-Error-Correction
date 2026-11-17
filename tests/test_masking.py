"""D-028 mask policy and the resumable data sampler (no downloaded assets needed)."""

import torch
import pytest

from diffusion_correction.models.alignment import build_layout
from diffusion_correction.training.data import EpochSampler, select_subset
from diffusion_correction.training.masking import T_MAX, T_MIN, apply_masks, sample_response_masks
from diffusion_correction.training.rng import make_generator, stream_seed

from tiny_tasks import make_tiny_examples

PAD, MASK = 17, 18


def _layout(n=64, seed=0):
    exs = make_tiny_examples(n, seed)
    return build_layout([e.prompt_ids for e in exs], [e.response_ids for e in exs], PAD)


def test_masks_only_touch_editable_response_positions():
    lay = _layout()
    draw = sample_response_masks(lay, make_generator(1, "train_mask"))
    assert not bool((draw.masked_positions & ~lay.editable_mask).any())
    ids = apply_masks(lay.input_ids, draw.masked_positions, MASK)
    assert torch.equal(ids[~lay.editable_mask], lay.input_ids[~lay.editable_mask])  # prompt and pad unchanged
    assert draw.n_masked == draw.masked_positions.sum(1).tolist()


def test_rate_is_clamped_uniform_drawn_per_example():
    lay = _layout(n=4000)
    draw = sample_response_masks(lay, make_generator(2, "train_mask"))
    assert all(T_MIN <= t <= T_MAX for t in draw.t)
    assert all(t == min(max(u, T_MIN), T_MAX) for u, t in zip(draw.u, draw.t))
    assert len(set(draw.t)) > 3900  # one draw per example, not per batch
    mean_t = sum(draw.t) / len(draw.t)
    assert abs(mean_t - 0.5) < 0.03  # E[clamp(u)] = 0.5 by symmetry


def test_bernoulli_masking_and_kept_zero_mask_examples():
    """Mask fraction tracks t, and zero-mask examples are kept, at about E[(1-t)^L]."""
    lay = _layout(n=6000, seed=3)
    draw = sample_response_masks(lay, make_generator(3, "train_mask"))
    lengths = lay.editable_mask.sum(1).tolist()
    frac = [m / n for m, n in zip(draw.n_masked, lengths)]
    lo = [f for f, t in zip(frac, draw.t) if t < 0.2]
    hi = [f for f, t in zip(frac, draw.t) if t > 0.8]
    assert sum(lo) / len(lo) < 0.15 and sum(hi) / len(hi) > 0.85
    zero = sum(m == 0 for m in draw.n_masked)
    expected = sum(1 / (n + 1) for n in lengths)  # E[(1-u)^n] = 1/(n+1), ignoring the clamp
    assert zero > 0 and abs(zero - expected) < 4 * expected ** 0.5 + 10
    assert len(draw.n_masked) == 6000  # nothing resampled or dropped


def test_extreme_rates_and_determinism():
    lay = _layout()
    none = sample_response_masks(lay, make_generator(0, "m"), t_min=0.0, t_max=0.0)
    assert none.total_masked == 0
    allm = sample_response_masks(lay, make_generator(0, "m"), t_min=1.0, t_max=1.0)
    assert torch.equal(allm.masked_positions, lay.editable_mask)
    a = sample_response_masks(lay, make_generator(5, "train_mask"))
    b = sample_response_masks(lay, make_generator(5, "train_mask"))
    assert torch.equal(a.masked_positions, b.masked_positions) and a.t == b.t


def test_streams_are_independent():
    assert len({stream_seed(7, s) for s in ("data_order", "train_mask", "eval")}) == 3
    lay = _layout()
    g = make_generator(7, "train_mask")
    first = sample_response_masks(lay, g)
    g2 = make_generator(7, "train_mask")
    _ = torch.rand(100, generator=make_generator(7, "eval"))  # using another stream changes nothing
    again = sample_response_masks(lay, g2)
    assert torch.equal(first.masked_positions, again.masked_positions)


def test_epoch_sampler_covers_each_epoch_and_resumes_exactly():
    s = EpochSampler(10, make_generator(0, "data_order"))
    drawn = s.next_indices(4) + s.next_indices(4) + s.next_indices(4)  # straddles the epoch boundary
    assert sorted(drawn[:10]) == list(range(10))
    state = s.state_dict()
    gstate = s.generator.get_state()
    cont = [s.next_indices(3) for _ in range(5)]
    s2 = EpochSampler(10, make_generator(99, "other"))
    s2.generator.set_state(gstate)
    s2.load_state_dict(state)
    assert [s2.next_indices(3) for _ in range(5)] == cont
    with pytest.raises(ValueError):
        EpochSampler(11, make_generator(0, "x")).load_state_dict(state)


def test_subset_is_stratified_and_deterministic():
    exs = make_tiny_examples(200, 1)
    a, b = select_subset(exs, 12, 0), select_subset(exs, 12, 0)
    assert [e.problem_id for e in a] == [e.problem_id for e in b]
    keys = {(e.family, e.setting["length"]) for e in a}
    assert len(keys) == 6  # 2 families x 3 lengths all represented
    assert [e.problem_id for e in select_subset(exs, 12, 1)] != [e.problem_id for e in a]
