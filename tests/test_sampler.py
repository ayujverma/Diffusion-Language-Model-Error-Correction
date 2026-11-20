"""D-029 monotone sampler: quotas, immutability, ties, completion, no reference leakage."""

import copy
import math

import pytest
import torch
from torch import nn

from diffusion_correction.evaluation.generation import generate, public_response_length
from diffusion_correction.models.adapter import DiffusionAdapter
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.sampling.monotone import allowed_token_mask, monotone_unmask, reveal_schedule

from tiny_tasks import SLOT_LEN, make_tiny_examples

MASK, PAD, EOS = TINY_TOKENS.mask_token_id, TINY_TOKENS.pad_token_id, TINY_TOKENS.eos_token_id
V = TINY_TOKENS.vocab_size


def tiny_allowed():
    return allowed_token_mask(V, V, special_ids=[PAD, EOS], extra_forbidden=[MASK])


@pytest.mark.parametrize("L,T,expected", [
    (10, 1, [10]), (10, 2, [5, 5]), (10, 4, [3, 2, 3, 2]), (10, 8, [2, 1, 1, 1, 2, 1, 1, 1]),
    (10, 10, [1] * 10), (20, 8, [3, 2, 3, 2, 3, 2, 3, 2]), (30, 4, [8, 7, 8, 7]), (7, 3, [3, 2, 2]),
])
def test_reveal_schedule_hand_values(L, T, expected):
    assert reveal_schedule(L, T) == expected


def test_reveal_schedule_invariants_and_infeasible_budgets():
    for L in range(1, 61):
        for T in range(1, L + 1):
            sch = reveal_schedule(L, T)
            assert len(sch) == T and sum(sch) == L and min(sch) >= 1
            assert all(sum(sch[:s]) == math.ceil(s * L / T) for s in range(1, T + 1))
    for L, T in ((10, 11), (10, 0), (5, -1)):
        with pytest.raises(ValueError, match="infeasible"):
            reveal_schedule(L, T)


def _run(adapter, exs, T, **kw):
    return monotone_unmask(adapter, [e.prompt_ids for e in exs],
                           [public_response_length(e, SLOT_LEN) for e in exs], T, tiny_allowed(), **kw)


@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_traces_quotas_prompt_pad_and_no_revision(T):
    adapter = build_tiny_adapter(seed=1)
    exs = make_tiny_examples(6, seed=2, n_slots_choices=(2, 3, 4))  # L in {4, 6, 8}, ragged batch
    res = _run(adapter, exs, T, record_traces=True)
    init, final = res.initial_input_ids, res.final_input_ids
    for b, ex in enumerate(exs):
        L, P = public_response_length(ex, SLOT_LEN), len(ex.prompt_ids)
        assert final[b, :P].tolist() == ex.prompt_ids                       # prompt unchanged
        assert init[b, P:P + L].tolist() == [MASK] * L                      # starts fully masked
        assert final[b, P + L:].tolist() == [PAD] * (final.shape[1] - P - L)  # padding unchanged
        trace = res.traces[b]
        assert len(trace) == T == res.calls_per_example[b]
        prev = [MASK] * L
        for s, state in enumerate(trace, start=1):
            revealed = [i for i in range(L) if prev[i] == MASK and state[i] != MASK]
            assert len(revealed) == res.schedule[b][s - 1] == reveal_schedule(L, T)[s - 1]
            assert all(state[i] == prev[i] for i in range(L) if prev[i] != MASK)  # no revision
            assert sum(t != MASK for t in state) == math.ceil(s * L / T)
            prev = state
        assert MASK not in trace[-1] and trace[-1] == res.responses[b]      # complete, no extra call
    assert res.batched_forward_calls == T
    assert all(bool(tiny_allowed()[t]) for r in res.responses for t in r)


class ConstModel(nn.Module):
    """Returns the same raw logits row at every position (forces ties)."""

    def __init__(self, row):
        super().__init__()
        self.row = nn.Parameter(torch.tensor(row, dtype=torch.float32))
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        B, L = input_ids.shape
        return self.row.expand(B, L, -1).clone()


def test_tie_rules_smallest_token_then_leftmost_position():
    row = [0.0] * V
    row[3] = row[5] = 2.0      # tied maximal candidates -> token 3
    row[MASK] = 9.0            # forbidden even though largest
    row[EOS] = 9.0
    model = ConstModel(row)
    adapter = DiffusionAdapter(model=model, tokens=TINY_TOKENS)
    res = monotone_unmask(adapter, [[10, 11, 12]], [5], 2, tiny_allowed(), record_traces=True)
    # Every position has the same score: call 1 reveals the 3 leftmost, call 2 the rest.
    assert res.traces[0][0] == [3, 3, 3, MASK, MASK]
    assert res.traces[0][1] == [3] * 5
    assert model.calls == 2 == res.batched_forward_calls
    ev = res.events[0]
    raw = torch.log_softmax(torch.tensor(row), -1)[3].item()
    allowed = tiny_allowed()
    pol = torch.log_softmax(torch.tensor(row).masked_fill(~allowed, float("-inf")), -1)[3].item()
    assert ev.raw_logp == pytest.approx(raw, abs=1e-6) and ev.policy_logp == pytest.approx(pol, abs=1e-6)
    assert ev.raw_logp < ev.policy_logp  # raw and restricted probabilities are kept distinct


def test_ranking_uses_raw_full_vocabulary_score_not_restricted_policy():
    """Position scores differ only through forbidden mass in the unrestricted normaliser."""

    class PosModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            B, L = input_ids.shape
            out = torch.zeros(B, L, V)
            out[..., 4] = 1.0
            # Raw row i-1 predicts position i. Give response position 2 (raw row P+1) huge forbidden mass.
            out[:, 4, MASK] = 8.0
            return out

    adapter = DiffusionAdapter(model=PosModel(), tokens=TINY_TOKENS)
    res = monotone_unmask(adapter, [[10, 11, 12]], [4], 4, tiny_allowed(), record_traces=True)
    # Restricted probabilities are identical everywhere; the raw score is lowest at position 2,
    # so it is revealed last; others go left to right.
    order = [next(i for i in range(4) if t[i] != MASK and (s == 0 or res.traces[0][s - 1][i] == MASK))
             for s, t in enumerate(res.traces[0])]
    assert order == [0, 1, 3, 2]


def test_ranking_sort_key_is_float32_log_softmax_not_probability():
    """The unchanged D-029 sampler ranks by float32 log_softmax (raw_logp); it never calls exp.

    D-029's decision text said "probability"; the code that satisfied it (and every generation
    since, including job 3595133) sorts by ``-score`` with ``score = log_softmax(rows)[cand]``.
    Ranking by float32 ``exp(score)`` is not equivalent: distinct logs can share one float32
    probability (underflow to 0, or other non-injective exp), so ties and then position order
    would differ. Generation must not be rewritten to rank by probability.
    """
    # Float32 fact used below: two distinct logs both exp to 0.
    lo = torch.tensor(-120.0, dtype=torch.float32)
    hi = torch.tensor(-104.0, dtype=torch.float32)
    assert float(lo) != float(hi)
    assert float(torch.exp(lo)) == 0.0 == float(torch.exp(hi))

    class UnderflowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            B, L = input_ids.shape
            out = torch.full((B, L, V), -1.0e9, dtype=torch.float32)
            out[..., 4] = 0.0
            # Prompt length 3: response pos 0 <- raw row 2; pos 1 <- raw row 3.
            # Worse log at the LEFT position so log-ranking and prob-ranking (tie -> left) disagree.
            out[:, 2, MASK] = 120.0   # log_softmax(cand) ≈ -120 at response pos 0
            out[:, 3, MASK] = 104.0   # log_softmax(cand) ≈ -104 at response pos 1
            return out

    adapter = DiffusionAdapter(model=UnderflowModel(), tokens=TINY_TOKENS)
    # T=2: call 1 reveals one position. Log ranking picks pos 1; float32-prob ranking would
    # tie at 0 and then pick the left position (pos 0).
    res = monotone_unmask(adapter, [[10, 11, 12]], [2], 2, tiny_allowed(), record_traces=True)
    assert res.traces[0][0] == [MASK, 4]
    assert res.traces[0][1] == [4, 4]
    ev = {e.position: e for e in res.events}
    assert ev[1].raw_logp > ev[0].raw_logp
    assert float(torch.exp(torch.tensor(ev[0].raw_logp, dtype=torch.float32))) == 0.0
    assert float(torch.exp(torch.tensor(ev[1].raw_logp, dtype=torch.float32))) == 0.0
    # Stored score is the log value, not the probability.
    row1 = torch.full((V,), -1.0e9, dtype=torch.float32)
    row1[4] = 0.0
    row1[MASK] = 104.0
    assert ev[1].raw_logp == float(torch.log_softmax(row1, -1)[4])
    assert ev[1].raw_logp != float(torch.softmax(row1, -1)[4])


def test_outputs_do_not_depend_on_reference_answers():
    adapter = build_tiny_adapter(seed=4)
    exs = make_tiny_examples(8, seed=5)
    allowed = tiny_allowed()
    base = generate(adapter, exs, 2, SLOT_LEN, allowed, batch_size=3)
    altered = copy.deepcopy(exs)
    for e in altered:
        e.response_ids = [(t + 3) % 10 for t in e.response_ids]
        e.values = [v + 1 for v in e.values]
        e.final_answer += 7
    again = generate(adapter, altered, 2, SLOT_LEN, allowed, batch_size=3)
    assert again.responses == base.responses and again.calls == base.calls


def test_one_token_per_call_and_infeasible_budget_reporting():
    adapter = build_tiny_adapter(seed=6)
    exs = make_tiny_examples(6, seed=7, n_slots_choices=(2, 4))  # L = 4 or 8
    ref = generate(adapter, exs, "L", SLOT_LEN, tiny_allowed(), batch_size=4)
    assert all(ref.calls[i] == public_response_length(e, SLOT_LEN) for i, e in enumerate(exs))
    big = generate(adapter, exs, 6, SLOT_LEN, tiny_allowed())
    short = [i for i, e in enumerate(exs) if public_response_length(e, SLOT_LEN) < 6]
    assert sorted(big.infeasible) == short and not set(big.responses) & set(short)


def test_observer_sees_commitment_pass_without_changing_output():
    adapter = build_tiny_adapter(seed=8)
    exs = make_tiny_examples(3, seed=9)
    seen = []

    def observer(call, ids, logits, events):
        rows = torch.log_softmax(logits.float(), -1)
        for ev in events:
            pos = len(exs[ev.example].prompt_ids) + ev.position
            assert ids[ev.example, pos].item() == MASK          # committed from a masked state
            assert rows[ev.example, pos, ev.token_id].item() == pytest.approx(ev.raw_logp, abs=1e-6)
        seen.append(call)

    plain = _run(adapter, exs, 3)
    watched = _run(adapter, exs, 3, observer=observer)
    assert plain.responses == watched.responses and seen == [1, 2, 3]
