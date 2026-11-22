"""Milestone 6 samplers: NFE accounting, ProSeCo-style correction, D3IM, extra-unmask (D-049).

Invariants, not snapshots of an arbitrary implementation:
- every model forward is one NFE, and a refused call does not run the model;
- S=0 correction unmasking matches D-029 monotone unmasking;
- the corrector is fed a temporary complete draft while the main state can still hold masks;
- a visible token can change, which monotone unmasking never does;
- D3IM can remask a visible token; the extra-unmask control scores the current token and does not;
- a shorter call budget is its own horizon, not a prefix of a longer one;
- targets do not affect the tokens; test splits are refused.
"""

import copy
import inspect

import pytest
import torch
import yaml
from torch import nn

from diffusion_correction.cli import main
from diffusion_correction.data.format import ResponseFormat, TokenizedFormat
from diffusion_correction.evaluation.call_budget import require_eval_split, run_call_budget, score_call_budget
from diffusion_correction.evaluation.repair import repair_drafts, repair_drafts_proseco
from diffusion_correction.models.adapter import DiffusionAdapter
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.sampling.candidates import greedy_allowed
from diffusion_correction.sampling.d3im import d3im_unmask, keep_count
from diffusion_correction.sampling.extra_unmask import extra_unmask
from diffusion_correction.sampling.monotone import monotone_unmask
from diffusion_correction.sampling.nfe import ForwardMeter
from diffusion_correction.sampling.proseco import (
    DEFAULT_CORRECTOR_STEPS, DEFAULT_OMEGA, completing_unmask_steps, proseco_unmask, worst_case_nfe)
from diffusion_correction.training.bank_data import BankDraft

from tiny_tasks import SLOT_LEN, make_tiny_examples

MASK, PAD, EOS = TINY_TOKENS.mask_token_id, TINY_TOKENS.pad_token_id, TINY_TOKENS.eos_token_id
V = TINY_TOKENS.vocab_size
ROOT_CFG = "configs/eval/m6_sampler.yaml"


def tiny_allowed():
    from diffusion_correction.sampling.monotone import allowed_token_mask
    return allowed_token_mask(V, V, special_ids=[PAD, EOS], extra_forbidden=[MASK])


class Scripted(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.inputs = []
        self.fn = fn

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        self.inputs.append(input_ids.detach().cpu().clone())
        return self.fn(input_ids)


def adapter_of(model) -> DiffusionAdapter:
    return DiffusionAdapter(model=model, tokens=TINY_TOKENS)


def resp(ids, prompt_len, length):
    return ids[0, prompt_len:prompt_len + length].tolist()


def test_meter_refuses_without_a_second_forward():
    model = Scripted(lambda ids: torch.zeros(ids.shape[0], ids.shape[1], V))
    adapter = adapter_of(model)
    layout = build_layout([[10, 11]], [[MASK, MASK]], PAD)
    meter = ForwardMeter(adapter, max_calls=1)
    assert meter.forward(layout, kind="predictor", step=1) is not None
    assert meter.forward(layout, kind="corrector", step=1) is None
    assert model.calls == 1
    assert meter.nfe == 1 and meter.kinds == ["predictor"]


def test_monotone_meter_records_exactly_t_and_refuses_a_short_cap():
    adapter = build_tiny_adapter(seed=0)
    prompts, lengths = [[10, 11, 12]], [4]
    plain = monotone_unmask(adapter, prompts, lengths, 3, tiny_allowed())
    meter = ForwardMeter(adapter, max_calls=3)
    recorded = monotone_unmask(adapter, prompts, lengths, 3, tiny_allowed(), meter=meter)
    assert recorded.responses == plain.responses
    assert meter.nfe == 3 and meter.kinds == ["predictor"] * 3
    short = ForwardMeter(adapter, max_calls=1)
    with pytest.raises(RuntimeError, match="mid-schedule"):
        monotone_unmask(adapter, prompts, lengths, 3, tiny_allowed(), meter=short)
    assert short.nfe == 1


def test_completing_plan_and_yaml_defaults():
    assert worst_case_nfe(4, omega=2, corrector_steps=2) == 4 + 2 * 2
    assert completing_unmask_steps(10, max_calls=5, omega=2, corrector_steps=2) == 3
    assert completing_unmask_steps(10, max_calls=3, omega=1, corrector_steps=4) == 0
    assert DEFAULT_OMEGA == 2 and DEFAULT_CORRECTOR_STEPS == 4
    cfg = yaml.safe_load(open(ROOT_CFG))
    assert cfg["omega"] == DEFAULT_OMEGA and cfg["corrector_steps"] == DEFAULT_CORRECTOR_STEPS
    assert cfg["split"] == "val"
    assert cfg["controls"]["remdm"] == "not_implemented"
    assert len(cfg["family_a_runs"]) == 6


def test_zero_corrector_matches_monotone_and_never_revises():
    adapter = build_tiny_adapter(seed=4)
    exs = make_tiny_examples(5, seed=3)
    for budget in (1, 2, 4):
        mono = run_call_budget(adapter, exs, "monotone", budget, tiny_allowed(), SLOT_LEN, batch_size=2)
        corr = run_call_budget(adapter, exs, "proseco", budget, tiny_allowed(), SLOT_LEN, batch_size=2,
                               omega=2, corrector_steps=0)
        assert corr.responses == mono.responses
        assert corr.calls == mono.calls
        for i, ex in enumerate(exs):
            length = len(ex.response_ids)
            assert corr.calls[i] == min(budget, length)
            assert corr.masks_remaining[i] == 0
            assert MASK not in corr.responses[i]


def test_corrector_sees_a_complete_draft_and_can_revise_a_visible_token():
    """Step 1 reveals without a corrector. Step 2's corrector input has no mask,
    and the already-visible token changes. Monotone unmasking leaves it."""
    prompt, length = [10, 11, 12], 4
    P = len(prompt)

    def logits(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        out[..., 3] = 2.0
        for b in range(ids.shape[0]):
            if int(ids[b, P]) == 3:
                out[b, P - 1, 4] = 8.0
                out[b, P - 1, 3] = 0.0
        return out

    model = Scripted(logits)
    adapter = adapter_of(model)
    run = proseco_unmask(adapter, [prompt], [length], max_calls=3, allowed=tiny_allowed(),
                         omega=2, corrector_steps=1, record_trajectory=True)
    assert model.calls == 3
    assert run.call_kinds == ["predictor", "predictor", "corrector"]
    assert MASK in resp(model.inputs[0], P, length)
    assert MASK in resp(model.inputs[1], P, length)
    assert MASK not in resp(model.inputs[2], P, length)          # temporary complete draft
    assert run.trajectory[0][0].count(MASK) == length - 2        # main state still masked
    assert run.responses[0][0] == 4                              # visible token revised
    assert MASK not in run.responses[0]
    mono = monotone_unmask(adapter_of(Scripted(logits)), [prompt], [length], 2, tiny_allowed())
    assert mono.responses[0][0] == 3
    final = run.final_input_ids
    assert final[0, :P].tolist() == prompt
    assert all(t not in (MASK, PAD, EOS) for t in run.responses[0])


def test_fixed_point_does_not_backfill_and_partial_budget_stops_mid_loop():
    prompt, length = [10, 11, 12], 4

    def copy_or_fill(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        out[..., 3] = 1.0
        for i in range(1, ids.shape[1]):
            for b in range(ids.shape[0]):
                tok = int(ids[b, i])
                if tok != MASK:
                    out[b, i - 1, tok] = 6.0
        return out

    copy_model = Scripted(copy_or_fill)
    run = proseco_unmask(adapter_of(copy_model), [prompt], [length], max_calls=20, allowed=tiny_allowed(),
                         omega=2, corrector_steps=3)
    # Four unmask steps, corrector due on steps 2 and 4, one iteration each (then fixed).
    assert copy_model.calls == 6
    assert run.call_kinds.count("corrector") == 2
    assert run.call_kinds.count("predictor") == 4
    assert 20 - run.batched_forward_calls == 14
    assert MASK not in run.responses[0]

    class Cycle(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, attention_mask):
            self.calls += 1
            return _prefer(input_ids, token=(self.calls % 7) + 1)

    cyc = Cycle()
    partial = proseco_unmask(adapter_of(cyc), [prompt], [length], max_calls=3, allowed=tiny_allowed(),
                             omega=1, corrector_steps=5)
    assert completing_unmask_steps(length, 3, omega=1, corrector_steps=5) == 0
    assert cyc.calls == 3
    assert partial.call_kinds == ["predictor", "corrector", "corrector"]
    assert MASK not in partial.responses[0]
    assert len(set(partial.responses[0])) == 1


def _prefer(ids, token):
    out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
    out[..., token] = 4.0
    return out


def test_sampler_signatures_take_no_targets():
    for fn in (proseco_unmask, d3im_unmask, extra_unmask, monotone_unmask):
        names = set(inspect.signature(fn).parameters)
        assert names.isdisjoint({"targets", "target_ids", "response_ids", "reference_ids", "error_mask"})


def test_targets_do_not_change_tokens_and_test_splits_are_refused():
    adapter = build_tiny_adapter(seed=2)
    exs = make_tiny_examples(3, seed=5)
    for ex in exs:
        ex.split = "val"
    first = run_call_budget(adapter, exs, "proseco", 4, tiny_allowed(), SLOT_LEN, corrector_steps=1)
    changed = copy.deepcopy(exs)
    for ex in changed:
        ex.response_ids = [(t + 1) % 10 for t in ex.response_ids]
        ex.values = [8] * len(ex.values)
    second = run_call_budget(adapter, changed, "proseco", 4, tiny_allowed(), SLOT_LEN, corrector_steps=1)
    assert second.responses == first.responses
    boom = Scripted(lambda ids: (_ for _ in ()).throw(AssertionError("sampler ran")))
    for split in ("test_id", "test_long", "test_unseen"):
        bad = copy.deepcopy(exs)
        for ex in bad:
            ex.split = split
        with pytest.raises(ValueError, match="refuses"):
            run_call_budget(adapter_of(boom), bad, "monotone", 1, tiny_allowed(), SLOT_LEN)
        with pytest.raises(ValueError, match="refuses"):
            require_eval_split(bad)


def test_d3im_remasks_and_a_shorter_horizon_is_not_a_prefix():
    prompt, length = [10, 11, 12], 4
    P = len(prompt)

    def flip(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        for b in range(ids.shape[0]):
            seen = ids[b, P:P + length]
            scores = [0.0, 1.0, 2.0, 5.0] if bool((seen != MASK).any()) else [4.0, 3.0, 2.0, 1.0]
            for rel, score in enumerate(scores):
                out[b, P + rel - 1, 5] = score
        return out

    model = Scripted(flip)
    run = d3im_unmask(adapter_of(model), [prompt], [length], max_calls=3, allowed=tiny_allowed(),
                      record_trajectory=True)
    assert model.calls == 3 and run.call_kinds == ["d3im"] * 3
    assert run.trajectory[0][0] == [5, MASK, MASK, MASK]
    assert run.trajectory[1][0][0] == MASK                      # previously kept token remasked
    assert sorted(t for t in run.trajectory[1][0] if t != MASK) == [5, 5]
    assert MASK not in run.responses[0]
    long = d3im_unmask(adapter_of(Scripted(flip)), [prompt], [length], max_calls=4, allowed=tiny_allowed(),
                       record_trajectory=True)
    short = d3im_unmask(adapter_of(Scripted(flip)), [prompt], [length], max_calls=2, allowed=tiny_allowed())
    assert MASK not in short.responses[0]
    assert long.trajectory[1][0].count(MASK) == 2               # step 2 of horizon 4 keeps K=2
    assert short.responses[0] != long.trajectory[1][0]
    assert keep_count(4, 1, 3) == 1 and keep_count(4, 2, 3) == 2 and keep_count(4, 3, 3) == 4
    assert keep_count(4, 1, 5) == 0


def test_d3im_freezes_earlier_blocks():
    prompt, length = [10, 11, 12], 4
    P = len(prompt)

    def logits(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        for b in range(ids.shape[0]):
            block = ids[b, P:P + 2]
            token = 9 if int(block[0]) != MASK and int(block[1]) != MASK else 3
            out[b, :, token] = 3.0
        return out

    run = d3im_unmask(adapter_of(Scripted(logits)), [prompt], [length], max_calls=4, allowed=tiny_allowed(),
                      block_length=2, record_trajectory=True)
    assert run.responses[0] == [3, 3, 9, 9]
    assert run.trajectory[1][0][:2] == [3, 3]
    assert run.trajectory[2][0][:2] == [3, 3]                   # frozen while the next block is active
    assert run.batched_forward_calls == 4
    with pytest.raises(ValueError, match="divisible"):
        d3im_unmask(adapter_of(Scripted(logits)), [prompt], [length], max_calls=3, allowed=tiny_allowed(),
                    block_length=2)


def test_extra_unmask_spends_the_budget_on_the_current_token():
    prompt, length = [10, 11, 12], 2
    P = len(prompt)

    def logits(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        out[..., 3] = 2.0
        for b in range(ids.shape[0]):
            if bool((ids[b, P:P + length] == MASK).any()):
                continue
            out[b, P, 3] = -4.0          # current token at response position 1
            out[b, P, 4] = 6.0           # new candidate there
        return out

    model = Scripted(logits)
    run = extra_unmask(adapter_of(model), [prompt], [length], max_calls=3, allowed=tiny_allowed())
    assert model.calls == 3
    assert run.call_kinds == ["predictor", "predictor", "revision"]
    assert run.responses[0] == [3, 4]
    assert MASK not in run.responses[0]
    # The matched monotone decode at T=L does not revise.
    mono = monotone_unmask(adapter_of(Scripted(logits)), [prompt], [length], 2, tiny_allowed())
    assert mono.responses[0] == [3, 3]
    full = run_call_budget(build_tiny_adapter(seed=1), make_tiny_examples(2, seed=1), "extra_unmask",
                           6, tiny_allowed(), SLOT_LEN)
    for i, ex in enumerate(make_tiny_examples(2, seed=1)):
        assert full.calls[i] == 6
        assert full.masks_remaining[i] == 0


def test_repair_proseco_first_step_matches_greedy_and_later_steps_count():
    prompt = [10, 11, 12]
    P = len(prompt)

    def logits(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        out[..., 4] = 2.0
        for b in range(ids.shape[0]):
            if int(ids[b, P]) in (4, 5):
                out[b, :, 4] = 0.0
                out[b, :, 5] = 3.0
        return out

    model = Scripted(logits)
    adapter = adapter_of(model)
    draft = _draft("a", prompt, [3, 3], [1, 1])
    greedy = repair_drafts(adapter, [draft], allowed=tiny_allowed())
    assert model.calls == 1
    once = repair_drafts_proseco(adapter, [draft], max_calls=1, allowed=tiny_allowed())
    assert once[0]["repaired_ids"] == greedy[0]["repaired_ids"] == [4, 4]
    assert once[0]["calls"] == 1
    assert "calls" not in greedy[0]
    model.calls = 0
    twice = repair_drafts_proseco(adapter, [draft], max_calls=4, allowed=tiny_allowed())
    assert twice[0]["repaired_ids"] == [5, 5]
    assert twice[0]["calls"] == 3                          # 3→4, 4→5, then the fixed point
    # A draft that is already the argmax stops after one call and is not backfilled.
    stable = _draft("b", prompt, [5, 5], [1, 1])
    model.calls = 0
    held = repair_drafts_proseco(adapter, [stable], max_calls=4, allowed=tiny_allowed())
    assert held[0]["calls"] == 1 and held[0]["repaired_ids"] == [5, 5]


def test_repair_call_charge_is_per_batch():
    prompt = [10, 11, 12]

    def logits(ids):
        out = torch.full((ids.shape[0], ids.shape[1], V), -1.0)
        out[..., 4] = 1.0
        return out

    adapter = adapter_of(Scripted(logits))
    changing = _draft("c", prompt, [3, 3], [0, 0])
    stable = _draft("s", prompt, [4, 4], [0, 0])
    together = repair_drafts_proseco(adapter, [changing, stable], max_calls=3, batch_size=2,
                                    allowed=tiny_allowed())
    assert [r["calls"] for r in together] == [2, 2]
    apart = repair_drafts_proseco(adapter, [changing, stable], max_calls=3, batch_size=1,
                                  allowed=tiny_allowed())
    assert [r["calls"] for r in apart] == [2, 1]


def test_greedy_tie_matches_allowed_argmax():
    rows = torch.zeros(1, 2, V)
    rows[0, :, 5] = 1.0
    rows[0, :, 3] = 1.0
    rows[0, :, MASK] = 9.0
    from diffusion_correction.evaluation.repair import _greedy_response
    layout_editable = torch.tensor([[False, True, True, False]])
    # Build a logit tensor whose editable rows are the tied rows. Use greedy_allowed on rows
    # placed at those positions.
    full = torch.zeros(1, 4, V)
    full[0, 1] = rows[0, 0]
    full[0, 2] = rows[0, 1]
    allowed = tiny_allowed()
    cand, _ = greedy_allowed(full, allowed)
    pred = _greedy_response(full, layout_editable, allowed)
    assert pred == [[3, 3]]
    assert cand[0, 1].item() == 3 and cand[0, 2].item() == 3


def test_score_records_calls_and_cli_refuses_test_splits():
    adapter = build_tiny_adapter(seed=7)
    exs = make_tiny_examples(2, seed=8)
    for ex in exs:
        ex.split = "val"
    run = run_call_budget(adapter, exs, "monotone", 2, tiny_allowed(), SLOT_LEN)
    fmt = ResponseFormat(value_bound=9)
    tf = TokenizedFormat(fmt=fmt, sign_ids={"+": 6, "-": 7}, digit_ids=tuple(range(10)),
                         special_ids=frozenset({MASK, PAD, EOS}), tokenizer_size=V)
    records, agg = score_call_budget(exs, run, tf)
    assert len(records) == 2
    assert agg["method"] == "monotone" and agg["max_calls"] == 2
    assert all(r["calls"] == 2 and r["masks_remaining"] == 0 for r in records)
    assert "prefix" in agg["budget_note"] and "No primary budget is selected" in agg["budget_note"]
    for split in ("test_id", "test_long", "test_unseen"):
        with pytest.raises(SystemExit, match=split):
            main(["evaluate-calls", "--checkpoint", "x", "--data-dir", "y", "--manifest-sha256", "z",
                  "--split", split, "--method", "monotone", "--budgets", "1", "--out", "o"])


def _draft(pid, prompt, draft, ref) -> BankDraft:
    n = len(draft)
    return BankDraft(
        problem_id=pid, split="val", budget="4", budget_code=4, family="arithmetic", length=1,
        operand_digits=1, setting="arithmetic:1:1", prompt_ids=list(prompt), draft_ids=list(draft),
        reference_ids=list(ref), log_c_gen=[0.0] * n, log_q_gen=[0.0] * n,
        mask_fraction_before=[0.0] * n, token_correct=[a == b for a, b in zip(draft, ref)],
        final_correct=False, well_formed=True, n_token_errors=sum(a != b for a, b in zip(draft, ref)),
        shard="", identity_sha256="abc")
