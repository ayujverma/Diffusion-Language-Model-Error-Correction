"""Fixed model-call budgets for full generation (D-049).

Every method is run at a declared ``max_calls`` cap. A point on an accuracy-versus-calls
curve is a separate run at that cap. It is not a prefix of a longer horizon: monotone
quotas, the ProSeCo unmask plan, and D3IM keep-counts all depend on the horizon.

Only ``train`` (wiring checks) and ``val`` are accepted. Test splits are refused, so
a budget cannot be selected from them. Targets are read by the checker after sampling
and are not passed into the sampler.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..env import release_cached_memory, synchronize
from ..models.adapter import DiffusionAdapter
from ..sampling.d3im import d3im_unmask
from ..sampling.extra_unmask import extra_unmask
from ..sampling.monotone import monotone_unmask
from ..sampling.nfe import ForwardMeter
from ..sampling.proseco import DEFAULT_CORRECTOR_STEPS, DEFAULT_OMEGA, proseco_unmask
from ..sampling.results import SamplerRun
from ..sampling.state import masks_remaining
from .generation import GenerationRun, public_response_length, score

EVAL_SPLITS = ("train", "val")
METHODS = ("monotone", "proseco", "d3im", "extra_unmask")

# Written into every summary so a curve is not mistaken for a selected primary budget.
BUDGET_NOTE = (
    "Each budget is an independent horizon under max_calls. "
    "A prefix of a longer run is not the result at the shorter budget. "
    "No primary budget is selected here."
)


@dataclass
class CallBudgetRun:
    method: str
    max_calls: int
    responses: dict[int, list[int]]
    calls: dict[int, int]
    masks_remaining: dict[int, int]
    batched_forward_calls: int
    model_seconds: float
    wall_seconds: float
    omega: int | None = None
    corrector_steps: int | None = None
    block_length: int | None = None


def require_eval_split(examples: list[Example] | tuple[Example, ...]) -> None:
    bad = sorted({ex.split for ex in examples if ex.split not in EVAL_SPLITS})
    if bad:
        raise ValueError(
            f"call-budget eval refuses split(s) {bad}; use val (or train for wiring checks). "
            "test_id, test_long, and test_unseen are never used to select a budget.")


def run_call_budget(adapter: DiffusionAdapter, examples: list[Example], method: str, max_calls: int,
                    allowed: torch.Tensor, slot_len: int, *, batch_size: int = 32,
                    omega: int = DEFAULT_OMEGA, corrector_steps: int = DEFAULT_CORRECTOR_STEPS,
                    block_length: int | None = None) -> CallBudgetRun:
    """Run one method at one call cap. Examples are grouped by response length."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if method != "d3im" and block_length is not None:
        raise ValueError("block_length is a D3IM setting")
    require_eval_split(examples)
    t_wall = time.perf_counter()
    lengths = [public_response_length(ex, slot_len) for ex in examples]
    for ex, n in zip(examples, lengths, strict=True):
        if n != len(ex.response_ids):
            raise ValueError(f"{ex.problem_id}: public length {n} != stored response length")
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        groups.setdefault(n, []).append(i)
    responses: dict[int, list[int]] = {}
    calls: dict[int, int] = {}
    masks: dict[int, int] = {}
    n_fwd, model_s = 0, 0.0
    device = adapter.device
    for _length, idx in sorted(groups.items()):
        for k in range(0, len(idx), batch_size):
            chunk = idx[k:k + batch_size]
            prompts = [examples[i].prompt_ids for i in chunk]
            resp_len = [lengths[i] for i in chunk]
            synchronize(device)
            t0 = time.perf_counter()
            run = _dispatch(adapter, method, prompts, resp_len, max_calls, allowed, omega,
                            corrector_steps, block_length)
            synchronize(device)
            model_s += time.perf_counter() - t0
            release_cached_memory(device)
            n_fwd += run.batched_forward_calls
            for j, i in enumerate(chunk):
                responses[i] = run.responses[j]
                calls[i] = run.calls_per_example[j]
                masks[i] = run.masks_remaining[j]
                if calls[i] > max_calls:
                    raise AssertionError(f"example {i} used {calls[i]} calls, cap is {max_calls}")
    return CallBudgetRun(
        method=method, max_calls=max_calls, responses=responses, calls=calls, masks_remaining=masks,
        batched_forward_calls=n_fwd, model_seconds=model_s, wall_seconds=time.perf_counter() - t_wall,
        omega=omega if method == "proseco" else None,
        corrector_steps=corrector_steps if method == "proseco" else None,
        block_length=block_length if method == "d3im" else None)


def _dispatch(adapter, method, prompts, lengths, max_calls, allowed, omega, corrector_steps, block_length):
    if method == "monotone":
        # D-029 cannot spend more calls than the response length. The remainder is unused.
        calls = min(max_calls, lengths[0])
        meter = ForwardMeter(adapter, max_calls=max_calls)
        mono = monotone_unmask(adapter, prompts, lengths, calls, allowed, meter=meter)
        return SamplerRun(
            responses=mono.responses, calls_per_example=mono.calls_per_example,
            batched_forward_calls=mono.batched_forward_calls, call_kinds=meter.kinds,
            max_calls=max_calls,
            masks_remaining=masks_remaining(mono.responses, adapter.tokens.mask_token_id),
            final_input_ids=mono.final_input_ids)
    if method == "proseco":
        return proseco_unmask(adapter, prompts, lengths, max_calls=max_calls, allowed=allowed,
                              omega=omega, corrector_steps=corrector_steps)
    if method == "d3im":
        return d3im_unmask(adapter, prompts, lengths, max_calls=max_calls, allowed=allowed,
                           block_length=block_length)
    if method == "extra_unmask":
        return extra_unmask(adapter, prompts, lengths, max_calls=max_calls, allowed=allowed)
    raise AssertionError(method)


def score_call_budget(examples: list[Example], run: CallBudgetRun, tf: TokenizedFormat) -> tuple[list[dict], dict]:
    """Checker scores after sampling. Permuting targets does not change ``run.responses``."""
    require_eval_split(examples)
    gen = GenerationRun(budget=run.max_calls, responses=run.responses, calls=run.calls, infeasible=[],
                        batched_forward_calls=run.batched_forward_calls, model_seconds=run.model_seconds,
                        wall_seconds=run.wall_seconds)
    records, agg = score(examples, gen, tf)
    by_id: dict[str, int] = {}
    for i, ex in enumerate(examples):
        if ex.problem_id in by_id:
            raise ValueError(f"duplicate problem_id {ex.problem_id}")
        by_id[ex.problem_id] = i
    for rec in records:
        i = by_id[rec["problem_id"]]
        rec["method"] = run.method
        rec["max_calls"] = run.max_calls
        rec["calls_unused"] = run.max_calls - rec["calls"]
        rec["masks_remaining"] = run.masks_remaining[i]
        rec["omega"] = run.omega
        rec["corrector_steps"] = run.corrector_steps
        rec["block_length"] = run.block_length
    agg["method"] = run.method
    agg["max_calls"] = run.max_calls
    agg["calls_unused_total"] = sum(run.max_calls - c for c in run.calls.values())
    agg["budget_note"] = BUDGET_NOTE
    agg["model_seconds"] = round(run.model_seconds, 6)
    agg["wall_seconds"] = round(run.wall_seconds, 6)
    return records, agg
